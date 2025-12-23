# CLEETS-SMART Dashboard
# EV chargers + flood overlays + RCSP routing (optimized)
# Adapted from:
# Naeima (2022). Correlation-between-Female-Fertility-and-Employment-Status.
# GitHub repository: https://github.com/Naeima/Correlation-between-Female-Fertility-and-Employment-Status

import io
import os
import time
import json
import tempfile
import math
import heapq
from io import StringIO, BytesIO
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import dash
from dash import Dash, dcc, html, Input, Output, State
from flask import Response

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.ops import split as shp_split
from functools import lru_cache

import plotly.graph_objs as go
import folium
from folium.plugins import MarkerCluster, Draw, BeautifyIcon
from folium.raster_layers import WmsTileLayer

from osmnx.distance import nearest_nodes
from dash import exceptions
import os
import gdown
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import xarray as xr
import plotly.express as px


def heat_colour(val, vmin=5, vmax=25):
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    rgba = cm.get_cmap("inferno")(norm(val))
    return mcolors.to_hex(rgba)

def value_to_hex(val, vmin=5, vmax=25, cmap="inferno"):
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    rgba = cm.get_cmap(cmap)(norm(val))
    return mcolors.to_hex(rgba)

# =========================
# Heat datasets 
# =========================

DATA_DIR = "heat_data"
os.makedirs(DATA_DIR, exist_ok=True)

HEAT_DATASETS = {
    "HEAT 2020–2030": {
        "fid": "10O85ZkQOLcBBOjvYcFQsjjR0AYsVHiE_",
        "path": os.path.join(DATA_DIR, "heat_2020_2030.nc"),
    },
    "HEAT 2030–2040": {
        "fid": "1OOEUaevo0VVUE5MPZtFKrarfB0RK6Kec",
        "path": os.path.join(DATA_DIR, "heat_2030_2040.nc"),
    },
    "HEAT 2040–2050": {
        "fid": "128chSSQv_O9wiv_f3lBGppIQ3cGqGlPp",
        "path": os.path.join(DATA_DIR, "heat_2040_2050.nc"),
    },
}

# Download once if missing
for label, meta in HEAT_DATASETS.items():
    if not os.path.exists(meta["path"]):
        gdown.download(
            f"https://drive.google.com/uc?id={meta['fid']}",
            meta["path"],
            quiet=False,
        )

# Convenience lookup used by callbacks
HEAT_FILES = {label: meta["path"] for label, meta in HEAT_DATASETS.items()}

HEAT_TS_COLORS = {
    "HEAT 2020–2030": "#1f77b4",  # blue
    "HEAT 2030–2040": "#d62728",  # red
    "HEAT 2040–2050": "#9467bd",  # purple
}

# =========================
# Google Drive helpers
# =========================

def _gd_url(x: str) -> str:
    """
    Build a proper download URL for Google Drive / Google Sheets.

    - If it's a Google Sheets URL: force CSV export.
    - If it's a generic drive 'file/d' URL: use uc?export=download&id=...
    - If it's a bare id: assume Google Sheets and export as CSV.
    """
    x = str(x).strip()

    # Full URL
    if "drive.google.com" in x or "docs.google.com" in x:
        # Google Sheets URL
        if "spreadsheets" in x and "/d/" in x:
            fid = x.split("/d/")[1].split("/")[0]
            return f"https://docs.google.com/spreadsheets/d/{fid}/export?format=csv"

        # Generic Drive file URL
        if "/file/d/" in x:
            fid = x.split("/file/d/")[1].split("/")[0]
            return f"https://drive.google.com/uc?export=download&id={fid}"

        # Fallback: pass through
        return x

    # Bare id: treat as Google Sheet and export CSV
    return f"https://docs.google.com/spreadsheets/d/{x}/export?format=csv"


def _requests_session():
    sess = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
    )
    sess.mount("https://", HTTPAdapter(max_retries=retry))
    sess.headers.update({"User-Agent": "Mozilla/5.0 (EV-Dashboard)"})
    return sess


def read_csv_resilient_gdrive(file_id_or_url: str, **kw) -> pd.DataFrame:
    url = _gd_url(file_id_or_url)
    sess = _requests_session()
    try:
        r = sess.get(url, timeout=30, stream=True)
        r.raise_for_status()
        token = next(
            (v for k, v in r.cookies.items() if k.startswith("download_warning")),
            None,
        )
        if token:
            r = sess.get(url, params={"confirm": token}, timeout=30, stream=True)
            r.raise_for_status()
        return pd.read_csv(io.BytesIO(r.content), low_memory=False, **kw)
    except Exception as e:
        raise RuntimeError(f"Google Drive fetch failed: {e}")


def read_bytes_resilient_gdrive(file_id_or_url: str) -> bytes:
    url = _gd_url(file_id_or_url)
    sess = _requests_session()
    r = sess.get(url, timeout=30, stream=True)
    r.raise_for_status()
    token = next(
        (v for k, v in r.cookies.items() if k.startswith("download_warning")),
        None,
    )
    if token:
        r = sess.get(url, params={"confirm": token}, timeout=30, stream=True)
        r.raise_for_status()
    return r.content

# =========================
# Optional libs / config
# =========================

MAPBOX_API_KEY = os.environ.get(
    "MAPBOX_API_KEY",
    os.environ.get(
        "pk.eyJ1IjoibmFlaW1hIiwiYSI6ImNsNDRoa295ZDAzMmkza21tdnJrNWRqNmwifQ.-cUTmhr1Q03qUXJfQoIKGQ",
        "",
    ).strip(),
)

try:
    import pydeck as pdk
    HAS_PYDECK = True
except Exception:
    HAS_PYDECK = False

try:
    import osmnx as ox
    import networkx as nx
    HAS_OSMNX = True
except Exception:
    HAS_OSMNX = False

# =========================
# Geocoding helpers
# =========================

def geocode_postcode_uk(pc: str):
    """Fast UK postcode lookup (Postcodes.io). Returns (lat, lon) or None."""
    try:
        pc = (pc or "").strip()
        if not pc:
            return None
        url = f"https://api.postcodes.io/postcodes/{requests.utils.quote(pc)}"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            j = r.json()
            res = (j.get("result") or {})
            lat = res.get("latitude")
            lon = res.get("longitude")
            if lat is not None and lon is not None:
                return float(lat), float(lon)
    except Exception:
        pass
    return None


def geocode_text_osm(q: str):
    """General forward geocoder (Nominatim). Returns (lat, lon) or None."""
    try:
        q = (q or "").strip()
        if not q:
            return None
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": q, "format": "jsonv2", "limit": 1, "addressdetails": 0}
        r = requests.get(
            url,
            params=params,
            headers={"User-Agent": "CLEETS-EV/1.0"},
            timeout=15,
        )
        if r.status_code == 200:
            arr = r.json()
            if arr:
                return float(arr[0]["lat"]), float(arr[0]["lon"])
    except Exception:
        pass
    return None


def geocode_start_end(start_pc_or_text: str, end_pc_or_text: str):
    """Try UK postcode first; fall back to OSM text search."""
    s = geocode_postcode_uk(start_pc_or_text) or geocode_text_osm(start_pc_or_text)
    e = geocode_postcode_uk(end_pc_or_text) or geocode_text_osm(end_pc_or_text)
    return s, e

# =========================
# Global config + caches
# =========================

EV_GDRIVE_FILE_ID = "1P3smzZTMBbLzM7F49wkOJivNBbTqFd1m"

CACHE_DIR = ".cache_wfs"
os.makedirs(CACHE_DIR, exist_ok=True)

GRAPH_CACHE_DIR = os.path.join(CACHE_DIR, "graphs")
FLOOD_CACHE_DIR = os.path.join(CACHE_DIR, "flood_unions")
os.makedirs(GRAPH_CACHE_DIR, exist_ok=True)
os.makedirs(FLOOD_CACHE_DIR, exist_ok=True)

LOGO_GDRIVE_FILE_OR_URL = (
    "https://drive.google.com/file/d/1QLQPln4dRyWXh65E5ua_rC3CGTChwKxc/view?usp=sharing"
)
LOGO_CACHE_PATH = os.path.join(CACHE_DIR, "cleets_logo-01.png")

OWS_BASE = "https://datamap.gov.wales/geoserver/ows"

FRAW_WMS = {
    "FRAW – Rivers": "inspire-nrw:NRW_FLOOD_RISK_FROM_RIVERS",
    "FRAW – Sea": "inspire-nrw:NRW_FLOOD_RISK_FROM_SEA",
    "FRAW – Surface": "inspire-nrw:NRW_FLOOD_RISK_FROM_SURFACE_WATER_SMALL_WATERCOURSES",
}
FMFP_WMS = {
    "FMfP – Rivers & Sea": "inspire-nrw:NRW_FLOODZONE_RIVERS_SEAS_MERGED",
    "FMfP – Surface/Small Watercourses": "inspire-nrw:NRW_FLOODZONE_SURFACE_WATER_AND_SMALL_WATERCOURSES",
}
LIVE_WMS = {
    "Live – Warning Areas": "inspire-nrw:NRW_FLOOD_WARNING",
    "Live – Alert Areas": "inspire-nrw:NRW_FLOOD_WATCH_AREAS",
}
CONTEXT_WMS = {"Historic Flood Extents": "inspire-nrw:NRW_HISTORIC_FLOODMAP"}

FRAW_WFS = {
    "FRAW Rivers": "inspire-nrw:NRW_FLOOD_RISK_FROM_RIVERS",
    "FRAW Sea": "inspire-nrw:NRW_FLOOD_RISK_FROM_SEA",
    "FRAW Surface": "inspire-nrw:NRW_FLOOD_RISK_FROM_SURFACE_WATER_SMALL_WATERCOURSES",
}
FMFP_WFS = {
    "FMfP Rivers & Sea": "inspire-nrw:NRW_FLOODZONE_RIVERS_SEAS_MERGED",
    "FMfP Surface/Small": "inspire-nrw:NRW_FLOODZONE_SURFACE_WATER_AND_SMALL_WATERCOURSES",
}
LIVE_WFS = {
    "Warnings": "inspire-nrw:NRW_FLOOD_WARNING",
    "Alerts": "inspire-nrw:NRW_FLOOD_WATCH_AREAS",
}

SIM_DEFAULTS = dict(
    start_lat=51.4816,
    start_lon=-3.1791,
    end_lat=51.6214,
    end_lon=-3.9436,
    battery_kwh=64.0,
    init_soc=90.0,
    reserve_soc=10.0,
    target_soc=80.0,
    kwh_per_km=0.18,
    max_charger_offset_km=1.5,
    min_leg_km=20.0,
    route_buffer_m=30,
    wfs_pad_m=800,
    wfs_pad_m_fast=120,
    soc_step_normal=0.05,
    soc_step_fast=0.10,
)

FAST_MODE_DEFAULT = bool(int(os.getenv("ONS_FAST_MODE", "0")))

# RCSP knobs
SOC_STEP = 0.025
CHARGE_STEP = 0.10
DEFAULT_POWER_KW = 50.0
BASE_RISK_PENALTY_PER_KM = 60.0
EXTREME_RISK_PENALTY_PER_KM = 240.0
EXTREME_BUFFER_M = 60.0
MAX_GRAPH_BBOX_DEG = 1.0
ROUTE_BUFFER_M = 30

ZONE_COLORS = {
    "Zone 3": "#D32F2F",
    "High": "#D32F2F",
    "Zone 2": "#FFC107",
    "Medium": "#FFC107",
    "Zone 1": "#2E7D32",
    "Low": "#2E7D32",
    "Very Low": "#2E7D32",
    "Outside": "#2E7D32",
    "Unknown": "#2E7D32",
}
ZONE_PRIORITY = ["Zone 3", "High", "Zone 2", "Medium", "Zone 1", "Low", "Very Low", "Outside", "Unknown"]
_PRI = {z: i for i, z in enumerate(ZONE_PRIORITY)}

# Optimisation helper: thinning
MAX_FOLIUM_POINTS = 5000      # max chargers on overview map
MAX_ROUTE_POINTS = 3000       # max chargers on route map
ENABLE_ROUTE_FLOOD_UNION = False  # expensive flood union for routing (False = fast)

# =========================
# Vehicle presets
# =========================

@dataclass
class EVParams:
    battery_kWh: float = 75.0
    start_soc: float = 0.80
    reserve_soc: float = 0.10
    target_soc: float = 0.80
    kWh_per_km: float = 0.20
    max_charge_kW: float = 120.0


@dataclass
class StopInfo:
    lat: float
    lon: float
    name: str = "Charger"
    postcode: str = ""
    ZoneLabel: str = "Outside"
    ZoneColor: str = ZONE_COLORS["Outside"]
    Operational: bool = True
    soc_before: float = 0.0
    soc_after: float = 0.0
    energy_kWh: float = 0.0
    charge_time_min: float = 0.0

VEHICLE_PRESETS = {
    "Hatchback": {
        "Nissan Leaf 40":   {"battery_kWh": 40.0, "kWh_per_km": 0.16, "max_charge_kW": 50.0},
        "Renault Zoe R135": {"battery_kWh": 52.0, "kWh_per_km": 0.15, "max_charge_kW": 50.0},
        "VW ID.3 Pro":      {"battery_kWh": 58.0, "kWh_per_km": 0.16, "max_charge_kW": 120.0},
    },
    "Sedan": {
        "Tesla Model 3 RWD":   {"battery_kWh": 57.5, "kWh_per_km": 0.145, "max_charge_kW": 170.0},
        "Hyundai Ioniq 6":     {"battery_kWh": 77.4, "kWh_per_km": 0.14,  "max_charge_kW": 220.0},
        "Polestar 2 Long":     {"battery_kWh": 82.0, "kWh_per_km": 0.17,  "max_charge_kW": 155.0},
    },
    "SUV / Crossover": {
        "Kia EV6 RWD":         {"battery_kWh": 77.4, "kWh_per_km": 0.18, "max_charge_kW": 230.0},
        "Hyundai Kona 64":     {"battery_kWh": 64.0, "kWh_per_km": 0.155, "max_charge_kW": 75.0},
        "VW ID.4 Pro":         {"battery_kWh": 77.0, "kWh_per_km": 0.19, "max_charge_kW": 125.0},
    },
    "Van / MPV": {
        "VW ID. Buzz":         {"battery_kWh": 77.0, "kWh_per_km": 0.23, "max_charge_kW": 170.0},
        "Peugeot e-Traveller": {"battery_kWh": 75.0, "kWh_per_km": 0.26, "max_charge_kW": 100.0},
        "Renault Kangoo E-Tech": {"battery_kWh": 45.0, "kWh_per_km": 0.20, "max_charge_kW": 80.0},
    },
}

# =========================
# Utilities
# =========================

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * R * math.asin(math.sqrt(a))


def bbox_expand(bounds, pad_m):
    minx, miny, maxx, maxy = bounds
    pad_deg = max(0.002, pad_m / 111_320.0)
    return (minx - pad_deg, miny - pad_deg, maxx + pad_deg, maxy + pad_deg)


def _bbox_for(df_like, pad_m=SIM_DEFAULTS["wfs_pad_m"]):
    if isinstance(df_like, gpd.GeoDataFrame) and not df_like.empty:
        minx, miny, maxx, maxy = df_like.total_bounds
    else:
        minx, miny, maxx, maxy = gdf_ev.total_bounds
    return bbox_expand((minx, miny, maxx, maxy), pad_m)

def coords_2d(coords):
    """Safely drop Z if present (lon, lat[, z]) → (lon, lat)."""
    return [(c[0], c[1]) for c in coords]

def _cache_path(layer, bbox):
    safe = f"{layer}_{bbox[0]:.4f}_{bbox[1]:.4f}_{bbox[2]:.4f}_{bbox[3]:.4f}".replace(
        ":", "_"
    ).replace(",", "_")
    return os.path.join(CACHE_DIR, f"{safe}.geojson")


def fetch_wfs_layer_cached(layer, bbox, ttl_h=48):
    p = _cache_path(layer, bbox)
    if os.path.exists(p) and time.time() - os.path.getmtime(p) < ttl_h * 3600:
        try:
            gj = json.load(open(p, "r", encoding="utf-8"))
            return gpd.GeoDataFrame.from_features(
                gj.get("features", []), crs="EPSG:4326"
            )
        except Exception:
            pass
    from urllib.parse import urlencode

    params = {
        "service": "WFS",
        "request": "GetFeature",
        "version": "2.0.0",
        "typenames": layer,
        "outputFormat": "application/json",
        "srsName": "EPSG:4326",
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},EPSG:4326",
    }
    url = f"{OWS_BASE}?{urlencode(params)}"
    try:
        gj = requests.get(url, timeout=30).json()
        with open(p, "w", encoding="utf-8") as f:
            json.dump(gj, f)
        return gpd.GeoDataFrame.from_features(
            gj.get("features", []), crs="EPSG:4326"
        )
    except Exception:
        if os.path.exists(p):
            try:
                gj = json.load(open(p, "r", encoding="utf-8"))
                return gpd.GeoDataFrame.from_features(
                    gj.get("features", []), crs="EPSG:4326"
                )
            except Exception:
                pass
        return gpd.GeoDataFrame(
            columns=["geometry"], geometry="geometry", crs="EPSG:4326"
        )

# =========================
# Weather
# =========================

@lru_cache(maxsize=128)
def cached_get(url, headers_tuple=(), params_tuple=()):
    headers = dict(headers_tuple) if headers_tuple else {}
    params = dict(params_tuple) if params_tuple else {}
    r = requests.get(url, headers=headers, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def _ts():
    # 5-minute buckets to keep cache keys stable but refresh periodically
    return str(int(time.time() // 300))

METOFFICE_KEY = os.environ.get("METOFFICE_KEY", "").strip()
METOFFICE_SITE_API = os.environ.get("METOFFICE_SITE_API", "").strip()  # optional

def get_weather(lat=51.48, lon=-3.18):
    """
    If Met Office key + site API are configured, use that; otherwise fall back to Open-Meteo.
    Returns dict: {"provider": "...", "raw": {...}} or {"provider":"error", "error": "..."}.
    """
    try:
        if METOFFICE_KEY and METOFFICE_SITE_API:
            headers = (("apikey", METOFFICE_KEY),)
            params = (("latitude", str(lat)), ("longitude", str(lon)))
            data = cached_get(METOFFICE_SITE_API, headers, params)
            return {"provider": "Met Office", "raw": data}
        else:
            url = "https://api.open-meteo.com/v1/forecast"
            params = (
                ("latitude", str(lat)), ("longitude", str(lon)),
                ("current", "temperature_2m,precipitation,wind_speed_10m"),
                ("hourly", "temperature_2m,precipitation_probability,wind_speed_10m"),
                ("timezone", "Europe/London"),
                ("_ts", _ts()),
            )
            data = cached_get(url, (), params)
            return {"provider": "Open-Meteo", "raw": data}
    except Exception as e:
        return {"provider": "error", "error": str(e)}

def _parse_metoffice_timeseries(raw):
    """
    Normalise Met Office time-series into {time[], temp[], pop[]} for plotting.
    """
    try:
        feats = raw.get("features") or []
        if feats and isinstance(feats, list):
            ts = feats[0].get("properties", {}).get("timeSeries") or []
            times = [r.get("time") for r in ts if "time" in r][:24]
            temps = [r.get("screenTemperature") for r in ts][:24]
            pops  = [r.get("precipitationProbability") or r.get("precipProb") for r in ts][:24]
            if times and temps:
                return {
                    "time": times,
                    "temp": temps,
                    "pop": pops or [None] * len(times),
                }
    except Exception:
        pass
    return {}

# =========================
# Flood model zones
# =========================

def _norm_zone(props: dict, layer_name: str) -> str:
    txt = " ".join([str(v) for v in props.values() if v is not None]).lower()
    if "zone 3" in txt:
        return "Zone 3"
    if "zone 2" in txt:
        return "Zone 2"
    if "zone 1" in txt:
        return "Zone 1"
    if "very low" in txt:
        return "Very Low"
    if "high" in txt:
        return "High"
    if "medium" in txt:
        return "Medium"
    if "low" in txt:
        return "Low"
    return "Unknown"


def fetch_model_zones_gdf(ev_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    bbox = _bbox_for(ev_gdf, pad_m=SIM_DEFAULTS.get("wfs_pad_m", 800))
    chunks = []
    for title, layer in {**FMFP_WFS, **FRAW_WFS}.items():
        g = fetch_wfs_layer_cached(layer, bbox)
        if g.empty:
            continue
        props_df = g.drop(columns=["geometry"], errors="ignore")
        zlabs = [_norm_zone(r.to_dict(), title) for _, r in props_df.iterrows()]
        g = g.assign(
            zone=zlabs,
            color=[ZONE_COLORS.get(z, "#2E7D32") for z in zlabs],
            model=title,
        )
        try:
            g["geometry"] = g["geometry"].buffer(0)
        except Exception:
            pass
        try:
            g = g.explode(index_parts=False).reset_index(drop=True)
        except Exception:
            pass
        chunks.append(g[["zone", "color", "model", "geometry"]])
    if not chunks:
        return gpd.GeoDataFrame(
            columns=["zone", "color", "model", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )
    G = pd.concat(chunks, ignore_index=True)
    return gpd.GeoDataFrame(G, geometry="geometry", crs="EPSG:4326")


def compute_model_zones_for_points(ev_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    zones = fetch_model_zones_gdf(ev_gdf)
    out = ev_gdf[["ROW_ID"]].copy()
    out["ZoneLabel"] = "Outside"
    out["ZoneColor"] = ZONE_COLORS["Outside"]
    if zones.empty or ev_gdf.empty:
        return out[["ROW_ID", "ZoneLabel", "ZoneColor"]]
    try:
        ev_m = ev_gdf.to_crs("EPSG:27700")
        zn_m = zones.to_crs("EPSG:27700")
    except Exception:
        ev_m = ev_gdf.to_crs("EPSG:3857")
        zn_m = zones.to_crs("EPSG:3857")
    try:
        joined = gpd.sjoin(
            ev_m[["ROW_ID", "geometry"]],
            zn_m,
            how="left",
            predicate="within",
        )
    except Exception:
        joined = gpd.sjoin(
            ev_m[["ROW_ID", "geometry"]],
            zn_m,
            how="left",
            predicate="intersects",
        )
    if joined.empty:
        return out[["ROW_ID", "ZoneLabel", "ZoneColor"]]
    joined["pri"] = joined["zone"].map(_PRI).fillna(_PRI["Unknown"])
    idx = (
        joined.sort_values(["ROW_ID", "pri"])
        .groupby("ROW_ID", as_index=False)
        .first()
    )
    lut = idx.set_index("ROW_ID")
    out.loc[out["ROW_ID"].isin(lut.index), "ZoneLabel"] = lut["zone"]
    out.loc[out["ROW_ID"].isin(lut.index), "ZoneColor"] = (
        lut["zone"].map(ZONE_COLORS).fillna("#2E7D32")
    )
    return out[["ROW_ID", "ZoneLabel", "ZoneColor"]]


def safe_compute_zones():
    try:
        return compute_model_zones_for_points(gdf_ev)
    except Exception:
        return pd.DataFrame(
            {
                "ROW_ID": gdf_ev["ROW_ID"],
                "ZoneLabel": "Outside",
                "ZoneColor": ZONE_COLORS["Outside"],
            }
        )

# =========================
# Load EV data
# =========================

df = read_csv_resilient_gdrive(EV_GDRIVE_FILE_ID)

TARGET_AREAS = [
    "Blaenau Gwent",
    "Bridgend",
    "Caerphilly",
    "Cardiff",
    "Carmarthenshire",
    "Merthyr Tydfil",
    "Monmouthshire",
    "Neath Port Talbot",
    "Newport",
    "Pembrokeshire",
    "Rhondda Cynon Taf",
    "Swansea",
    "The Vale Of Glamorgan",
    "Torfaen",
]
area_col = (
    "country"
    if "country" in df.columns
    else ("adminArea" if "adminArea" in df.columns else "town")
)
df[area_col] = df[area_col].astype(str).str.strip().str.title()

df["Latitude"] = pd.to_numeric(df.get("latitude", df.get("Latitude")), errors="coerce")
df["Longitude"] = pd.to_numeric(df.get("longitude", df.get("Longitude")), errors="coerce")
df = df.dropna(subset=["Latitude", "Longitude"])
df["country"] = df[area_col]


def classify_availability(s):
    s = str(s).lower().strip()
    if any(
        k in s
        for k in [
            "available",
            "in service",
            "operational",
            "working",
            "ok",
            "service",
        ]
    ):
        return True
    if any(
        k in s
        for k in [
            "not operational",
            "fault",
            "out of service",
            "offline",
            "unavailable",
            "down",
        ]
    ):
        return False
    return None


_df_status = df.get("chargeDeviceStatus", pd.Series(index=df.index))
df["Available"] = _df_status.apply(classify_availability)
df["AvailabilityLabel"] = (
    df["Available"]
    .map({True: "Operational", False: "Not operational"})
    .fillna("Unknown")
)
df["Operator"] = df.get("deviceControllerName", df.get("Operator", "Unknown"))
df["Postcode"] = df.get("postcode", df.get("Postcode", "N/A"))
df["dateCreated"] = pd.to_datetime(
    df.get("dateCreated", df.get("DateCreated")),
    errors="coerce",
    dayfirst=True,
)

df["geometry"] = [Point(xy) for xy in zip(df["Longitude"], df["Latitude"])]
gdf_ev = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
gdf_ev["ROW_ID"] = gdf_ev.index.astype(int)
country_OPTIONS = sorted(
    [t for t in gdf_ev["country"].dropna().astype(str).unique() if t]
)

# =========================
# Routing helpers
# =========================

def _requests_session_osrm():
    s = requests.Session()
    r = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )
    s.mount("https://", HTTPAdapter(max_retries=r))
    s.headers.update({"User-Agent": "CLEETS-EV/1.0"})
    return s


def _osrm_try(base_url, sl, so, el, eo, want_steps=True):
    url = f"{base_url}/route/v1/driving/{so},{sl};{eo},{el}"
    params = {
        "overview": "full",
        "geometries": "geojson",
        "alternatives": "false",
        "steps": "true" if want_steps else "false",
    }
    sess = _requests_session_osrm()
    r = sess.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if not data.get("routes"):
        raise RuntimeError("No routes")
    rt = data["routes"][0]
    coords = rt["geometry"]["coordinates"]
    ln = LineString([(c[0], c[1]) for c in coords])
    dist_m = float(rt["distance"])
    dur_s = float(rt["duration"])
    steps = []
    if want_steps:
        for leg in rt.get("legs", []):
            for st in leg.get("steps", []):
                nm = st.get("name") or ""
                m = st.get("maneuver", {})
                kind = m.get("modifier") or m.get("type") or ""
                t = " ".join([w for w in [kind, nm] if w]).strip()
                if t:
                    steps.append(t)
    return ln, dist_m, dur_s, steps


def osrm_route(sl, so, el, eo):
    for base in (
        "https://router.project-osrm.org",
        "https://routing.openstreetmap.de/routed-car",
    ):
        try:
            ln, d, t, steps = _osrm_try(base, sl, so, el, eo, want_steps=True)
            return ln, d, t, steps, base
        except Exception:
            continue
    raise RuntimeError("OSRM routing failed on both endpoints")


def get_flood_union(bounds, include_live=True, include_fraw=True,
                    include_fmfp=True, pad_m=SIM_DEFAULTS["wfs_pad_m"]):
    bbox = bbox_expand(bounds, pad_m)
    chunks = []
    if include_fmfp:
        for lyr in FMFP_WFS.values():
            g = fetch_wfs_layer_cached(lyr, bbox)
            if not g.empty:
                chunks.append(g[["geometry"]])
    if include_fraw:
        for lyr in FRAW_WFS.values():
            g = fetch_wfs_layer_cached(lyr, bbox)
            if not g.empty:
                chunks.append(g[["geometry"]])
    if include_live:
        for lyr in LIVE_WFS.values():
            g = fetch_wfs_layer_cached(lyr, bbox)
            if not g.empty:
                chunks.append(g[["geometry"]])
    if not chunks:
        return None
    G = gpd.GeoDataFrame(
        pd.concat(chunks, ignore_index=True),
        geometry="geometry",
        crs="EPSG:4326",
    )
    try:
        G["geometry"] = G["geometry"].buffer(0)
    except Exception:
        pass
    try:
        G = G.explode(index_parts=False).reset_index(drop=True)
    except Exception:
        pass
    try:
        return G.to_crs("EPSG:27700").union_all()
    except Exception:
        return G.to_crs("EPSG:27700").unary_union


def _graph_point_cache_path(lat, lon, dist_m):
    return os.path.join(
        GRAPH_CACHE_DIR,
        f"pt_{round(lat,5)}_{round(lon,5)}_{int(dist_m)}.graphml",
    )


def _graph_bbox_cache_path(north, south, east, west):
    key = f"bbox_{round(north,5)}_{round(south,5)}_{round(east,5)}_{round(west,5)}.graphml"
    return os.path.join(GRAPH_CACHE_DIR, key)


def _ox_save_graphml(G, path):
    try:
        return ox.io.save_graphml(G, path)
    except Exception:
        try:
            return ox.save_graphml(G, path)
        except Exception:
            return None


def _ox_load_graphml(path):
    try:
        return ox.io.load_graphml(path)
    except Exception:
        try:
            return ox.load_graphml(path)
        except Exception:
            return None


def graph_from_point_cached(lat, lon, dist_m=15000, ttl_days=30):
    if not HAS_OSMNX:
        raise RuntimeError("OSMnx not installed")
    path = _graph_point_cache_path(lat, lon, dist_m)
    if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < ttl_days * 86400:
        G = _ox_load_graphml(path)
        if G is not None:
            return G
    G = ox.graph_from_point((lat, lon), dist=dist_m, network_type="drive", simplify=True)
    try:
        _ox_save_graphml(G, path)
    except Exception:
        pass
    return G


def graph_from_bbox_cached(north, south, east, west, ttl_days=30):
    if not HAS_OSMNX:
        raise RuntimeError("OSMnx not installed")
    path = _graph_bbox_cache_path(north, south, east, west)
    if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < ttl_days * 86400:
        G = _ox_load_graphml(path)
        if G is not None:
            return G
    try:
        G = ox.graph_from_bbox(
            north=north,
            south=south,
            east=east,
            west=west,
            network_type="drive",
            simplify=True,
        )
    except TypeError:
        try:
            G = ox.graph_from_bbox(
                (north, south, east, west),
                network_type="drive",
                simplify=True,
            )
        except TypeError:
            G = ox.graph_from_bbox(north, south, east, west, "drive", True)
    try:
        _ox_save_graphml(G, path)
    except Exception:
        pass
    return G


def _build_graph_bbox(north, south, east, west):
    try:
        return graph_from_bbox_cached(north, south, east, west)
    except TypeError:
        pass
    try:
        return ox.graph_from_bbox(
            (north, south, east, west),
            network_type="drive",
            simplify=True,
        )
    except TypeError:
        pass
    try:
        return ox.graph_from_bbox(north, south, east, west, "drive", True)
    except TypeError as e:
        raise RuntimeError(f"OSMnx graph_from_bbox signature not recognised: {e}")


def _graph_two_points(sl, so, el, eo, dist_m=15000):
    G1 = graph_from_point_cached(sl, so, dist_m)
    G2 = graph_from_point_cached(el, eo, dist_m)
    try:
        return nx.compose(G1, G2)
    except Exception:
        G = nx.MultiDiGraph()
        G.update(G1)
        G.update(G2)
        return G


def segment_route_by_risk(line_wgs84, risk_union_metric, buffer_m=ROUTE_BUFFER_M):
    if risk_union_metric is None:
        return [line_wgs84], []
    try:
        line_m = (
            gpd.GeoSeries([line_wgs84], crs="EPSG:4326")
            .to_crs("EPSG:27700")
            .iloc[0]
        )
    except Exception:
        line_m = (
            gpd.GeoSeries([line_wgs84], crs="EPSG:4326")
            .to_crs("EPSG:3857")
            .iloc[0]
        )
    hit = risk_union_metric.buffer(buffer_m)
    try:
        pieces = list(shp_split(line_m, hit.boundary))
    except Exception:
        pieces = [line_m]
    safe_m, risk_m = [], []
    for seg in pieces:
        (risk_m if seg.intersects(hit) else safe_m).append(seg)
    safe = (
        gpd.GeoSeries(safe_m, crs="EPSG:27700")
        .to_crs("EPSG:4326")
        .tolist()
        if safe_m
        else []
    )
    risk = (
        gpd.GeoSeries(risk_m, crs="EPSG:27700")
        .to_crs("EPSG:4326")
        .tolist()
        if risk_m
        else []
    )
    return safe, risk


def fetch_graph_and_chargers(center_lat, center_lon, dist_m=15000):
    if not HAS_OSMNX:
        raise RuntimeError("OSMnx not installed")
    G = graph_from_point_cached(center_lat, center_lon, dist_m)
    try:
        pois = ox.geometries_from_point(
            (center_lat, center_lon),
            tags={"amenity": "charging_station"},
            dist=dist_m,
        )
        chargers = []
        if not pois.empty:
            for _, row in pois.iterrows():
                geom = row.geometry
                if geom is None:
                    continue
                pt = geom.centroid if hasattr(geom, "centroid") else geom
                chargers.append(
                    {
                        "Latitude": float(pt.y),
                        "Longitude": float(pt.x),
                        "Name": row.get("name", "Charging station"),
                        "ROW_ID": len(chargers) + 1,
                        "AvailabilityLabel": "Operational",
                    }
                )
        chargers_df = pd.DataFrame(
            chargers,
            columns=[
                "ROW_ID",
                "Latitude",
                "Longitude",
                "Name",
                "AvailabilityLabel",
            ],
        )
    except Exception:
        chargers_df = pd.DataFrame(
            columns=[
                "ROW_ID",
                "Latitude",
                "Longitude",
                "Name",
                "AvailabilityLabel",
            ]
        )
    return G, chargers_df


def _soc_to_frac(x: float) -> float:
    """
    Ensure state-of-charge is represented as a fraction in [0, 1].

    Accepts:
      - fractions (0.3 → 30%)
      - legacy percentages (30 → 30%)

    Returns:
      float in [0, 1]
    """
    x = float(x)
    return x / 100.0 if x > 1.0 else x

def rcsp_optimize(
    start_lat,
    start_lon,
    end_lat,
    end_lon,
    battery_kwh,
    init_soc,
    reserve_soc,
    target_soc,
    kwh_per_km,
    chargers_df,
    flood_union_m,
    extreme=False,
    risk_penalty_per_km=None,
    max_seconds=5.0,
    soc_step=None,
):
    if not HAS_OSMNX:
        raise RuntimeError("OSMnx not installed")

   # --------------------------------------------------
# Graph construction (FIXED)
# --------------------------------------------------
    minlat, maxlat = sorted([float(start_lat), float(end_lat)])
    minlon, maxlon = sorted([float(start_lon), float(end_lon)])

    # base bbox (no diag_km yet)
    south0, north0 = minlat - 0.05, maxlat + 0.05
    west0,  east0  = minlon - 0.05, maxlon + 0.05

    # compute diagonal safely
    diag_km = haversine_km(south0, west0, north0, east0)

    # adaptive padding
    pad = max(0.05, diag_km / 110.0)
    south, north = minlat - pad, maxlat + pad
    west,  east  = minlon - pad, maxlon + pad

    # choose graph strategy
    if (
        (east - west) > MAX_GRAPH_BBOX_DEG
        or (north - south) > MAX_GRAPH_BBOX_DEG
        or diag_km > 60.0
    ):
        # long-distance fallback
        G = _graph_two_points(start_lat, start_lon, end_lat, end_lon, dist_m=30000)
    else:
        # preferred: single connected bbox graph
        G = _build_graph_bbox(north, south, east, west)

    # enrich graph
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    Gm = ox.project_graph(G, to_crs="EPSG:27700")
    edges_m = ox.graph_to_gdfs(Gm, nodes=False, edges=True, fill_edge_geometry=True)
    edges   = ox.graph_to_gdfs(G,  nodes=False, edges=True, fill_edge_geometry=True)

    # --------------------------------------------------
    # Flood risk tagging
    # --------------------------------------------------
    if flood_union_m is not None:
        buf = EXTREME_BUFFER_M if extreme else 0.0
        edges_m["risk"] = edges_m.geometry.intersects(flood_union_m.buffer(buf))
    else:
        edges_m["risk"] = False

    edges_m["length_m"] = edges_m.geometry.length.astype(float)

    edges_join = edges.join(edges_m[["risk", "length_m"]])
    edges_lookup = {}

    for (u, v, k), row in edges_join.iterrows():
        L = float(row.get("length_m", 0.0))
        if not math.isfinite(L) or L == 0:
            L = float(row.get("length", 0.0)) * 111_000.0
        T = float(row.get("travel_time", L / 13.9))
        R = bool(row.get("risk", False))
        edges_lookup[(u, v, k)] = (L, T, R)

    try:
        nn = ox.nearest_nodes
    except AttributeError:
        from osmnx.distance import nearest_nodes as nn

    u0 = nearest_nodes(G, start_lon, start_lat)
    v0 = nearest_nodes(G, end_lon, end_lat)

    # --------------------------------------------------
    # Chargers
    # --------------------------------------------------
    chargers = {}
    if isinstance(chargers_df, (pd.DataFrame, gpd.GeoDataFrame)) and not chargers_df.empty:
        for _, r in chargers_df.iterrows():
            try:
                nid = nn(G, float(r["Longitude"]), float(r["Latitude"]))
                p_kw = r.get("power_kW")
                p_kw = float(p_kw) if pd.notna(p_kw) and p_kw > 0 else DEFAULT_POWER_KW
                chargers[nid] = dict(
                    power_kW=p_kw,
                    operational=(str(r.get("AvailabilityLabel", "")) == "Operational"),
                )
            except Exception:
                continue

    # --------------------------------------------------
    # RCSP state space
    # --------------------------------------------------
    step = soc_step or SOC_STEP
    Q = [round(i * step, 2) for i in range(0, int(1 / step) + 1)]

    def q_to_idx(q):
        return max(0, min(len(Q) - 1, int(round(q / step))))

    # ✅ FIXED SoC HANDLING (canonical fractions)
    init_q    = _soc_to_frac(init_soc)
    reserve_q = _soc_to_frac(reserve_soc)
    tgt_q     = _soc_to_frac(target_soc)

    assert 0 <= reserve_q <= init_q <= 1.0, "Invalid SoC bounds"

    # --------------------------------------------------
    # RCSP search
    # --------------------------------------------------
    INF = 1e18
    best = {}
    pred = {}
    pareto = {}
    pq = []

    start_key = (u0, q_to_idx(init_q))
    best[start_key] = 0.0
    heapq.heappush(pq, (0.0, u0, q_to_idx(init_q)))

    risk_penalty = (
        risk_penalty_per_km
        if risk_penalty_per_km is not None
        else (EXTREME_RISK_PENALTY_PER_KM if extreme else BASE_RISK_PENALTY_PER_KM)
    )

    t0 = time.time()

    adj = {}
    for (u, v, _), (L, T, R) in edges_lookup.items():
        adj.setdefault(u, []).append((v, L, T, R))

    while pq:
        if time.time() - t0 > max_seconds:
            raise TimeoutError("RCSP time limit exceeded")

        cost, node, qi = heapq.heappop(pq)
        if best.get((node, qi), INF) < cost:
            continue

        if node == v0 and Q[qi] >= reserve_q:
            goal = (node, qi)
            break

        # Drive
        for v, L, T, R in adj.get(node, []):
            dq = (L / 1000.0) * kwh_per_km / battery_kwh
            if Q[qi] - dq < reserve_q:
                continue
            qj = q_to_idx(Q[qi] - dq)
            c2 = cost + T + (risk_penalty * (L / 1000.0) if R else 0.0)
            if c2 < best.get((v, qj), INF):
                best[(v, qj)] = c2
                pred[(v, qj)] = (node, qi, "drive", None)
                heapq.heappush(pq, (c2, v, qj))

        # Charge
        ch = chargers.get(node)
        if ch and ch["operational"]:
            p_kw = ch["power_kW"]
            for dq in (CHARGE_STEP, 2 * CHARGE_STEP, 3 * CHARGE_STEP):
                qn = min(1.0, Q[qi] + dq)
                dt = 3600.0 * battery_kwh * (qn - Q[qi]) / p_kw
                k2 = (node, q_to_idx(qn))
                c2 = cost + dt
                if c2 < best.get(k2, INF):
                    best[k2] = c2
                    pred[k2] = (node, qi, "charge", dict(dt=dt))
                    heapq.heappush(pq, (c2, node, k2[1]))
    else:
        raise RuntimeError("No feasible RCSP solution: try increasing bbox padding, "
        "reducing reserve_soc, or enabling direct OSRM fallback.")

    # --------------------------------------------------
    # Path reconstruction
    # --------------------------------------------------
    path = []
    k = goal
    while k in pred:
        path.append(k[0])
        k = (pred[k][0], pred[k][1])
    path.append(u0)
    path.reverse()

    lats = [G.nodes[n]["y"] for n in path]
    lons = [G.nodes[n]["x"] for n in path]
    line = LineString(coords_2d(zip(lons, lats)))

    safe, risk = segment_route_by_risk(
        line,
        flood_union_m,
        buffer_m=(EXTREME_BUFFER_M if extreme else ROUTE_BUFFER_M),
    )

    return line, safe, risk, [], best[goal]

# =========================
# Planner (helper)
# =========================

def line_to_latlon_list(line: LineString) -> List[Tuple[float, float]]:
    return [(lat, lon) for lon, lat in coords_2d(line.coords)]

# =========================
# Folium helpers
# =========================

def add_wms_group(fmap, title_to_layer: dict, visible=True, opacity=0.55):
    for title, layer in title_to_layer.items():
        try:
            WmsTileLayer(
                url=OWS_BASE,
                layers=layer,
                name=f"{title} (WMS)",
                fmt="image/png",
                transparent=True,
                opacity=opacity,
                version="1.3.0",
                show=visible,
            ).add_to(fmap)
        except Exception:
            pass

def add_base_tiles(m):
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        name="OpenStreetMap",
        attr="© OpenStreetMap contributors",
        control=True,
        overlay=False,
        max_zoom=19,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        name="CartoDB Positron",
        attr="© OpenStreetMap contributors, © CARTO",
        control=True,
        overlay=False,
        max_zoom=19,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        name="CartoDB Dark Matter",
        attr="© OpenStreetMap contributors, © CARTO",
        control=True,
        overlay=False,
        max_zoom=19,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
        name="OSM Humanitarian",
        attr="© OpenStreetMap contributors, Tiles courtesy of HOT",
        control=True,
        overlay=False,
        max_zoom=19,
    ).add_to(m)
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        name="Esri WorldImagery",
        attr="Tiles © Esri & contributors",
        control=True,
        overlay=False,
        max_zoom=19,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        name="OpenTopoMap",
        attr="© OpenStreetMap contributors, SRTM; style © OpenTopoMap (CC-BY-SA)",
        control=True,
        overlay=False,
        max_zoom=17,
    ).add_to(m)


def make_beautify_icon(color_hex: str):
    border_map = {
        "#D32F2F": "#B71C1C",
        "#FFC107": "#FF8F00",
        "#2E7D32": "#1B5E20",
    }
    border = border_map.get(color_hex, "#1B5E20")
    return BeautifyIcon(
        icon="bolt",
        icon_shape="marker",
        background_color=color_hex,
        border_color=border,
        border_width=3,
        text_color="white",
        inner_icon_style="font-size:22px;padding-top:2px;",
    )

def _row_to_tooltip_html(row, title=None):
    s = f"<b>{title or 'Data Point'}</b><br>"
    for k, v in row.items():
        if pd.isna(v) or v is None or str(v).strip() == "":
            continue
        s += f"<b>{k}:</b> {v}<br>"
    return s


def _thin_for_folium(
    df: pd.DataFrame, max_points: int = MAX_FOLIUM_POINTS, zone_col: str = "ZoneLabel"
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if len(df) <= max_points:
        return df

    df = df.copy()
    if zone_col not in df.columns:
        df[zone_col] = "Outside"

    out = []
    per_zone = max(1, max_points // df[zone_col].nunique())
    for z, grp in df.groupby(zone_col):
        if len(grp) <= per_zone:
            out.append(grp)
        else:
            out.append(grp.sample(n=per_zone, random_state=42))
    thinned = pd.concat(out, ignore_index=True)
    if len(thinned) > max_points:
        thinned = thinned.sample(n=max_points, random_state=42)
    return thinned
def add_heat_overlay(m, heat_data, vmin=5, vmax=25, opacity=0.55):
    if not heat_data:
        return

    fg = folium.FeatureGroup(name="Heat (UKCP tas)", show=True)

    lon = np.array(heat_data["lon"])
    lat = np.array(heat_data["lat"])
    z   = np.array(heat_data["z"])

    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            val = z[i, j]
            if not np.isfinite(val):
                continue

            folium.CircleMarker(
                location=[lat[i, j], lon[i, j]],
                radius=5,
                fill=True,
                fill_color=value_to_hex(val, vmin, vmax),
                fill_opacity=opacity,
                color=None,
            ).add_to(fg)

    fg.add_to(m)

# =========================
# Map rendering
# =========================

def render_map_html_ev(df_map, show_fraw, show_fmfp, show_live, show_ctx,
                       light=False, heat_data=None):
    if isinstance(df_map, gpd.GeoDataFrame):
        df_plot = pd.DataFrame(df_map.drop(columns=["geometry"], errors="ignore"))
    else:
        df_plot = pd.DataFrame(df_map)

    df_plot = _thin_for_folium(df_plot, max_points=MAX_FOLIUM_POINTS, zone_col="ZoneLabel")

    m = folium.Map(
        location=[51.6, -3.2],
        zoom_start=9,
        tiles=None,
        control_scale=True,
    )
    add_base_tiles(m)

    red_group = folium.FeatureGroup(name="Chargers: Zone 3 / High (red)", show=True).add_to(m)
    amber_group = folium.FeatureGroup(name="Chargers: Zone 2 / Medium (amber)", show=True).add_to(m)
    green_group = folium.FeatureGroup(
        name="Chargers: Zone 1 / Low–Outside (green)",
        show=True,
    ).add_to(m)

    red_cluster = MarkerCluster(name="Cluster: Zone 3 / High").add_to(red_group)
    amber_cluster = MarkerCluster(name="Cluster: Zone 2 / Medium").add_to(amber_group)
    green_cluster = MarkerCluster(name="Cluster: Zone 1 / Low–Outside").add_to(green_group)

    Draw(
        export=False,
        position="topleft",
        draw_options={
            "polygon": {
                "allowIntersection": False,
                "showArea": True,
            },
            "rectangle": True,
            "polyline": False,
            "circle": False,
            "circlemarker": False,
            "marker": False,
        },
        edit_options={"edit": True},
    ).add_to(m)

    for _, row in df_plot.iterrows():
        zlabel = (row.get("ZoneLabel") or "Outside")
        if zlabel in ("Zone 3", "High"):
            color_hex = ZONE_COLORS["Zone 3"]
            group_cluster = red_cluster
        elif zlabel in ("Zone 2", "Medium"):
            color_hex = ZONE_COLORS["Zone 2"]
            group_cluster = amber_cluster
        else:
            color_hex = ZONE_COLORS["Outside"]
            group_cluster = green_cluster

        title = f"{row.get('Operator','')} ({row.get('country','')})"
        try:
            tooltip_html = _row_to_tooltip_html(row, title=title)
            tooltip_obj = folium.Tooltip(tooltip_html, sticky=True)
        except Exception:
            tooltip_obj = title

        lat = row.get("Latitude")
        lon = row.get("Longitude")
        if pd.isna(lat) or pd.isna(lon):
            continue

        marker = folium.Marker(
            [lat, lon],
            tooltip=tooltip_obj,
            icon=make_beautify_icon(color_hex),
        )
        group_cluster.add_child(marker)

    if show_fraw:
        add_wms_group(m, FRAW_WMS, True, 0.50)
    if show_fmfp:
        add_wms_group(m, FMFP_WMS, True, 0.55)
    if show_ctx:
        add_wms_group(m, CONTEXT_WMS, False, 0.45)
    if show_live:
        add_wms_group(m, LIVE_WMS, True, 0.65)
    if heat_data:
        add_heat_overlay(m, heat_data, vmin=5, vmax=25, opacity=0.55)
    if show_live:
        add_wms_group(m, LIVE_WMS, True, 0.65)
    
    # Heat overlay (UKCP tas)
    if heat_data:
        add_heat_overlay(
        m,
        heat_data,
        vmin=5,
        vmax=25,
        opacity=0.55,
    )
    # temp_legend = """
    # <div style="
    # position: fixed;
    # bottom: 20px;
    # left: 360px;
    # z-index: 9999;
    # background: white;
    # padding: 10px 12px;
    # border: 1px solid #ccc;
    # border-radius: 6px;
    # font-size: 12px;">
    # <b>Mean temperature (°C)</b><br>
    # <div style="margin-top:6px">
    # <span style="display:inline-block;width:14px;height:14px;
    #         background:#000004;"></span> 5 °C<br>
    # <span style="display:inline-block;width:14px;height:14px;
    #         background:#420a68;"></span> 10 °C<br>
    # <span style="display:inline-block;width:14px;height:14px;
    #         background:#932567;"></span> 15 °C<br>
    # <span style="display:inline-block;width:14px;height:14px;
    #         background:#dd513a;"></span> 20 °C<br>
    # <span style="display:inline-block;width:14px;height:14px;
    #         background:#fca50a;"></span> 25 °C
    # </div>
    # </div>
    # """
    # m.get_root().html.add_child(folium.Element(temp_legend))

    folium.LayerControl(collapsed=True).add_to(m)
    
    heat_legend = """
    <div style="position: fixed; bottom:20px; right:20px; z-index:9999;
                background:white; padding:10px; border:1px solid #ccc;
                border-radius:6px; font-size:13px;">
    <b>Mean temperature (tas, °C)</b>
    <div style="height:10px; background:linear-gradient(to right,
    #000004,#1b0c41,#4a0c6b,#781c6d,#a52c60,#cf4446,#ed6925,#fb9b06,#f7d13d);
    margin-top:6px;"></div>
    <div style="display:flex; justify-content:space-between;">
    <span>5</span><span>25</span>
    </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(heat_legend))

    legend_html = (
        f"<div style='position: fixed; bottom:20px; left:20px; z-index:9999; "
        f"background:white; padding:10px 12px; border:1px solid #ccc; border-radius:6px; font-size:13px;'>"
        f"<b>Chargers by Flood Model Zone</b>"
        f"<div style='margin-top:6px'><span style='display:inline-block;width:12px;height:12px;"
        f"background:{ZONE_COLORS['Zone 3']};margin-right:6px;border:1px solid #555;'></span> Zone 3 / High</div>"
        f"<div><span style='display:inline-block;width:12px;height:12px;"
        f"background:{ZONE_COLORS['Zone 2']};margin-right:6px;border:1px solid #555;'></span> Zone 2 / Medium</div>"
        f"<div><span style='display:inline-block;width:12px;height:12px;"
        f"background:{ZONE_COLORS['Outside']};margin-right:6px;border:1px solid #555;'></span> "
        f"Zone 1 / Low–Very Low / Outside</div>"
        f"</div>"
    )
    m.get_root().html.add_child(folium.Element(legend_html))
    return m.get_root().render()

def render_map_html_route(
    full_line,
    route_safe,
    route_risk,
    start,
    end,
    chargers,
    all_chargers_df=None,
    animate=False,
    speed_kmh=50,
    show_live_backdrops=False,
):
    m = folium.Map(
        location=[(start[0] + end[0]) / 2, (start[1] + end[1]) / 2],
        zoom_start=11,
        tiles=None,
        control_scale=True,
    )
    add_base_tiles(m)

    if all_chargers_df is not None and not all_chargers_df.empty:
        df_all = pd.DataFrame(all_chargers_df)
        df_all = _thin_for_folium(df_all, max_points=MAX_ROUTE_POINTS, zone_col="ZoneLabel")

        red_group = folium.FeatureGroup(
            name="All Chargers: Zone 3 / High (red)", show=True
        ).add_to(m)
        amber_group = folium.FeatureGroup(
            name="All Chargers: Zone 2 / Medium (amber)", show=True
        ).add_to(m)
        green_group = folium.FeatureGroup(
            name="All Chargers: Zone 1 / Low–Outside (green)",
            show=True,
        ).add_to(m)

        red_cluster = MarkerCluster(name="Cluster: Zone 3 / High").add_to(red_group)
        amber_cluster = MarkerCluster(name="Cluster: Zone 2 / Medium").add_to(amber_group)
        green_cluster = MarkerCluster(name="Cluster: Zone 1 / Low–Outside").add_to(green_group)

        for _, row in df_all.iterrows():
            zlabel = (row.get("ZoneLabel") or "Outside")
            if zlabel in ("Zone 3", "High"):
                color_hex = ZONE_COLORS["Zone 3"]
                group_cluster = red_cluster
            elif zlabel in ("Zone 2", "Medium"):
                color_hex = ZONE_COLORS["Zone 2"]
                group_cluster = amber_cluster
            else:
                color_hex = ZONE_COLORS["Outside"]
                group_cluster = green_cluster

            title = f"{row.get('Operator','')} ({row.get('country','')})"
            try:
                tooltip_html = _row_to_tooltip_html(row, title=title)
                tooltip_obj = folium.Tooltip(tooltip_html, sticky=True)
            except Exception:
                tooltip_obj = title

            lat = row.get("Latitude")
            lon = row.get("Longitude")
            if pd.isna(lat) or pd.isna(lon):
                continue

            marker = folium.Marker(
                [lat, lon],
                tooltip=tooltip_obj,
                icon=make_beautify_icon(color_hex),
            )
            group_cluster.add_child(marker)

    def add_lines(lines, color, name):
        if not lines:
            return
        fg = folium.FeatureGroup(name=name).add_to(m)
        for ln in lines:
            coords = [(lat, lon) for lon, lat in coords_2d(ln.coords)]
            folium.PolyLine(coords, color=color, weight=6, opacity=0.9).add_to(fg)

    if isinstance(full_line, LineString):
        coords_full = [(lat, lon) for lon, lat in coords_2d(full_line.coords)]
        folium.PolyLine(
            coords_full,
            color="#999999",
            weight=3,
            opacity=0.5,
            tooltip="Planned route",
        ).add_to(m)

    add_lines(route_safe, "#2b8cbe", "Route – safe")
    add_lines(route_risk, "#e31a1c", "Route – flood risk")
    folium.Marker(start, tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(end, tooltip="End", icon=folium.Icon(color="blue")).add_to(m)

    cluster = MarkerCluster(name="Planned route stops").add_to(m)
    for st in chargers:
        try:
            row = gdf_ev.loc[gdf_ev["ROW_ID"].eq(st.get("ROW_ID", -1))].iloc[0]
        except Exception:
            continue
        zlabel = (row.get("ZoneLabel") or "Outside")
        if zlabel in ("Zone 3", "High"):
            color_hex = ZONE_COLORS["Zone 3"]
        elif zlabel in ("Zone 2", "Medium"):
            color_hex = ZONE_COLORS["Zone 2"]
        else:
            color_hex = ZONE_COLORS["Outside"]
        title = f"{row.get('Operator','')} ({row.get('country','')})"
        try:
            tooltip_html = _row_to_tooltip_html(row, title=title)
            tooltip_obj = folium.Tooltip(tooltip_html, sticky=True)
        except Exception:
            tooltip_obj = title
        folium.Marker(
            [row["Latitude"], row["Longitude"]],
            tooltip=tooltip_obj,
            icon=make_beautify_icon(color_hex),
        ).add_to(cluster)
    if show_live_backdrops:
        add_wms_group(m, LIVE_WMS, visible=True, opacity=0.65)
    
    folium.LayerControl(collapsed=True).add_to(m)
    
    legend_html = """
    <div style="position: fixed; bottom:20px; left:20px; z-index:9999; background:white;
                padding:10px 12px; border:1px solid #ccc; border-radius:6px; font-size:13px;">
      <b>Charger icon colour — Flood model zone</b>
      <div style="margin-top:6px"><span style="display:inline-block;width:12px;height:12px;background:#D32F2F;
           margin-right:6px;"></span> Red: Zone 3 / High</div>
      <div><span style="display:inline-block;width:12px;height:12px;background:#FFC107;
           margin-right:6px;"></span> Amber: Zone 2 / Medium</div>
      <div><span style="display:inline-block;width:12px;height:12px;background:#2E7D32;
           margin-right:6px;"></span> Green: Zone 1 / Low–Very Low / Outside</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m.get_root().render()

# =========================
# 3D (pydeck) stub
# =========================

def render_map_html_ev_3d(df_map=None, start=None, end=None,
                          route_full=None, route_safe=None, route_risk=None):
    if not HAS_PYDECK or not MAPBOX_API_KEY:
        return "<html><body>3D mode unavailable (pydeck/Mapbox missing).</body></html>"
    return "<html><body>3D map placeholder.</body></html>"

# =========================
# Dash app + logo
# =========================

app = Dash(__name__)
server = app.server

@server.route("/__logo")
def _serve_logo():
    try:
        if (not os.path.exists(LOGO_CACHE_PATH) or
            (time.time() - os.path.getmtime(LOGO_CACHE_PATH)) > 7 * 24 * 3600):
            content = read_bytes_resilient_gdrive(LOGO_GDRIVE_FILE_OR_URL)
            with open(LOGO_CACHE_PATH, "wb") as f:
                f.write(content)
        with open(LOGO_CACHE_PATH, "rb") as f:
            return Response(f.read(), mimetype="image/png")
    except Exception:
        return Response(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06"
            b"\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\x0bIDATx\x9cc``\x00\x00\x00\x02\x00\x01"
            b"\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82",
            mimetype="image/png",
        )

WALES_LOCS = {
    "Cardiff": (51.4816, -3.1791),
    "Swansea": (51.6214, -3.9436),
    "Newport": (51.5842, -2.9977),
    "Aberystwyth": (52.4153, -4.0829),
    "Bangor": (53.2290, -4.1294),
}

def preload_zones_json() -> str:
    try:
        if os.path.exists("cache_model_zones.parquet"):
            cache = pd.read_parquet("cache_model_zones.parquet")
            if {"ROW_ID", "ZoneLabel", "ZoneColor"}.issubset(cache.columns) and not cache.empty:
                return cache[["ROW_ID", "ZoneLabel", "ZoneColor"]].to_json(orient="records")
        zones = safe_compute_zones()
        zones.to_parquet("cache_model_zones.parquet")
        return zones[["ROW_ID", "ZoneLabel", "ZoneColor"]].to_json(orient="records")
    except Exception:
        return "[]"

def _kml_escape(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def _kml_argb_from_hex(rgb: str) -> str:
    try:
        if not (isinstance(rgb, str) and rgb.startswith("#") and len(rgb) == 7):
            raise ValueError
        rr, gg, bb = rgb[1:3], rgb[3:5], rgb[5:7]
        return f"ff{bb}{gg}{rr}"
    except Exception:
        return "ff327d2e"

def build_kml(route_data: dict) -> str:
    name = "EV Journey Simulator"
    coords = (route_data or {}).get("route") or []
    if coords:
        line_coords = " ".join(f"{c['lon']:.6f},{c['lat']:.6f},0" for c in coords)
        linestring = f"""
  <Placemark>
    <name>Route</name>
    <Style><LineStyle><color>ff1f78b4</color><width>4</width></LineStyle></Style>
    <LineString><coordinates>{line_coords}</coordinates></LineString>
  </Placemark>"""
    else:
        linestring = ""

    def mk_pt(title, lat, lon, color_hex=None):
        kml_color = _kml_argb_from_hex(color_hex or "#2E7D32")
        return f"""
  <Placemark>
    <name>{_kml_escape(title)}</name>
    <Style><IconStyle><color>{kml_color}</color></IconStyle></Style>
    <Point><coordinates>{lon:.6f},{lat:.6f},0</coordinates></Point>
  </Placemark>"""

    pts = []
    s = (route_data or {}).get("start")
    e = (route_data or {}).get("end")
    if s:
        pts.append(mk_pt("Start", float(s["lat"]), float(s["lon"]), "#2E7D32"))
    if e:
        pts.append(mk_pt("End", float(e["lat"]), float(e["lon"]), "#1f78b4"))
    for i, st in enumerate((route_data or {}).get("stops") or [], 1):
        title = st.get("name") or f"Stop {i}"
        color_hex = st.get("ZoneColor") or "#2E7D32"
        pts.append(mk_pt(title, float(st["lat"]), float(st["lon"]), color_hex))

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>{_kml_escape(name)}</name>{linestring}
  {''.join(pts)}
</Document>
</kml>"""

# =========================
# Layout
# =========================

app.layout = html.Div(
    [
        html.Div(
            [
                html.Img(
                    src="/__logo",
                    style={"height": "200px", "marginRight": "8px"},
                ),
                html.H1(
                    "CLEETS-SMART: Sustainable Mobility and Resilient Transport",
                    style={"margin": "4px"},
                ),
            ],
            style={
                "display": "flex",
                "alignItems": "center",
                "gap": "100px",
                "marginBottom": "8px",
            },
        ),
        html.H2("A) Chargers & Flood Overlays", style={"margin": "24px 7px 8px"}),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("country(s)"),
                        dcc.Dropdown(
                            id="f-country",
                            options=[{"label": t, "value": t} for t in country_OPTIONS],
                            value=[],
                            multi=True,
                            placeholder="All countrys",
                        ),
                    ],
                    style={"minWidth": "260px"},
                ),
                html.Div(
                    [
                        html.Label("country contains"),
                        dcc.Input(
                            id="f-country-like",
                            type="text",
                            placeholder="substring",
                            debounce=True,
                        ),
                    ],
                    style={"minWidth": "220px"},
                ),
                html.Div(
                    [
                        html.Label("Operational"),
                        dcc.Checklist(
                            id="f-op",
                            options=[
                                {"label": "Operational", "value": "op"},
                                {"label": "Not operational", "value": "down"},
                                {"label": "Unknown", "value": "unk"},
                            ],
                            value=["op", "down", "unk"],
                            inputStyle={"marginRight": "6px"},
                        ),
                    ],
                    style={"minWidth": "320px"},
                ),
                html.Div(
                    [
                        html.Label("Show overlays"),
                        dcc.Checklist(
                            id="layers",
                            options=[
                                {"label": "FRAW", "value": "fraw"},
                                {"label": "FMfP", "value": "fmfp"},
                                {"label": "Live warnings", "value": "live"},
                                {"label": "Context", "value": "ctx"},
                            ],
                            value=["fraw", "fmfp"],
                            inputStyle={"marginRight": "6px"},
                        ),
                    ],
                    style={"minWidth": "360px"},
                ),
                html.Div(
                    [
                        html.Label("Start-up mode"),
                        dcc.Checklist(
                            id="light",
                            options=[
                                {
                                    "label": "Light mode (fast start)",
                                    "value": "on",
                                }
                            ],
                            value=["on"],
                        ),
                    ],
                    style={"minWidth": "260px"},
                ),
                html.Button(
                    "Compute/Update zones",
                    id="btn-zones",
                    n_clicks=0,
                    style={"height": "38px", "marginLeft": "8px"},
                ),
                html.Button(
                    "Refresh overlays",
                    id="btn-refresh",
                    n_clicks=0,
                    style={"height": "38px", "marginLeft": "8px"},
                ),
            ],
            style={
                "display": "flex",
                "gap": "12px",
                "alignItems": "end",
                "flexWrap": "wrap",
                "margin": "6px 0 12px",
            },
        ),

        # Section B – Weather
        html.H2("B) Weather for Wales"),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Location"),
                        dcc.Dropdown(
                            id="wales-location",
                            value="Cardiff",
                            options=[{"label": k, "value": k} for k in WALES_LOCS.keys()],
                            clearable=False,
                            style={"width": "220px"},
                        ),
                    ],
                    style={"marginRight": "16px"},
                ),
                dcc.Interval(
                    id="wx-refresh",
                    interval=5 * 60 * 1000,
                    n_intervals=0,
                ),
            ],
            style={
                "display": "flex",
                "alignItems": "center",
                "gap": "12px",
            },
        ),

        html.Div(
            id="weather-split",
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr",
                "gap": "12px",
                "alignItems": "stretch",
            },
        ),

        html.H2("C) Heat (UK Temperature Predictions)"),

        html.Div(
            [
                dcc.Dropdown(
                    id="heat-period",
                    options=[{"label": k, "value": k} for k in HEAT_FILES],
                    value="HEAT 2020–2030",
                    clearable=False,
                    style={"width": "320px"},
                )
            ],
            style={"marginBottom": "12px"},
        ),
 
        html.Div(
            [
                dcc.Graph(id="heat-timeseries"),
            ],
            style={
                "display": "grid",
                "gridTemplateRows": "3fr 2fr",
                "gap": "12px",
            },
        ),

        html.H2("D) Journey Simulator (EV Route Planner)"),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Start (lat, lon)"),
                        dcc.Input(
                            id="sla",
                            type="number",
                            value=SIM_DEFAULTS["start_lat"],
                            step=0.001,
                            style={"width": "45%"},
                        ),
                        dcc.Input(
                            id="slo",
                            type="number",
                            value=SIM_DEFAULTS["start_lon"],
                            step=0.001,
                            style={"width": "45%", "marginLeft": "4px"},
                        ),
                        html.Small(
                            "Lat/Lon or use geocoder in future",
                            style={"color": "#666"},
                        ),
                    ],
                    style={"minWidth": "260px"},
                ),
                html.Div(
                    [
                        html.Label("End (lat, lon)"),
                        dcc.Input(
                            id="ela",
                            type="number",
                            value=SIM_DEFAULTS["end_lat"],
                            step=0.001,
                            style={"width": "45%"},
                        ),
                        dcc.Input(
                            id="elo",
                            type="number",
                            value=SIM_DEFAULTS["end_lon"],
                            step=0.001,
                            style={"width": "45%", "marginLeft": "4px"},
                        ),
                    ],
                    style={"minWidth": "260px"},
                ),
                html.Div(
                    [
                        html.Label("Battery size (kWh)"),
                        dcc.Input(
                            id="batt",
                            type="number",
                            value=64.0,
                            step=1,
                            min=20,
                            max=120,
                            style={"width": "100%"},
                        ),
                        html.Small("Gross usable capacity", style={"color": "#666"}),
                    ],
                    style={"minWidth": "180px"},
                ),
            ],
            style={
                "display": "flex",
                "gap": "12px",
                "alignItems": "flex-start",
                "flexWrap": "wrap",
            },
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Initial SoC"),
                        dcc.Slider(
                            id="si",
                            min=0.1,
                            max=1.0,
                            step=0.05,
                            value=0.90,
                        ),
                        html.Div(
                            id="si-label",
                            style={
                                "textAlign": "right",
                                "fontSize": "12px",
                                "color": "#666",
                            },
                            children="90%",
                        ),
                    ],
                    style={"minWidth": "220px"},
                ),
                html.Div(
                    [
                        html.Label("Reserve SoC"),
                        dcc.Slider(
                            id="sres",
                            min=0.05,
                            max=0.30,
                            step=0.05,
                            value=0.10,
                        ),
                        html.Div(
                            id="sres-label",
                            style={
                                "textAlign": "right",
                                "fontSize": "12px",
                                "color": "#666",
                            },
                            children="10%",
                        ),
                    ],
                    style={"minWidth": "220px"},
                ),
                html.Div(
                    [
                        html.Label("Target SoC"),
                        dcc.Slider(
                            id="stgt",
                            min=0.5,
                            max=1.0,
                            step=0.05,
                            value=0.80,
                        ),
                        html.Div(
                            id="stgt-label",
                            style={
                                "textAlign": "right",
                                "fontSize": "12px",
                                "color": "#666",
                            },
                            children="80%",
                        ),
                    ],
                    style={"minWidth": "220px"},
                ),
                html.Div(
                    [
                        html.Label("Consumption (kWh/km)"),
                        dcc.Slider(
                            id="kwhkm",
                            min=0.10,
                            max=0.30,
                            step=0.005,
                            value=0.20,
                        ),
                        html.Div(
                            id="kwhkm-label",
                            style={
                                "textAlign": "right",
                                "fontSize": "12px",
                                "color": "#666",
                            },
                            children="0.20 kWh/km · ≈ 5.0 km/kWh",
                        ),
                    ],
                    style={"minWidth": "260px"},
                ),
                html.Div(
                    [
                        html.Label("Max charge power (kW)"),
                        dcc.Input(
                            id="pmax",
                            type="number",
                            value=120.0,
                            step=5,
                            min=20,
                            max=350,
                            style={"width": "100%"},
                        ),
                        html.Small("Peak DC rate", style={"color": "#666"}),
                    ],
                    style={"minWidth": "180px"},
                ),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(3, minmax(220px, 1fr))",
                "gap": "12px",
                "marginTop": "8px",
            },
        ),
        html.Div(
            [
                dcc.Checklist(
                    id="show_leg_details",
                    options=[
                        {
                            "label": "Show per-leg details",
                            "value": "details",
                        }
                    ],
                    value=["details"],
                    inline=True,
                ),
                dcc.RadioItems(
                    id="units",
                    options=[
                        {
                            "label": "km / kWh / %",
                            "value": "metric",
                        },
                        {
                            "label": "miles / kWh / %",
                            "value": "imperial",
                        },
                    ],
                    value="metric",
                    inline=True,
                    style={"marginLeft": "16px"},
                ),
            ],
            style={
                "display": "flex",
                "alignItems": "center",
                "gap": "12px",
                "flexWrap": "wrap",
            },
        ),
        html.Div(
            [
                html.Button(
                    "Optimise",
                    id="simulate",
                    n_clicks=0,
                    style={"marginTop": "10px"},
                ),
                html.Button(
                    "Download KML",
                    id="btn-kml",
                    n_clicks=0,
                    style={"marginLeft": "8px", "marginTop": "10px"},
                ),
            ],
            style={"display": "flex", "gap": "8px"},
        ),
        html.Div(id="status", style={"marginTop": "10px"}),
        html.Div(id="explain", style={"marginTop": "10px", "whiteSpace": "pre-line"}),
        html.Div(
            [
                html.Label("Map mode"),
                dcc.RadioItems(
                    id="map-mode",
                    options=[
                        {
                            "label": "2D (Folium)",
                            "value": "2d",
                        },
                        {
                            "label": "3D (beta)",
                            "value": "3d",
                        },
                    ],
                    value="2d",
                    inline=True,
                ),
            ],
            style={"marginTop": "8px"},
        ),
        dcc.Loading(
            html.Iframe(
                id="map",
                srcDoc=(
                    "<html><body style='font-family:sans-serif;padding:10px'>"
                    "Loading…</body></html>"
                ),
                style={
                    "width": "100%",
                    "height": "99vh",
                    "border": "1px solid #ddd",
                    "borderRadius": "8px",
                },
            )
        ),
        html.Div(id="itinerary", style={"marginTop": "12px"}),
        dcc.Store(id="zones-json", data=preload_zones_json()),
        dcc.Store(id="store-route"),
        dcc.Store(id="heat-store"),
        dcc.Download(id="dl-kml"),
    ]
)
# =========================
# Callbacks
# =========================

@app.callback(
    Output("si-label", "children"),
    Input("si", "value"),
)
def _update_si_label(v):
    try:
        return f"{int(100 * float(v))}%"
    except Exception:
        return "–"

@app.callback(
    Output("sres-label", "children"),
    Input("sres", "value"),
)
def _update_sres_label(v):
    try:
        return f"{int(100 * float(v))}%"
    except Exception:
        return "–"

@app.callback(
    Output("stgt-label", "children"),
    Input("stgt", "value"),
)
def _update_stgt_label(v):
    try:
        return f"{int(100 * float(v))}%"
    except Exception:
        return "–"

@app.callback(
    Output("kwhkm-label", "children"),
    Input("kwhkm", "value"),
)
def _update_kwhkm_label(v):
    try:
        v = float(v)
        km_per_kwh = 1.0 / v if v > 0 else 0.0
        return f"{v:.2f} kWh/km · ≈ {km_per_kwh:.1f} km/kWh"
    except Exception:
        return "–"

@app.callback(
    Output("zones-json", "data"),
    Input("btn-zones", "n_clicks"),
    prevent_initial_call=True,
)
def _compute_zones(_n):
    zones = safe_compute_zones()
    try:
        zones.to_parquet("cache_model_zones.parquet")
    except Exception:
        pass
    return zones[["ROW_ID", "ZoneLabel", "ZoneColor"]].to_json(orient="records")

@app.callback(
    Output("heat-store", "data"),
    Output("heat-timeseries", "figure"),
    Input("heat-period", "value"),
)

def update_heat(period_label):

    if period_label is None:
        raise exceptions.PreventUpdate

    ds = xr.open_dataset(HEAT_FILES[period_label], engine="netcdf4")

    # ---- interactive time series ----
    tas_ts = ds["tas"].mean(
        dim=[
            "projection_x_coordinate",
            "projection_y_coordinate",
            "ensemble_member",
        ],
        skipna=True,
    )

    ts_df = tas_ts.to_dataframe(name="tas").reset_index()

    # colour by period (stable & obvious)
    PERIOD_COLOURS = {
    "HEAT 2020–2030": "#1f77b4",
    "HEAT 2030–2040": "#ff7f0e",
    "HEAT 2040–2050": "#d62728",
    }
    line_colour = PERIOD_COLOURS.get(period_label, "#333333")

#     ts_fig = go.Figure()

#     ts_fig.add_trace(
#     go.Scatter(
#         x=ts_df["time"],
#         y=ts_df["tas"],
#         mode="lines",
#         line=dict(color=line_colour, width=2),
#         hovertemplate=(
#             "<b>Date:</b> %{x}<br>"
#             "<b>tas:</b> %{y:.2f} °C<br>"
#             "<extra></extra>"
#         ),
#         name=period_label,
#     )
# )

#     ts_fig.update_layout(
#     title=(
#         f"{period_label} – Daily mean near-surface air temperature (tas) "
#         "averaged over the UK land domain"
#     ),
#     hovermode="x unified",
#     xaxis_title="Time (360-day calendar)",
#     yaxis_title="tas (°C)",
#     margin=dict(l=50, r=20, t=60, b=40),
# )

    # --- interactive time series (FINAL, SINGLE FIGURE) ---
    ts_fig = px.line(
    ts_df,
    x="time",
    y="tas",
    title=(
        f"{period_label} – Daily mean near-surface air temperature (tas) "
        "averaged over the UK land domain"
    ),
)

    ts_fig.update_traces(
    line=dict(
        color=HEAT_TS_COLORS.get(period_label, "#1f77b4"),
        width=3,
    ),
    hovertemplate=(
        "<b>Date:</b> %{x|%Y-%m-%d}<br>"
        "<b>Day of year:</b> %{x|%j}<br>"
        "<b>Temperature:</b> %{y:.2f} °C<br>"
        "<extra></extra>"
    ),
)

    ts_fig.update_layout(
    hovermode="x unified",
    xaxis_title="Date (360-day calendar)",
    yaxis_title="tas (°C)",
)

    hovertemplate=(
        "<b>Date:</b> %{x}<br>"
        "<b>Temperature:</b> %{y:.2f} °C<br>"
        "<extra></extra>"
    ),

    ts_fig.update_layout(
    hovermode="x unified",
    xaxis_title="Date (360-day calendar)",
    yaxis_title="tas (°C)",
    )

    # ---- heat grid (DATA ONLY) ----
    tas_map = (
        ds["tas"]
        .mean(dim=["time", "ensemble_member"], skipna=True)
        .squeeze(drop=True)
    )

    heat_data = {
        "lon": ds["grid_longitude"].values[::8, ::8].tolist(),
        "lat": ds["grid_latitude"].values[::8, ::8].tolist(),
        "z": tas_map.values[::8, ::8].tolist(),
    }

    return heat_data, ts_fig

def render_heat_map(heat_data):

    if not heat_data:
        raise exceptions.PreventUpdate

    fig = go.Figure(
        data=go.Heatmap(
            z=heat_data["z"],
            x=heat_data["lon"][0],   # longitude axis
            y=[row[0] for row in heat_data["lat"]],  # latitude axis
            colorscale="Inferno",
            colorbar=dict(title="tas (°C)"),
        )
    )

    fig.update_layout(
        title="Mean near-surface temperature (tas)",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        height=500,
    )

    return fig

@app.callback(
    Output("map", "srcDoc"),
    Output("itinerary", "children"),
    Output("store-route", "data"),
    Input("f-country", "value"),
    Input("f-country-like", "value"),
    Input("f-op", "value"),
    Input("layers", "value"),
    Input("light", "value"),
    Input("zones-json", "data"),
    Input("simulate", "n_clicks"),
    State("sla", "value"),
    State("slo", "value"),
    State("ela", "value"),
    State("elo", "value"),
    State("batt", "value"),
    State("si", "value"),
    State("sres", "value"),
    State("stgt", "value"),
    State("kwhkm", "value"),
    State("pmax", "value"),
    State("show_leg_details", "value"),
    State("units", "value"),
    State("map-mode", "value"),
    State("heat-store", "data"), 
    )

def _update_map(
    countrys,
    country_like,
    op_vals,
    layers_vals,
    light_vals,
    zones_json,
    sim_clicks,
    sla,
    slo,
    ela,
    elo,
    batt,
    si,
    sres,
    stgt,
    kwhkm,
    pmax,
    show_leg_details,
    units,
    map_mode,
    heat_store,
):
    light = "on" in (light_vals or [])

    d = gdf_ev.copy()
    zones_df = None
    if zones_json and zones_json != "[]":
        try:
            zextra = pd.read_json(StringIO(zones_json))
            if {"ROW_ID", "ZoneLabel", "ZoneColor"}.issubset(zextra.columns) and not zextra.empty:
                zones_df = zextra[["ROW_ID", "ZoneLabel", "ZoneColor"]]
        except Exception:
            zones_df = None
    if zones_df is None:
        zones_df = pd.DataFrame(
            {
                "ROW_ID": gdf_ev["ROW_ID"],
                "ZoneLabel": "Outside",
                "ZoneColor": ZONE_COLORS["Outside"],
            }
        )

    d = d.merge(zones_df, on="ROW_ID", how="left")
    d["ZoneLabel"] = d["ZoneLabel"].fillna("Outside")
    d["ZoneColor"] = d["ZoneColor"].fillna(ZONE_COLORS["Outside"])

    if countrys:
        d = d[d["country"].isin(countrys)]
    else:
        d = gdf_ev.merge(zones_df, on="ROW_ID", how="left")
        d["ZoneLabel"] = d["ZoneLabel"].fillna("Outside")
        d["ZoneColor"] = d["ZoneColor"].fillna(ZONE_COLORS["Outside"])

    if country_like:
        s = str(country_like).strip().lower()
        if s:
            d = d[d["country"].str.lower().str.contains(s, na=False)]

    op_vals = set(op_vals or [])
    if op_vals and len(op_vals) < 3:
        mask = pd.Series(False, index=d.index)
        if "op" in op_vals:
            mask |= d["AvailabilityLabel"].eq("Operational")
        if "down" in op_vals:
            mask |= d["AvailabilityLabel"].eq("Not operational")
        if "unk" in op_vals:
            mask |= d["AvailabilityLabel"].eq("Unknown") | d["AvailabilityLabel"].isna()
        d = d[mask]

    layers_vals = set(layers_vals or [])
    show_fraw = "fraw" in layers_vals
    show_fmfp = "fmfp" in layers_vals
    show_live = "live" in layers_vals
    show_ctx = "ctx" in layers_vals

    itinerary_children = html.Div()
    route_store = {}

    # Chargers to show during routing: operational only, with zones
    route_chargers = d[d["AvailabilityLabel"].eq("Operational")].copy()



    if sim_clicks:
        try:
            flood_union_m = None
            if (not light) and ENABLE_ROUTE_FLOOD_UNION:
                bounds = (
                    min(slo, elo),
                    min(sla, ela),
                    max(slo, elo),
                    max(sla, ela),
                )
                flood_union_m = get_flood_union(
                    bounds,
                    include_live=True,
                    include_fraw=True,
                    include_fmfp=True,
                    pad_m=(SIM_DEFAULTS["wfs_pad_m_fast"] if FAST_MODE_DEFAULT else SIM_DEFAULTS["wfs_pad_m"]),
                )

            # Fast path: light mode or no OSMnx -> OSRM only
            if light or not HAS_OSMNX:
                line, dist_m, dur_s, step_text, src = osrm_route(
                    float(sla), float(slo), float(ela), float(elo)
                )
                safe_lines, risk_lines = segment_route_by_risk(
                    line, flood_union_m, buffer_m=ROUTE_BUFFER_M
                )

                html_str = render_map_html_route(
                    full_line=line,
                    route_safe=safe_lines,
                    route_risk=risk_lines,
                    start=(float(sla), float(slo)),
                    end=(float(ela), float(elo)),
                    chargers=[],
                    all_chargers_df=route_chargers,
                    animate=False,
                    speed_kmh=45,
                    show_live_backdrops=False,
                )

                msg = [
                    f"**Routing plan (fast/OSRM)** — {dist_m/1000.0:.1f} km • {dur_s/3600.0:.2f} h",
                ]
                if step_text:
                    msg.append("---")
                    msg.extend([f"- {t}" for t in step_text[:12]])
                itinerary_children = dcc.Markdown("\n".join(msg))

                coords_latlng = [{"lat": lat, "lon": lon} for lon, lat in coords_2d(line.coords)]

                route_store = dict(
                    start={"lat": float(sla), "lon": float(slo)},
                    end={"lat": float(ela), "lon": float(elo)},
                    route=coords_latlng,
                    stops=[],
                    created_ts=time.time(),
                )
                return html_str, itinerary_children, route_store

            # Full RCSP optimiser
            if HAS_OSMNX:
                line, safe_lines, risk_lines, stops, total_cost = rcsp_optimize(
                    float(sla),
                    float(slo),
                    float(ela),
                    float(elo),
                    float(batt),
                    float(si),
                    float(sres),
                    float(stgt),
                    float(kwhkm),
                    d if not d.empty else gdf_ev,
                    flood_union_m,
                    extreme=False,
                    max_seconds=4.0,
                )

                if (map_mode or "2d") == "3d":
                    html_str = render_map_html_ev_3d(
                        df_map=None,
                        start=(float(sla), float(slo)),
                        end=(float(ela), float(elo)),
                        route_full=line,
                        route_safe=safe_lines,
                        route_risk=risk_lines,
                    )
                else:
                    html_str = render_map_html_route(
                      full_line=line,
                      route_safe=safe_lines,
                      route_risk=risk_lines,
                      start=(float(sla), float(slo)),
                      end=(float(ela), float(elo)),
                      chargers=stops,
                      all_chargers_df=route_chargers,
                      animate=False,
                      speed_kmh=45,
                      show_live_backdrops=False,
                  )


                rows = [
                    f"**Routing & charging plan** — RCSP on OSM road network; generalised cost ≈ {total_cost/60:.1f} min",
                ]
                if stops:
                    rows.append("---")
                    for i, st in enumerate(stops, 1):
                        row = gdf_ev.loc[gdf_ev["ROW_ID"].eq(st["ROW_ID"])].iloc[0]
                        rows.append(
                            f"**Stop {i}** — {row.get('Operator','')} ({row.get('country','')}) "
                            f"{row.get('Postcode','')} • Zone: {row.get('ZoneLabel','Outside')} • "
                            f"+{st['energy_kWh']:.1f} kWh to {int(100*st['soc_after'])}% "
                            f"in ~{st['charge_time_min']:.0f} min"
                        )
                itinerary_children = dcc.Markdown("\n\n".join(rows))

                coords_latlng = [{"lat": lat, "lon": lon} for lon, lat in coords_2d(line.coords)]

                route_store = dict(
                    start={"lat": float(sla), "lon": float(slo)},
                    end={"lat": float(ela), "lon": float(elo)},
                    route=coords_latlng,
                    stops=[
                        {
                            "lat": gdf_ev.loc[gdf_ev["ROW_ID"].eq(s["ROW_ID"])].iloc[0]["Latitude"],
                            "lon": gdf_ev.loc[gdf_ev["ROW_ID"].eq(s["ROW_ID"])].iloc[0]["Longitude"],
                            "name": gdf_ev.loc[gdf_ev["ROW_ID"].eq(s["ROW_ID"])].iloc[0].get("Operator", "Charger"),
                            "ZoneColor": gdf_ev.loc[gdf_ev["ROW_ID"].eq(s["ROW_ID"])].iloc[0].get(
                                "ZoneColor", ZONE_COLORS["Outside"]
                            ),
                        }
                        for s in stops
                    ],
                    created_ts=time.time(),
                )
                return html_str, itinerary_children, route_store

            # Fallback OSRM if RCSP path fails unexpectedly
            line, dist_m, dur_s, step_text, src = osrm_route(
                float(sla), float(slo), float(ela), float(elo)
            )
            safe_lines, risk_lines = segment_route_by_risk(
                line, flood_union_m, buffer_m=ROUTE_BUFFER_M
            )
            html_str = render_map_html_route(
                full_line=line,
                route_safe=safe_lines,
                route_risk=risk_lines,
                start=(float(sla), float(slo)),
                end=(float(ela), float(elo)),
                chargers=[],
                all_chargers_df=route_chargers,
                animate=False,
                speed_kmh=45,
                show_live_backdrops=False,
            )
            msg = [
                f"**Routing plan (OSRM fallback)** — {dist_m/1000.0:.1f} km • {dur_s/3600.0:.2f} h (source: {src})"
            ]
            if step_text:
                msg.append("---")
                msg.extend([f"- {t}" for t in step_text[:12]])
            itinerary_children = dcc.Markdown("\n".join(msg))

            coords_latlng = [{"lat": lat, "lon": lon} for lon, lat in coords_2d(line.coords)]

            route_store = dict(
                start={"lat": float(sla), "lon": float(slo)},
                end={"lat": float(ela), "lon": float(elo)},
                route=coords_latlng,
                stops=[],
                created_ts=time.time(),
            )
            return html_str, itinerary_children, route_store

        except Exception as e:
            itinerary_children = dcc.Markdown(f"**Routing error:** {e}")
            html_str = render_map_html_ev(
                d,
                show_fraw,
                show_fmfp,
                show_live,
                show_ctx,
                light=light,
                heat_data=heat_store,       
            )
            if heat_store:
                add_heat_overlay(m, heat_store)
            return html_str, itinerary_children, {}

    # No optimise yet: just charger map
    if (map_mode or "2d") == "3d":
        html_str = render_map_html_ev_3d(d)
    else:
        html_str = render_map_html_ev(
        d,
        show_fraw,
        show_fmfp,
        show_live,
        show_ctx,
        light=light,
        heat_data=heat_store,   # <-- ADD THIS
    )

    return html_str, itinerary_children, {}

@app.callback(
    Output("dl-kml", "data"),
    Input("btn-kml", "n_clicks"),
    State("store-route", "data"),
    prevent_initial_call=True,
)
def _download_kml(_n, route_data):
    if not route_data or not (
        route_data.get("route")
        and route_data.get("start")
        and route_data.get("end")
    ):
        return dash.no_update
    kml = build_kml(route_data)
    return dict(
        content=kml,
        filename="ev_journey.kml",
        type="application/vnd.google-earth.kml+xml",
    )

@app.callback(
    Output("weather-split", "children"),
    Input("wales-location", "value"),
    Input("wx-refresh", "n_intervals"),
)
def _wx_split(loc, _n):
    lat, lon = WALES_LOCS.get(loc, (51.60, -3.20))
    data = get_weather(lat, lon)
    prov = data.get("provider", "?")
    raw = data.get("raw") or {}

    # Left column
    left_children = [html.H3(f"{loc} – Current ({prov})")]

    if prov == "Open-Meteo":
        cur = raw.get("current", {})
        if cur:
            left_children += [
                html.Div(f"Temperature: {cur.get('temperature_2m', '?')} °C"),
                html.Div(f"Precipitation: {cur.get('precipitation', '?')} mm"),
                html.Div(f"Wind: {cur.get('wind_speed_10m', '?')} m/s"),
            ]

        hrs = raw.get("hourly", {})
        times = (hrs.get("time") or [])[:6]
        temps = (hrs.get("temperature_2m") or [])[:6]
        pops = (hrs.get("precipitation_probability") or [])[:6]
        if times:
            tbl_rows = [
                html.Tr(
                    [
                        html.Th("Time"),
                        html.Th("Temp (°C)"),
                        html.Th("Precip (%)"),
                    ]
                )
            ]
            for t, temp, pop in zip(times, temps, pops):
                tbl_rows.append(
                    html.Tr(
                        [
                            html.Td(t),
                            html.Td(f"{temp}" if temp is not None else "–"),
                            html.Td(f"{pop}" if pop is not None else "–"),
                        ]
                    )
                )
            left_children.append(
                html.Div(
                    [
                        html.H4("Next 6 hours"),
                        html.Table(tbl_rows, style={"fontSize": "12px"}),
                    ],
                    style={"marginTop": "8px"},
                )
            )

    elif prov == "Met Office":
        left_children += [
            html.Div(
                "Raw Met Office JSON (truncated):",
                style={"fontWeight": "bold", "marginTop": "6px"},
            ),
            html.Pre(json.dumps(raw, indent=2)[:2000]),
        ]

    elif prov == "error":
        left_children += [
            html.Div("Weather error: " + data.get("error", ""))
        ]

    left = html.Div(
        style={
            "border": "1px solid #eee",
            "borderRadius": "10px",
            "padding": "10px",
        },
        children=left_children,
    )

    # Right column (24h forecast chart)
    try:
        if prov == "Open-Meteo":
            hrs = raw.get("hourly", {})
            times = (hrs.get("time") or [])[:24]
            temps = (hrs.get("temperature_2m") or [])[:24]
            pops = (hrs.get("precipitation_probability") or [])[:24]

            df2 = pd.DataFrame({"time": times, "temp": temps, "pop": pops})
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=df2["time"],
                    y=df2["pop"],
                    name="Precip %",
                    yaxis="y2",
                    opacity=0.5,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=df2["time"], y=df2["temp"], name="Temp °C"
                )
            )
            fig.update_layout(
                margin=dict(l=10, r=10, t=30, b=10),
                height=320,
                xaxis_title="",
                yaxis_title="Temp (°C)",
                yaxis2=dict(
                    title="Precip (%)", overlaying="y", side="right"
                ),
            )

        elif prov == "Met Office":
            ts = _parse_metoffice_timeseries(raw)
            fig = go.Figure()
            if ts:
                df2 = pd.DataFrame(
                    {
                        "time": ts.get("time", []),
                        "temp": ts.get("temp", []),
                        "pop": ts.get("pop", []),
                    }
                )
                if any(df2.get("pop", [])):
                    fig.add_trace(
                        go.Bar(
                            x=df2["time"],
                            y=df2["pop"],
                            name="Precip %",
                            yaxis="y2",
                            opacity=0.5,
                        )
                    )
                fig.add_trace(
                    go.Scatter(
                        x=df2["time"], y=df2["temp"], name="Temp °C",
                    )
                )
                fig.update_layout(
                    margin=dict(l=10, r=10, t=30, b=10),
                    height=320,
                    xaxis_title="",
                    yaxis_title="Temp (°C)",
                    yaxis2=dict(
                        title="Precip (%)",
                        overlaying="y",
                        side="right",
                    ),
                )
            else:
                fig.update_layout(title="Met Office: timeseries not found", height=320)
        else:
            fig = go.Figure()
            fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=320)

    except Exception as e:
        fig = go.Figure()
        fig.update_layout(title=f"Weather chart error: {e}", height=320)

    right = html.Div(
        style={
            "border": "1px solid #eee",
            "borderRadius": "10px",
            "padding": "10px",
            "overflowY": "auto",
            "height": "400px",
        },
        children=[
            html.H3("Next 24h forecast"),
            dcc.Graph(
                figure=fig,
                config={"displayModeBar": False},
                style={"height": "320px", "width": "100%"},
            ),
        ],
    )

    return [left, right]

# =========================
# Main
# =========================

if __name__ == "__main__":
    app.run(debug=True)
