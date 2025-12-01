# CLEETS-SMART [South Wales]: Towards a Support System for Electric Vehicles Travel Planning During Extreme Weather Events

Vehicles Travel Planning During Extreme Weather Events  

A decision support dashboard is proposed for South Wales in the United Kingdom to reduce range anxiety and enable safer and more resilient EV travel planning during extreme weather events. The dashboard integrates heterogeneous data sources, including the UK National ChargePoint Registry, Welsh Government flood-risk layers, and 24-hour forecasts from the Met Office DataHub and the Open-Meteo API as a fallback. Users specify trip origin, destination, and battery State-Of-Charge (SOC); the dashboard then simulates routes, highlights exposure risks, and recommends safe charging stops. Two routing modes are supported: an exact Resource Constrained Shortest Path (RCSP) solver, which provides charge-feasible routes that minimize travel time and flood exposure; and a fallback Open Source Routing Machine (OSRM) mode, which returns only the fastest path and overlays flood warnings, without guaranteeing feasibility or optimality. 

# EV Chargers & Flood Risk — South Wales  

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)  [![Dash](https://img.shields.io/badge/Dash-2.x-brightgreen.svg)](https://dash.plotly.com/)  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![OGL v3.0](https://img.shields.io/badge/Data%20License-OGL--UK--3.0-lightgrey.svg)](http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/) [![ODbL](https://img.shields.io/badge/Data%20License-ODbL-orange.svg)](https://www.openstreetmap.org/copyright)  

A Dash + Folium web app that integrates:  
- **Welsh Government flood-risk maps** (FRAW, FMfP, live warnings via GeoServer).  
- **UK National ChargePoint Registry (NCR) data** for public EV chargers.  
- **Met Office DataHub and Open-Meteo forecasts** for 24-hour conditions.  
- **Journey simulator** with exact RCSP routing + fallback OSRM.  
- **Chatbot explanations** for transparency and scenario testing.  

Chargers are visualised with overlays for flood zones, live warnings, and weather impact.  

---

## Attribution  

Contains Natural Resources Wales information © Natural Resources Wales and database right. All rights reserved. Some features of this information are based on digital spatial data licensed from the UK Centre for Ecology & Hydrology © UKCEH. Defra, Met Office and DARD Rivers Agency © Crown copyright. ©Cranfield University. © James Hutton Institute. Contains OS data © Crown copyright and database right.[Flood Risk Assessment Wales (FRAW)](https://datamap.gov.wales/layergroups/inspire-nrw:FloodRiskAssessmentWales) [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).    
- Contains data from the UK National ChargePoint Registry © OZEV.  
- Weather data © Met Office DataHub / Open-Meteo.  
- Contains OS data © Crown copyright and database right.  

---

## Screenshots  

### Dashboard with flood overlays, journey simulator and weather forecast  
![CLEETS-SMART Dashboard](dash11.png)  

---

## Features  
- **Live flood overlays** (FRAW, FMfP, NRW warnings).  
- **CLEETS-SMART EV Routing Optimization Method: RCSP solver (battery-aware) + fallback OSRM routes.

Electric-vehicle (EV) routing under flood conditions is formulated as an Exact Resource-Constrained  
Shortest Path (RCSP) problem designed to mitigate *range anxiety*, defined as the driver’s concern  
that the vehicle may not have sufficient energy to reach the next charger or destination. The road  
network is represented as a directed graph \(G=(V,A)\) where each arc has attributes for driving  
time, distance, and flood status. The optimization jointly minimizes travel time and charging time  
while enforcing battery-feasibility constraints.

- **Optimization Problem

\[
\begin{aligned}
\min_{x,q}\quad   
    &\sum_{a\in A}\big(t_a^{\mathrm{drive}} + \lambda d_a F_a\big) x_a
    + \sum_{v\in C} t_v^{\mathrm{charge}}  \\
\text{s.t.}\quad  
    &q_j = q_i - \frac{\kappa d_a}{B},
        &&\forall a=(i,j)\in A,\; x_a = 1, \\
    &q_j \ge q_{\min},
        &&\forall j\in V,\\
    &t_v^{\mathrm{charge}} = 3600\,\frac{B(q_k-q_j)}{P_v},
        &&\forall v\in C,\\
    &\lambda >
        \frac{\max_{P\in\mathcal{P}_0}\sum_{a\in P} t_a^{\mathrm{drive}} + 1}
             {d_{\min}}, \\
    &x_a\in\{0,1\},\quad q\in[0,1].
\end{aligned}
\]

- **Variable definitions:**  
- \(x_a\): arc-selection variable  
- \(t_a^{\mathrm{drive}}\): driving time  
- \(d_a\): distance  
- \(F_a\): flood indicator  
- \(\lambda\): flood-penalty coefficient  
- \(q_i, q_j\): state of charge (SOC)  
- \(\kappa\): energy-consumption rate  
- \(B\): battery capacity  
- \(q_{\min}\): minimum SOC  
- \(P_v\): charger power  
- \(t_v^{\mathrm{charge}}\): charging time  
- \(C\): set of charger nodes  
- \(\mathcal{P}_0\): set of unflooded paths  
- \(d_{\min}\): minimum arc distance  

- **How the Method Works

Optimization is solved using a label-setting dynamic program that tracks both the vehicle’s location  
and its battery level in an augmented state space \((v,q)\). Each label represents a possible state  
of the journey, recording the node position and its SOC. Moving along a road segment reduces SOC  
according to the energy-consumption model, while charging stations increase SOC following the  
charging-time expression. Dominated labels—those slower or less energy efficient than another at the  
same node—are pruned throughout the search. Flood-affected arcs are kept but given a large soft  
penalty \(\lambda d_a F_a\), making them unattractive unless no unflooded option exists.

  
1. **Build the road network**
   - Model the map as a graph $G=(V,E)$ with nodes $V$ (intersections/chargers) 
     and edges $E$ (road segments).
   - Each edge $e$ has: length $d_e$, driving time $t^{\mathrm{drive}}_e$, 
     and a flood flag $F_e \in \{0,1\}$ (1 if flooded/penalized).

2. **Energy model per edge**
   - Energy used on edge $e$: $\Delta E_e = \kappa d_e$ (kWh), where $\kappa$ is 
     average consumption (kWh/km).
   - State of charge (SOC) update: $q' = q - \Delta E_e / B$, where $B$ is 
     battery capacity (kWh).  
     Enforce minimum reserve: $q' \ge q_{\min}$.

3. **Travel cost per edge**
   - Cost combines drive time and flood penalty:  
     $c_e = t^{\mathrm{drive}}_e + \lambda d_e F_e$.  
     Flooded edges ($F_e=1$) get an extra penalty proportional to length.

4. **Charging decision at nodes**
   - If SOC would drop below reserve before the next node, add a charging stop at node $v$.
   - Charging time:  
     $t^{\mathrm{chg}}(q,q';v) = \frac{B(q'-q)}{P_v}\times 3600$ seconds,  
     where $P_v$ is charger power (kW) and 3600 converts hours → seconds.  
     Enforce $q' \le 1$ and $q' \ge q$.

5. **Route objective (RCSP)**
   - Minimize total travel cost:  
     $\min \sum_{e \in \text{path}} c_e + \sum_{\text{stops } v} t^{\mathrm{chg}}(q,q';v)$  
     subject to SOC dynamics and reserve constraints.

6. **Two ways to compute routes**
   - (i) Use OSRM to compute a baseline route and time.  
   - (ii) Solve an exact resource-constrained shortest path (RCSP) on OSMnx graphs 
     to jointly optimize path and charging stops under SOC and flood penalties.

7. **High-level workflow**
   - Start with initial SOC $q_0$.  
   - For each candidate path, propagate SOC (step 2), insert charging (step 4), 
     and accumulate cost (steps 3 & 4).  
   - RCSP picks the feasible path/charging plan with minimum cost while avoiding flooded edges.

- **Weather forecasts** from Met Office/Open-Meteo, shown alongside maps.  
- **Downloadable routes** with summaries of time, distance, charging stops, and risk level.  

---

## Repository contents  
- `app.py` — single-file Dash app with Folium map, routing, flood overlays, and chatbot interface.  
- `AquaEV.png` — screenshot of the main dashboard.  
- [Colab Notebook] https://colab.research.google.com/drive/1uw6xeN1H6tFmyBEMJUoxDH-qDAU7Hb9a?usp=sharing.
---

## Installation  
```bash
git clone https://github.com/Naeima/CLEETS-SMART.git
cd CLEETS-SMART
pip install dash pandas geopandas folium shapely requests plotly osmnx
python app.py
