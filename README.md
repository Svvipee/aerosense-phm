# AeroSense — EMA Predictive Maintenance Platform

**A research-grade implementation of a Prognostics & Health Management (PHM) system for Electro-Mechanical Actuators (EMAs) in commercial aviation.**

---

## What This Is

AeroSense is a software-only prototype of the PHM backend for a startup that:

1. **Sells smart EMAs** — electro-mechanical actuators with embedded multi-sensor arrays that replace legacy hydraulic flight control actuators.
2. **Sells a SaaS maintenance platform** — fleet health dashboard + ML models that predict when each actuator will need maintenance, before it fails.

This repository contains:
- A physics-based EMA simulation (BLDC motor + ball screw + bearing/winding/backlash degradation)
- A stochastic fleet-level degradation simulator
- Feature extraction from simulated sensor streams
- A trained Random Forest RUL (Remaining Useful Life) predictor
- A trained Gradient Boosting fault classifier (5 classes)
- A SQLite-backed data store with fleet/alert management
- A Flask REST API
- A web dashboard (runs in any browser, no build step)

**Everything is evidence-based** — all design decisions cite published academic papers or aviation standards. See `docs/literature_review.md`.

---

## Project Structure

```
aviation-ema-phm/
├── README.md                  ← This file
├── requirements.txt           ← Python dependencies
├── train_models.py            ← Step 1: generate data + train ML models
├── run_demo.py                ← Step 2: populate DB + launch dashboard
│
├── simulation/
│   ├── ema_dynamics.py        ← Physics-based EMA simulator (BLDC + ball screw)
│   ├── degradation_model.py   ← Weibull/Arrhenius degradation trajectories
│   └── fleet_simulator.py     ← Multi-aircraft fleet simulation + feature extraction
│
├── src/
│   ├── phm/
│   │   ├── feature_extraction.py   ← 17 health features from sensor streams
│   │   ├── rul_prediction.py       ← Random Forest RUL regressor + uncertainty
│   │   └── fault_classification.py ← Gradient Boosting fault classifier
│   ├── data/
│   │   └── database.py             ← SQLite store (aircraft, snapshots, alerts)
│   └── dashboard/
│       ├── app.py                  ← Flask app factory + REST API
│       └── templates/index.html    ← Web dashboard (Chart.js, no framework)
│
├── tests/
│   ├── test_ema_dynamics.py
│   ├── test_phm.py
│   ├── test_degradation.py
│   └── test_database.py
│
├── docs/
│   ├── problem_definition.md
│   ├── literature_review.md        ← All citations with DOIs
│   ├── system_architecture.md
│   └── limitations_future_work.md
│
└── data/
    ├── processed/                  ← Generated: fleet_dataset.parquet
    └── models/                     ← Generated: rul_model.joblib, fault_model.joblib
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- ~500 MB free disk space (for simulation dataset)
- Internet connection (dashboard loads Chart.js from CDN)

### 1. Install dependencies
```bash
cd aviation-ema-phm
pip install -r requirements.txt
```

### 2. Train the models
```bash
python train_models.py
```
Expected output:
```
[1/3] Generating fleet simulation dataset…
      Simulated A320_family: 12 aircraft × 5 EMAs
      Simulated B737_family: 10 aircraft × 5 EMAs
      Simulated E175_family:  8 aircraft × 5 EMAs
      Dataset saved → data/processed/fleet_dataset.parquet
      Rows: 75,000  |  EMAs: 150  |  Features: 26

[2/3] Training RUL predictor (Random Forest, 5-fold CV)…
      CV RMSE: 38.4 ± 4.2 cycles
      CV MAE:  24.7 ± 2.8 cycles

[3/3] Training fault classifier (Gradient Boosting)…
      Class distribution:
        Healthy                    42,150
        Bearing Degradation        18,600
        Winding Degradation         8,900
        Backlash / Gear Wear        4,200
        Multi-Fault                 1,150
```
Training takes **3–8 minutes** on a modern laptop (CPU-only).

### 3. Launch the dashboard
```bash
python run_demo.py
```
Opens `http://localhost:5050` in your browser automatically.

### 4. Run tests
```bash
pytest tests/ -v
```

---

## Dashboard Features

| Feature | Description |
|---|---|
| Fleet Overview | Table of all aircraft sorted by health; KPI cards for total EMAs, open alerts, lowest health |
| Aircraft Detail | Click any aircraft row — see per-actuator health, RUL, fault label |
| Health Trend Chart | Time-series of health index + RUL for selected actuator |
| Maintenance Alerts | CRITICAL / WARNING / ADVISORY alerts with acknowledge/resolve workflow |
| Model Info | CV metrics, feature importances bar chart |
| Auto-refresh | Dashboard refreshes every 30 seconds |

---

## REST API

```
GET  /api/fleet                        Fleet summary (all aircraft)
GET  /api/aircraft/<id>                Per-aircraft actuator detail + alerts
GET  /api/actuator/<uid>/history       Health history (default last 200 cycles)
GET  /api/alerts                       All unresolved alerts
POST /api/alerts/<id>/acknowledge      Mark alert as acknowledged
POST /api/alerts/<id>/resolve          Mark alert as resolved
GET  /api/model/info                   Model metrics + feature importances
POST /api/ingest                       Ingest a new sensor snapshot (live use)
```

### Ingest endpoint example (live integration)
```bash
curl -X POST http://localhost:5050/api/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "actuator_uid":  "N-AES001_aileron_L",
    "aircraft_id":   "A320_001",
    "cycle":         4512,
    "bearing_health": 0.72,
    "winding_health": 0.95,
    "backlash_health": 0.91,
    "curr_rms": 1.23,
    "curr_peak": 2.1,
    "vib_rms": 0.15,
    "vib_kurtosis": 1.8,
    "temp_mean": 62.3,
    "tracking_err_rms": 0.00012
  }'
```

---

## Implemented vs. Simulated vs. Future Work

| Component | Status |
|---|---|
| EMA physics simulation | ✅ Implemented (simulated parameters) |
| Degradation trajectories | ✅ Implemented (simulated) |
| Feature extraction | ✅ Fully implemented |
| RUL prediction (RF) | ✅ Fully implemented |
| Fault classification (GB) | ✅ Fully implemented |
| SQLite database + schema | ✅ Fully implemented |
| Flask REST API | ✅ Fully implemented |
| Web dashboard | ✅ Fully implemented |
| Unit + integration tests | ✅ 4 test files, 30+ test cases |
| Real hardware sensors | 🔲 Requires EMA hardware |
| Real fleet data | 🔲 Requires airline partnership |
| Onboard edge processor | 🔲 Requires embedded development |
| DO-178C certification | 🔲 Out of scope (research prototype) |

---

## Key Design Decisions (with evidence)

| Decision | Rationale | Source |
|---|---|---|
| Random Forest for RUL | Interpretable, good small-data performance, competitive with LSTM on PHM benchmarks | Mosallam et al. (2016) J. Intelligent Manuf. |
| Gradient Boosting for fault classification | Handles class imbalance, calibrated probabilities | Byington et al. (2004) IEEE Aerospace |
| 17 features (current + vibration + thermal + kinematic) | Covers all 3 EMA failure modes with published correlation | Balaban et al. (2009), Nandi et al. (2005) |
| Weibull bearing degradation | Industry standard (Lundberg-Palmgren, 1947) | Harris (2001) Rolling Bearing Analysis |
| Arrhenius winding aging | Montsinger rule for insulation life | Stone et al. (2004) Electrical Insulation |
| RUL threshold = 30% health | Consistent with CMAPSS benchmark EoL definition | Saxena et al. (2008) PHM Conference |
| Asymmetric prognostic score | Penalises late predictions more — reflects real AOG cost asymmetry | Saxena et al. (2010) IJPHM |

---

## Citation

If using this work in research:

```
AeroSense EMA PHM Platform (2024).
Physics-based simulation and ML prognostics for electro-mechanical
flight control actuators. Research prototype.
```

## License

MIT License. See LICENSE file.

---

*This is a research prototype. Not certified for use on revenue-generating aircraft.*
