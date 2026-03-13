# System Architecture

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         AIRCRAFT (Edge)                          │
│                                                                  │
│  ┌─────────────────┐    ┌──────────────┐    ┌────────────────┐  │
│  │  Smart EMA      │    │  EMA         │    │  EMA           │  │
│  │  (aileron L)    │    │  (elevator L)│    │  (rudder)      │  │
│  │  ┌───────────┐  │    │              │    │                │  │
│  │  │ BLDC Motor│  │    │  (×2)        │    │  (×1)          │  │
│  │  │ Ball Screw│  │    └──────────────┘    └────────────────┘  │
│  │  │ Sensors:  │  │         │                    │             │
│  │  │  Current  │  │         └────────────────────┘             │
│  │  │  Vibration│  │                     │                       │
│  │  │  Temp     │  │         ┌──────────────────────┐            │
│  │  │  Position │  │         │  Onboard Data Logger  │           │
│  │  └─────┬─────┘  │         │  (per ATA 31 ACARS)   │           │
│  └────────┼────────┘         └───────────┬───────────┘           │
│           └──────────────────────────────┘                       │
└──────────────────────────────────────┬──────────────────────────┘
                                       │  (Ground download at gate)
                                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PHM BACKEND (Ground / Cloud)                  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                  Data Ingestion Pipeline                  │   │
│  │   Raw sensor streams → Feature extraction → DB storage   │   │
│  └──────────────────────────┬───────────────────────────────┘   │
│                             │                                    │
│  ┌─────────────────────┐    │    ┌─────────────────────────┐    │
│  │   RUL Predictor     │◄───┤    │   Fault Classifier       │   │
│  │   (Random Forest)   │    │    │   (Gradient Boosting)    │   │
│  └──────────┬──────────┘    │    └───────────┬─────────────┘    │
│             │               │                │                    │
│  ┌──────────▼───────────────▼────────────────▼──────────────┐   │
│  │                     SQLite Database                        │   │
│  │   Tables: aircraft, actuators, health_snapshots, alerts   │   │
│  └──────────────────────────────────┬────────────────────────┘   │
│                                     │                              │
│  ┌──────────────────────────────────▼────────────────────────┐   │
│  │              Flask REST API (port 5050)                    │   │
│  │   GET /api/fleet  |  GET /api/aircraft/<id>               │   │
│  │   GET /api/alerts |  POST /api/ingest                     │   │
│  └──────────────────────────────────┬────────────────────────┘   │
└─────────────────────────────────────┼──────────────────────────┘
                                      │  HTTP
                                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MRO DASHBOARD (Browser)                        │
│                                                                   │
│   Fleet Overview → Aircraft Detail → Actuator Trend Charts       │
│   Alert Management → Maintenance Scheduling                      │
│   Model Info → Feature Importances → CV Metrics                  │
└─────────────────────────────────────────────────────────────────┘
```

## 2. Module Descriptions

### 2.1 `simulation/` — Physics Simulation Layer

| Module | Purpose |
|---|---|
| `ema_dynamics.py` | Physics-based BLDC/ball-screw EMA simulation. Implements voltage-current-torque-position dynamics with degradation-dependent parameters. |
| `degradation_model.py` | Stochastic degradation trajectory generator (Weibull bearing, Arrhenius winding, linear backlash). |
| `fleet_simulator.py` | Fleet-level simulation orchestrator. Generates the labelled training dataset. |

### 2.2 `src/phm/` — PHM Algorithms

| Module | Purpose |
|---|---|
| `feature_extraction.py` | Converts raw time-series to 17 scalar features (current, vibration, thermal, kinematic groups). |
| `rul_prediction.py` | Random Forest RUL regression with 5-fold CV and bootstrap uncertainty. |
| `fault_classification.py` | Gradient Boosting fault classifier (5 classes: healthy, bearing, winding, backlash, multi-fault). |

### 2.3 `src/data/` — Persistence Layer

| Module | Purpose |
|---|---|
| `database.py` | SQLite store: aircraft registry, health snapshots, alert CRUD, fleet/actuator summary queries. |

### 2.4 `src/dashboard/` — Web Application

| Module | Purpose |
|---|---|
| `app.py` | Flask app factory with all REST routes and alert auto-generation logic. |
| `templates/index.html` | Single-page dashboard: fleet table, actuator detail panel, health trend chart (Chart.js), alert list. |

## 3. Data Flow

```
Simulation generates synthetic sensor streams
    ↓
fleet_simulator.py calls ema_dynamics.run_profile() for each cycle
    ↓
FeatureVector extracted from N time-series samples
    ↓
(aircraft_id, actuator_id, cycle, features, ground_truth) → DataFrame
    ↓
train_models.py: RULPredictor.train() + FaultClassifier.train()
    ↓
Trained models saved to data/models/
    ↓
run_demo.py: loads models, writes snapshots + alerts to SQLite DB
    ↓
Flask API serves data to browser dashboard
    ↓
Chart.js renders health trends; auto-refresh every 30 s
```

## 4. Feature Vector (17 dimensions)

| Feature | Group | Physical meaning |
|---|---|---|
| `curr_rms` | Current | Winding load indicator |
| `curr_peak` | Current | Peak current demand |
| `curr_crest` | Current | Impulsive load indicator |
| `curr_thd` | Current | Harmonic distortion (winding fault) |
| `curr_f1` | Current | Fundamental current amplitude |
| `vib_rms` | Vibration | Overall vibration level |
| `vib_peak` | Vibration | Peak vibration event |
| `vib_kurtosis` | Vibration | Bearing fault impulsiveness |
| `vib_crest` | Vibration | Bearing impulsiveness normalised |
| `vib_energy_high` | Vibration | High-frequency bearing harmonics |
| `temp_mean` | Thermal | Operating temperature |
| `temp_max` | Thermal | Peak temperature |
| `temp_rise` | Thermal | Thermal excursion |
| `tracking_err_rms` | Kinematic | Position accuracy |
| `tracking_err_max` | Kinematic | Worst-case tracking error |
| `tracking_err_p2p` | Kinematic | Peak-to-peak backlash indicator |
| `pos_rms` | Kinematic | Work amplitude normalisation |

## 5. Alert Severity Tiers

| Tier | Health Index | Trigger | Recommended Action |
|---|---|---|---|
| ADVISORY | Any | RUL < 150 cycles | Schedule inspection at next C-check |
| WARNING | < 0.55 | Significant degradation | Plan maintenance at next layover |
| CRITICAL | < 0.35 | Severe degradation | Inspect before next revenue flight |

## 6. What Is Simulated vs. Real

| Component | Status |
|---|---|
| EMA physics model | Simulated (representative parameters, not from a specific OEM datasheet) |
| Degradation trajectories | Simulated (Weibull/Arrhenius models, not from real fleet data) |
| Sensor noise | Simulated (Gaussian additive noise, representative levels) |
| Feature extraction | Fully implemented — runs on real or simulated data |
| RUL prediction model | Fully implemented — trained on simulated data |
| Fault classifier | Fully implemented — trained on simulated data |
| SQLite database | Fully implemented — production-ready schema |
| REST API | Fully implemented — handles real POST /api/ingest payloads |
| Web dashboard | Fully implemented — works with real or simulated data |
| Onboard data logger | Not implemented (requires hardware) |
| Real fleet data | Not available — requires airline partnership |
| DO-178C certification | Not applicable (research prototype) |
