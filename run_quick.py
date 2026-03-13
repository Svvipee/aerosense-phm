"""
Quick-Start Demo (no full simulation required)
===============================================

Generates synthetic demo data and trains minimal models in ~15-30 seconds,
then launches the dashboard at http://localhost:5050.

For full-fidelity training with physics simulation, run train_models.py first.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.phm.feature_extraction import FEATURE_NAMES
from src.phm.rul_prediction import RULPredictor
from src.phm.fault_classification import FaultClassifier, assign_fault_class
from src.data.database import Database
from src.dashboard.app import create_app


# ---------------------------------------------------------------------------
# Synthetic demo fleet
# ---------------------------------------------------------------------------

DEMO_AIRCRAFT = [
    {"id": "A320_000", "type": "A320",     "tail": "N-AES001", "age": 0.82},
    {"id": "A320_001", "type": "A321",     "tail": "N-AES002", "age": 0.45},
    {"id": "B737_000", "type": "B737-800", "tail": "N-AES010", "age": 0.91},
    {"id": "B737_001", "type": "B737 MAX", "tail": "N-AES011", "age": 0.33},
    {"id": "B737_002", "type": "B737-800", "tail": "N-AES012", "age": 0.60},
    {"id": "E175_000", "type": "E175",     "tail": "N-AES020", "age": 0.70},
    {"id": "E175_001", "type": "E175",     "tail": "N-AES021", "age": 0.25},
    {"id": "A320_002", "type": "A320neo",  "tail": "N-AES003", "age": 0.15},
]
ACTUATOR_TYPES = ["aileron_L", "aileron_R", "elevator_L", "elevator_R", "rudder"]


def _make_health_series(age: float, seed: int, n: int = 80) -> tuple:
    """Generate a realistic degradation history for one EMA."""
    rng = np.random.default_rng(seed)
    # Start health at install and degrade to current age
    start_h = min(1.0, 1.0 - age * rng.uniform(0.1, 0.6))
    end_h   = max(0.05, start_h - rng.uniform(0.05, 0.40))
    noise   = rng.normal(0, 0.015, n).cumsum() * 0.005

    bh  = np.clip(np.linspace(min(1.0, start_h + 0.1), end_h, n) + noise, 0.05, 1.0)
    wh  = np.clip(np.linspace(min(1.0, start_h + 0.05), end_h + 0.1, n) + rng.normal(0,0.01,n).cumsum()*0.003, 0.1, 1.0)
    blh = np.clip(np.linspace(min(1.0, start_h + 0.08), end_h + 0.15, n) + rng.normal(0,0.01,n).cumsum()*0.002, 0.1, 1.0)
    hi  = (bh * wh * blh) ** (1/3)
    return bh, wh, blh, hi


def generate_synthetic_dataset(n_per_ema: int = 80) -> pd.DataFrame:
    records = []
    rng_global = np.random.default_rng(2024)

    for ac in DEMO_AIRCRAFT:
        for act in ACTUATOR_TYPES:
            seed = int(rng_global.integers(0, 2**31))
            rng = np.random.default_rng(seed)
            bh, wh, blh, hi = _make_health_series(ac["age"], seed, n_per_ema)
            base_cycle = int(ac["age"] * 5000)

            for i in range(n_per_ema):
                rul = max(0.0, (1.0 - (base_cycle + i) / 6000) * 500 + rng.normal(0, 15))
                feat = {
                    "curr_rms":          0.8 + (1 - bh[i]) * 0.6 + rng.normal(0, 0.05),
                    "curr_peak":         1.5 + (1 - bh[i]) * 1.0 + rng.normal(0, 0.1),
                    "curr_crest":        1.4 + (1 - wh[i]) * 0.8 + rng.normal(0, 0.05),
                    "curr_thd":          0.05 + (1 - wh[i]) * 0.20 + rng.normal(0, 0.01),
                    "curr_f1":           12.0 + rng.normal(0, 0.5),
                    "vib_rms":           0.05 + (1 - bh[i]) * 1.5 + rng.normal(0, 0.02),
                    "vib_peak":          0.15 + (1 - bh[i]) * 3.0 + rng.normal(0, 0.05),
                    "vib_kurtosis":      0.1  + (1 - bh[i]) * 8.0 + rng.normal(0, 0.2),
                    "vib_crest":         2.0  + (1 - bh[i]) * 4.0 + rng.normal(0, 0.1),
                    "vib_energy_high":   50   + (1 - bh[i]) * 800 + rng.normal(0, 10),
                    "temp_mean":         45   + (1 - wh[i]) * 30  + rng.normal(0, 1),
                    "temp_max":          55   + (1 - wh[i]) * 40  + rng.normal(0, 1),
                    "temp_rise":         10   + (1 - wh[i]) * 20  + rng.normal(0, 0.5),
                    "tracking_err_rms":  0.0001 + (1 - blh[i]) * 0.002 + rng.normal(0, 0.00005),
                    "tracking_err_max":  0.0005 + (1 - blh[i]) * 0.005 + rng.normal(0, 0.0001),
                    "tracking_err_p2p":  0.001  + (1 - blh[i]) * 0.008 + rng.normal(0, 0.0001),
                    "pos_rms":           0.008  + rng.normal(0, 0.001),
                }
                feat = {k: float(np.clip(v, 0, None)) for k, v in feat.items()}
                records.append({
                    "aircraft_id": ac["id"], "actuator_id": act,
                    "cycle": base_cycle + i,
                    "health_index": float(hi[i]),
                    "bearing_health": float(bh[i]),
                    "winding_health": float(wh[i]),
                    "backlash_health": float(blh[i]),
                    "rul": float(rul),
                    **feat,
                })
    return pd.DataFrame(records)


def main():
    print("=" * 60)
    print("  AeroSense PHM — Quick Demo Launcher")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Generate synthetic data + train fast models                        #
    # ------------------------------------------------------------------ #
    rul_path   = ROOT / "data" / "models" / "rul_model.joblib"
    fault_path = ROOT / "data" / "models" / "fault_model.joblib"

    if rul_path.exists() and fault_path.exists():
        print("\n[1/3] Loading existing trained models…")
        rul_model   = RULPredictor.load(rul_path)
        fault_model = FaultClassifier.load(fault_path)
    else:
        print("\n[1/3] Training quick models on synthetic data (~20 s)…")
        df_train = generate_synthetic_dataset(n_per_ema=150)

        rul_model = RULPredictor(n_estimators=100, max_depth=15, random_state=42)
        m = rul_model.train(df_train, cv_folds=3)
        print(f"      RUL CV RMSE: {m['cv_rmse_mean']:.1f} ± {m['cv_rmse_std']:.1f} cycles")
        rul_model.save(rul_path)

        fault_model = FaultClassifier()
        fault_model.train(df_train)
        fault_model.save(fault_path)
        print("      Models trained and saved.")

    # ------------------------------------------------------------------ #
    # Populate database                                                   #
    # ------------------------------------------------------------------ #
    db_path = ROOT / "data" / "ema_phm.db"
    db = Database(db_path)

    existing = db.fleet_summary()
    if existing:
        print(f"\n[2/3] Database has {len(existing)} aircraft — skipping population.")
        print("      Delete data/ema_phm.db to reset.")
    else:
        print("\n[2/3] Populating database with demo fleet…")
        df_demo = generate_synthetic_dataset(n_per_ema=80)

        for ac in DEMO_AIRCRAFT:
            db.upsert_aircraft(ac["id"], ac["type"], ac["tail"])
            ac_rows = df_demo[df_demo["aircraft_id"] == ac["id"]]

            for act in ACTUATOR_TYPES:
                uid = f"{ac['id']}_{act}"
                db.upsert_actuator(uid, ac["id"], act)
                act_rows = ac_rows[ac_rows["actuator_id"] == act].sort_values("cycle")

                for _, row in act_rows.iterrows():
                    feat = np.array([row[k] for k in FEATURE_NAMES])
                    rul_med, rul_lo, rul_hi = rul_model.predict_with_uncertainty(feat)
                    fr = fault_model.predict(feat)

                    bh, wh, blh = row["bearing_health"], row["winding_health"], row["backlash_health"]
                    hi = (bh * wh * blh) ** (1/3)

                    db.insert_snapshot(
                        actuator_uid=uid, cycle=int(row["cycle"]),
                        health_index=hi, bearing_health=bh, winding_health=wh,
                        backlash_health=blh,
                        rul_predicted=max(0.0, rul_med),
                        rul_lower=max(0.0, rul_lo), rul_upper=max(0.0, rul_hi),
                        fault_class=fr["fault_class"], fault_label=fr["fault_label"],
                        fault_confidence=fr["confidence"],
                        features={k: float(row[k]) for k in FEATURE_NAMES},
                    )

                    # Generate alerts for degraded actuators
                    from src.dashboard.app import _maybe_alert
                    _maybe_alert(db, uid, ac["id"], int(row["cycle"]), hi,
                                 max(0.0, rul_med), fr["fault_class"], fr["fault_label"])

            print(f"   OK {ac['type']} {ac['tail']}")

        print("   Database populated.")

    # ------------------------------------------------------------------ #
    # Launch dashboard                                                    #
    # ------------------------------------------------------------------ #
    import threading, time, webbrowser
    print("\n[3/3] Starting dashboard at http://localhost:5050 …\n")
    app = create_app(str(db_path))

    def open_browser():
        time.sleep(1.2)
        webbrowser.open("http://localhost:5050")

    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host="0.0.0.0", port=5050, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
