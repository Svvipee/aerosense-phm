"""
Demo Population + Dashboard Launcher
======================================

1. Loads (or trains) the PHM models.
2. Populates the SQLite database with a representative set of aircraft
   and actuator health snapshots from the simulation.
3. Starts the Flask dashboard on http://localhost:5050

Run after train_models.py:
    python run_demo.py
"""

import sys
import time
import threading
import webbrowser
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.data.database import Database
from src.phm.rul_prediction import RULPredictor
from src.phm.fault_classification import FaultClassifier
from src.phm.feature_extraction import FEATURE_NAMES
from src.dashboard.app import create_app


# ---------------------------------------------------------------------------
# Aircraft / tail definitions for the demo
# ---------------------------------------------------------------------------

DEMO_FLEET = [
    {"aircraft_id": "A320_family_000", "ac_type": "A320", "tail_number": "N-AES001"},
    {"aircraft_id": "A320_family_001", "ac_type": "A320", "tail_number": "N-AES002"},
    {"aircraft_id": "A320_family_002", "ac_type": "A321", "tail_number": "N-AES003"},
    {"aircraft_id": "B737_family_000", "ac_type": "B737-800", "tail_number": "N-AES010"},
    {"aircraft_id": "B737_family_001", "ac_type": "B737-800", "tail_number": "N-AES011"},
    {"aircraft_id": "B737_family_002", "ac_type": "B737 MAX", "tail_number": "N-AES012"},
    {"aircraft_id": "E175_family_000", "ac_type": "E175",   "tail_number": "N-AES020"},
    {"aircraft_id": "E175_family_001", "ac_type": "E175",   "tail_number": "N-AES021"},
]

ACTUATOR_TYPES = ["aileron_L", "aileron_R", "elevator_L", "elevator_R", "rudder"]


def populate_database(db: Database, df: pd.DataFrame, rul_model: RULPredictor, fault_model: FaultClassifier):
    """
    Write a representative snapshot history for each demo aircraft/actuator.
    Uses the simulation dataset to find matching rows.
    """
    print("\n[2/3] Populating database…")

    for ac in DEMO_FLEET:
        db.upsert_aircraft(ac["aircraft_id"], ac["ac_type"], ac["tail_number"])
        ac_rows = df[df["aircraft_id"] == ac["aircraft_id"]]

        if len(ac_rows) == 0:
            # Aircraft not in dataset — synthesize plausible data
            ac_rows = _synthesise_aircraft_data(ac["aircraft_id"])

        for act_type in ACTUATOR_TYPES:
            uid = f"{ac['aircraft_id']}_{act_type}"
            db.upsert_actuator(uid, ac["aircraft_id"], act_type)

            act_rows = ac_rows[ac_rows["actuator_id"] == act_type]
            if len(act_rows) == 0:
                act_rows = _synthesise_actuator_data(act_type)

            # Take up to last 100 cycles
            act_rows = act_rows.sort_values("cycle").tail(100)

            for _, row in act_rows.iterrows():
                feat = np.array([row.get(k, 0.0) for k in FEATURE_NAMES])
                rul_med, rul_lo, rul_hi = rul_model.predict_with_uncertainty(feat)
                fault_result = fault_model.predict(feat)

                db.insert_snapshot(
                    actuator_uid=uid,
                    cycle=int(row["cycle"]),
                    health_index=float(row.get("health_index", 1.0)),
                    bearing_health=float(row.get("bearing_health", 1.0)),
                    winding_health=float(row.get("winding_health", 1.0)),
                    backlash_health=float(row.get("backlash_health", 1.0)),
                    rul_predicted=max(0.0, rul_med),
                    rul_lower=max(0.0, rul_lo),
                    rul_upper=max(0.0, rul_hi),
                    fault_class=fault_result["fault_class"],
                    fault_label=fault_result["fault_label"],
                    fault_confidence=fault_result["confidence"],
                    features={k: float(row.get(k, 0.0)) for k in FEATURE_NAMES},
                )

        print(f"   Populated: {ac['aircraft_id']} ({ac['tail_number']})")

    print("   Database population complete.")


def _synthesise_aircraft_data(aircraft_id: str) -> pd.DataFrame:
    """Create plausible synthetic rows when the aircraft isn't in the sim dataset."""
    from simulation.fleet_simulator import FleetSimulator, ACTUATOR_TYPES as ACT_TYPES
    rng = np.random.default_rng(abs(hash(aircraft_id)) % 2**32)
    records = []
    for act_name in ACTUATOR_TYPES:
        n_cycles = 400
        for i in range(n_cycles):
            age = i / n_cycles
            bh  = np.clip(1.0 - age * rng.uniform(0.3, 0.9), 0.05, 1.0)
            wh  = np.clip(1.0 - age * rng.uniform(0.1, 0.5), 0.1, 1.0)
            blh = np.clip(1.0 - age * rng.uniform(0.05, 0.4), 0.1, 1.0)
            hi  = (bh * wh * blh) ** (1/3)
            row = {
                "aircraft_id": aircraft_id, "actuator_id": act_name,
                "cycle": i, "age_fraction": age,
                "rul": max(0, (1.0 - age) * 600),
                "health_index": hi,
                "bearing_health": bh, "winding_health": wh, "backlash_health": blh,
            }
            for feat in FEATURE_NAMES:
                row[feat] = float(rng.normal(0.5, 0.1))
            records.append(row)
    return pd.DataFrame(records)


def _synthesise_actuator_data(act_type: str) -> pd.DataFrame:
    rng = np.random.default_rng(abs(hash(act_type)) % 2**32)
    n_cycles = 200
    records = []
    for i in range(n_cycles):
        age = i / n_cycles
        bh  = np.clip(1.0 - age * 0.6, 0.05, 1.0)
        wh  = np.clip(1.0 - age * 0.3, 0.1, 1.0)
        blh = np.clip(1.0 - age * 0.2, 0.1, 1.0)
        hi  = (bh * wh * blh) ** (1/3)
        row = {
            "aircraft_id": "unknown", "actuator_id": act_type,
            "cycle": i, "age_fraction": age,
            "rul": max(0, (1.0 - age) * 400),
            "health_index": hi,
            "bearing_health": bh, "winding_health": wh, "backlash_health": blh,
        }
        for feat in FEATURE_NAMES:
            row[feat] = float(rng.normal(0.5, 0.1))
        records.append(row)
    return pd.DataFrame(records)


def main():
    print("=" * 60)
    print("  AeroSense PHM — Demo Launcher")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # [1/3] Load models                                                   #
    # ------------------------------------------------------------------ #
    print("\n[1/3] Loading trained models…")
    rul_path   = ROOT / "data" / "models" / "rul_model.joblib"
    fault_path = ROOT / "data" / "models" / "fault_model.joblib"

    if not rul_path.exists() or not fault_path.exists():
        print("  Models not found. Running training pipeline first…")
        import subprocess, sys as _sys
        subprocess.run([_sys.executable, str(ROOT / "train_models.py")], check=True)

    rul_model   = RULPredictor.load(rul_path)
    fault_model = FaultClassifier.load(fault_path)
    print("  Models loaded.")

    # ------------------------------------------------------------------ #
    # [2/3] Populate DB                                                   #
    # ------------------------------------------------------------------ #
    db_path = ROOT / "data" / "ema_phm.db"
    db = Database(db_path)

    existing = db.fleet_summary()
    if existing:
        print(f"\n[2/3] Database already has {len(existing)} aircraft — skipping population.")
        print("      Delete data/ema_phm.db to re-populate from scratch.")
    else:
        dataset_path = ROOT / "data" / "processed" / "fleet_dataset.parquet"
        if dataset_path.exists():
            df = pd.read_parquet(dataset_path)
        else:
            print("  No dataset found — synthesising demo data…")
            df = pd.DataFrame()
        populate_database(db, df, rul_model, fault_model)

    # ------------------------------------------------------------------ #
    # [3/3] Launch dashboard                                              #
    # ------------------------------------------------------------------ #
    print("\n[3/3] Starting dashboard…")
    app = create_app(str(db_path))

    url = "http://localhost:5050"

    def open_browser():
        time.sleep(1.2)
        webbrowser.open(url)

    threading.Thread(target=open_browser, daemon=True).start()

    print(f"\n  Dashboard running at {url}")
    print("  Press Ctrl+C to stop.\n")
    app.run(host="0.0.0.0", port=5050, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
