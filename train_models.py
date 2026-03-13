"""
Train PHM Models
=================

Generates the simulated fleet dataset, trains the RUL predictor and
fault classifier, and saves them to data/models/.

Run time on a modern laptop: ~3–8 minutes depending on CPU.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

from simulation.fleet_simulator import FleetSimulator
from src.phm.rul_prediction import RULPredictor
from src.phm.fault_classification import FaultClassifier


def main():
    print("=" * 60)
    print("  AeroSense PHM — Model Training Pipeline")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Step 1: Generate simulation dataset                                 #
    # ------------------------------------------------------------------ #
    dataset_path = ROOT / "data" / "processed" / "fleet_dataset.parquet"

    if dataset_path.exists():
        print(f"\n[1/3] Loading existing dataset from {dataset_path}")
        df = pd.read_parquet(dataset_path)
        print(f"      Loaded {len(df):,} rows, {len(df.columns)} columns")
    else:
        print("\n[1/3] Generating fleet simulation dataset…")
        print("      This may take 3–8 minutes on a standard laptop.")
        t0 = time.time()
        sim = FleetSimulator(output_dir=ROOT / "data" / "processed", seed=2024)
        # 5 s per cycle at 5 ms dt = 1000 steps/cycle — fast enough for batch training
        df = sim.generate_dataset(
            cycles_per_aircraft=500,
            test_flight_duration_s=5.0,
            dt=0.005,
            verbose=True,
        )
        print(f"      Generated in {time.time()-t0:.1f}s")

    print(f"      Dataset shape: {df.shape}")
    print(f"      Health index range: [{df['health_index'].min():.3f}, {df['health_index'].max():.3f}]")
    print(f"      RUL range (valid): [{df[df['rul']>=0]['rul'].min():.0f}, {df[df['rul']>=0]['rul'].max():.0f}] cycles")

    # ------------------------------------------------------------------ #
    # Step 2: Train RUL predictor                                        #
    # ------------------------------------------------------------------ #
    print("\n[2/3] Training RUL predictor (Random Forest, 5-fold CV)…")
    t0 = time.time()
    rul_predictor = RULPredictor(n_estimators=200, max_depth=20, random_state=42)
    rul_metrics = rul_predictor.train(df, cv_folds=5)
    print(f"      Training time: {time.time()-t0:.1f}s")
    print(f"      CV RMSE: {rul_metrics['cv_rmse_mean']:.1f} ± {rul_metrics['cv_rmse_std']:.1f} cycles")
    print(f"      CV MAE:  {rul_metrics['cv_mae_mean']:.1f} ± {rul_metrics['cv_mae_std']:.1f} cycles")

    rul_path = ROOT / "data" / "models" / "rul_model.joblib"
    rul_predictor.save(rul_path)
    print(f"      Saved → {rul_path}")

    # Top 5 feature importances
    imps = sorted(rul_predictor.feature_importances().items(), key=lambda x: -x[1])
    print("      Top 5 features:")
    for name, imp in imps[:5]:
        print(f"        {name:<30} {imp*100:.1f}%")

    # ------------------------------------------------------------------ #
    # Step 3: Train fault classifier                                     #
    # ------------------------------------------------------------------ #
    print("\n[3/3] Training fault classifier (Gradient Boosting)…")
    t0 = time.time()
    fault_clf = FaultClassifier()
    fault_metrics = fault_clf.train(df)
    print(f"      Training time: {time.time()-t0:.1f}s")
    print(f"      Training size: {fault_metrics['n_train']:,} samples")
    print("      Class distribution:")
    for label, count in fault_metrics["class_distribution"].items():
        print(f"        {label:<30} {count:,}")
    print("\n      Classification report (training set):")
    for line in fault_metrics["classification_report"].split("\n"):
        print(f"        {line}")

    fault_path = ROOT / "data" / "models" / "fault_model.joblib"
    fault_clf.save(fault_path)
    print(f"\n      Saved → {fault_path}")

    print("\n" + "=" * 60)
    print("  Training complete.")
    print("  Next: python run_demo.py   (populate DB + launch dashboard)")
    print("=" * 60)


if __name__ == "__main__":
    main()
