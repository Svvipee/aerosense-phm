"""
Model Validation and Comparison Study
======================================

Compares five ML approaches for EMA Remaining Useful Life (RUL) regression
on synthetic data that follows the same physics-based degradation model
used in training (representative of CMAPSS-style benchmark methodology,
Saxena et al. 2008).

Algorithms compared
-------------------
1. Random Forest (RF)        — chosen algorithm in production
2. Gradient Boosting (GBR)   — tree ensemble baseline
3. Support Vector Regression (SVR) — kernel-based baseline
4. Multilayer Perceptron (MLP)     — neural network baseline
5. Ridge Regression          — linear baseline (lower bound)

Metrics reported (all from Saxena et al. 2010 PHM scoring framework)
----------------------------------------------------------------------
- CV RMSE (5-fold)     : primary accuracy metric
- CV MAE  (5-fold)     : mean absolute error
- Prognostic Score     : asymmetric score penalising late predictions more
                         than early (penalty constants a1=13, a2=10)
- Training time [s]    : wall-clock fit time

Results are written to data/validation_results.json.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from run_quick import generate_synthetic_dataset
from src.phm.feature_extraction import FEATURE_NAMES
from src.phm.rul_prediction import prognostic_score

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

print("=" * 65)
print("  AeroSense PHM — Model Validation Study")
print("=" * 65)
print("\nGenerating synthetic dataset (n_per_ema=150) …")
df = generate_synthetic_dataset(n_per_ema=150)
print(f"  Dataset: {len(df):,} samples, {df['aircraft_id'].nunique()} aircraft, "
      f"{df['actuator_id'].nunique()} actuator types")

X = df[FEATURE_NAMES].values
y = df["rul"].values

# ---------------------------------------------------------------------------
# Model zoo
# ---------------------------------------------------------------------------

models = {
    "Random Forest (n=200, depth=20)": Pipeline([
        ("rf", RandomForestRegressor(
            n_estimators=200, max_depth=20, min_samples_leaf=3,
            n_jobs=-1, random_state=42,
        ))
    ]),
    "Gradient Boosting (n=200, depth=5)": Pipeline([
        ("gbr", GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42,
        ))
    ]),
    "SVR (RBF kernel, C=100)": Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=100.0, gamma="scale", epsilon=5.0)),
    ]),
    "MLP (100-50, relu, adam)": Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(100, 50), activation="relu", solver="adam",
            max_iter=300, random_state=42, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=15,
        ))
    ]),
    "Ridge Regression (alpha=1.0)": Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ]),
}

# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

cv = KFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print(f"\n{'Model':<42} {'CV RMSE':>9} {'± std':>7} {'CV MAE':>8} {'P-Score':>9} {'Time(s)':>8}")
print("-" * 85)

for name, pipe in models.items():
    t0 = time.time()

    # RMSE scores
    rmse_scores = np.sqrt(-cross_val_score(
        pipe, X, y, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1
    ))
    mae_scores = -cross_val_score(
        pipe, X, y, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1
    )
    train_time = time.time() - t0

    # Prognostic score: fit on 80%, score on 20% (one representative fold)
    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
    pipe_clone = pipe.__class__(**pipe.get_params()) if False else \
        type(pipe)(steps=pipe.steps)  # fresh clone

    from sklearn.base import clone as sk_clone
    pipe_scored = sk_clone(pipe)
    pipe_scored.fit(X_tr, y_tr)
    y_pred_te = pipe_scored.predict(X_te)
    pscore = prognostic_score(y_te, y_pred_te)

    results[name] = {
        "cv_rmse_mean": float(rmse_scores.mean()),
        "cv_rmse_std":  float(rmse_scores.std()),
        "cv_mae_mean":  float(mae_scores.mean()),
        "cv_mae_std":   float(mae_scores.std()),
        "prognostic_score": float(pscore),
        "train_time_s": round(train_time, 2),
    }

    print(f"{name:<42} {rmse_scores.mean():>9.2f} {rmse_scores.std():>7.2f} "
          f"{mae_scores.mean():>8.2f} {pscore:>9.1f} {train_time:>8.2f}")

# ---------------------------------------------------------------------------
# Identify winner and save results
# ---------------------------------------------------------------------------

best = min(results, key=lambda k: results[k]["cv_rmse_mean"])
print(f"\nBest model by CV RMSE: {best}")
print(f"  RMSE: {results[best]['cv_rmse_mean']:.2f} ± {results[best]['cv_rmse_std']:.2f}")
print(f"  MAE:  {results[best]['cv_mae_mean']:.2f}")
print(f"  Prognostic Score: {results[best]['prognostic_score']:.1f}")

out_path = ROOT / "data" / "validation_results.json"
out_path.parent.mkdir(parents=True, exist_ok=True)

payload = {
    "dataset_info": {
        "n_samples": int(len(df)),
        "n_aircraft": int(df["aircraft_id"].nunique()),
        "n_actuator_types": int(df["actuator_id"].nunique()),
        "n_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "cv_folds": 5,
        "methodology": (
            "Synthetic dataset following CMAPSS-style degradation "
            "(Saxena et al. 2008). 5-fold CV, shuffled. "
            "Prognostic score from Saxena et al. (2010) with a1=13, a2=10."
        ),
    },
    "models": results,
    "best_model_by_rmse": best,
}

with open(out_path, "w") as f:
    json.dump(payload, f, indent=2)
print(f"\nResults saved to {out_path.relative_to(ROOT)}")
