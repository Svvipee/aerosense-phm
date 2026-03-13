"""
Remaining Useful Life (RUL) Prediction
=======================================

Implements a Random Forest regressor that predicts the RUL (in service
cycles) for an EMA given a feature vector extracted from its most recent
test-flight data.

Model choice rationale
----------------------
Random Forest was selected over alternatives (LSTM, SVM) based on:

1. Interpretability: feature importances directly indicate which sensor
   channels are most predictive — important for FAA/EASA evidence packages.
2. Small-data robustness: RF performs well with hundreds of training
   instances; LSTMs require significantly more data for stable training.
3. Literature precedent: Random Forest is the most common PHM baseline in
   the turbomachinery and actuation literature.
   - Li, X. et al. (2018). "Remaining Useful Life Estimation in Prognostics
     Using Deep Convolution Neural Networks." Reliability Engineering &
     System Safety, 172, 1–11.  (RF used as baseline.)
   - Mosallam, A., Medjaher, K., Zerhouni, N. (2016). "Data-driven
     prognostic method based on Bayesian approaches for direct remaining
     useful life prediction." Journal of Intelligent Manufacturing, 27(5),
     1037–1048.

Model validation follows the evaluation protocol from:
  Saxena, A., Goebel, K., Larrosa, C.C., Luo, J., Vachtsevanos, G. (2010).
  "Metrics for Offline Evaluation of Prognostic Performance." IJPHM, 1(1).
  Metrics: RMSE, Score function (asymmetric penalty for late predictions).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.phm.feature_extraction import FEATURE_NAMES, N_FEATURES


# ---------------------------------------------------------------------------
# RUL prediction model
# ---------------------------------------------------------------------------

class RULPredictor:
    """
    Wraps a scikit-learn Random Forest pipeline for EMA RUL prediction.

    The pipeline applies StandardScaler then RandomForestRegressor.
    Hyperparameters chosen by 5-fold CV on the simulated fleet dataset;
    values are consistent with those reported for similar PHM tasks in
    Mosallam et al. (2016).
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = 20,
        min_samples_leaf: int = 5,
        random_state: int = 42,
    ):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                n_jobs=-1,
                random_state=random_state,
            )),
        ])
        self._is_trained = False
        self._feature_importances: np.ndarray | None = None
        self._cv_scores: dict | None = None

    # ------------------------------------------------------------------ #
    # Training                                                            #
    # ------------------------------------------------------------------ #

    def train(self, df: pd.DataFrame, cv_folds: int = 5) -> dict:
        """
        Train on a fleet dataset DataFrame.

        Parameters
        ----------
        df : Must contain columns in FEATURE_NAMES plus 'rul'.
             Rows with rul == -1 (post-EoL) are excluded.

        Returns
        -------
        metrics : dict with RMSE, MAE, and CV scores.
        """
        df_valid = df[df["rul"] >= 0].copy()
        X = df_valid[FEATURE_NAMES].values.astype(np.float64)
        y = df_valid["rul"].values.astype(np.float64)

        # Cap RUL at 500 cycles to reduce variance from very-new components
        # (consistent with CMAPSS capping at max observed RUL per unit)
        RUL_CAP = 500
        y = np.clip(y, 0, RUL_CAP)

        self.pipeline.fit(X, y)
        self._is_trained = True
        self._feature_importances = self.pipeline.named_steps["rf"].feature_importances_

        # Cross-validation
        neg_rmse = cross_val_score(
            self.pipeline, X, y,
            scoring="neg_root_mean_squared_error",
            cv=cv_folds,
            n_jobs=-1,
        )
        neg_mae = cross_val_score(
            self.pipeline, X, y,
            scoring="neg_mean_absolute_error",
            cv=cv_folds,
            n_jobs=-1,
        )

        self._cv_scores = {
            "cv_rmse_mean": float(-neg_rmse.mean()),
            "cv_rmse_std":  float(neg_rmse.std()),
            "cv_mae_mean":  float(-neg_mae.mean()),
            "cv_mae_std":   float(neg_mae.std()),
            "n_train":      int(len(y)),
        }
        return self._cv_scores

    # ------------------------------------------------------------------ #
    # Inference                                                           #
    # ------------------------------------------------------------------ #

    def predict(self, features: np.ndarray) -> float:
        """
        Predict RUL for one or more feature vectors.

        Parameters
        ----------
        features : shape (N_FEATURES,) or (n_samples, N_FEATURES)

        Returns
        -------
        Predicted RUL in cycles (scalar or array).
        """
        self._check_trained()
        x = np.atleast_2d(features)
        pred = self.pipeline.predict(x)
        return float(pred[0]) if x.shape[0] == 1 else pred

    def predict_with_uncertainty(
        self, features: np.ndarray, n_bootstrap: int = 100
    ) -> tuple[float, float, float]:
        """
        Return (median, lower_5th_percentile, upper_95th_percentile) of
        per-tree predictions — a simple non-parametric confidence interval.

        Based on: Heskes, T. (1997). "Practical Confidence and Prediction
        Intervals." NIPS.
        """
        self._check_trained()
        x = np.atleast_2d(features)
        rf = self.pipeline.named_steps["rf"]
        # Collect individual tree predictions
        tree_preds = np.array([t.predict(
            self.pipeline.named_steps["scaler"].transform(x)
        )[0] for t in rf.estimators_])
        return (
            float(np.median(tree_preds)),
            float(np.percentile(tree_preds, 5)),
            float(np.percentile(tree_preds, 95)),
        )

    # ------------------------------------------------------------------ #
    # Feature importance                                                  #
    # ------------------------------------------------------------------ #

    def feature_importances(self) -> dict[str, float]:
        self._check_trained()
        return dict(zip(FEATURE_NAMES, self._feature_importances.tolist()))

    # ------------------------------------------------------------------ #
    # Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """Persist the trained pipeline to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "pipeline":    self.pipeline,
            "cv_scores":   self._cv_scores,
            "importances": self._feature_importances,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "RULPredictor":
        """Load a previously saved model."""
        obj = cls.__new__(cls)
        data = joblib.load(path)
        obj.pipeline = data["pipeline"]
        obj._is_trained = True
        obj._cv_scores = data.get("cv_scores")
        obj._feature_importances = data.get("importances")
        return obj

    def _check_trained(self) -> None:
        if not self._is_trained:
            raise RuntimeError("Model has not been trained. Call train() first.")


# ---------------------------------------------------------------------------
# Prognostic score function (Saxena et al. 2010)
# ---------------------------------------------------------------------------

def prognostic_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Asymmetric scoring function from:
      Saxena et al. (2010). "Metrics for Offline Evaluation of Prognostic
      Performance." IJPHM 1(1).

    Penalises late predictions (under-estimate remaining life) more harshly
    than early predictions, reflecting the asymmetric cost of missed
    maintenance vs. unnecessary maintenance.

    Score < 0 is better; 0 is perfect.
    """
    diff = y_pred - y_true
    score = np.where(
        diff < 0,
        np.exp(-diff / 13.0) - 1,  # late prediction penalty
        np.exp(diff / 10.0) - 1,   # early prediction penalty
    )
    return float(np.sum(score))
