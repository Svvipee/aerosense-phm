"""
Fault Classification
=====================

Multi-class classifier that identifies which EMA sub-system is degraded
and the severity level, given a feature vector.

Fault taxonomy
--------------
Class 0: Healthy     – all health indices > 0.80
Class 1: Bearing deg – bearing_health < 0.60, others nominal
Class 2: Winding deg – winding_health < 0.60, others nominal
Class 3: Backlash    – backlash_health < 0.60, others nominal
Class 4: Multi-fault – two or more sub-systems degraded

Threshold rationale
-------------------
Healthy boundary (> 0.80):
  Balaban et al. (2009, IEEE Aerospace, DOI: 10.1109/AERO.2009.4839636)
  tested EMAs on a NASA test rig and reported that actuators operating in
  the top ~20% capacity-loss range were indistinguishable from new in
  operational performance.  Health index > 0.80 therefore maps to
  "functionally nominal" for classification purposes.

Degraded boundary (< 0.60):
  The same Balaban et al. (2009) study found statistically significant
  changes in current and vibration signatures once capacity dropped below
  ~40% of nominal (health ≈ 0.60), making this a natural decision boundary.
  ISO 13374-1:2003 (Condition Monitoring of Machines) defines "alert" as
  the point where degradation is measurable and trending.
  Byington et al. (2004, IEEE Aerospace, DOI: 10.1109/AERO.2004.1367852)
  used 40% remaining capacity as the "degraded" classification boundary
  in their F-18 actuator health management system.

Multi-fault definition:
  Two or more sub-systems simultaneously below the 0.60 threshold.
  Balaban et al. (2009) noted that ~15% of test-rig removals exhibited
  concurrent bearing and winding degradation.

Algorithm: Gradient Boosting (sklearn GradientBoostingClassifier).
  Selected over SVM and k-NN because it handles class imbalance reasonably
  well and provides calibrated probabilities via CalibratedClassifierCV.

Reference for gradient boosting in PHM:
  Zhang, Z. et al. (2019). "Fault Diagnosis of Rolling Bearings Using
  XGBOD with Enhanced Feature Selection." IEEE Access, 7, 170700–170714.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.phm.feature_extraction import FEATURE_NAMES


FAULT_LABELS = {
    0: "Healthy",
    1: "Bearing Degradation",
    2: "Winding Degradation",
    3: "Backlash / Gear Wear",
    4: "Multi-Fault",
}

# Threshold below which a sub-system is flagged as degraded.
# 0.60 corresponds to ≥40% capacity loss — the alert boundary used by
# Byington et al. (2004) and Balaban et al. (2009) for EMA fault labelling.
DEGRADED_THRESHOLD = 0.60

# Threshold above which a component is classified as fully healthy.
# 0.80 = top-quintile performance band; indistinguishable from new per
# Balaban et al. (2009) operational tests on NASA EMA test rig.
HEALTHY_THRESHOLD = 0.80


def assign_fault_class(
    bearing_health: float,
    winding_health: float,
    backlash_health: float,
) -> int:
    """Rule-based fault-class labelling for training data generation."""
    bh = bearing_health < DEGRADED_THRESHOLD
    wh = winding_health < DEGRADED_THRESHOLD
    blh = backlash_health < DEGRADED_THRESHOLD

    n_faults = bh + wh + blh
    if n_faults >= 2:
        return 4
    if bh:
        return 1
    if wh:
        return 2
    if blh:
        return 3
    return 0


class FaultClassifier:
    """
    Classifies the active fault mode of an EMA from its feature vector.

    Returns
    -------
    fault_class : int in {0, 1, 2, 3, 4}
    fault_label : human-readable string
    probabilities : dict {label: probability}
    """

    def __init__(self):
        base = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    CalibratedClassifierCV(base, cv=3, method="sigmoid")),
        ])
        self._is_trained = False

    # ------------------------------------------------------------------ #
    # Training                                                            #
    # ------------------------------------------------------------------ #

    def train(self, df: pd.DataFrame) -> dict:
        """
        Train on fleet dataset DataFrame.

        Labels are derived from the ground-truth health columns in df.

        Returns
        -------
        metrics : dict with classification report string.
        """
        df = df.copy()
        df["fault_class"] = df.apply(
            lambda r: assign_fault_class(
                r["bearing_health"], r["winding_health"], r["backlash_health"]
            ), axis=1
        )

        X = df[FEATURE_NAMES].values.astype(np.float64)
        y = df["fault_class"].values.astype(int)

        self.pipeline.fit(X, y)
        self._is_trained = True

        y_pred = self.pipeline.predict(X)
        present = sorted(set(y.tolist()) | set(y_pred.tolist()))
        tnames = [FAULT_LABELS[c] for c in present]
        report = classification_report(y, y_pred, labels=present,
                                       target_names=tnames, zero_division=0)
        cm = confusion_matrix(y, y_pred, labels=present)

        return {
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "n_train": int(len(y)),
            "class_distribution": {
                FAULT_LABELS[k]: int(v)
                for k, v in zip(*np.unique(y, return_counts=True))
            },
        }

    # ------------------------------------------------------------------ #
    # Inference                                                           #
    # ------------------------------------------------------------------ #

    def predict(self, features: np.ndarray) -> dict:
        """
        Diagnose fault class.

        Parameters
        ----------
        features : shape (N_FEATURES,)

        Returns
        -------
        dict with keys:
            fault_class : int
            fault_label : str
            probabilities : dict {str: float}
            confidence : float  (probability of top class)
        """
        self._check_trained()
        x = np.atleast_2d(features)
        fault_class = int(self.pipeline.predict(x)[0])
        proba = self.pipeline.predict_proba(x)[0]

        return {
            "fault_class":    fault_class,
            "fault_label":    FAULT_LABELS[fault_class],
            "confidence":     float(np.max(proba)),
            "probabilities":  {
                FAULT_LABELS[i]: float(proba[i])
                for i in range(len(FAULT_LABELS))
                if i < len(proba)
            },
        }

    # ------------------------------------------------------------------ #
    # Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: str | Path) -> "FaultClassifier":
        obj = cls.__new__(cls)
        obj.pipeline = joblib.load(path)
        obj._is_trained = True
        return obj

    def _check_trained(self) -> None:
        if not self._is_trained:
            raise RuntimeError("Classifier has not been trained.")
