"""
PHM Feature Extraction
=======================

Converts raw time-series sensor data into scalar features suitable
for machine-learning models.

The feature set is divided into four groups aligned with the three
EMA sub-system health indicators:

  Group A – Current signature analysis  → winding / magnetic health
  Group B – Vibration analysis          → bearing health
  Group C – Thermal analysis            → overall thermal margin
  Group D – Kinematic analysis          → mechanical / backlash health

Feature selection methodology follows:
  Nandi, S., Toliyat, H.A., Li, X. (2005). "Condition Monitoring and
  Fault Diagnosis of Electrical Motors — A Review." IEEE Trans. Energy
  Conversion, 20(4), 719–729. DOI: 10.1109/TEC.2005.847955

Vibration statistical features:
  Randall, R.B. (2011). "Vibration-based Condition Monitoring:
  Industrial, Automotive and Aerospace Applications." Wiley. Ch.5.

Kurtosis as bearing fault indicator:
  McFadden, P.D., Smith, J.D. (1984). "Vibration monitoring of rolling
  element bearings by the high-frequency resonance technique — a review."
  Tribology International, 17(1), 3–10.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class FeatureVector:
    """All features extracted from one test-flight data snapshot."""

    # ---- Group A: Current signature features ----
    curr_rms:     float
    curr_peak:    float
    curr_crest:   float    # crest factor = peak / RMS
    curr_thd:     float    # total harmonic distortion proxy
    curr_f1:      float    # fundamental FFT amplitude

    # ---- Group B: Vibration features ----
    vib_rms:          float
    vib_peak:         float
    vib_kurtosis:     float
    vib_crest:        float
    vib_energy_high:  float   # high-frequency band energy (bearing indicator)

    # ---- Group C: Thermal features ----
    temp_mean:  float
    temp_max:   float
    temp_rise:  float

    # ---- Group D: Kinematic / tracking features ----
    tracking_err_rms:  float
    tracking_err_max:  float
    tracking_err_p2p:  float
    pos_rms:           float

    def to_array(self) -> np.ndarray:
        """Return features as a 1-D numpy array (order matches FEATURE_NAMES)."""
        return np.array([
            self.curr_rms, self.curr_peak, self.curr_crest, self.curr_thd, self.curr_f1,
            self.vib_rms, self.vib_peak, self.vib_kurtosis, self.vib_crest, self.vib_energy_high,
            self.temp_mean, self.temp_max, self.temp_rise,
            self.tracking_err_rms, self.tracking_err_max, self.tracking_err_p2p, self.pos_rms,
        ], dtype=np.float64)

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureVector":
        return cls(**{k: d[k] for k in FEATURE_NAMES})


FEATURE_NAMES: list[str] = [
    "curr_rms", "curr_peak", "curr_crest", "curr_thd", "curr_f1",
    "vib_rms", "vib_peak", "vib_kurtosis", "vib_crest", "vib_energy_high",
    "temp_mean", "temp_max", "temp_rise",
    "tracking_err_rms", "tracking_err_max", "tracking_err_p2p", "pos_rms",
]

N_FEATURES = len(FEATURE_NAMES)


def extract_features(
    current:  np.ndarray,
    vibration: np.ndarray,
    temperature: np.ndarray,
    position: np.ndarray,
    position_cmd: np.ndarray,
    dt: float = 0.001,
) -> FeatureVector:
    """
    Extract all scalar health features from one flight's raw sensor streams.

    Parameters
    ----------
    current      : Phase current [A], shape (N,)
    vibration    : Housing vibration RMS [g], shape (N,)
    temperature  : Winding temperature [°C], shape (N,)
    position     : Measured output position [m], shape (N,)
    position_cmd : Commanded position [m], shape (N,)
    dt           : Sample interval [s]

    Returns
    -------
    FeatureVector
    """
    # ---- Group A: Current ---------------------------------------------------
    curr_rms  = _rms(current)
    curr_peak = float(np.max(np.abs(current)))
    curr_cf   = _crest_factor(current)
    fft_mag   = np.abs(np.fft.rfft(current))
    f1_idx    = int(np.argmax(fft_mag[1:]) + 1) if len(fft_mag) > 1 else 1
    curr_f1   = float(fft_mag[f1_idx])
    f3_idx    = min(f1_idx * 3, len(fft_mag) - 1)
    curr_thd  = float(fft_mag[f3_idx]) / (curr_f1 + 1e-12)

    # ---- Group B: Vibration -------------------------------------------------
    vib_rms  = _rms(vibration)
    vib_peak = float(np.max(vibration))
    vib_kurt = _kurtosis(vibration)
    vib_cf   = _crest_factor(vibration)
    vib_fft  = np.abs(np.fft.rfft(vibration))
    # High-frequency band = upper 25 % of FFT (bearing fault harmonics)
    hf_start = len(vib_fft) * 3 // 4
    vib_energy_high = float(np.sum(vib_fft[hf_start:] ** 2))

    # ---- Group C: Temperature -----------------------------------------------
    temp_mean  = float(np.mean(temperature))
    temp_max   = float(np.max(temperature))
    temp_rise  = float(temp_max - temperature[0])

    # ---- Group D: Kinematic -------------------------------------------------
    tracking_err = position - position_cmd
    terr_rms = _rms(tracking_err)
    terr_max = float(np.max(np.abs(tracking_err)))
    terr_p2p = float(np.ptp(tracking_err))
    pos_rms  = _rms(position)

    return FeatureVector(
        curr_rms=curr_rms, curr_peak=curr_peak, curr_crest=curr_cf,
        curr_thd=curr_thd, curr_f1=curr_f1,
        vib_rms=vib_rms, vib_peak=vib_peak, vib_kurtosis=vib_kurt,
        vib_crest=vib_cf, vib_energy_high=vib_energy_high,
        temp_mean=temp_mean, temp_max=temp_max, temp_rise=temp_rise,
        tracking_err_rms=terr_rms, tracking_err_max=terr_max,
        tracking_err_p2p=terr_p2p, pos_rms=pos_rms,
    )


# ---------------------------------------------------------------------------
# Statistical utilities
# ---------------------------------------------------------------------------

def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))


def _crest_factor(x: np.ndarray) -> float:
    rms = _rms(x)
    peak = float(np.max(np.abs(x)))
    return peak / (rms + 1e-12)


def _kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis (Fisher's definition)."""
    mu    = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-12:
        return 0.0
    return float(np.mean(((x - mu) / sigma) ** 4) - 3.0)
