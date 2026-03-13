"""
Fleet-Level Simulation
=======================

Simulates a mixed fleet of aircraft, each with multiple EMAs at
different stages of their service life.  Generates the labelled
dataset used to train the PHM models.

Fleet composition is representative of a 30-aircraft narrow-body
operator (e.g., A320 / B737 family) where each aircraft has:
  - 2 aileron EMAs (L/R)
  - 2 elevator EMAs (L/R)
  - 1 rudder EMA

= 5 EMAs per aircraft, 150 total for a 30-aircraft fleet.

Design of simulation experiment follows the approach used in:
  Saxena, A., Goebel, K., Simon, D., Eklund, N. (2008).
  "Damage Propagation Modeling for Aircraft Engine Run-to-Failure
  Simulation." PHM Society Conference. (CMAPSS methodology adapted
  for actuator systems.)
"""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from simulation.ema_dynamics import EMASimulator, EMAParameters, sine_sweep_profile
from simulation.degradation_model import DegradationModel, DegradationProfile


# ---------------------------------------------------------------------------
# Aircraft & fleet definitions
# ---------------------------------------------------------------------------

ACTUATOR_TYPES = {
    "aileron_L":  {"peak_force_N": 4500, "stroke_m": 0.025},
    "aileron_R":  {"peak_force_N": 4500, "stroke_m": 0.025},
    "elevator_L": {"peak_force_N": 6000, "stroke_m": 0.030},
    "elevator_R": {"peak_force_N": 6000, "stroke_m": 0.030},
    "rudder":     {"peak_force_N": 8000, "stroke_m": 0.040},
}

AIRCRAFT_TYPES = {
    "A320_family": {"n_aircraft": 12, "design_cycles": 65_000},
    "B737_family": {"n_aircraft": 10, "design_cycles": 75_000},
    "E175_family": {"n_aircraft": 8,  "design_cycles": 60_000},
}


class FleetSimulator:
    """
    Generates a labelled dataset from a simulated fleet of aircraft.

    Each row in the output dataset corresponds to one 'inspection event'
    (one test flight's worth of sensor data, ~90 seconds), and includes:
      - Aircraft and actuator identifiers
      - Extracted signal features (RMS, kurtosis, THD, tracking error, …)
      - Ground-truth health index and RUL (for model training/evaluation)
    """

    def __init__(
        self,
        output_dir: str | Path = "data/processed",
        seed: int = 2024,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(seed)
        self._records: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Main entry point                                                    #
    # ------------------------------------------------------------------ #

    def generate_dataset(
        self,
        cycles_per_aircraft: int = 800,
        test_flight_duration_s: float = 5.0,
        dt: float = 0.005,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Generate the full fleet dataset.

        Parameters
        ----------
        cycles_per_aircraft : Number of service cycles to simulate per EMA.
        test_flight_duration_s : Duration of each simulated test flight [s].
        dt : Integration time step [s].

        Returns
        -------
        DataFrame with one row per (aircraft, actuator, cycle).
        """
        aircraft_id = 0
        total_emas = 0

        for ac_type, ac_cfg in AIRCRAFT_TYPES.items():
            n_ac = ac_cfg["n_aircraft"]
            design_cycles = ac_cfg["design_cycles"]

            for ac_idx in range(n_ac):
                # Each aircraft starts at a random point in its service life
                age_fraction = self.rng.uniform(0.05, 0.85)

                for act_name in ACTUATOR_TYPES:
                    seed_unit = int(self.rng.integers(0, 2**31))
                    self._simulate_ema(
                        aircraft_id=f"{ac_type}_{ac_idx:03d}",
                        actuator_id=act_name,
                        age_fraction=age_fraction,
                        cycles_per_aircraft=cycles_per_aircraft,
                        design_cycles=design_cycles,
                        test_flight_duration_s=test_flight_duration_s,
                        dt=dt,
                        seed=seed_unit,
                    )
                    total_emas += 1

                aircraft_id += 1

            if verbose:
                print(f"  Simulated {ac_type}: {n_ac} aircraft × {len(ACTUATOR_TYPES)} EMAs")

        df = pd.DataFrame(self._records)
        out_path = self.output_dir / "fleet_dataset.parquet"
        df.to_parquet(out_path, index=False)
        if verbose:
            print(f"\nDataset saved → {out_path}")
            print(f"  Rows: {len(df):,}  |  EMAs: {total_emas}  |  Features: {len(df.columns)}")
        return df

    # ------------------------------------------------------------------ #
    # Per-EMA simulation                                                  #
    # ------------------------------------------------------------------ #

    def _simulate_ema(
        self,
        aircraft_id: str,
        actuator_id: str,
        age_fraction: float,
        cycles_per_aircraft: int,
        design_cycles: int,
        test_flight_duration_s: float,
        dt: float,
        seed: int,
    ) -> None:
        """Simulate one EMA over its monitored service history."""

        rng = np.random.default_rng(seed)
        degradation_model = DegradationModel(seed=seed)
        total_cycles = int(age_fraction * design_cycles) + cycles_per_aircraft

        trajectory = degradation_model.generate(n_cycles=total_cycles)
        rul_array = degradation_model.remaining_useful_life(trajectory)

        # Only keep the monitored window (last cycles_per_aircraft cycles)
        start_idx = total_cycles - cycles_per_aircraft

        ema_params = EMAParameters()
        sim = EMASimulator(params=ema_params, dt=dt, seed=int(rng.integers(0, 2**31)))

        n_time = int(test_flight_duration_s / dt)
        cmd_profile = sine_sweep_profile(
            amplitude=ACTUATOR_TYPES[actuator_id]["stroke_m"] * 0.6,
            duration=test_flight_duration_s,
            dt=dt,
        )
        # Pad / truncate to exactly n_time
        if len(cmd_profile) < n_time:
            cmd_profile = np.pad(cmd_profile, (0, n_time - len(cmd_profile)))
        else:
            cmd_profile = cmd_profile[:n_time]

        for cycle_offset in range(cycles_per_aircraft):
            cycle_abs = start_idx + cycle_offset

            # Set degradation state from trajectory
            _, bh, wh, blh = trajectory[cycle_abs]
            sim.bearing_health = float(bh)
            sim.winding_health = float(wh)
            sim.backlash_health = float(blh)

            # Aerodynamic load: random within ±70 % of rated
            peak_load = ACTUATOR_TYPES[actuator_id]["peak_force_N"]
            load_profile = rng.uniform(-0.7, 0.7, n_time) * peak_load

            snapshots = sim.run_profile(cmd_profile, test_flight_duration_s, load_profile)

            features = self._extract_features(snapshots, dt)
            rul_val = rul_array[cycle_abs]

            self._records.append({
                "aircraft_id":   aircraft_id,
                "actuator_id":   actuator_id,
                "cycle":         cycle_abs,
                "age_fraction":  cycle_abs / design_cycles,
                "rul":           float(rul_val) if not np.isnan(rul_val) else -1.0,
                "health_index":  float(trajectory[cycle_abs, 1:4].prod() ** (1 / 3)),
                "bearing_health": float(bh),
                "winding_health": float(wh),
                "backlash_health": float(blh),
                **features,
            })

    # ------------------------------------------------------------------ #
    # Feature extraction from raw sensor stream                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_features(snapshots: list, dt: float) -> dict[str, float]:
        """
        Extract scalar health-indicative features from one flight's
        sensor data stream.

        Features are identical to those used by the online PHM system,
        ensuring training/inference consistency.

        Feature rationale documented in:
          Nandi, S., Toliyat, H.A., Li, X. (2005). "Condition Monitoring
          and Fault Diagnosis of Electrical Motors — A Review." IEEE Trans.
          Energy Conversion, 20(4), 719–729.
        """
        pos   = np.array([s.position        for s in snapshots])
        curr  = np.array([s.current         for s in snapshots])
        temp  = np.array([s.temperature     for s in snapshots])
        vib   = np.array([s.vibration_rms   for s in snapshots])
        terr  = np.array([s.tracking_error  for s in snapshots])

        # --- Current features (motor winding health) ---
        curr_rms   = float(np.sqrt(np.mean(curr ** 2)))
        curr_peak  = float(np.max(np.abs(curr)))
        curr_cf    = curr_peak / (curr_rms + 1e-12)           # crest factor
        curr_fft   = np.abs(np.fft.rfft(curr))
        freqs      = np.fft.rfftfreq(len(curr), d=dt)
        # Fundamental and 3rd harmonic (THD proxy)
        f1_idx     = int(np.argmax(curr_fft[1:]) + 1)
        curr_f1    = float(curr_fft[f1_idx]) if f1_idx < len(curr_fft) else 0.0
        curr_f3    = float(curr_fft[min(f1_idx * 3, len(curr_fft) - 1)])
        curr_thd   = curr_f3 / (curr_f1 + 1e-12)

        # --- Vibration features (bearing health) ---
        vib_rms    = float(np.sqrt(np.mean(vib ** 2)))
        vib_peak   = float(np.max(vib))
        vib_kurt   = float(_kurtosis(vib))
        vib_cf     = vib_peak / (vib_rms + 1e-12)
        vib_fft    = np.abs(np.fft.rfft(vib))
        vib_energy_high = float(np.sum(vib_fft[len(vib_fft) // 4:] ** 2))

        # --- Temperature features (thermal health) ---
        temp_mean  = float(np.mean(temp))
        temp_max   = float(np.max(temp))
        temp_rise  = float(temp_max - temp[0])

        # --- Tracking error features (mechanical integrity) ---
        terr_rms   = float(np.sqrt(np.mean(terr ** 2)))
        terr_max   = float(np.max(np.abs(terr)))
        terr_p2p   = float(np.ptp(terr))  # peak-to-peak

        # --- Position features (backlash / stiction) ---
        pos_rms    = float(np.sqrt(np.mean(pos ** 2)))

        return {
            "curr_rms":         curr_rms,
            "curr_peak":        curr_peak,
            "curr_crest":       curr_cf,
            "curr_thd":         curr_thd,
            "curr_f1":          curr_f1,
            "vib_rms":          vib_rms,
            "vib_peak":         vib_peak,
            "vib_kurtosis":     vib_kurt,
            "vib_crest":        vib_cf,
            "vib_energy_high":  vib_energy_high,
            "temp_mean":        temp_mean,
            "temp_max":         temp_max,
            "temp_rise":        temp_rise,
            "tracking_err_rms": terr_rms,
            "tracking_err_max": terr_max,
            "tracking_err_p2p": terr_p2p,
            "pos_rms":          pos_rms,
        }


def _kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis (Fisher's definition), robust to near-constant signals."""
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < 1e-12:
        return 0.0
    return float(np.mean(((x - mu) / sigma) ** 4) - 3.0)
