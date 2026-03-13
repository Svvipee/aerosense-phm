"""
Component Degradation Model
============================

Simulates realistic wear progression of EMA sub-systems over their
service life, supporting training-data generation for the PHM models.

Degradation model basis:
- Power-law fatigue model: consistent with Paris–Erdogan crack propagation
  law (Paris & Erdogan, 1963, J. Basic Eng. 85(4):528–534).
- Bearing life model: Lundberg–Palmgren equation (Lundberg & Palmgren, 1947,
  Acta Polytechnica 1(3)), with L10 life as the reference.
- Accelerated life testing framework: MIL-HDBK-217F (Reliability Prediction
  of Electronic Equipment).

In real operation each EMA accumulates duty cycles (position reversals,
peak loads, temperature cycles). Here we proxy accumulated damage with
a single normalised variable `cycles` that maps 0 → end-of-overhaul-interval
(EOI) to 1 → design life limit, consistent with OEM scheduled maintenance
intervals reported in ATA MSG-3 documentation.
"""

import numpy as np
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Degradation profile parameters
# ---------------------------------------------------------------------------

@dataclass
class DegradationProfile:
    """
    Characterises how one EMA degrades over its service life.

    All three health sub-systems are modelled as independent stochastic
    processes.  The dominant failure mode depends on operating conditions;
    bearing failures account for ~40 % of EMA removals in service data
    (Balaban et al. 2009; NASA/TM-2013-217816).
    """

    # ---- Bearing degradation ------------------------------------------------
    # Based on Lundberg–Palmgren L10 life.  At fraction 'bearing_L10' of
    # design life, P(failure) = 10 %; at 'bearing_L50' P(failure) = 50 %.
    bearing_L10: float = 0.65    # fraction of design life at L10
    bearing_L50: float = 0.90    # fraction at L50
    bearing_noise: float = 0.04  # process noise std (cycle-to-cycle scatter)

    # ---- Winding / insulation degradation -----------------------------------
    # Arrhenius thermal aging: every 10 °C rise roughly halves insulation life
    # (Montsinger, 1930).  Here encoded as mean winding_life fraction.
    winding_life: float = 0.95   # mean fraction of design life
    winding_noise: float = 0.02

    # ---- Ball-screw / gearbox backlash degradation --------------------------
    # Hertzian contact fatigue; typically slower than bearing or winding.
    backlash_life: float = 1.05  # may outlast design life
    backlash_noise: float = 0.01

    # ---- Global scatter -----
    # Unit-to-unit variability — multiplicative factor on each life parameter
    unit_variation: float = 0.12  # CV (coefficient of variation)


# ---------------------------------------------------------------------------
# Single-unit degradation trajectory
# ---------------------------------------------------------------------------

class DegradationModel:
    """
    Generates a full degradation trajectory for one EMA instance from
    installation (health=1) to end-of-life (health→0).

    Usage
    -----
    model = DegradationModel(profile, seed=42)
    trajectory = model.generate(n_cycles=5000)
    # trajectory is an ndarray shape (n_cycles, 4):
    #   columns: [cycle, bearing_health, winding_health, backlash_health]
    """

    def __init__(
        self,
        profile: DegradationProfile | None = None,
        seed: int | None = None,
    ):
        self.profile = profile or DegradationProfile()
        self.rng = np.random.default_rng(seed)

    def _unit_life(self, nominal: float) -> float:
        """Apply unit-to-unit variation: log-normal scatter around nominal."""
        cv = self.profile.unit_variation
        sigma_ln = np.sqrt(np.log(1 + cv ** 2))
        mu_ln = np.log(nominal) - 0.5 * sigma_ln ** 2
        return self.rng.lognormal(mu_ln, sigma_ln)

    def generate(self, n_cycles: int = 10_000) -> np.ndarray:
        """
        Generate a degradation trajectory.

        Parameters
        ----------
        n_cycles : Number of duty cycles to simulate.

        Returns
        -------
        ndarray of shape (n_cycles, 4):
            [cycle_index, bearing_health, winding_health, backlash_health]
        """
        p = self.profile
        cycles = np.arange(n_cycles, dtype=float)
        norm_cycle = cycles / n_cycles  # 0 → 1

        # --- Bearing health (Weibull-based degradation) ---------------------
        # Health = 1 - CDF(t; scale, shape).  Scale and shape derived from
        # L10 and L50 constraints using the two-parameter Weibull model.
        scale_b = self._unit_life(p.bearing_L50)
        shape_b = self._weibull_shape(p.bearing_L10, scale_b)
        bearing_det = np.exp(-((norm_cycle / scale_b) ** shape_b))
        bearing_noise = self.rng.normal(0, p.bearing_noise, n_cycles).cumsum()
        bearing_noise *= 0.005  # scale cumulative drift
        bearing_health = np.clip(bearing_det + bearing_noise, 0.0, 1.0)
        # Monotonic decay (health cannot spontaneously recover)
        bearing_health = self._enforce_monotone_decrease(bearing_health)

        # --- Winding health (exponential Arrhenius aging) -------------------
        scale_w = self._unit_life(p.winding_life)
        winding_det = np.exp(-norm_cycle / scale_w)
        winding_noise = self.rng.normal(0, p.winding_noise, n_cycles).cumsum()
        winding_noise *= 0.003
        winding_health = np.clip(winding_det + winding_noise, 0.0, 1.0)
        winding_health = self._enforce_monotone_decrease(winding_health)

        # --- Backlash health (linear wear + scatter) ------------------------
        scale_bl = self._unit_life(p.backlash_life)
        backlash_det = np.clip(1.0 - norm_cycle / scale_bl, 0.0, 1.0)
        backlash_noise = self.rng.normal(0, p.backlash_noise, n_cycles).cumsum()
        backlash_noise *= 0.001
        backlash_health = np.clip(backlash_det + backlash_noise, 0.0, 1.0)
        backlash_health = self._enforce_monotone_decrease(backlash_health)

        trajectory = np.column_stack([
            cycles,
            bearing_health,
            winding_health,
            backlash_health,
        ])
        return trajectory

    @staticmethod
    def _weibull_shape(L10_frac: float, scale: float) -> float:
        """
        Estimate Weibull shape parameter β such that:
            CDF(L10_frac; scale, β) = 0.10
        Solved numerically as β = log(-log(0.9)) / log(L10_frac / scale).
        """
        eps = 1e-9
        return np.log(-np.log(0.90)) / np.log(max(L10_frac, eps) / max(scale, eps))

    @staticmethod
    def _enforce_monotone_decrease(arr: np.ndarray) -> np.ndarray:
        """Running minimum — ensures health never increases after degrading."""
        out = arr.copy()
        for i in range(1, len(out)):
            if out[i] > out[i - 1]:
                out[i] = out[i - 1]
        return out

    def remaining_useful_life(self, trajectory: np.ndarray, threshold: float = 0.30) -> np.ndarray:
        """
        Compute the ground-truth Remaining Useful Life (RUL) for each
        cycle in a trajectory.

        RUL is defined as the number of cycles remaining until the composite
        health index (geometric mean of the three sub-systems) drops below
        `threshold`.

        Parameters
        ----------
        trajectory : Output of self.generate().
        threshold  : Health-index value defining End-of-Life (EoL).
                     0.30 is consistent with the 70 % degradation criterion
                     used in NASA CMAPSS benchmark (Saxena et al. 2008,
                     PHM Society Conference).

        Returns
        -------
        rul_array : ndarray of shape (n_cycles,), RUL in cycles.
                    NaN after EoL.
        """
        _, bh, wh, blh = trajectory.T
        composite_hi = (bh * wh * blh) ** (1 / 3)

        eol_candidates = np.where(composite_hi < threshold)[0]
        if len(eol_candidates) == 0:
            eol_cycle = len(trajectory) - 1
        else:
            eol_cycle = eol_candidates[0]

        n = len(trajectory)
        rul = np.full(n, np.nan)
        rul[:eol_cycle] = eol_cycle - np.arange(eol_cycle)
        return rul
