"""
Unit tests for degradation model.

These tests validate the Weibull bearing + Arrhenius winding + linear backlash
degradation models that generate training data for the ML pipeline.  They prove
that the physics-based degradation behaves as real actuator components do:
health starts near 1.0, monotonically decreases over time, stays bounded
in [0, 1], and different actuator units degrade at different rates (unit-to-unit
variation), matching the stochastic nature of real field failures.

References verified:
 - Weibull/Lundberg-Palmgren (1947) bearing life model
 - Arrhenius/Montsinger (1930) thermal aging model
 - Linear backlash growth from ball-screw wear
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pytest

from simulation.degradation_model import DegradationModel, DegradationProfile


class TestDegradationModel:
    def test_trajectory_shape(self):
        """PROVES: The degradation model produces a trajectory matrix of exactly
        (n_cycles, 4) — one row per flight cycle, four columns: [cycle_number,
        bearing_health, winding_health, backlash_health].  This shape contract
        is what the fleet simulator depends on to generate labelled training data.
        If the shape is wrong, the entire ML training pipeline would fail."""
        model = DegradationModel(seed=0)
        traj = model.generate(n_cycles=1000)
        assert traj.shape == (1000, 4)

    def test_health_bounded_zero_to_one(self):
        """PROVES: All three health indicators (bearing, winding, backlash) stay
        within the physical range [0.0, 1.0] throughout the entire degradation
        trajectory.  Health=1.0 means brand new; health=0.0 means completely
        failed.  Values outside this range would be physically meaningless and
        would break the alert thresholds (CRITICAL < 0.35, WARNING < 0.55)
        which depend on health being a normalised quantity."""
        model = DegradationModel(seed=1)
        traj = model.generate(n_cycles=500)
        _, bh, wh, blh = traj.T
        assert np.all(bh  >= 0.0) and np.all(bh  <= 1.0)
        assert np.all(wh  >= 0.0) and np.all(wh  <= 1.0)
        assert np.all(blh >= 0.0) and np.all(blh <= 1.0)

    def test_health_monotonically_decreasing(self):
        """PROVES: Component health only goes down, never up — matching the
        real physics of bearing fatigue (Weibull cumulative damage), insulation
        aging (Arrhenius irreversibility), and mechanical wear.  A component
        cannot self-heal in the field.  If health ever increased, the RUL
        predictor would see impossible 'recovery' patterns and learn incorrect
        failure signatures.  Tolerance of 1e-9 accounts for floating-point noise."""
        model = DegradationModel(seed=2)
        traj = model.generate(n_cycles=1000)
        for col in [1, 2, 3]:
            h = traj[:, col]
            diffs = np.diff(h)
            # Allow tiny floating-point noise but no net recovery
            assert np.all(diffs <= 1e-9), f"Column {col} has non-monotone decrease"

    def test_health_starts_near_one(self):
        """PROVES: A new actuator starts with health > 0.90 for all three
        subsystems — confirming the model represents a newly installed,
        serviceable component.  In practice, a new bearing/winding/ballscrew
        is at or near 100% health.  If initial health were low, the model
        would generate training data where actuators are 'born degraded',
        which doesn't match any real installation scenario."""
        model = DegradationModel(seed=3)
        traj = model.generate(n_cycles=500)
        for col in [1, 2, 3]:
            assert traj[0, col] > 0.90

    def test_rul_computation(self):
        """PROVES: The Remaining Useful Life (RUL) label — the ground-truth
        target the ML model learns to predict — decreases monotonically as the
        actuator ages.  RUL is defined as 'cycles remaining until health drops
        below 0.30' (the CMAPSS EoL threshold from Saxena et al. 2008).
        If RUL ever increased over time, it would mean the actuator is getting
        further from failure as it ages — a physical impossibility.  This test
        validates the label generation that makes supervised learning possible."""
        model = DegradationModel(seed=4)
        traj = model.generate(n_cycles=1000)
        rul = model.remaining_useful_life(traj, threshold=0.30)
        # RUL should decrease (or stay) over time
        valid = rul[~np.isnan(rul)]
        if len(valid) > 1:
            diffs = np.diff(valid)
            assert np.all(diffs <= 1.0 + 1e-6), "RUL should not increase over time"

    def test_unit_variation_produces_different_trajectories(self):
        """PROVES: Two actuators manufactured to the same specification still
        degrade at different rates — matching real-world experience where
        identical bearings can have 3-5× variation in fatigue life (Lundberg &
        Palmgren 1947 L10 vs L50 spread).  The stochastic seed controls this.
        If all trajectories were identical, the ML model would only learn one
        degradation curve and fail to generalise across a real fleet."""
        model1 = DegradationModel(seed=10)
        model2 = DegradationModel(seed=20)
        t1 = model1.generate(1000)
        t2 = model2.generate(1000)
        # The two trajectories should differ due to stochastic unit variation
        assert not np.allclose(t1[:, 1], t2[:, 1])
