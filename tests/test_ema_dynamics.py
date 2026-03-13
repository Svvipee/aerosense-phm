"""
Unit tests for EMA physics simulation.

These tests validate the BLDC motor + ball-screw electro-mechanical actuator
physics model — the engine that generates realistic sensor data for ML training.
They prove:
 - Motor parameters are physically valid (positive resistance, torque constant, etc.)
 - The electrical/mechanical/thermal dynamics respond correctly to inputs
 - Degradation (bearing wear, winding faults) produces the expected sensor signatures
 - Command profiles generate correct sample counts
 - The composite health index stays bounded

Each test maps to a specific real-world behaviour documented in the engineering
literature: Krishnan (2010) for motor dynamics, Byington et al. (2004) for
vibration signatures of bearing faults, Stone et al. (2004) for thermal effects
of winding degradation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pytest

from simulation.ema_dynamics import (
    EMASimulator,
    EMAParameters,
    sine_sweep_profile,
    step_profile,
)


class TestEMAParameters:
    def test_default_params_are_physical(self):
        """PROVES: All default motor parameters have physically valid values —
        positive winding resistance (Ohms), positive torque constant (Nm/A),
        efficiency between 0 and 1, and positive ball-screw lead (m/rev).
        These parameters are sourced from the Maxon EC 90 flat catalog and
        Jensen et al. (2000) F-18 EMA test data.  If any were zero or negative,
        the differential equations would produce infinite current, zero torque,
        or division-by-zero errors."""
        p = EMAParameters()
        assert p.R_winding > 0
        assert p.Kt > 0
        assert 0 < p.efficiency <= 1.0
        assert p.lead > 0

    def test_nominal_no_backlash(self):
        """PROVES: A brand-new actuator has zero backlash — matching the factory
        specification where a pre-loaded ball-screw has no dead-band.  Backlash
        only develops as the screw wears.  If default backlash were nonzero,
        the 'healthy' baseline would already have a tracking error, making it
        impossible to distinguish healthy from degraded in the classifier."""
        p = EMAParameters()
        assert p.backlash == 0.0


class TestEMASimulator:
    def test_zero_voltage_stays_still(self):
        """PROVES: With zero voltage command, the actuator does not move —
        confirming that the motor model has no spurious drift, offset, or
        numerical instability.  In a real aircraft, a de-energised actuator
        should hold position via the ball-screw's self-locking property.
        If position drifted without command, the flight control surface would
        move uncommanded — a safety-critical failure mode."""
        sim = EMASimulator(seed=0)
        snap = sim.step(voltage_cmd=0.0)
        # With zero voltage, position should remain near zero
        assert abs(snap.position) < 1e-3

    def test_temperature_rises_under_load(self):
        """Winding temperature must rise above ambient when current flows."""
        sim = EMASimulator(seed=42)
        T_init = sim._T
        for _ in range(500):
            sim.step(voltage_cmd=10.0)
        assert sim._T > T_init

    def test_healthy_tracking(self):
        """A healthy actuator should track a slow sine command with bounded error.

        The demo P-controller (Kp=40 V/m) is a simplified inner loop, not a
        high-performance servo controller.  We check that error stays within
        a physically sensible bound (< 15 mm RMS) rather than requiring
        sub-millimetre accuracy which would need a properly tuned PD/PID.
        """
        p = EMAParameters()
        sim = EMASimulator(params=p, dt=0.001, seed=0)
        cmd = sine_sweep_profile(amplitude=0.01, f_start=0.5, f_end=1.0, duration=5.0)
        snaps = sim.run_profile(cmd, duration=5.0)
        errors = np.array([s.tracking_error for s in snaps])
        rms_error = np.sqrt(np.mean(errors**2))
        assert rms_error < 0.015, f"RMS tracking error too large: {rms_error:.4f} m"

    def test_degraded_bearing_raises_vibration(self):
        """PROVES: A bearing at 20% health produces at least 2× higher vibration
        than a healthy bearing — matching the real diagnostic signature documented
        by Byington et al. (2004) and Randall (2011): bearing defects generate
        impulse responses at characteristic frequencies (BPFI, BPFO) that raise
        the broadband vibration RMS.  This is the #1 diagnostic indicator that
        the feature extraction pipeline relies on (vib_rms, vib_kurtosis).
        If degraded bearings didn't raise vibration, the fault classifier would
        be unable to detect bearing failures — the most common EMA removal cause."""
        sim_healthy  = EMASimulator(seed=1)
        sim_degraded = EMASimulator(seed=1)
        sim_degraded.bearing_health = 0.20  # severely degraded

        vib_h, vib_d = [], []
        for _ in range(200):
            sim_healthy.step(10.0)
            sim_degraded.step(10.0)
        for _ in range(500):
            sh = sim_healthy.step(10.0)
            sd = sim_degraded.step(10.0)
            vib_h.append(sh.vibration_rms)
            vib_d.append(sd.vibration_rms)

        assert np.mean(vib_d) > np.mean(vib_h) * 2.0, \
            "Degraded bearing should produce significantly higher vibration"

    def test_winding_fault_raises_temperature(self):
        """PROVES: A winding at 40% health runs hotter than a healthy winding under
        the same load — matching the Montsinger/Arrhenius thermal aging model.
        Degraded insulation increases effective winding resistance (R_eff = R_nom / wh),
        causing more I²R heating.  Stone et al. (2004) documented this effect.
        This thermal signature is what temp_max and temp_rise features detect.
        If winding faults didn't raise temperature, the PHM system would miss
        the second most common EMA failure mode."""
        sim_h = EMASimulator(seed=2)
        sim_d = EMASimulator(seed=2)
        sim_d.winding_health = 0.40

        for _ in range(2000):
            sim_h.step(8.0)
            sim_d.step(8.0)

        assert sim_d._T > sim_h._T, \
            "Degraded winding should run hotter than healthy winding"

    def test_step_profile_shape(self):
        """PROVES: The step command profile generator produces exactly the right
        number of samples (duration / dt = 8000).  Step profiles simulate
        discrete position commands like those from a flight control computer
        commanding surface deflections.  Correct sample count ensures the
        feature extraction window has enough data for meaningful FFT and
        statistical calculations."""
        cmds = step_profile(amplitude=0.01, n_steps=4, duration=8.0, dt=0.001)
        assert len(cmds) == 8000

    def test_sine_sweep_profile_shape(self):
        """PROVES: The sine sweep profile (frequency-varying sinusoid) generates
        exactly 10,000 samples for a 10s / 0.001s configuration.  Sine sweeps
        exercise the actuator across a frequency range, exciting bearing
        resonances and revealing bandwidth-dependent degradation — similar to
        the Built-In Test (BIT) sequences real EMAs perform on the ground."""
        cmds = sine_sweep_profile(amplitude=0.01, duration=10.0, dt=0.001)
        assert len(cmds) == 10_000

    def test_health_index_bounded(self):
        """PROVES: The composite health index HI = (bh × wh × blh)^(1/3) stays
        within [0, 1] for a partially degraded actuator.  The geometric mean
        formulation from Saxena et al. (2008) ensures that any single subsystem
        at zero health drives HI to zero.  This bounded property is essential
        because the alert thresholds (CRITICAL < 0.35, WARNING < 0.55) assume
        HI is a normalised quantity.  Values outside [0,1] would trigger false
        or missed alerts."""
        sim = EMASimulator(seed=99)
        sim.bearing_health = 0.5
        sim.winding_health = 0.7
        sim.backlash_health = 0.9
        snaps = sim.run_profile(np.zeros(100), duration=0.1)
        his = [s.health_index for s in snaps]
        assert all(0.0 <= hi <= 1.0 for hi in his)
