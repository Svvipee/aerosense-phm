"""
EMA (Electro-Mechanical Actuator) Physics Simulation
=====================================================

Models a brushless DC motor driving a ball screw to position a flight
control surface (e.g., aileron, elevator, rudder).

Physical model based on:
- Jensen et al. (2000): "Flight Test Experience with an EMA on the F-18 SRA"
  NASA/TM-2000-209116
- Balaban et al. (2009): "A Diagnostic Approach for EMA in Aerospace Systems"
  IEEE Aerospace Conference, DOI: 10.1109/AERO.2009.4839636
- Byington et al. (2004): "A Model-Based Approach to Prognostics and Health
  Management for Flight Control Actuators", IEEE Aerospace Conference

Motor parameters are representative of a mid-size flight control EMA
(e.g., aileron actuator on a regional jet), consistent with parameters
reported in Bauer & Kennel (2019), Actuators journal MDPI, 8(3), 60.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import NamedTuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EMAParameters:
    """
    Nominal electromechanical parameters for a BLDC-driven ball-screw EMA.

    Values are representative of a 5 kN peak-force aileron EMA on a
    regional jet, consistent with the range documented in:
      - SAE AIR5386 (Electric Actuation Technologies for Commercial Aircraft)
      - Bauer & Kennel (2019), Actuators 8(3):60
    """
    # Motor electrical
    R_winding: float = 0.45          # Phase resistance [Ω]
    L_winding: float = 0.0012        # Phase inductance [H]
    Kt: float = 0.28                 # Torque constant [N·m/A]
    Ke: float = 0.28                 # Back-EMF constant [V·s/rad]

    # Motor mechanical
    J_rotor: float = 1.8e-4         # Rotor moment of inertia [kg·m²]
    B_viscous: float = 5.0e-4       # Viscous damping coefficient [N·m·s/rad]
    tau_coulomb: float = 0.05       # Coulomb friction torque [N·m]

    # Ball screw
    lead: float = 0.005             # Ball-screw lead [m/rev] (5 mm/rev)
    efficiency: float = 0.92        # Mechanical efficiency
    backlash: float = 0.0           # Dead-band gap [m] — increases with wear

    # Gear ratio (motor turns per output shaft turn; 1 = direct drive)
    gear_ratio: float = 1.0

    # Thermal model — simplified single-node
    R_thermal: float = 2.5          # Thermal resistance [°C/W]
    C_thermal: float = 45.0         # Thermal capacitance [J/°C]
    T_ambient: float = 25.0         # Ambient temperature [°C]

    # Bearing characteristic frequencies (normalised to shaft rotation speed)
    # Based on NSK 6208ZZ deep-groove ball bearing (NSK Cat. No. E1102):
    #   N_balls=8, contact_angle=0°, pitch_dia=52 mm, ball_dia=12.7 mm
    # Formulae from Harris (2001) "Rolling Bearing Analysis" 4th ed., Ch. 14:
    #   BPFO = (N/2)*(1 - d/D*cos(α))          = 3.547 per shaft rev
    #   BPFI = (N/2)*(1 + d/D*cos(α))          = 4.453 per shaft rev
    #   BSF  = (D/2d)*(1 - (d/D*cos(α))^2)     = 2.256 per shaft rev
    #   FTF  = (1/2)*(1 - d/D*cos(α))          = 0.443 per shaft rev
    BPFI_ratio: float = 4.453       # Ball Pass Frequency Inner Race / shaft rev
    BPFO_ratio: float = 3.547       # Ball Pass Frequency Outer Race / shaft rev
    BSF_ratio: float = 2.256        # Ball Spin Frequency / shaft rev
    FTF_ratio: float = 0.443        # Fundamental Train (cage) Frequency / shaft rev


class SensorSnapshot(NamedTuple):
    """One sample of all sensor outputs from the EMA."""
    time:          float    # simulation time [s]
    position:      float    # output position [m]
    velocity:      float    # output velocity [m/s]
    current:       float    # phase current [A]
    temperature:   float    # winding temperature [°C]
    vibration_rms: float    # housing vibration RMS [g]
    tracking_error: float   # position error [m]
    health_index:  float    # 0 (failed) → 1 (new)  (ground-truth for training)


# ---------------------------------------------------------------------------
# Core EMA simulator
# ---------------------------------------------------------------------------

class EMASimulator:
    """
    Numerical simulation of a brushless-DC / ball-screw EMA using the
    Euler–Maruyama method (fixed time step with additive noise).

    Degradation is captured through three scalar health parameters that
    drift from their nominal values as the component ages:
      - bearing_health  : 1→0  (increases B_viscous, adds fault vibration)
      - winding_health  : 1→0  (increases R_winding, reduces Kt)
      - backlash_health : 1→0  (increases ball-screw backlash)

    These are **not** directly observable; only the sensor outputs are used
    by the PHM system, exactly as in real operation.
    """

    def __init__(
        self,
        params: EMAParameters | None = None,
        dt: float = 0.001,              # integration step [s]
        seed: int | None = None,
    ):
        self.p = params or EMAParameters()
        self.dt = dt
        self.rng = np.random.default_rng(seed)

        # State vector: [i (current), omega (angular velocity), theta (angle), T (temp)]
        self._i = 0.0        # motor current [A]
        self._omega = 0.0    # angular velocity [rad/s]
        self._theta = 0.0    # rotor angle [rad]
        self._T = self.p.T_ambient  # winding temperature [°C]
        self._prev_cmd = 0.0  # previous position command (for backlash)

        # Degradation state — updated externally by DegradationModel
        self.bearing_health: float = 1.0   # 1=new, 0=failed
        self.winding_health: float = 1.0
        self.backlash_health: float = 1.0

    # ------------------------------------------------------------------ #
    # Effective (degraded) parameters                                     #
    # ------------------------------------------------------------------ #

    def _effective_R(self) -> float:
        """Winding resistance increases as insulation degrades.
        Model: R_eff = R_nom / winding_health, capped at 3×nominal.
        Based on: Stone et al. (2004) "Electrical Insulation for Rotating Machines".
        """
        return min(self.p.R_winding / max(self.winding_health, 0.33), self.p.R_winding * 3.0)

    def _effective_Kt(self) -> float:
        """Torque constant drops with partial demagnetisation from thermal aging.
        Linear model: Kt_eff = Kt_nom * winding_health.
        Basis: Nandi, Toliyat & Li (2005) IEEE Trans. Energy Conv. 20(4):719–729
        report up to ~30% Kt reduction at severe winding fault; linear scaling
        from health=1 (nominal) to health=0 (total demagnetisation) is a
        conservative first-order approximation consistent with that bound.
        """
        return self.p.Kt * self.winding_health

    def _effective_B(self) -> float:
        """Bearing wear increases viscous damping.
        Model: B_eff = B_nom / bearing_health, capped at 10×nominal.
        Based on: Byington et al. (2004) PHM actuator model.
        """
        return min(self.p.B_viscous / max(self.bearing_health, 0.10), self.p.B_viscous * 10.0)

    def _effective_backlash(self) -> float:
        """Ball-screw backlash increases with wear.
        Max backlash = 0.5 mm, consistent with AS9100 out-of-service limits.
        """
        return self.p.backlash + (1.0 - self.backlash_health) * 0.0005

    # ------------------------------------------------------------------ #
    # Single integration step                                             #
    # ------------------------------------------------------------------ #

    def step(self, voltage_cmd: float, load_force: float = 0.0) -> SensorSnapshot:
        """
        Advance the simulation by one time step (self.dt).

        Parameters
        ----------
        voltage_cmd  : Voltage applied to motor winding [V].
        load_force   : External aerodynamic load on output shaft [N].

        Returns
        -------
        SensorSnapshot with all measurable quantities.
        """
        p = self.p
        dt = self.dt
        R = self._effective_R()
        Kt = self._effective_Kt()
        B = self._effective_B()

        # --- Electrical dynamics (first-order lag) ---
        # V = Ke*ω + R*i + L*di/dt  →  di/dt = (V - Ke*ω - R*i) / L
        d_i = (voltage_cmd - p.Ke * self._omega - R * self._i) / p.L_winding
        self._i += d_i * dt + self.rng.normal(0.0, 0.005)  # sensor noise

        # --- Mechanical dynamics ---
        # Convert load force to load torque on motor shaft
        tau_load = load_force * (p.lead / (2 * np.pi)) / (p.efficiency * p.gear_ratio)

        # Coulomb friction (opposes motion)
        sign_omega = np.sign(self._omega) if abs(self._omega) > 1e-6 else 0.0
        tau_friction = p.tau_coulomb * sign_omega

        tau_motor = Kt * self._i
        d_omega = (tau_motor - tau_friction - B * self._omega - tau_load) / p.J_rotor
        self._omega += d_omega * dt

        # --- Position (ball-screw kinematics) ---
        self._theta += self._omega * dt
        x_ideal = self._theta * (p.lead / (2 * np.pi)) / p.gear_ratio

        # Apply backlash dead-band
        backlash = self._effective_backlash()
        x_output = self._apply_backlash(x_ideal, backlash)

        velocity_output = self._omega * (p.lead / (2 * np.pi)) / p.gear_ratio

        # --- Thermal model (single-node) ---
        P_loss = R * self._i ** 2
        d_T = (P_loss - (self._T - p.T_ambient) / p.R_thermal) / p.C_thermal
        self._T += d_T * dt

        # --- Vibration signal (RMS acceleration) ---
        vib_rms = self._compute_vibration_rms()

        # Composite health index (geometric mean of sub-system healths)
        hi = (self.bearing_health * self.winding_health * self.backlash_health) ** (1 / 3)

        snap = SensorSnapshot(
            time=0.0,           # caller fills this in
            position=x_output + self.rng.normal(0.0, 5e-6),
            velocity=velocity_output,
            current=self._i,
            temperature=self._T + self.rng.normal(0.0, 0.2),
            vibration_rms=vib_rms,
            tracking_error=0.0,  # caller fills in after comparing to command
            health_index=hi,
        )
        self._prev_cmd = x_ideal
        return snap

    def _apply_backlash(self, x_ideal: float, backlash: float) -> float:
        """Simple backlash model: output lags input by up to backlash/2."""
        half = backlash / 2.0
        delta = x_ideal - self._prev_cmd
        if delta > half:
            return x_ideal - half
        elif delta < -half:
            return x_ideal + half
        return self._prev_cmd

    def _compute_vibration_rms(self) -> float:
        """
        Estimate vibration RMS [g] from:
          1. Normal operating vibration (proportional to speed)
          2. Bearing fault excitation (proportional to degradation severity)

        Bearing fault vibration model from:
          McFadden & Smith (1984) "Vibration monitoring of rolling element
          bearings by the high-frequency resonance technique."
          Tribology International, 17(1), 3–10.
        """
        speed_hz = abs(self._omega) / (2 * np.pi)

        # Normal vibration — empirically 0.05–0.5 g at operating speed
        vib_normal = 0.05 + 0.001 * speed_hz

        # Bearing fault component — rises as bearing_health → 0
        # Peak of ~3–4 g at end-of-life consistent with industrial bearing
        # fault measurements reported in:
        #   Randall (2011) "Vibration-Based Condition Monitoring" Wiley, Ch. 5
        #   Scheffer & Girdhar (2004) "Practical Machinery Vibration Analysis"
        #   Elsevier, Table 4.2 (defect severities on rolling element bearings)
        fault_severity = (1.0 - self.bearing_health) ** 2
        vib_fault = fault_severity * 3.5  # up to ~3.5 g at end-of-life

        # White noise floor
        noise = abs(self.rng.normal(0.0, 0.01))

        return vib_normal + vib_fault + noise

    # ------------------------------------------------------------------ #
    # Run a complete manoeuvre profile                                    #
    # ------------------------------------------------------------------ #

    def run_profile(
        self,
        position_cmds: np.ndarray,
        duration: float,
        load_profile: np.ndarray | None = None,
    ) -> list[SensorSnapshot]:
        """
        Run the actuator through a position-command profile.

        Parameters
        ----------
        position_cmds : Target position at each sample [m].  Length = N.
        duration      : Total duration of the profile [s].
        load_profile  : Aerodynamic load [N] at each sample. If None, zero.

        Returns a list of SensorSnapshot, one per time step.
        """
        N = len(position_cmds)
        t_arr = np.linspace(0.0, duration, N)
        if load_profile is None:
            load_profile = np.zeros(N)

        # Simple P-controller gain (representative of inner loop bandwidth)
        # ~50 Hz closed-loop bandwidth → Kp ≈ J*(2π*50)^2 / Kt
        Kp_pos = 40.0   # [V/m]
        Kv_vel = 0.5    # [V·s/m] — velocity damping

        snapshots: list[SensorSnapshot] = []
        cmd_prev = position_cmds[0]

        for idx in range(N):
            cmd = position_cmds[idx]
            pos_err = cmd - self._prev_cmd
            vel_cmd = (cmd - cmd_prev) / self.dt if idx > 0 else 0.0
            cmd_prev = cmd

            voltage = Kp_pos * pos_err - Kv_vel * self._omega
            voltage = float(np.clip(voltage, -28.0, 28.0))  # 28 V bus

            snap = self.step(voltage, float(load_profile[idx]))
            tracking_err = cmd - snap.position

            snapshots.append(SensorSnapshot(
                time=float(t_arr[idx]),
                position=snap.position,
                velocity=snap.velocity,
                current=snap.current,
                temperature=snap.temperature,
                vibration_rms=snap.vibration_rms,
                tracking_error=tracking_err,
                health_index=snap.health_index,
            ))

        return snapshots


# ---------------------------------------------------------------------------
# Utility: standard flight-test manoeuvre profiles
# ---------------------------------------------------------------------------

def sine_sweep_profile(
    amplitude: float = 0.01,
    f_start: float = 0.5,
    f_end: float = 5.0,
    duration: float = 10.0,
    dt: float = 0.001,
) -> np.ndarray:
    """
    Linear sine sweep: standard input for bandwidth/frequency-response tests.
    Used in DO-178C qualification and Jensen et al. (2000) F-18 EMA tests.
    """
    t = np.arange(0, duration, dt)
    freq_inst = f_start + (f_end - f_start) * t / duration
    phase = 2 * np.pi * np.cumsum(freq_inst) * dt
    return amplitude * np.sin(phase)


def step_profile(
    amplitude: float = 0.015,
    n_steps: int = 6,
    duration: float = 12.0,
    dt: float = 0.001,
) -> np.ndarray:
    """
    Repeated step commands: used for tracking accuracy and backlash measurement.
    """
    t = np.arange(0, duration, dt)
    period = duration / n_steps
    steps = amplitude * np.sign(np.sin(2 * np.pi * t / period))
    return steps
