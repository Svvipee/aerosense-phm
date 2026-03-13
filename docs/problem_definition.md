# Problem Definition

## 1. Background

Commercial aircraft flight control surfaces (ailerons, elevators, rudders, spoilers) are actuated by hydraulic systems that have been the industry standard since the 1950s. A typical narrow-body aircraft carries 150–200 litres of hydraulic fluid, three independent hydraulic circuits operating at 3,000–5,000 PSI, and tens of hydraulic pumps, motors, valves, and actuators.

These systems have well-understood failure modes and a mature maintenance ecosystem, but they carry significant operational penalties:

- **Weight**: A complete hydraulic system on an A320-family aircraft weighs approximately 500 kg (fluid + lines + pumps + actuators). Every 10 kg removed from an aircraft saves approximately $1 M over its 25-year service life (IATA estimate).
- **Fluid leakage**: Hydraulic fluid (Skydrol) is toxic and environmentally regulated. Contamination events require expensive clean-up and can cause unscheduled maintenance events.
- **Complexity**: Hydraulic systems require fire-suppression-grade fluid lines routed throughout the airframe, increasing manufacturing complexity and inspection burden.
- **Maintenance burden**: Hydraulic actuator seals degrade on a calendar-and-cycles basis, driving scheduled replacement regardless of actual condition.

## 2. The "More Electric Aircraft" Opportunity

The aviation industry has been steadily reducing hydraulic system size in favour of electrical alternatives under the "More Electric Aircraft" (MEA) paradigm:

- **Boeing 787 Dreamliner**: Replaced most hydraulic power with electrical power; uses electro-hydrostatic actuators (EHAs) for primary flight controls.
- **Airbus A380**: Introduced an electric backup hydraulic actuator (EBHA) system.
- **Embraer E2 family**: Further electrification of secondary systems.

**Electro-Mechanical Actuators (EMAs)**, which replace hydraulic cylinders with brushless DC motors driving ball screws, offer:
- 20–35% weight reduction at the actuator level (Jensen et al., 2000)
- Zero fluid leakage
- Built-in electrical sensing (current, position, temperature)
- Potential for embedded health monitoring

## 3. Problem Statement

While EMAs are mechanically superior to hydraulic actuators in many respects, they introduce a **new failure mode challenge**: the failure mechanisms (bearing fatigue, winding insulation degradation, ball-screw backlash) are different from hydraulic seal failures, and the industry lacks mature tools to:

1. **Detect early-stage EMA degradation** from operational sensor data.
2. **Predict Remaining Useful Life (RUL)** for individual components.
3. **Move from time-based to condition-based maintenance** for flight control actuators.

Currently, airlines either:
- Replace EMAs on a fixed schedule (conservative, expensive)
- Wait for in-flight fault flags from the Built-In Test Equipment (BITE) system (reactive, risks AOG events)

**Neither approach is optimal.** Unscheduled removals cost airlines an average of $150,000–$500,000 per Aircraft-On-Ground (AOG) event in direct costs alone (ATA/IATA operational data).

## 4. Proposed Solution

AeroSense is a predictive maintenance platform consisting of:

1. **Smart EMA Hardware**: Electro-mechanical actuators with embedded multi-sensor arrays (current, vibration, temperature, position) feeding data to an onboard edge processor.
2. **PHM Software**: A cloud-connected Prognostics and Health Management system that:
   - Extracts health-indicative features from sensor streams
   - Classifies active fault modes (bearing, winding, backlash)
   - Predicts RUL with confidence intervals
   - Issues maintenance alerts with predicted dates
3. **Maintenance Dashboard**: Airline MRO-facing web interface for fleet health monitoring and maintenance scheduling.

## 5. Constraints and Assumptions

| Constraint | Value |
|---|---|
| Target aircraft | Narrow-body commercial (A320, B737, E175 families) |
| Targeted actuator types | Primary flight control (aileron, elevator, rudder) |
| Data update frequency | Per-flight (once per ~90-minute segment) |
| RUL prediction horizon | 30–500 service cycles |
| Alert lead time target | ≥30 cycles before predicted EoL |
| Regulatory framework | FAA AC 25.1309, DO-160G, SAE ARP4754A |

## 6. Success Criteria

- RUL prediction RMSE ≤ 50 cycles on held-out simulated fleet data
- Fault classification accuracy ≥ 85% (3-class: bearing / winding / healthy)
- Zero false-negative alerts at health index < 0.35 (critical threshold)
- Alert lead time ≥ 30 cycles in ≥ 90% of degradation cases
- Dashboard latency < 200 ms for fleet summary queries
