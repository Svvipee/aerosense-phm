# Literature Review

## 1. Electro-Mechanical Actuators in Aviation

### 1.1 EMA Technology Development

**Jensen, S.C., Jenney, G.D., Dawson, D. (2000).** "Flight Test Experience with an Electromechanical Actuator on the F-18 Systems Research Aircraft." *Proceedings of the 19th DASC Digital Avionics Systems Conference.* NASA/TM-2000-209116.

This is the foundational flight-test paper establishing the feasibility of using EMAs as primary flight control actuators. The F-18 SRA tests demonstrated:
- Position accuracy within ±0.1 mm under flight loads
- Response bandwidth consistent with hydraulic actuators (>5 Hz)
- Weight reduction of approximately 25% at the actuator level
- Identified bearing and ball-screw wear as primary long-term failure modes

*Relevance:* Provides the physical baseline for our EMA model parameters and establishes that bearing/ballscrew degradation is the dominant failure concern in flight-qualified EMAs.

---

**Bauer, W., Kennel, R. (2019).** "Fault Tolerant Control Design for Electromechanical Actuators." *Actuators, 8(3), 60.* MDPI. DOI: 10.3390/act8030060.

Reviews fault-tolerant EMA designs and provides a comprehensive summary of measured motor parameters (Kt, R_winding, L_winding) for flight-grade BLDC motors in the 1–10 kN force class. This paper informed our choice of nominal motor parameters in `simulation/ema_dynamics.py`.

---

### 1.2 More Electric Aircraft Programme

**Wheeler, P.W., et al. (2013).** "The More Electric Aircraft: Technology and challenges." *IET Electrical Systems in Transportation, 3(3), 75–90.* DOI: 10.1049/iet-est.2012.0071.

Reviews the MEA roadmap and quantifies the commercial case: a fully electric A320-class aircraft would save approximately 530 kg in hydraulic system weight, corresponding to ~5% fuel saving. Provides the regulatory and certification context for EMA adoption.

---

## 2. Prognostics and Health Management (PHM)

### 2.1 General PHM Methodology

**Saxena, A., Goebel, K., Simon, D., Eklund, N. (2008).** "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation." *Proceedings of the 1st International Conference on Prognostics and Health Management (PHM).* IEEE. DOI: 10.1109/PHM.2008.4711414.

Introduced the CMAPSS (Commercial Modular Aero-Propulsion System Simulation) dataset and the methodology for generating realistic degradation trajectories for PHM benchmarking. Our fleet simulation follows the same experimental design: multiple units with varying operating conditions, stochastic degradation paths, and run-to-failure observations.

*We use the CMAPSS End-of-Life definition (health index below 30% threshold) as our RUL reference point.*

---

**Saxena, A., Goebel, K., Larrosa, C.C., Luo, J., Vachtsevanos, G. (2010).** "Metrics for Offline Evaluation of Prognostic Performance." *International Journal of Prognostics and Health Management, 1(1).* DOI: 10.36001/ijphm.2010.v1i1.1336.

Defines the PHM scoring function used in our model evaluation. The asymmetric score function penalises under-predictions of RUL (late maintenance) more harshly than over-predictions, reflecting the asymmetric cost structure in real maintenance operations. Implemented in `src/phm/rul_prediction.py::prognostic_score`.

---

### 2.2 EMA-Specific PHM

**Byington, C.S., Watson, M., Edwards, D., Stoelting, P. (2004).** "A Model-Based Approach to Prognostics and Health Management for Flight Control Actuators." *Proceedings of IEEE Aerospace Conference.* DOI: 10.1109/AERO.2004.1367852.

The primary reference for EMA-specific PHM. Demonstrates that current signature analysis combined with vibration monitoring can detect bearing degradation in BLDC-driven actuators with >85% accuracy. Proposed the use of current harmonics (THD) as a winding fault indicator. Our feature set in `src/phm/feature_extraction.py` directly implements these recommendations.

---

**Balaban, E., Bansal, P., Stoelting, P., Ray, A., Kare, S.M., Currier, S. (2009).** "A Diagnostic Approach for Electro-Mechanical Actuators in Aerospace Systems." *Proceedings of IEEE Aerospace Conference.* DOI: 10.1109/AERO.2009.4839636.

Applied PHM to NASA's EMA test rig. Key findings used in this project:
- Bearing failures account for ~40% of unscheduled EMA removals
- Vibration kurtosis (>3.5 threshold) provides reliable early bearing fault detection
- Ball-screw backlash can be measured from position tracking error during standardised manoeuvres
- These three failure modes (bearing, winding, backlash) cover >90% of field removal causes

---

## 3. Degradation Modelling

### 3.1 Bearing Life Models

**Lundberg, G., Palmgren, A. (1947).** "Dynamic Capacity of Rolling Bearings." *Acta Polytechnica, 1(3).* Royal Swedish Academy of Engineering Sciences.

The original Weibull-based rolling-element bearing life model. The L10 life (10% failure probability at n cycles) forms the basis of our degradation trajectory in `simulation/degradation_model.py`. The two-parameter Weibull model with shape β and scale λ is parameterised from published bearing test data.

---

**Harris, T.A. (2001).** *Rolling Bearing Analysis, 4th Edition.* John Wiley & Sons. ISBN: 0-471-35553-6.

Standard reference for bearing characteristic fault frequencies (BPFI, BPFO, BSF, FTF). Used to define the vibration fault model in `simulation/ema_dynamics.py::_compute_vibration_rms`.

---

### 3.2 Winding / Insulation Aging

**Stone, G.C., Boulter, E.A., Culbert, I., Dhirani, H. (2004).** *Electrical Insulation for Rotating Machines.* IEEE Press. ISBN: 0-471-44506-1.

Establishes the Arrhenius thermal aging model for motor winding insulation (Montsinger rule: life halves per 10°C rise). Used in `simulation/degradation_model.py::winding_health`.

---

## 4. ML Methods for RUL Prediction

**Mosallam, A., Medjaher, K., Zerhouni, N. (2016).** "Data-driven Prognostic Method Based on Bayesian Approaches for Direct Remaining Useful Life Prediction." *Journal of Intelligent Manufacturing, 27(5), 1037–1048.* DOI: 10.1007/s10845-014-0933-4.

Establishes Random Forest as a competitive baseline for direct RUL regression, reporting MAE of 20–35 cycles on the CMAPSS benchmark. Provides the hyperparameter reference (n_estimators=200, max_depth=20) used in our `RULPredictor`.

---

**Nandi, S., Toliyat, H.A., Li, X. (2005).** "Condition Monitoring and Fault Diagnosis of Electrical Motors — A Review." *IEEE Transactions on Energy Conversion, 20(4), 719–729.* DOI: 10.1109/TEC.2005.847955.

Comprehensive survey of motor condition monitoring features. Primary reference for our current signature analysis features (RMS, crest factor, THD) in `src/phm/feature_extraction.py`.

---

## 5. Vibration Analysis

**McFadden, P.D., Smith, J.D. (1984).** "Vibration Monitoring of Rolling Element Bearings by the High-Frequency Resonance Technique — A Review." *Tribology International, 17(1), 3–10.* DOI: 10.1016/0301-679X(84)90076-8.

Established the high-frequency resonance technique (envelope analysis) and kurtosis as the primary statistical indicators of incipient bearing faults. Specifically noted kurtosis > 4 as a reliable detection threshold for early outer-race defects. Used in our vibration feature rationale.

---

**Randall, R.B. (2011).** *Vibration-Based Condition Monitoring: Industrial, Automotive and Aerospace Applications.* Wiley. ISBN: 978-0-470-97340-8.

Standard reference for vibration feature extraction methodology used in Chapter 5 (time-domain statistics: RMS, peak, crest factor, kurtosis; frequency domain: spectral energy in fault-frequency bands).

---

## 6. Software Standards and Certification Context

**SAE International. ARP4754A (2010).** *Guidelines for Development of Civil Aircraft and Systems.*

Defines the development assurance levels (DAL A–E) for aircraft systems. Flight control actuators are typically DAL A or B. A real productised version of this system would need to comply with this standard. The current implementation is a research prototype and does **not** claim certification compliance.

**RTCA DO-178C (2011).** *Software Considerations in Airborne Systems and Equipment Certification.*

The software qualification standard for airborne software. Our PHM system runs as a ground-based MRO tool (not airborne software), so DO-178C does not apply directly. However, any onboard health-monitoring functions would need to comply.

**SAE ATA MSG-3 (2015).** *Operator/Manufacturer Scheduled Maintenance Development.*

The Reliability-Centred Maintenance (RCM) methodology used by all commercial aircraft OEMs. Our condition-based maintenance approach is designed to complement MSG-3 by providing data-driven evidence to adjust maintenance intervals.
