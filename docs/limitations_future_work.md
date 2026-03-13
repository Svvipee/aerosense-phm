# Limitations and Future Work

## 1. Current Limitations

### 1.1 Simulated Training Data

The most significant limitation of the current system is that all ML models are trained entirely on **simulated data**. The physics model captures the dominant first-order dynamics of a BLDC/ball-screw EMA, but:

- **Unmodelled phenomena**: Gearbox tooth wear, resolver/LVDT sensor degradation, connector corrosion, and seal aging are not modelled.
- **Environmental factors**: Altitude-dependent lubricant viscosity changes, ice/moisture ingestion effects, and EMI from adjacent avionics are not represented.
- **Simulation-to-reality gap**: Models trained purely on simulation data routinely underperform on real hardware (a well-documented challenge in PHM literature, discussed in Saxena et al. 2010). A domain adaptation step (transfer learning or fine-tuning on real removal data) would be required before operational deployment.

**Mitigation path**: Partner with an MRO organisation to collect data from EMAs removed from service, label them with post-removal inspection findings, and retrain or fine-tune the models on real data.

### 1.2 Single-Component Health Model

The current model treats each EMA independently. In reality:

- Correlated failure modes exist across actuators on the same aircraft (same environmental exposure, same operating load profile).
- A fleet-level model that accounts for cross-actuator correlations could improve RUL predictions.

### 1.3 No Onboard Processing

The current architecture assumes data is downloaded at the gate and processed on the ground. This means:

- No real-time in-flight health monitoring.
- A ground-time delay between data generation and alert issuance.

A future version would run the feature extraction on an onboard edge processor (consistent with the emerging ATA Chapter 45 / IVHM architecture) and transmit compressed feature vectors via ACARS or future connected-aircraft datalinks.

### 1.4 RUL in Cycles, Not Calendar Time

The RUL predictor outputs results in service cycles (equivalent to flights). Airlines schedule maintenance in calendar time and flight hours. A production system would need a mapping from cycles to calendar/FH units based on each operator's route structure.

### 1.5 No Formal Certification

The system is a research prototype. Any deployment on revenue-generating aircraft would require:
- FAA/EASA Supplemental Type Certificate (STC) for the hardware.
- Ground-based PHM software qualification under DO-178C/DO-278A.
- Integration with the airline's existing Aircraft Health Monitoring (AHM) platform (typically Airbus AIRMAN or Boeing Airplane Health Management).

## 2. Future Work

### 2.1 Hardware Integration
- Develop the embedded sensor node (STM32/RISC-V based) with BLDC driver, ADC, and compressed feature extraction firmware.
- Partner with an EMA OEM (e.g., Parker Hannifin, Curtiss-Wright, Moog) for avionics-grade hardware integration.

### 2.2 Real Data Collection and Model Refinement
- Deploy sensor nodes on an EMA test rig (ground-based) to collect real degradation data.
- Run accelerated life tests (per MIL-HDBK-217F) to generate run-to-failure datasets.
- Apply transfer learning to adapt the simulated-data model to real-hardware distributions.

### 2.3 LSTM-Based Sequence Model
- The current Random Forest model uses only the most recent feature snapshot.
- An LSTM would model the **trajectory** of degradation, potentially improving RUL accuracy for actuators with unusual degradation progression patterns.
- Requires sufficient run-to-failure sequences (recommend ≥200 per fault class).

### 2.4 Maintenance Optimisation Layer
- Given RUL predictions for all actuators across a fleet, formulate a maintenance scheduling optimisation problem that minimises total downtime cost subject to aircraft availability and hangar capacity constraints.
- This is a combinatorial optimisation problem amenable to integer programming or constraint satisfaction.

### 2.5 Digital Twin Integration
- Build a high-fidelity digital twin of each EMA instance (parameterised from its installation data and operational history) to support "what-if" maintenance planning.
- Align with SAE AIR6988 (Digital Twin Lifecycle Management standard, in development).

### 2.6 Regulatory Pathway
- Engage with FAA Aircraft Certification Office (ACO) to define the acceptable means of compliance for PHM-based maintenance credit.
- Reference: FAA AC 120-17A (Maintenance Program Development) and the emerging FAA/EASA guidance on CBM+ (Condition-Based Maintenance Plus).
