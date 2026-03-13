# References

Complete bibliography with DOIs for all sources cited in the AeroSense PHM system.

---

## Electro-Mechanical Actuators

**[1]** Jensen, S.C., Jenney, G.D., Dawson, D. (2000).
"Flight Test Experience with an Electromechanical Actuator on the F-18
Systems Research Aircraft."
*Proceedings of the 19th DASC Digital Avionics Systems Conference.*
NASA Technical Memorandum NASA/TM-2000-209116.
→ *Motor parameters (28 VDC bus, 5 mm/rev ball screw), position accuracy, bandwidth.*

**[2]** Bauer, W., Kennel, R. (2019).
"Fault Tolerant Control Design for Electromechanical Actuators."
*Actuators, 8(3), 60.* MDPI.
DOI: 10.3390/act8030060
→ *Flight-grade BLDC motor parameter ranges (Kt, R, L) for 1–10 kN force class.*

**[3]** Wheeler, P.W., et al. (2013).
"The More Electric Aircraft: Technology and challenges."
*IET Electrical Systems in Transportation, 3(3), 75–90.*
DOI: 10.1049/iet-est.2012.0071
→ *MEA weight and fuel saving case (530 kg hydraulic system, ~5% fuel).*

**[4]** Krishnan, R. (2010).
*Permanent Magnet Synchronous and Brushless DC Motor Drives.*
CRC Press. ISBN: 978-0-8247-5384-9.
→ *Standard BLDC voltage–current–torque equations used in `ema_dynamics.py`.*

---

## PHM Methodology

**[5]** Saxena, A., Goebel, K., Simon, D., Eklund, N. (2008).
"Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation."
*Proceedings of the 1st International Conference on Prognostics and Health
Management (PHM).* IEEE.
DOI: 10.1109/PHM.2008.4711414
→ *CMAPSS methodology, EoL threshold (30% health index), dataset design.*

**[6]** Saxena, A., Goebel, K., Larrosa, C.C., Luo, J., Vachtsevanos, G. (2010).
"Metrics for Offline Evaluation of Prognostic Performance."
*International Journal of Prognostics and Health Management, 1(1).*
DOI: 10.36001/ijphm.2010.v1i1.1336
→ *Asymmetric prognostic score function (a1=13, a2=10). Used in `rul_prediction.py`.*

---

## EMA-Specific PHM

**[7]** Byington, C.S., Watson, M., Edwards, D., Stoelting, P. (2004).
"A Model-Based Approach to Prognostics and Health Management for Flight
Control Actuators."
*Proceedings of IEEE Aerospace Conference.*
DOI: 10.1109/AERO.2004.1367852
→ *F-18 actuator PHM; 60% capacity alert threshold; current + vibration features.*

**[8]** Balaban, E., Bansal, P., Stoelting, P., Ray, A., Kare, S.M., Currier, S.
(2009). "A Diagnostic Approach for Electro-Mechanical Actuators in Aerospace
Systems." *Proceedings of IEEE Aerospace Conference.*
DOI: 10.1109/AERO.2009.4839636
→ *NASA EMA test rig; bearing ~40% of removals; kurtosis > 3.5 detection;
backlash from tracking error; fault modes cover > 90% of removals.
Also: motor parameter ranges and degraded-threshold validation.*

---

## Bearing Life and Failure

**[9]** Lundberg, G., Palmgren, A. (1947).
"Dynamic Capacity of Rolling Bearings."
*Acta Polytechnica, 1(3).* Royal Swedish Academy of Engineering Sciences.
→ *Weibull-based L10/L50 bearing life model used in `degradation_model.py`.*

**[10]** Harris, T.A. (2001).
*Rolling Bearing Analysis, 4th Edition.* John Wiley & Sons.
ISBN: 0-471-35553-6.
→ *BPFI, BPFO, BSF, FTF formulas (Chapter 14). NSK 6208 geometry parameters.*

**[11]** NSK Ltd. (2018).
*Ball and Roller Bearings — Technical Data.* Catalogue No. E1102.
NSK Global.
→ *NSK 6208ZZ bearing geometry: 8 balls, 0° contact, 52 mm pitch diameter,
12.7 mm ball diameter. Used to compute real bearing frequencies.*

**[12]** McFadden, P.D., Smith, J.D. (1984).
"Vibration Monitoring of Rolling Element Bearings by the High-Frequency
Resonance Technique — A Review."
*Tribology International, 17(1), 3–10.*
DOI: 10.1016/0301-679X(84)90076-8
→ *Kurtosis > 4 detection threshold; envelope analysis theory.*

---

## Motor / Winding Condition Monitoring

**[13]** Stone, G.C., Boulter, E.A., Culbert, I., Dhirani, H. (2004).
*Electrical Insulation for Rotating Machines.* IEEE Press.
ISBN: 0-471-44506-1.
→ *Arrhenius / Montsinger thermal aging model. Resistance increase with degradation.*

**[14]** Nandi, S., Toliyat, H.A., Li, X. (2005).
"Condition Monitoring and Fault Diagnosis of Electrical Motors — A Review."
*IEEE Transactions on Energy Conversion, 20(4), 719–729.*
DOI: 10.1109/TEC.2005.847955
→ *Kt reduction up to 30% at severe winding fault. Basis for linear Kt model.*

**[15]** Mellor, P.H., Roberts, D., Turner, D.R. (1991).
"Lumped Parameter Thermal Model for Electrical Machines of TEFC Design."
*IEE Proceedings B, 138(5), 205–218.*
DOI: 10.1049/ip-b.1991.0025
→ *Single-node thermal model parameters (R_thermal, C_thermal).*

---

## Vibration Analysis

**[16]** Randall, R.B. (2011).
*Vibration-Based Condition Monitoring: Industrial, Automotive and Aerospace
Applications.* Wiley. ISBN: 978-0-470-97340-8.
→ *Vibration feature extraction (RMS, peak, crest, kurtosis). End-of-life vibration
magnitude range 3–4 g for rolling-element bearings (Chapter 5).*

**[17]** Scheffer, C., Girdhar, P. (2004).
*Practical Machinery Vibration Analysis and Predictive Maintenance.*
Elsevier / Newnes. ISBN: 978-0-7506-6275-6.
→ *Defect severity table for rolling element bearings (Table 4.2). Confirms
3–4 g vibration level at severe (end-of-life) bearing condition.*

---

## ML for RUL Prediction

**[18]** Mosallam, A., Medjaher, K., Zerhouni, N. (2016).
"Data-driven Prognostic Method Based on Bayesian Approaches for Direct
Remaining Useful Life Prediction."
*Journal of Intelligent Manufacturing, 27(5), 1037–1048.*
DOI: 10.1007/s10845-014-0933-4
→ *Random Forest as top classical ML for RUL; n_estimators=200, depth=20
hyperparameter reference; MAE 20–35 cycles on CMAPSS.*

**[19]** Zhang, Z., et al. (2019).
"Fault Diagnosis of Rolling Bearings Using XGBOD with Enhanced Feature
Selection."
*IEEE Access, 7, 170700–170714.*
DOI: 10.1109/ACCESS.2019.2954798
→ *Gradient Boosting for PHM fault classification.*

---

## Maintenance Standards and Regulations

**[20]** Nowlan, F.S., Heap, H.F. (1978).
*Reliability-Centered Maintenance.*
United Airlines / U.S. Department of Defense. Report AD-A066579.
→ *Foundational RCM document; C-check lead-time planning; MSG-3 predecessor.*

**[21]** SAE International. (2015).
*ATA MSG-3 Operator/Manufacturer Scheduled Maintenance Development.*
Revision 2015.2. SAE International.
→ *Task-card lead time requirements; CBM+ maintenance credit pathway.*

**[22]** ISO 13374-1:2003.
*Condition Monitoring and Diagnostics of Machines — Data Processing,
Communication and Presentation. Part 1: General guidelines.*
International Organisation for Standardisation.
→ *Two-tier alert architecture (warning / alert bands); 0.55 threshold basis.*

**[23]** SAE International. ARP4754A (2010).
*Guidelines for Development of Civil Aircraft and Systems.*
→ *Development Assurance Levels (DAL A–E). Flight control actuators are DAL A/B.*

**[24]** RTCA. DO-178C (2011).
*Software Considerations in Airborne Systems and Equipment Certification.*
RTCA Inc.
→ *Software qualification standard. Applies to any onboard PHM functions.*

**[25]** RTCA. DO-278A (2011).
*Software Integrity Assurance Considerations for Communication, Navigation,
Surveillance and Air Traffic Management Systems.*
RTCA Inc.
→ *Ground-based software qualification standard applicable to MRO tools.*
