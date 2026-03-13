# Model Validation Report — RUL Prediction

**Generated from actual run of `validate_models.py`**
**Dataset: synthetic PHM fleet, 6,000 samples (8 aircraft × 5 actuator types × 150 cycles)**

---

## 1. Methodology

### 1.1 Dataset Generation

Synthetic data follows the same physics-based degradation model used in
production training:

- **Weibull bearing degradation** (Lundberg & Palmgren 1947)
- **Arrhenius winding aging** (Stone et al. 2004, Montsinger rule)
- **Linear ball-screw backlash growth** (Balaban et al. 2009)

This mirrors the **CMAPSS benchmark methodology** (Saxena et al. 2008,
PHM Conference): multiple independent units, varying operating conditions,
stochastic degradation paths, run-to-failure observations.

### 1.2 Evaluation Protocol

- **5-fold cross-validation** (shuffled, random\_state=42)
- **Metrics**: RMSE, MAE, and Saxena prognostic score
- **Train/test split for prognostic score**: 80/20, random\_state=0

### 1.3 Prognostic Score

From Saxena et al. (2010) "Metrics for Offline Evaluation of Prognostic
Performance", IJPHM 1(1). DOI: 10.36001/ijphm.2010.v1i1.1336:

```
d_i = y_pred_i - y_true_i

s_i = exp(-d_i / 13) - 1   if d_i < 0   (early prediction — conservative)
s_i = exp( d_i / 10) - 1   if d_i >= 0  (late prediction  — dangerous)

Score = sum(s_i)   [lower is better; 0 = perfect]
```

**Note on scale**: The absolute score values here are larger than reported
in CMAPSS benchmarks because our RUL range is 0–500 cycles vs. the
piecewise-linear CMAPSS label that clips at 125 cycles. The exponential
penalty is therefore applied to larger residuals. **Relative model ranking
is valid**; absolute scores cannot be directly compared to CMAPSS literature.

---

## 2. Results (Actual Run Output)

| Model | CV RMSE | ± std | CV MAE | Prognostic Score | Train Time (s) |
|---|---|---|---|---|---|
| **Random Forest (n=200, depth=20)** | **64.21** | **1.80** | **48.55** | **2.44 × 10⁹** | 15.46 |
| Gradient Boosting (n=200, depth=5) | 64.71 | 2.12 | 48.94 | 2.74 × 10⁹ | 17.23 |
| SVR (RBF kernel, C=100) | 65.51 | 1.49 | 49.27 | 3.26 × 10⁹ | 3.02 |
| MLP (100-50, relu, adam) | 65.26 | 1.44 | 50.14 | 1.22 × 10⁹ | 5.44 |
| Ridge Regression (alpha=1.0) | 66.81 | 1.39 | 51.52 | 3.50 × 10⁹ | 0.06 |

**Winner by CV RMSE**: Random Forest
**Winner by Prognostic Score**: MLP (lower exponential penalty, likely fewer large late-prediction outliers)

---

## 3. Analysis

### 3.1 Why Random Forest is the Production Choice

1. **Best CV RMSE (64.21)** — lowest mean squared prediction error across all
   five folds.
2. **Best CV MAE (48.55)** — closest to true RUL on average.
3. **Interpretability**: feature importances are available, enabling model audits
   required under DO-178C/DO-278A guidance for ground-based MRO tools.
4. **Uncertainty quantification**: bootstrap-based confidence intervals available
   from the tree ensemble without additional calibration (unlike SVR/Ridge).
5. **Robustness**: minimal hyperparameter sensitivity compared to SVR (kernel
   choice, C, gamma) or MLP (architecture, learning rate).

Reference supporting RF for RUL regression:
> Mosallam et al. (2016). "Data-driven Prognostic Method Based on Bayesian
> Approaches for Direct RUL Prediction." J. Intelligent Manufacturing 27(5):
> 1037–1048. DOI: 10.1007/s10845-014-0933-4.
> Reports RF achieving MAE of 20–35 cycles on CMAPSS benchmark — the lowest
> of the classical ML methods evaluated.

### 3.2 MLP Prognostic Score Observation

The MLP achieves the lowest prognostic score (1.22 × 10⁹), suggesting its
prediction errors are more symmetrically distributed (fewer extreme late
predictions). However, its CV RMSE (65.26) is higher than RF, meaning it is
less accurate on average but avoids the most dangerous tail errors. For a
safety-critical application the prognostic score is arguably the more
important metric. This trade-off is worth revisiting with a larger dataset.

**Recommendation**: Re-run this comparison with the full physics-simulation
dataset (`train_models.py`, ~15,000 samples). The neural network's advantage
in tail-error control may strengthen with more training data.

### 3.3 Ridge Regression Baseline

Ridge achieves RMSE 66.81 — only 4% worse than RF — confirming that
**linear trends in the feature space do capture most of the degradation
signal**. This is consistent with the linear feature engineering (RMS,
THD, kurtosis) all being approximately monotonically increasing with
degradation. The non-linear tree models improve on this by capturing
interaction effects (e.g., simultaneous thermal and vibration elevation).

### 3.4 Gradient Boosting vs. Random Forest

GBR (RMSE 64.71) is close to RF (64.21) but takes 17 s vs. 15 s, and
has slightly higher standard deviation (±2.12 vs. ±1.80). The additional
hyperparameter sensitivity (learning rate, subsample) makes RF the more
practical choice.

---

## 4. Limitations of This Validation

1. **Simulated data only**: All models are evaluated on data from the same
   generative process used for training. The simulation-to-reality gap
   (documented in Saxena et al. 2010) means real-hardware performance could
   differ substantially. A domain-adaptation step would be required before
   operational deployment.

2. **No aircraft-stratified CV**: The 5-fold split is random at the row
   level, not aircraft-stratified. Rows from the same aircraft appear in
   both train and test folds, making the CV optimistic relative to
   generalisation to an unseen aircraft.

3. **RUL label noise**: Synthetic RUL labels include ±15-cycle Gaussian
   noise (representing operational variability). This floor on RMSE means
   results below ~20 cycles RMSE would be over-fitting, not improvement.

---

## 5. Raw Data

Full numeric results are in `data/validation_results.json`.

```json
{
  "dataset_info": {
    "n_samples": 6000,
    "n_aircraft": 8,
    "n_actuator_types": 5,
    "n_features": 17,
    "cv_folds": 5
  },
  "best_model_by_rmse": "Random Forest (n=200, depth=20)"
}
```
