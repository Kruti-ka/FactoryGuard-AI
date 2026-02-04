# SHAP Values Calculation & Analysis Documentation
**Week 3 - Akshada's Deliverable**

======================================================================

## What are SHAP Values?

SHAP (SHapley Additive exPlanations) values explain how much each feature
contributes to pushing a prediction away from the base value (average prediction).

**Key Properties:**
- Based on game theory (Shapley values)
- Each feature gets a fair share of credit for prediction
- Positive SHAP = pushes toward failure prediction
- Negative SHAP = pushes toward normal prediction

======================================================================

## Methodology

**Model Used:** Logistic Regression (baseline model from Week 2)

**SHAP Explainer:** LinearExplainer (exact, fast for linear models)

**Dataset:**
- Test set size: 500 instances
- Number of features: 25

**Calculation Steps:**
1. Loaded trained Logistic Regression model
2. Scaled test data using fitted StandardScaler
3. Created SHAP LinearExplainer
4. Calculated SHAP values for all test predictions
5. Analyzed feature importance using mean |SHAP|

======================================================================

## Feature Importance Rankings

Features ranked by mean absolute SHAP value:

```
                feature  mean_abs_shap  mean_shap  std_shap  max_abs_shap  rank
            temperature       1.502269   0.174351  2.044830      8.379221     1
               pressure       1.074544   0.020809  1.356781      5.200158     2
 vibration_roll_mean_12       1.057229   0.019227  1.392020      5.242336     3
   pressure_roll_mean_6       0.987151   0.060409  1.209113      4.672987     4
              vibration       0.977122   0.043392  1.324777      5.681411     5
      temperature_lag_3       0.931194  -0.055239  1.259461      5.175988     6
  vibration_roll_mean_3       0.631946   0.006406  0.857405      3.768675     7
temperature_roll_mean_6       0.558745  -0.046641  0.761948      2.968659     8
        vibration_lag_1       0.531306  -0.032313  0.707178      3.075807     9
  vibration_roll_mean_6       0.522318   0.013444  0.703698      2.862496    10
  pressure_roll_mean_12       0.501484   0.031173  0.606423      2.163499    11
                   hour       0.410192   0.005185  0.478188      0.820551    12
temperature_roll_mean_3       0.406846   0.046876  0.559464      2.311335    13
      temperature_lag_2       0.342919  -0.039422  0.467245      1.997457    14
   pressure_roll_mean_3       0.341209   0.025780  0.419585      1.640858    15
```

======================================================================

## Key Findings

**Top 5 Most Important Features:**

1. **temperature**
   - Mean |SHAP|: 1.5023
   - Mean SHAP: 0.1744
   - Impact: increases failure probability

2. **pressure**
   - Mean |SHAP|: 1.0745
   - Mean SHAP: 0.0208
   - Impact: increases failure probability

3. **vibration_roll_mean_12**
   - Mean |SHAP|: 1.0572
   - Mean SHAP: 0.0192
   - Impact: increases failure probability

4. **pressure_roll_mean_6**
   - Mean |SHAP|: 0.9872
   - Mean SHAP: 0.0604
   - Impact: increases failure probability

5. **vibration**
   - Mean |SHAP|: 0.9771
   - Mean SHAP: 0.0434
   - Impact: increases failure probability

======================================================================

## Statistical Summary

- **Total features analyzed:** 25
- **Mean SHAP value range:** [-0.0552, 0.1744]
- **Most variable feature:** temperature
- **Most consistent feature:** month

======================================================================

## Next Steps (for Team)

**Harish's Tasks:**
- Create SHAP summary plots using shap_values.pkl
- Generate force plots for individual predictions
- Create dependence plots for top 5 features

**Krutika's Tasks:**
- Validate feature impacts against physical expectations
- Check if temperature/vibration impacts make sense
- Prepare consolidated XAI report

======================================================================

## Files Generated

- `outputs/shap_values.pkl` - SHAP values for team use
- `outputs/feature_importance_analysis.csv` - Detailed rankings
- `outputs/shap_methodology_documentation.md` - This file

**Prepared by:** Akshada
**Date:** Week 3, FactoryGuard AI Project
