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
             temperature       1.334548   0.078046  1.858959      7.502545     1
                pressure       1.148521  -0.048675  1.451067      5.294458     2
   pressure_roll_mean_12       0.981170   0.081006  1.182746      3.805513     3
   vibration_roll_mean_6       0.858650  -0.088456  1.127669      4.389547     4
  vibration_roll_mean_12       0.653923  -0.074085  0.846256      3.172183     5
 temperature_roll_mean_3       0.587403   0.027174  0.878336      3.665018     6
    pressure_roll_mean_3       0.581912  -0.018666  0.719196      2.426629     7
    pressure_roll_mean_6       0.415480   0.020092  0.512046      1.667046     8
          pressure_lag_2       0.411921  -0.015566  0.510761      1.831415     9
               vibration       0.309519  -0.044932  0.393004      1.440035    10
                    hour       0.284586  -0.015154  0.329671      0.575007    11
temperature_roll_mean_12       0.271985  -0.003258  0.409764      1.519343    12
       temperature_lag_1       0.250529   0.023234  0.356382      1.502575    13
          pressure_lag_3       0.246385   0.019426  0.306928      1.024695    14
             day_of_week       0.235193   0.023442  0.263364      0.450808    15
```

======================================================================

## Key Findings

**Top 5 Most Important Features:**

1. **temperature**
   - Mean |SHAP|: 1.3345
   - Mean SHAP: 0.0780
   - Impact: increases failure probability

2. **pressure**
   - Mean |SHAP|: 1.1485
   - Mean SHAP: -0.0487
   - Impact: decreases failure probability

3. **pressure_roll_mean_12**
   - Mean |SHAP|: 0.9812
   - Mean SHAP: 0.0810
   - Impact: increases failure probability

4. **vibration_roll_mean_6**
   - Mean |SHAP|: 0.8587
   - Mean SHAP: -0.0885
   - Impact: decreases failure probability

5. **vibration_roll_mean_12**
   - Mean |SHAP|: 0.6539
   - Mean SHAP: -0.0741
   - Impact: decreases failure probability

======================================================================

## Statistical Summary

- **Total features analyzed:** 25
- **Mean SHAP value range:** [-0.0885, 0.0810]
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
