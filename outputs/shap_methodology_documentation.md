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
temperature_roll_mean_12       3.447581   0.171966  4.299312     14.375302     1
 temperature_roll_mean_6       1.318031  -0.115620  1.658347      5.566015     2
             temperature       1.144869  -0.118393  1.456776      5.527210     3
 temperature_roll_mean_3       0.792528  -0.093453  0.999481      3.588851     4
   vibration_roll_mean_6       0.755275  -0.127265  0.890722      3.348985     5
       temperature_lag_1       0.694214  -0.109758  0.878826      2.986163     6
  vibration_roll_mean_12       0.592989  -0.068318  0.699022      2.570210     7
       temperature_lag_3       0.552673  -0.064597  0.689495      2.454383     8
               vibration       0.453479  -0.084523  0.566020      1.988819     9
                pressure       0.437495  -0.074695  0.544792      1.977794    10
    pressure_roll_mean_6       0.426397  -0.059000  0.531426      2.051106    11
       temperature_lag_2       0.389111  -0.029175  0.492352      1.770649    12
   pressure_roll_mean_12       0.257216   0.035144  0.321664      1.256211    13
                    hour       0.213139  -0.002576  0.246371      0.423543    14
   vibration_roll_mean_3       0.189966  -0.036066  0.227964      0.850354    15
```

======================================================================

## Key Findings

**Top 5 Most Important Features:**

1. **temperature_roll_mean_12**
   - Mean |SHAP|: 3.4476
   - Mean SHAP: 0.1720
   - Impact: increases failure probability

2. **temperature_roll_mean_6**
   - Mean |SHAP|: 1.3180
   - Mean SHAP: -0.1156
   - Impact: decreases failure probability

3. **temperature**
   - Mean |SHAP|: 1.1449
   - Mean SHAP: -0.1184
   - Impact: decreases failure probability

4. **temperature_roll_mean_3**
   - Mean |SHAP|: 0.7925
   - Mean SHAP: -0.0935
   - Impact: decreases failure probability

5. **vibration_roll_mean_6**
   - Mean |SHAP|: 0.7553
   - Mean SHAP: -0.1273
   - Impact: decreases failure probability

======================================================================

## Statistical Summary

- **Total features analyzed:** 25
- **Mean SHAP value range:** [-0.1273, 0.1720]
- **Most variable feature:** temperature_roll_mean_12
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
