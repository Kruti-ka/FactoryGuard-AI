# Week 3 XAI Final Report - FactoryGuard AI
**Explainable AI (XAI) for Predictive Maintenance**  
**Team Deliverable - Comprehensive Analysis**

---

## Executive Summary

This report consolidates Week 3's Explainable AI (XAI) analysis for the FactoryGuard AI predictive maintenance system. The team applied SHAP (SHapley Additive exPlanations) to interpret model predictions, created comprehensive visualizations, and validated model logic against physical expectations.

### Team Contributions

| Team Member | Responsibility | Status |
|-------------|----------------|--------|
| **Akshada** | SHAP Values Calculation & Analysis | ✅ Complete |
| **Harish** | SHAP Visualizations (Force, Summary, Dependence Plots) | ✅ Complete |
| **Krutika** | Model Logic Validation & Physical Expectations | ✅ Complete |

### Key Findings

- **Model Type:** Logistic Regression (baseline from Week 2)
- **Features Analyzed:** 25 sensor and engineered features
- **Test Set Size:** 500 samples
- **Top Feature:** `temperature` (Mean |SHAP| = 1.50)
- **Physical Consistency:** 52% (moderate concerns identified)
- **Anomalies Detected:** 7 counterintuitive behaviors

---

## Part 1: SHAP Values Calculation (Akshada)

### Methodology

**SHAP Explainer:** LinearExplainer (exact, fast for linear models)  
**Dataset:** 500 test instances, 25 features  
**Calculation Approach:**
1. Loaded trained Logistic Regression model
2. Scaled test data using fitted StandardScaler
3. Created SHAP LinearExplainer
4. Calculated SHAP values for all test predictions
5. Analyzed feature importance using mean |SHAP|

### Top 10 Most Important Features

| Rank | Feature | Mean \|SHAP\| | Mean SHAP | Impact Direction |
|------|---------|---------------|-----------|------------------|
| 1 | `temperature` | 1.502 | +0.174 | ↑ Increases failure risk |
| 2 | `pressure` | 1.075 | +0.021 | ↑ Increases failure risk |
| 3 | `vibration_roll_mean_12` | 1.057 | +0.019 | ↑ Increases failure risk |
| 4 | `pressure_roll_mean_6` | 0.987 | +0.060 | ↑ Increases failure risk |
| 5 | `vibration` | 0.977 | +0.043 | ↑ Increases failure risk |
| 6 | `temperature_lag_3` | 0.931 | -0.055 | ↓ Decreases failure risk |
| 7 | `vibration_roll_mean_3` | 0.632 | +0.006 | ↑ Increases failure risk |
| 8 | `temperature_roll_mean_6` | 0.559 | -0.047 | ↓ Decreases failure risk |
| 9 | `vibration_lag_1` | 0.531 | -0.032 | ↓ Decreases failure risk |
| 10 | `vibration_roll_mean_6` | 0.522 | +0.013 | ↑ Increases failure risk |

### Key Insights

✅ **Temperature** is the most influential feature (1.5x more important than next feature)  
✅ **Vibration rolling means** show consistent positive impact on failure prediction  
⚠️ **Temperature lag features** show unexpected negative impacts (requires investigation)  
📊 **Pressure features** show moderate importance with mixed impacts

### Deliverables

- ✅ `scripts/shap_calculation.py` - SHAP calculation script (418 lines)
- ✅ `outputs/shap_values.pkl` - Saved SHAP values for team use
- ✅ `outputs/feature_importance_analysis.csv` - Feature importance rankings
- ✅ `outputs/shap_methodology_documentation.md` - Detailed methodology

---

## Part 2: SHAP Visualizations (Harish)

### Global Feature Importance

#### SHAP Summary Plot
![SHAP Summary Plot](file:///d:/Internship-project1/factoryguard-ai/outputs/plots/shap_summary.png)

**Interpretation:**
- Features ranked by mean absolute SHAP value
- Temperature dominates feature importance
- Vibration and pressure features show moderate importance
- Temporal features (hour, day) have minimal impact

#### SHAP Beeswarm Plot
![SHAP Beeswarm Plot](file:///d:/Internship-project1/factoryguard-ai/outputs/plots/shap_beeswarm.png)

**Interpretation:**
- Each dot represents one prediction
- Red = high feature value, Blue = low feature value
- Position shows SHAP value (impact on prediction)
- High temperature values push predictions toward failure

### Local Explanations (Individual Predictions)

#### Sample Force Plot - High Risk Prediction
![Local Explanation 0](file:///d:/Internship-project1/factoryguard-ai/outputs/plots/shap_force_plots/local_explanation_0.png)

**Interpretation:**
- Waterfall plot shows feature contributions for single prediction
- Red bars push toward failure, blue bars push toward normal
- Allows engineers to understand why specific predictions were made

**Additional Force Plots:** 10 representative cases generated
- High-risk predictions
- Low-risk predictions  
- Mixed cases showing feature interactions

### Feature Dependence Analysis

#### Temperature Dependence Plot
![Temperature Dependence](file:///d:/Internship-project1/factoryguard-ai/outputs/plots/shap_dependence/temperature_dependence.png)

**Interpretation:**
- Shows relationship between temperature values and SHAP values
- Higher temperatures → higher SHAP values (increased failure risk)
- Confirms expected physical relationship

#### Vibration Dependence Plot
![Vibration Dependence](file:///d:/Internship-project1/factoryguard-ai/outputs/plots/shap_dependence/vibration_dependence.png)

#### Pressure Dependence Plot
![Pressure Dependence](file:///d:/Internship-project1/factoryguard-ai/outputs/plots/shap_dependence/pressure_dependence.png)

#### Hour Dependence Plot
![Hour Dependence](file:///d:/Internship-project1/factoryguard-ai/outputs/plots/shap_dependence/hour_dependence.png)

#### Day Dependence Plot
![Day Dependence](file:///d:/Internship-project1/factoryguard-ai/outputs/plots/shap_dependence/day_dependence.png)

### Deliverables

- ✅ `scripts/shap_visualization.py` - Visualization generation script (137 lines)
- ✅ `outputs/plots/shap_summary.png` - Global feature importance
- ✅ `outputs/plots/shap_beeswarm.png` - Feature distribution plot
- ✅ `outputs/plots/shap_force_plots/` - 10 local explanation plots
- ✅ `outputs/plots/shap_dependence/` - 5 dependence plots (top features)
- ✅ `outputs/plots/shap_waterfall/` - 5 waterfall plots
- ✅ `outputs/visual_interpretation_guide.md` - Interpretation guide

---

## Part 3: Model Logic Validation (Krutika)

### Physical Expectations Framework

| Feature Type | Expected Impact | Rationale |
|--------------|-----------------|-----------|
| **Temperature** | Positive | Higher temps → thermal stress → increased wear |
| **Vibration** | Positive | Excessive vibration → mechanical imbalance → failure |
| **Pressure** | Mixed | Both high/low pressure indicate problems |
| **Temporal** | Mixed | May reflect operational schedules |

### Validation Results

#### Overall Consistency

- **Total Features:** 25
- **Consistent:** 8 (32%)
- **Inconsistent:** 6 (24%)
- **Acceptable (Mixed):** 11 (44%)
- **Consistency Rate:** 52% (when counting acceptable as consistent)

#### By Feature Category

| Category | Consistent | Inconsistent | Consistency Rate |
|----------|------------|--------------|------------------|
| Temperature (7 features) | 3 | 4 | 42.9% |
| Vibration (7 features) | 5 | 2 | 71.4% |
| Pressure (6 features) | 0 | 0 | 100% (all acceptable) |
| Temporal (4 features) | 0 | 0 | 100% (all acceptable) |

### Identified Anomalies

#### 🔴 High-Priority Anomalies (3)

1. **temperature_lag_3** (Rank #6)
   - Expected: Positive | Actual: Negative (-0.055)
   - Issue: Important lag feature shows opposite direction
   - Recommendation: Investigate temporal alignment

2. **temperature_roll_mean_6** (Rank #8)
   - Expected: Positive | Actual: Negative (-0.047)
   - Issue: Rolling mean contradicts base temperature
   - Recommendation: Check for multicollinearity

3. **vibration_lag_1** (Rank #9)
   - Expected: Positive | Actual: Negative (-0.032)
   - Issue: Recent vibration shows protective effect
   - Recommendation: Review post-maintenance patterns

#### 🟡 Medium-Priority Anomalies (4)

- `temperature_roll_mean_12` - Negative impact (unexpected)
- `temperature_lag_2` - Negative impact (unexpected)
- `vibration_lag_3` - Negative impact (unexpected)

### Recommendations

#### Priority 1: HIGH - Feature Engineering Review
- **Issue:** Multiple temperature features show unexpected behavior
- **Action:** Review feature scaling, check multicollinearity, validate temporal alignment

#### Priority 2: HIGH - Data Quality Investigation
- **Issue:** 7 anomalies detected
- **Action:** Audit sensor data, check calibration, verify preprocessing pipeline

#### Priority 3: MEDIUM - Domain Expert Review
- **Issue:** Need validation of unexpected patterns
- **Action:** Present findings to maintenance engineers, gather operational context

#### Priority 4: MEDIUM - Model Architecture
- **Issue:** Only 52% consistency rate
- **Action:** Consider non-linear models, interaction terms, ensemble methods

### Deliverables

- ✅ `scripts/model_validation.py` - Validation analysis script
- ✅ `outputs/physical_validation_report.md` - Detailed validation findings
- ✅ `outputs/validation_results.json` - Programmatic validation results
- ✅ `outputs/week3_xai_final_report.md` - This consolidated report

---

## Integrated Insights & Conclusions

### ✅ Strengths of Current Model

1. **Clear Feature Importance Hierarchy**
   - Temperature dominates (as expected physically)
   - Vibration and pressure show moderate importance
   - Model captures key sensor relationships

2. **Interpretable Predictions**
   - SHAP values provide clear explanations
   - Engineers can understand individual predictions
   - Visualizations support decision-making

3. **Base Sensor Values Consistent**
   - Temperature, vibration, pressure show expected directions
   - Core physical relationships captured

### ⚠️ Areas of Concern

1. **Engineered Feature Inconsistencies**
   - Lag features show unexpected negative impacts
   - Rolling means contradict base values
   - Suggests feature engineering issues

2. **Moderate Consistency Rate (52%)**
   - Lower than ideal for production deployment
   - Indicates potential model limitations
   - May require architecture changes

3. **Potential Multicollinearity**
   - Temperature features competing rather than complementing
   - Could indicate overfitting or redundancy

### 🎯 Recommended Next Steps

#### Immediate Actions (Week 4)
1. **Investigate Temperature Feature Engineering**
   - Analyze correlation matrix for temperature features
   - Consider PCA or feature selection
   - Validate temporal alignment of lag features

2. **Domain Expert Consultation**
   - Present findings to maintenance engineers
   - Validate unexpected patterns
   - Gather operational context (maintenance schedules, etc.)

#### Short-Term Actions (Week 5-6)
3. **Compare with Non-Linear Models**
   - Evaluate XGBoost SHAP values (from Week 2)
   - Compare feature importance across models
   - Assess if non-linear models capture physics better

4. **Data Quality Audit**
   - Review sensor calibration records
   - Check for systematic biases
   - Validate preprocessing pipeline

#### Long-Term Improvements
5. **Enhanced Feature Engineering**
   - Explore interaction terms
   - Consider polynomial features
   - Test domain-informed feature combinations

6. **Continuous Validation Framework**
   - Implement automated physical consistency checks
   - Monitor feature importance drift
   - Set up alerts for anomalous patterns

---

## Business Impact & Value

### Transparency & Trust
✅ **SHAP explanations** make model predictions interpretable to maintenance engineers  
✅ **Visual dashboards** support operational decision-making  
✅ **Validation framework** ensures model aligns with domain knowledge

### Risk Mitigation
⚠️ **Identified anomalies** prevent deployment of potentially flawed model  
⚠️ **Consistency checks** catch data quality issues early  
⚠️ **Recommendations** provide clear path to improvement

### Operational Readiness
📊 **52% consistency** suggests model needs refinement before production  
📊 **Clear action items** guide next development phase  
📊 **Comprehensive documentation** supports handoff and maintenance

---

## Technical Artifacts Summary

### Scripts
- `scripts/shap_calculation.py` (418 lines) - SHAP value calculation
- `scripts/shap_visualization.py` (137 lines) - Visualization generation
- `scripts/model_validation.py` - Physical validation analysis

### Data Files
- `outputs/shap_values.pkl` - SHAP values for 500 test samples
- `outputs/feature_importance_analysis.csv` - Feature rankings
- `outputs/validation_results.json` - Validation analysis results

### Visualizations
- `outputs/plots/shap_summary.png` - Global feature importance
- `outputs/plots/shap_beeswarm.png` - Feature distribution
- `outputs/plots/shap_force_plots/` - 10 local explanations
- `outputs/plots/shap_dependence/` - 5 dependence plots
- `outputs/plots/shap_waterfall/` - 5 waterfall plots

### Documentation
- `outputs/shap_methodology_documentation.md` - SHAP calculation methodology
- `outputs/visual_interpretation_guide.md` - Visualization interpretation
- `outputs/physical_validation_report.md` - Detailed validation findings
- `outputs/week3_xai_final_report.md` - This comprehensive report

---

## Conclusion

Week 3's XAI analysis successfully made the FactoryGuard AI model interpretable and validated its logic against physical expectations. While the model shows promise in capturing key sensor relationships, the **52% physical consistency rate** indicates that **feature engineering refinement is needed before production deployment**.

The team's comprehensive approach—combining SHAP calculation, visualization, and physical validation—provides a solid foundation for model improvement and establishes a framework for ongoing model interpretability.

### Final Status: ⚠️ **READY FOR REFINEMENT**

**Recommendation:** Proceed to Week 4 with focus on feature engineering improvements and domain expert consultation.

---

**Report Prepared By:** Akshada, Harish, Krutika  
**Date:** Week 3, FactoryGuard AI Project  
**Status:** Complete - Ready for stakeholder review
