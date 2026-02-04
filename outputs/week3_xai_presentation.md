# FactoryGuard AI - Week 3 XAI Insights
**Explainable AI for Predictive Maintenance**

---

## Slide 1: Project Overview

### FactoryGuard AI - Week 3 Objectives
- **Goal:** Make predictive maintenance model interpretable and trustworthy
- **Approach:** SHAP (SHapley Additive exPlanations) analysis
- **Team:** Akshada (Calculation), Harish (Visualization), Krutika (Validation)

### Key Deliverables
✅ SHAP value calculation for 500 test samples  
✅ Comprehensive visualizations (summary, force, dependence plots)  
✅ Physical validation against engineering expectations  
✅ Consolidated XAI report with recommendations

---

## Slide 2: Model & Data Summary

### Model Details
- **Type:** Logistic Regression (baseline from Week 2)
- **Features:** 25 (sensor readings + engineered features)
- **Test Set:** 500 samples
- **Target:** Predict failure within 24 hours

### Feature Categories
- **Temperature:** 7 features (base, rolling means, lags)
- **Vibration:** 7 features (base, rolling means, lags)
- **Pressure:** 6 features (base, rolling means, lags)
- **Temporal:** 4 features (hour, day, day_of_week, month)
- **Other:** 1 feature

---

## Slide 3: Top 10 Most Important Features

| Rank | Feature | Mean \|SHAP\| | Impact |
|------|---------|---------------|--------|
| 1️⃣ | **temperature** | 1.502 | ↑ Increases failure risk |
| 2️⃣ | **pressure** | 1.075 | ↑ Increases failure risk |
| 3️⃣ | **vibration_roll_mean_12** | 1.057 | ↑ Increases failure risk |
| 4️⃣ | **pressure_roll_mean_6** | 0.987 | ↑ Increases failure risk |
| 5️⃣ | **vibration** | 0.977 | ↑ Increases failure risk |
| 6️⃣ | **temperature_lag_3** | 0.931 | ⚠️ Decreases failure risk |
| 7️⃣ | **vibration_roll_mean_3** | 0.632 | ↑ Increases failure risk |
| 8️⃣ | **temperature_roll_mean_6** | 0.559 | ⚠️ Decreases failure risk |
| 9️⃣ | **vibration_lag_1** | 0.531 | ⚠️ Decreases failure risk |
| 🔟 | **vibration_roll_mean_6** | 0.522 | ↑ Increases failure risk |

### Key Insight
🌡️ **Temperature dominates** - 1.5x more important than next feature

---

## Slide 4: SHAP Visualizations - Global Importance

### Summary Plot
Shows feature importance across all predictions
- Temperature features dominate
- Vibration and pressure show moderate importance
- Temporal features have minimal impact

### Beeswarm Plot
Shows distribution of SHAP values
- Red dots = high feature values
- Blue dots = low feature values
- High temperature → pushes toward failure prediction

### Key Takeaway
📊 Model clearly prioritizes **temperature** and **vibration** sensors

---

## Slide 5: SHAP Visualizations - Local Explanations

### Force Plots (Waterfall)
Explain individual machine predictions
- **10 representative cases** generated
- Shows which features pushed prediction toward failure/normal
- Enables engineers to understand specific predictions

### Example Use Case
"Why did the model predict Machine #42 would fail?"
- High temperature (+0.85 SHAP)
- Elevated vibration (+0.32 SHAP)
- Normal pressure (-0.15 SHAP)
- **Result:** High failure risk prediction

### Key Takeaway
🔍 Engineers can **validate** and **trust** individual predictions

---

## Slide 6: Feature Dependence Analysis

### Temperature Dependence
- Higher temperature → higher SHAP values
- Confirms expected physical relationship
- Clear positive correlation

### Vibration Dependence
- Higher vibration → higher SHAP values
- Consistent with mechanical failure theory
- Some non-linearity observed

### Pressure Dependence
- Mixed relationship (expected)
- Both high and low pressure can indicate issues

### Key Takeaway
📈 Model captures **expected physical relationships** for base sensors

---

## Slide 7: Physical Validation Results

### Overall Consistency
- **Total Features:** 25
- **Consistent with Expectations:** 8 (32%)
- **Inconsistent:** 6 (24%)
- **Acceptable (Mixed):** 11 (44%)
- **Effective Consistency Rate:** 52%

### By Feature Category
| Category | Consistency Rate |
|----------|------------------|
| Temperature | 42.9% ⚠️ |
| Vibration | 71.4% ✅ |
| Pressure | 100% ✅ |
| Temporal | 100% ✅ |

### Key Takeaway
⚠️ **Temperature engineered features** show concerning inconsistencies

---

## Slide 8: Identified Anomalies

### 🔴 High-Priority Issues (3)

1. **temperature_lag_3** (Rank #6)
   - Expected: Positive | Actual: Negative
   - Issue: Important lag feature shows opposite direction

2. **temperature_roll_mean_6** (Rank #8)
   - Expected: Positive | Actual: Negative
   - Issue: Contradicts base temperature behavior

3. **vibration_lag_1** (Rank #9)
   - Expected: Positive | Actual: Negative
   - Issue: Recent vibration shows protective effect

### Possible Causes
- Multicollinearity between features
- Temporal misalignment in lag features
- Post-maintenance data patterns
- Feature scaling issues

---

## Slide 9: Recommendations

### Priority 1: HIGH - Feature Engineering Review
**Issue:** Multiple temperature features show unexpected behavior  
**Action:**
- Analyze correlation matrix for temperature features
- Consider PCA or feature selection
- Validate temporal alignment of lag features
- Review feature scaling methodology

### Priority 2: HIGH - Data Quality Investigation
**Issue:** 7 anomalies detected in feature impacts  
**Action:**
- Audit sensor data collection process
- Check calibration records
- Verify preprocessing pipeline
- Look for systematic biases

### Priority 3: MEDIUM - Domain Expert Review
**Issue:** Need validation of unexpected patterns  
**Action:**
- Present findings to maintenance engineers
- Gather operational context
- Validate if negative impacts have physical explanations

---

## Slide 10: Business Impact & Value

### ✅ Strengths
- **Transparency:** SHAP makes predictions interpretable
- **Trust:** Engineers can validate model logic
- **Insights:** Identified critical features (temperature, vibration)
- **Framework:** Established validation methodology

### ⚠️ Concerns
- **Consistency:** 52% rate suggests refinement needed
- **Anomalies:** 7 counterintuitive behaviors identified
- **Deployment:** Model needs improvement before production

### 💰 Value Delivered
- **Risk Mitigation:** Caught potential model flaws early
- **Clear Roadmap:** Actionable recommendations for improvement
- **Documentation:** Comprehensive artifacts for handoff

---

## Slide 11: Technical Achievements

### Scripts Developed
✅ `shap_calculation.py` (418 lines) - SHAP value calculation  
✅ `shap_visualization.py` (137 lines) - Visualization generation  
✅ `model_validation.py` - Physical validation analysis

### Visualizations Created
✅ Summary plot - Global feature importance  
✅ Beeswarm plot - Feature distribution  
✅ 10 Force plots - Individual predictions  
✅ 5 Dependence plots - Feature relationships  
✅ 5 Waterfall plots - Detailed explanations

### Documentation
✅ SHAP methodology documentation  
✅ Visual interpretation guide  
✅ Physical validation report  
✅ Consolidated XAI final report

---

## Slide 12: Next Steps & Roadmap

### Week 4 - Immediate Actions
1. **Investigate Temperature Features**
   - Correlation analysis
   - Feature selection/PCA
   - Temporal alignment validation

2. **Domain Expert Consultation**
   - Present findings to engineers
   - Validate unexpected patterns
   - Gather operational context

### Week 5-6 - Short-Term Actions
3. **Model Comparison**
   - Evaluate XGBoost SHAP values
   - Compare feature importance across models
   - Assess non-linear model performance

4. **Data Quality Audit**
   - Review sensor calibration
   - Check for biases
   - Validate preprocessing

### Long-Term - Continuous Improvement
5. **Enhanced Feature Engineering**
6. **Automated Validation Framework**

---

## Slide 13: Conclusion

### Summary
Week 3 successfully made FactoryGuard AI **interpretable** and **validated** its logic against physical expectations.

### Key Findings
- ✅ Temperature is the most critical feature
- ✅ Base sensors show expected behavior
- ⚠️ Engineered features need refinement
- ⚠️ 52% consistency rate requires improvement

### Final Status
⚠️ **READY FOR REFINEMENT**

### Recommendation
Proceed to Week 4 with focus on:
- Feature engineering improvements
- Domain expert consultation
- Data quality validation

---

## Slide 14: Q&A

### Common Questions

**Q: Can we deploy this model to production?**  
A: Not yet. The 52% consistency rate indicates feature engineering refinement is needed first.

**Q: What's the most important finding?**  
A: Temperature dominates predictions (1.5x more important than next feature), but engineered temperature features show concerning inconsistencies.

**Q: How do we fix the anomalies?**  
A: Priority actions: (1) Review temperature feature engineering, (2) Investigate multicollinearity, (3) Consult domain experts.

**Q: What's the business value of XAI?**  
A: Transparency builds trust with engineers, validates model logic, and catches potential flaws before deployment.

---

**Thank you!**

**Team:** Akshada, Harish, Krutika  
**Project:** FactoryGuard AI - Week 3 XAI Analysis  
**Contact:** [Your contact information]
