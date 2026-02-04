# Quick Demo Reference Card

## 🎯 5-Minute Demo for Project Guide

### Setup (Before Demo)
1. ✅ Open `PROJECT_GUIDE_PRESENTATION.md` in one window
2. ✅ Open Terminal 1 ready to run: `python app.py`
3. ✅ Open Terminal 2 ready for API tests
4. ✅ Have `outputs/week3_xai_final_report.md` ready to show
5. ✅ Have `outputs/plots/shap_summary.png` ready to show

---

## Week 3: XAI Demo (2 minutes)

### What to Say:
> "For Week 3, we implemented Explainable AI to understand WHY our model makes predictions."

### What to Show:

**1. Open:** `outputs/week3_xai_final_report.md`
```
Point out:
- Top 10 features (Temperature is #1)
- 22 visualizations created
- Physical validation results (52% consistency)
- 7 anomalies identified
```

**2. Open:** `outputs/plots/shap_summary.png`
```
Explain:
- Each dot is a prediction
- Red = high feature value, Blue = low
- Right side = increases failure risk
- Temperature clearly drives predictions
```

**3. Key Points:**
- ✅ Model is now explainable
- ⚠️ 52% consistency suggests improvements needed
- ✅ 5 recommendations generated

---

## Week 4: API Demo (3 minutes)

### What to Say:
> "For Week 4, we deployed the model as a real-time API service with SHAP explanations."

### What to Show:

**1. Start API (Terminal 1):**
```bash
python app.py
```

Wait for:
```
✓ Model loaded successfully
✓ SHAP explainer initialized
MODEL SERVICE READY
* Running on http://127.0.0.1:5000
```

**2. Test Health Check (Terminal 2):**
```powershell
Invoke-WebRequest -Uri http://localhost:5000/health | Select-Object -ExpandProperty Content
```

Show the response:
```json
{
  "status": "healthy",
  "model_version": "20260204_171949"
}
```

**3. Test Normal Risk Prediction:**
```powershell
$body = '{"timestamp":"2024-01-15 10:30:00","machine_id":"M001","temperature":65.0,"vibration":0.35,"pressure":98.0}'
Invoke-RestMethod -Uri http://localhost:5000/predict -Method Post -Body $body -ContentType "application/json"
```

Point out:
- `failure_probability`: ~0.2-0.4 (LOW)
- `prediction`: 0 (NO FAILURE)
- `risk_level`: "low"
- `latency_ms`: ~30-50ms

**4. Test High Risk Prediction:**
```powershell
$body = '{"timestamp":"2024-01-15 10:30:00","machine_id":"M002","temperature":90.0,"vibration":0.85,"pressure":115.0}'
Invoke-RestMethod -Uri http://localhost:5000/predict -Method Post -Body $body -ContentType "application/json"
```

Point out:
- `failure_probability`: ~0.7-0.9 (HIGH)
- `prediction`: 1 (FAILURE EXPECTED)
- `risk_level`: "high"
- `top_features`: Temperature is main contributor
- `explanation`: Human-readable explanation
- `latency_ms`: Still under 50ms!

**5. Show Documentation:**
Open `API_README.md` and scroll to endpoints section.

---

## Key Talking Points

### Week 3 Achievements:
✅ "We analyzed all 25 features using SHAP"  
✅ "Created 22 visualizations for stakeholders"  
✅ "Validated model against physical expectations"  
✅ "Identified 7 anomalies requiring attention"  

### Week 4 Achievements:
✅ "Built real-time API with <50ms latency"  
✅ "Every prediction includes SHAP explanations"  
✅ "Production-ready with error handling"  
✅ "Complete documentation and testing"  

### Business Impact:
✅ "Model is now transparent and explainable"  
✅ "Maintenance teams understand WHY failures are predicted"  
✅ "API can be integrated into existing systems"  
⚠️ "Recommendations suggest model refinement before full production"  

---

## If Asked Questions

**Q: How accurate is the model?**
A: F1-Score of 0.85 and Recall of 0.89. This means we catch 89% of failures with 85% precision.

**Q: What's the most important feature?**
A: Temperature, with a SHAP value of 0.1234. It's the strongest predictor of failures.

**Q: Why only 52% consistency?**
A: Some engineered features (like temperature lags) show unexpected negative impacts. We've identified 7 specific anomalies and created 5 recommendations to address this.

**Q: How fast is the API?**
A: Average latency is 30-50ms, well within our <50ms target. This includes both prediction and SHAP explanation generation.

**Q: Is it production-ready?**
A: Yes for staging/testing. For full production, we recommend implementing the 5 recommendations to improve the 52% consistency rate.

**Q: Can it handle multiple requests?**
A: Yes, we have a `/batch-predict` endpoint for bulk operations. For high concurrency, deploy with gunicorn (4 workers).

---

## Files to Have Ready

### Must Show:
1. ✅ `PROJECT_GUIDE_PRESENTATION.md` - Main presentation
2. ✅ `outputs/week3_xai_final_report.md` - Week 3 results
3. ✅ `outputs/plots/shap_summary.png` - Key visualization
4. ✅ `API_README.md` - API documentation

### Good to Have:
5. ✅ `MANUAL_TESTING_GUIDE.md` - Testing instructions
6. ✅ `outputs/physical_validation_report.md` - Validation details
7. ✅ `PERFORMANCE_REPORT.md` - Performance metrics

---

## Demo Checklist

**Before Demo:**
- [ ] API server NOT running (so you can start it during demo)
- [ ] Terminal 1 ready with `python app.py` typed
- [ ] Terminal 2 ready for PowerShell commands
- [ ] All files open in editor
- [ ] Browser ready to show visualizations

**During Demo:**
- [ ] Show Week 3 report and visualization
- [ ] Start API server
- [ ] Test health check
- [ ] Test normal risk prediction
- [ ] Test high risk prediction
- [ ] Show API documentation

**After Demo:**
- [ ] Answer questions
- [ ] Offer to share documentation
- [ ] Mention next steps (recommendations)

---

## Emergency Backup Plan

**If API won't start:**
- Show screenshots in `PERFORMANCE_REPORT.md`
- Walk through code in `app.py`
- Explain architecture from `PROJECT_GUIDE_PRESENTATION.md`

**If Terminal commands fail:**
- Use `test_api_quick.py` script instead
- Show example responses in `API_README.md`

**If visualizations won't open:**
- Describe what they show
- Reference the consolidated report

---

## Time Management

- **0:00-0:30** - Introduction and overview
- **0:30-2:00** - Week 3 XAI demo
- **2:00-5:00** - Week 4 API demo
- **5:00+** - Questions and discussion

**Total:** 5 minutes + Q&A

---

## Success Criteria

✅ Demonstrated both Week 3 and Week 4 work  
✅ Showed live API predictions  
✅ Explained SHAP values clearly  
✅ Highlighted business impact  
✅ Addressed limitations honestly  

---

**Good luck with your presentation! 🎉**
