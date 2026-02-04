# SHAP Visualization Interpretation – Week 3 (Harish)

## Objective
To visualize and interpret model predictions using Explainable AI (SHAP) techniques.

---

## Global Interpretation

### SHAP Summary Plot
Shows global feature importance across all test samples.
Temperature rolling mean and vibration-based features are the most influential in predicting machine failure.

---

### SHAP Beeswarm Plot
Displays distribution of SHAP values.
High temperature values contribute positively to failure risk, while low values reduce failure probability.

---

## Local Interpretation

### Waterfall (Local Explanation) Plots
Explain individual machine predictions.
Each bar shows how a feature pushes prediction towards failure or non-failure.

These plots help understand why the model made a particular prediction.

---

## Conclusion

SHAP visualizations improve transparency and trust in the predictive maintenance model.
They allow engineers to validate model behavior with real sensor patterns.
