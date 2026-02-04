Objective

To visualize and interpret model predictions using Explainable AI (SHAP) techniques and improve model transparency.

Global Interpretation
SHAP Summary Plot

Shows global feature importance across all test samples.
Temperature rolling mean and vibration-based features are the most influential in predicting machine failure. This helps identify the most critical sensor parameters affecting system health.

SHAP Beeswarm Plot

Displays the distribution of SHAP values across all samples.
High temperature values contribute positively to failure risk, while low values reduce failure probability. This plot also shows the spread and variability of feature impact.

Local Interpretation
Waterfall (Local Explanation) Plots

Explain individual machine predictions.
Each bar shows how a feature pushes the prediction towards failure or non-failure. This helps understand decision-making at the single machine level.

These plots are useful for diagnosing specific failure cases and validating model decisions.

Conclusion

SHAP visualizations improve transparency and trust in the predictive maintenance model.
They allow engineers to understand both global model behavior and individual predictions, enabling better preventive maintenance planning and decision support.
