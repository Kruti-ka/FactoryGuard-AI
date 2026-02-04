import shap
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------- PATH CONFIG ----------------

SHAP_PATH = "outputs/shap_values.pkl"
FEATURE_NAMES_PATH = "models/feature_names.pkl"

SUMMARY_OUT = "outputs/plots/shap_summary.png"
BEESWARM_OUT = "outputs/plots/shap_beeswarm.png"

FORCE_DIR = "outputs/plots/shap_force_plots"
DEPENDENCE_DIR = "outputs/plots/shap_dependence"
WATERFALL_DIR = "outputs/plots/shap_waterfall"

os.makedirs(FORCE_DIR, exist_ok=True)
os.makedirs(DEPENDENCE_DIR, exist_ok=True)
os.makedirs(WATERFALL_DIR, exist_ok=True)

# ---------------- LOAD SHAP VALUES ----------------

with open(SHAP_PATH, "rb") as f:
    shap_data = pickle.load(f)

print("SHAP file loaded")

# Handle dictionary or direct array
if isinstance(shap_data, dict):
    shap_values = shap_data["shap_values"]
else:
    shap_values = shap_data

# ---------------- LOAD FEATURE NAMES ----------------

with open(FEATURE_NAMES_PATH, "rb") as f:
    feature_names = pickle.load(f)

# ---------------- LOAD DATA ----------------

df = pd.read_csv("data/processed/model_ready_data.csv")

X = df[feature_names]

# Align rows with SHAP sampling size
X = X.iloc[:shap_values.shape[0]]

print("Data aligned with SHAP values")
print("SHAP shape:", shap_values.shape)
print("X shape:", X.shape)

# ---------------- SUMMARY PLOT ----------------

plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.savefig(SUMMARY_OUT, bbox_inches="tight", dpi=300)
plt.close()

print("Summary plot saved")

# ---------------- BEESWARM PLOT ----------------

plt.figure()
shap.summary_plot(shap_values, X, plot_type="dot", show=False)
plt.savefig(BEESWARM_OUT, bbox_inches="tight", dpi=300)
plt.close()

print("Beeswarm plot saved")

# ---------------- DEPENDENCE PLOTS (TOP 5 FEATURES) ----------------

top_features = feature_names[:5]

for feature in top_features:

    plt.figure()
    shap.dependence_plot(feature, shap_values, X, show=False)

    plt.savefig(f"{DEPENDENCE_DIR}/{feature}_dependence.png",
                bbox_inches="tight",
                dpi=300)
    plt.close()

print("Dependence plots saved")

# ---------------- LOCAL EXPLANATION (REPLACEMENT FOR FORCE PLOTS) ----------------

base_value = shap_values.mean(axis=0).sum()

for i in range(10):

    exp = shap.Explanation(
        values=shap_values[i],
        base_values=base_value,
        data=X.iloc[i],
        feature_names=X.columns
    )

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(exp, show=False)

    plt.savefig(f"{FORCE_DIR}/local_explanation_{i}.png",
                bbox_inches="tight",
                dpi=300)
    plt.close()

print("Local explanation plots saved")

# ---------------- EXTRA WATERFALL SET (FOR REPORT) ----------------

for i in range(5):

    exp = shap.Explanation(
        values=shap_values[i],
        base_values=base_value,
        data=X.iloc[i],
        feature_names=X.columns
    )

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(exp, show=False)

    plt.savefig(f"{WATERFALL_DIR}/waterfall_{i}.png",
                bbox_inches="tight",
                dpi=300)
    plt.close()

print("Waterfall plots saved")

# ---------------- DONE ----------------

print("\n======================================")
print("SHAP VISUALIZATION COMPLETED SUCCESSFULLY")
print("======================================")
