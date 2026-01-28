
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directories for outputs if they don't exist
os.makedirs('reports/figures', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Load modeling-ready data
# Assuming script is run from project root
try:
    df = pd.read_csv('data/processed/model_ready_data.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: data/processed/model_ready_data.csv not found. Please check the path.")
    exit(1)

# Prepare features and target
# Exclude non-numeric columns and target columns
exclude_cols = ['timestamp', 'machine_id', 'failure', 'failure_within_24h']
# Select only numeric columns for features, just in case there are other string columns
feature_cols = [col for col in df.columns if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])]
X = df[feature_cols]
y = df['failure_within_24h']

# Train-test split (temporal)
# Using 'timestamp' for temporal split
if 'timestamp' in df.columns:
    # Convert to datetime to avoid string interpolation errors in quantile
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Use quantile to finding splitting point
    train_cutoff = df['timestamp'].quantile(0.7)
    train_mask = df['timestamp'] <= train_cutoff
    
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]
    print(f"Train set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
else:
    print("Warning: 'timestamp' column not found. Using random split.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Correlation Matrix
print("Generating correlation matrix...")
corr_matrix = X_train.corr()
plt.figure(figsize=(20, 16))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
plt.title('Feature Correlation Matrix')
# Save to reports/figures instead of root to be cleaner
plt.savefig('reports/figures/correlation_matrix.png')
print("Correlation matrix saved to reports/figures/correlation_matrix.png")

# Logistic Regression Baseline
print("Training Logistic Regression Baseline...")
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train, y_train)

print("Evaluating model...")
y_pred = lr.predict(X_test)
print("\n=== Logistic Regression Baseline Results ===")
print(classification_report(y_test, y_pred))
print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
