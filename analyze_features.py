import pandas as pd
import json

# Load the dataset
df = pd.read_csv('data/processed/model_ready_data.csv')

# Create analysis dictionary
analysis = {
    "total_rows": len(df),
    "total_columns": len(df.columns),
    "missing_values": int(df.isnull().sum().sum()),
    "all_columns": df.columns.tolist(),
    "target_distribution": df['failure_within_24h'].value_counts().to_dict(),
    "rows_dropped": 7126 - len(df),
    "retention_percentage": round((len(df)/7126)*100, 2)
}

# Save to JSON for easy reading
with open('reports/feature_analysis.json', 'w') as f:
    json.dump(analysis, f, indent=2)

print("Analysis saved to reports/feature_analysis.json")
print(f"\nQuick Summary:")
print(f"Shape: {df.shape}")
print(f"Columns: {len(df.columns)}")
print(f"Missing: {df.isnull().sum().sum()}")
