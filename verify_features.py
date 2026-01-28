import pandas as pd

# Load the dataset
df = pd.read_csv('data/processed/model_ready_data.csv')

print("=" * 70)
print("FEATURE ENGINEERING VERIFICATION REPORT")
print("=" * 70)

print("\nDATASET OVERVIEW:")
print(f"Total Rows: {len(df):,}")
print(f"Total Columns: {len(df.columns)}")
print(f"Missing Values: {df.isnull().sum().sum()}")

print("\nALL COLUMNS:")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")

print("\nFEATURE BREAKDOWN:")

# Categorize features
original_features = ['timestamp', 'machine_id', 'vibration', 'temperature', 'pressure', 'failure', 'failure_within_24h']
temporal_features = [c for c in df.columns if c in ['hour', 'day', 'month', 'day_of_week']]
lag_features = [c for c in df.columns if 'lag' in c]
rolling_features = [c for c in df.columns if 'roll' in c]
other_features = [c for c in df.columns if c not in original_features + temporal_features + lag_features + rolling_features]

print(f"\n1. Original Features ({len(original_features)}):")
for f in original_features:
    if f in df.columns:
        print(f"   - {f}")

print(f"\n2. Temporal Features ({len(temporal_features)}):")
for f in temporal_features:
    print(f"   - {f}")

print(f"\n3. Lag Features ({len(lag_features)}):")
for f in lag_features:
    print(f"   - {f}")

print(f"\n4. Rolling Features ({len(rolling_features)}):")
for f in rolling_features:
    print(f"   - {f}")

if other_features:
    print(f"\n5. Other Features ({len(other_features)}):")
    for f in other_features:
        print(f"   - {f}")

print("\nTARGET VARIABLE DISTRIBUTION:")
print(df['failure_within_24h'].value_counts().sort_index())
print(f"\nClass balance: {(df['failure_within_24h']==0).sum()} normal vs {(df['failure_within_24h']==1).sum()} failure")

print("\nDATA QUALITY:")
print(f"Rows dropped due to NaN: {7126 - len(df)}")
print(f"Percentage retained: {(len(df)/7126)*100:.2f}%")

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)
