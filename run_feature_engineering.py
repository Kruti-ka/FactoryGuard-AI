import os
import pandas as pd


print("SCRIPT STARTED")


# ---------------- FEATURE FUNCTIONS ---------------- #

def create_time_features(df):
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    return df


def create_lag_features(df, lags=[1, 2, 3]):
    sensor_cols = ['vibration', 'temperature', 'pressure']

    for col in sensor_cols:
        for lag in lags:
            df[f'{col}_lag_{lag}'] = df.groupby('machine_id')[col].shift(lag)

    return df


def create_rolling_features(df, windows=[3, 6, 12]):
    sensor_cols = ['vibration', 'temperature', 'pressure']

    for col in sensor_cols:
        for window in windows:
            df[f'{col}_roll_mean_{window}'] = (
                df.groupby('machine_id')[col]
                .rolling(window)
                .mean()
                .reset_index(0, drop=True)
            )

    return df


# ---------------- MAIN PIPELINE ---------------- #

def main():

    print("=" * 60)
    print("HARISH FEATURE ENGINEERING PIPELINE")
    print("=" * 60)

    input_path = "data/processed/clean_data.csv"
    output_path = "data/processed/model_ready_data.csv"

    if not os.path.exists(input_path):
        print("ERROR: clean_data.csv not found")
        return

    print("Loading cleaned dataset...")
    df = pd.read_csv(input_path)

    print("Rows:", len(df))

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print("Creating time features...")
    df = create_time_features(df)

    print("Creating lag features...")
    df = create_lag_features(df)

    print("Creating rolling features...")
    df = create_rolling_features(df)

    df.dropna(inplace=True)

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(output_path, index=False)

    print("\nSUCCESS ")
    print("Saved:", output_path)
    print("Final Shape:", df.shape)


if __name__ == "__main__":
    main()
