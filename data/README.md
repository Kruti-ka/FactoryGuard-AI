# Data Directory

## Structure

- `raw/` - Contains raw CSV sensor logs
  - Expected format: timestamp, machine_id, vibration, temperature, pressure, failure
  
- `processed/` - Contains cleaned and feature-engineered datasets
  - `clean_data.csv` - After data cleaning
  - `model_ready_data.csv` - After feature engineering

## Data Requirements

### Sensor Logs (sensor_logs.csv)
Required columns:
- `timestamp` - ISO format datetime
- `machine_id` - Unique machine identifier (1-500)
- `vibration` - Vibration sensor reading
- `temperature` - Temperature in °C
- `pressure` - Pressure sensor reading
- `failure` - Binary (0 = normal, 1 = failure)

### Example Data Format
```csv
timestamp,machine_id,vibration,temperature,pressure,failure
2024-01-01 00:00:00,1,0.45,65.2,101.3,0
2024-01-01 01:00:00,1,0.48,66.1,101.5,0
2024-01-01 02:00:00,1,0.52,67.8,101.2,0
```

## Notes

- Place raw sensor data files in the `raw/` directory
- Processed data will be automatically saved to `processed/`
- Do not commit large data files to git (add to .gitignore)
