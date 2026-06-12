# monitoring/monitor.py
import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metric_preset import RegressionPreset

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

MLFLOW_TRACKING_URI = 'http://localhost:5000'
MODEL_NAME          = 'bike-sharing-best-model'
MODEL_VERSION       = '1'
DATA_PATH           = '../data/hour.csv'
REPORTS_DIR         = 'reports'

FEATURE_COLS = [
    'yr', 'mnth', 'hr', 'weekday', 'holiday', 'workingday',
    'weathersit', 'season', 'temp', 'atemp', 'hum', 'windspeed',
]

TARGET_COL = 'cnt'

# ─────────────────────────────────────────
# STEP 1 — LOAD AND PREPARE DATA
# ─────────────────────────────────────────

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Use raw features matching what the model was trained on
# Reference = training period (2011 + first half of 2012)
# Current control = last part of 2012 (test period)

df['dteday'] = pd.to_datetime(df['dteday'])

reference_raw = df[df['dteday'] <= '2012-08-07'].copy()
current_raw   = df[df['dteday'] >  '2012-08-07'].copy()

print(f"\nReference size: {len(reference_raw)} rows")
print(f"Current size:   {len(current_raw)} rows")


# ─────────────────────────────────────────
# STEP 2 — FEATURE ENGINEERING
# ─────────────────────────────────────────

def engineer_features(df):
    df = df.copy()

    # Rush hour flags
    df['is_morning_rush'] = df['hr'].between(7, 9).astype(int)
    df['is_evening_rush'] = df['hr'].between(16, 19).astype(int)

    # Weekend flag
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)

    # Peak season flag
    df['is_peak_season'] = df['season'].isin([2, 3]).astype(int)

    # Interaction features
    df['temp_hum_interaction']  = df['temp'] * (1 - df['hum'])
    df['temp_wind_interaction'] = df['temp'] * (1 - df['windspeed'])

    # Rename to match model training column names
    df.rename(columns={
        'weathersit': 'weathersit_original',
        'season':     'season_original',
    }, inplace=True)

    return df


MODEL_FEATURE_COLS = [
    'yr', 'mnth', 'hr', 'weekday', 'holiday', 'workingday',
    'weathersit_original', 'season_original',
    'temp', 'atemp', 'hum', 'windspeed',
    'is_morning_rush', 'is_evening_rush',
    'is_weekend', 'is_peak_season',
    'temp_hum_interaction', 'temp_wind_interaction',
]

reference_fe = engineer_features(reference_raw)
current_fe   = engineer_features(current_raw)

print("\nFeature engineering applied.")

# ─────────────────────────────────────────
# STEP 3 — LOAD MODEL FROM MLFLOW
# ─────────────────────────────────────────

print(f"\nLoading model from MLflow: {MODEL_NAME} v{MODEL_VERSION}...")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model = mlflow.xgboost.load_model(f'models:/{MODEL_NAME}/{MODEL_VERSION}')

print("Model loaded successfully.")

# ─────────────────────────────────────────
# STEP 4 — GENERATE PREDICTIONS
# ─────────────────────────────────────────

def get_predictions(df_fe):
    X      = df_fe[MODEL_FEATURE_COLS]
    y_pred = np.clip(model.predict(X), 0, None)
    return y_pred


reference_fe['prediction'] = get_predictions(reference_fe)
current_fe['prediction']   = get_predictions(current_fe)

print(f"\nReference prediction mean: {reference_fe['prediction'].mean():.2f}")
print(f"Current prediction mean:   {current_fe['prediction'].mean():.2f}")

# ─────────────────────────────────────────
# STEP 5 — SYNTHETIC DRIFT INJECTION
# ─────────────────────────────────────────

print("\nCreating synthetic drift batches...")

# (a) Control — current data unchanged
control = current_fe.copy()

# (b) Temp drift — shift temp and atemp up by 0.2
temp_drift = current_fe.copy()
temp_drift['temp']  = (temp_drift['temp']  + 0.2).clip(0, 1)
temp_drift['atemp'] = (temp_drift['atemp'] + 0.2).clip(0, 1)
temp_drift['temp_hum_interaction']  = (
    temp_drift['temp'] * (1 - temp_drift['hum'])
)
temp_drift['temp_wind_interaction'] = (
    temp_drift['temp'] * (1 - temp_drift['windspeed'])
)
temp_drift['prediction'] = get_predictions(temp_drift)

# (c) Hr drift — shift hr by +2 hours and update flags
hr_drift = current_fe.copy()
hr_drift['hr'] = (hr_drift['hr'] + 2) % 24
hr_drift['is_morning_rush'] = hr_drift['hr'].between(7, 9).astype(int)
hr_drift['is_evening_rush'] = hr_drift['hr'].between(16, 19).astype(int)
hr_drift['prediction'] = get_predictions(hr_drift)

print(f"Control temp mean:    {control['temp'].mean():.4f}")
print(f"Temp drift temp mean: {temp_drift['temp'].mean():.4f}")
print(f"Control hr mean:      {control['hr'].mean():.4f}")
print(f"Hr drift hr mean:     {hr_drift['hr'].mean():.4f}")

# ─────────────────────────────────────────
# STEP 6 — COLUMNS FOR REPORTS
# ─────────────────────────────────────────

REPORT_FEATURE_COLS = MODEL_FEATURE_COLS + ['prediction']
TARGET_COL          = 'cnt'

def prepare_report_df(df_fe, df_raw):
    """Combines features, predictions and target for reporting."""
    result = df_fe[REPORT_FEATURE_COLS].copy()
    result['target'] = df_raw[TARGET_COL].values
    return result

reference_report = prepare_report_df(reference_fe, reference_raw)
control_report   = prepare_report_df(control, current_raw)
temp_drift_report = prepare_report_df(temp_drift, current_raw)
hr_drift_report   = prepare_report_df(hr_drift, current_raw)

# ─────────────────────────────────────────
# STEP 7 — EVIDENTLY REPORTS
# ─────────────────────────────────────────

batches = {
    'control':    control_report,
    'temp_drift': temp_drift_report,
    'hr_drift':   hr_drift_report,
}

for name, current_batch in batches.items():

    print(f"\nGenerating reports for: {name}")

    # Data Drift Report
    drift_report = Report(metrics=[DataDriftPreset()])
    drift_report.run(
        reference_data=reference_report,
        current_data=current_batch
    )
    drift_path = f'{REPORTS_DIR}/drift_report_{name}.html'
    drift_report.save_html(drift_path)
    print(f"  Drift report saved: {drift_path}")

    # Regression Performance Report
    perf_report = Report(metrics=[RegressionPreset()])
    perf_report.run(
        reference_data=reference_report,
        current_data=current_batch
    )
    perf_path = f'{REPORTS_DIR}/performance_report_{name}.html'
    perf_report.save_html(perf_path)
    print(f"  Performance report saved: {perf_path}")

# ─────────────────────────────────────────
# STEP 8 — SUMMARY
# ─────────────────────────────────────────

print("\n=== MONITORING COMPLETE ===")
print(f"Reports saved to: {REPORTS_DIR}/")
print("Files generated:")
for name in batches.keys():
    print(f"  → drift_report_{name}.html")
    print(f"  → performance_report_{name}.html")
print("\nOpen any HTML file in a browser to view the interactive report.")