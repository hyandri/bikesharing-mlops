# tests/test_pipeline.py

import pytest
import requests
import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
import pymysql
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

DB_CONFIG = {
    'host':     'localhost',
    'port':     3306,
    'user':     'mlops_user',
    'password': 'mlops@123.',
    'database': 'bike_sharing',
}

MLFLOW_TRACKING_URI = 'http://localhost:5000'
MODEL_NAME          = 'bike-sharing-best-model'
MODEL_VERSION       = '1'
API_BASE_URL        = 'http://localhost:8000'

REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'db':   0,
}

SAMPLE_INPUT = {
    'yr':                    1,
    'mnth':                  6,
    'hr':                    8,
    'weekday':               2,
    'holiday':               0,
    'workingday':            1,
    'weathersit_original':   1,
    'season_original':       2,
    'temp':                  0.6,
    'atemp':                 0.58,
    'hum':                   0.4,
    'windspeed':             0.1,
    'is_morning_rush':       1,
    'is_evening_rush':       0,
    'is_weekend':            0,
    'is_peak_season':        1,
    'temp_hum_interaction':  0.36,
    'temp_wind_interaction': 0.54,
}

FEATURE_COLS = [
    'yr', 'mnth', 'hr', 'weekday', 'holiday', 'workingday',
    'weathersit_original', 'season_original',
    'temp', 'atemp', 'hum', 'windspeed',
    'is_morning_rush', 'is_evening_rush',
    'is_weekend', 'is_peak_season',
    'temp_hum_interaction', 'temp_wind_interaction',
]


# ─────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────

@pytest.fixture(scope='session')
def db_engine():
    """SQLAlchemy engine connected to ColumnStore."""
    url = URL.create(
        'mysql+pymysql',
        username=DB_CONFIG['user'],
        password=DB_CONFIG['password'],
        host=DB_CONFIG['host'],
        port=DB_CONFIG['port'],
        database=DB_CONFIG['database'],
    )
    engine = create_engine(url)
    yield engine
    engine.dispose()


@pytest.fixture(scope='session')
def mlflow_model():
    """Loads XGBoost model from MLflow registry once per session."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.xgboost.load_model(
        f'models:/{MODEL_NAME}/{MODEL_VERSION}'
    )
    return model


@pytest.fixture(scope='session')
def redis_splits():
    """Loads X_test and y_test from Redis once per session."""
    import redis
    import pyarrow as pa

    r = redis.Redis(
        host=REDIS_CONFIG['host'],
        port=REDIS_CONFIG['port'],
        db=REDIS_CONFIG['db'],
        decode_responses=False
    )

    def load(key):
        data = r.get(key)
        if data is None:
            pytest.skip(f"Redis key '{key}' not found. Run the pipeline DAG first.")
        reader = pa.ipc.open_stream(pa.py_buffer(data))
        return reader.read_pandas()

    return {
        'X_test': load('bike_sharing:preprocessing:X_test'),
        'y_test': load('bike_sharing:preprocessing:y_test'),
    }


# ─────────────────────────────────────────
# TEST 1 — DATA INGESTION
# ─────────────────────────────────────────

class TestDataIngestion:

    def test_fact_table_row_count(self, db_engine):
        """fact_hourly_rentals should have exactly 17379 rows."""
        with db_engine.connect() as conn:
            result = conn.execute(
                text("SELECT COUNT(*) FROM fact_hourly_rentals")
            ).scalar()
        assert result == 17379, (
            f"Expected 17379 rows in fact_hourly_rentals, got {result}"
        )

    def test_dim_datetime_row_count(self, db_engine):
        """dim_datetime should have exactly 17379 rows."""
        with db_engine.connect() as conn:
            result = conn.execute(
                text("SELECT COUNT(*) FROM dim_datetime")
            ).scalar()
        assert result == 17379, (
            f"Expected 17379 rows in dim_datetime, got {result}"
        )

    def test_dim_weather_row_count(self, db_engine):
        """dim_weather should have exactly 4 rows."""
        with db_engine.connect() as conn:
            result = conn.execute(
                text("SELECT COUNT(*) FROM dim_weather")
            ).scalar()
        assert result == 4, (
            f"Expected 4 rows in dim_weather, got {result}"
        )

    def test_dim_season_row_count(self, db_engine):
        """dim_season should have exactly 4 rows."""
        with db_engine.connect() as conn:
            result = conn.execute(
                text("SELECT COUNT(*) FROM dim_season")
            ).scalar()
        assert result == 4, (
            f"Expected 4 rows in dim_season, got {result}"
        )

    def test_no_null_fact_ids(self, db_engine):
        """fact_hourly_rentals should have no null fact_ids."""
        with db_engine.connect() as conn:
            result = conn.execute(
                text("SELECT COUNT(*) FROM fact_hourly_rentals WHERE fact_id IS NULL")
            ).scalar()
        assert result == 0, (
            f"Found {result} null fact_ids in fact_hourly_rentals"
        )

    def test_no_null_cnt(self, db_engine):
        """cnt column should have no null values."""
        with db_engine.connect() as conn:
            result = conn.execute(
                text("SELECT COUNT(*) FROM fact_hourly_rentals WHERE cnt IS NULL")
            ).scalar()
        assert result == 0, (
            f"Found {result} null cnt values in fact_hourly_rentals"
        )

    def test_cnt_non_negative(self, db_engine):
        """cnt values should all be non-negative."""
        with db_engine.connect() as conn:
            result = conn.execute(
                text("SELECT COUNT(*) FROM fact_hourly_rentals WHERE cnt < 0")
            ).scalar()
        assert result == 0, (
            f"Found {result} negative cnt values"
        )


# ─────────────────────────────────────────
# TEST 2 — MODEL PREDICTION API
# ─────────────────────────────────────────

class TestModelPredictionAPI:

    def test_health_endpoint_returns_200(self):
        """Health endpoint should return 200."""
        response = requests.get(f'{API_BASE_URL}/health')
        assert response.status_code == 200, (
            f"Health endpoint returned {response.status_code}"
        )

    def test_health_endpoint_model_loaded(self):
        """Health endpoint should confirm model is loaded."""
        response = requests.get(f'{API_BASE_URL}/health')
        data = response.json()
        assert data['model_loaded'] is True, (
            "Model is not loaded according to health endpoint"
        )

    def test_predict_endpoint_returns_200(self):
        """Predict endpoint should return 200 for valid input."""
        response = requests.post(
            f'{API_BASE_URL}/predict',
            json=SAMPLE_INPUT
        )
        assert response.status_code == 200, (
            f"Predict endpoint returned {response.status_code}: "
            f"{response.text}"
        )

    def test_predict_response_has_predicted_cnt(self):
        """Response should contain predicted_cnt field."""
        response = requests.post(
            f'{API_BASE_URL}/predict',
            json=SAMPLE_INPUT
        )
        data = response.json()
        assert 'predicted_cnt' in data, (
            f"Response missing predicted_cnt field: {data}"
        )

    def test_predict_cnt_is_non_negative(self):
        """Predicted count should never be negative."""
        response = requests.post(
            f'{API_BASE_URL}/predict',
            json=SAMPLE_INPUT
        )
        data = response.json()
        assert data['predicted_cnt'] >= 0, (
            f"Predicted cnt is negative: {data['predicted_cnt']}"
        )

    def test_predict_cnt_is_integer(self):
        """Predicted count should be an integer."""
        response = requests.post(
            f'{API_BASE_URL}/predict',
            json=SAMPLE_INPUT
        )
        data = response.json()
        assert isinstance(data['predicted_cnt'], int), (
            f"predicted_cnt is not an integer: {type(data['predicted_cnt'])}"
        )

    def test_predict_returns_model_name(self):
        """Response should include model name."""
        response = requests.post(
            f'{API_BASE_URL}/predict',
            json=SAMPLE_INPUT
        )
        data = response.json()
        assert data['model_name'] == MODEL_NAME, (
            f"Unexpected model name: {data['model_name']}"
        )

    def test_predict_rejects_missing_field(self):
        """Predict endpoint should return 422 for missing fields."""
        incomplete_input = {k: v for k, v in SAMPLE_INPUT.items() if k != 'hr'}
        response = requests.post(
            f'{API_BASE_URL}/predict',
            json=incomplete_input
        )
        assert response.status_code == 422, (
            f"Expected 422 for missing field, got {response.status_code}"
        )

    def test_predict_rejects_invalid_hr(self):
        """Predict endpoint should return 422 for hr out of range."""
        invalid_input = {**SAMPLE_INPUT, 'hr': 25}
        response = requests.post(
            f'{API_BASE_URL}/predict',
            json=invalid_input
        )
        assert response.status_code == 422, (
            f"Expected 422 for invalid hr, got {response.status_code}"
        )


# ─────────────────────────────────────────
# TEST 3 — MODEL PERFORMANCE
# ─────────────────────────────────────────

class TestModelPerformance:

    def test_r2_exceeds_threshold(self, mlflow_model, redis_splits):
        """Model R² on test set should exceed 0.75."""
        from sklearn.metrics import r2_score

        X_test = redis_splits['X_test'][FEATURE_COLS]
        y_test = redis_splits['y_test'].values.ravel()

        y_pred = np.clip(mlflow_model.predict(X_test), 0, None)
        r2     = r2_score(y_test, y_pred)

        print(f"\nR² on test set: {r2:.4f}")
        assert r2 >= 0.75, (
            f"R² {r2:.4f} is below threshold of 0.75"
        )

    def test_rmse_within_bound(self, mlflow_model, redis_splits):
        """Model RMSE on test set should be below 150."""
        from sklearn.metrics import mean_squared_error

        X_test = redis_splits['X_test'][FEATURE_COLS]
        y_test = redis_splits['y_test'].values.ravel()

        y_pred = np.clip(mlflow_model.predict(X_test), 0, None)
        rmse   = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"\nRMSE on test set: {rmse:.2f}")
        assert rmse <= 150, (
            f"RMSE {rmse:.2f} exceeds threshold of 150"
        )

    def test_mae_within_bound(self, mlflow_model, redis_splits):
        """Model MAE on test set should be below 100."""
        from sklearn.metrics import mean_absolute_error

        X_test = redis_splits['X_test'][FEATURE_COLS]
        y_test = redis_splits['y_test'].values.ravel()

        y_pred = np.clip(mlflow_model.predict(X_test), 0, None)
        mae    = mean_absolute_error(y_test, y_pred)

        print(f"\nMAE on test set: {mae:.2f}")
        assert mae <= 100, (
            f"MAE {mae:.2f} exceeds threshold of 100"
        )

    def test_no_negative_predictions(self, mlflow_model, redis_splits):
        """Model should never produce negative predictions after clipping."""
        X_test = redis_splits['X_test'][FEATURE_COLS]
        y_pred = np.clip(mlflow_model.predict(X_test), 0, None)

        assert (y_pred >= 0).all(), (
            "Found negative predictions after clipping"
        )

    def test_prediction_range_reasonable(self, mlflow_model, redis_splits):
        """Predictions should fall within reasonable range 0-1000."""
        X_test = redis_splits['X_test'][FEATURE_COLS]
        y_pred = np.clip(mlflow_model.predict(X_test), 0, None)

        assert y_pred.max() <= 1000, (
            f"Max prediction {y_pred.max():.0f} exceeds reasonable upper bound of 1000"
        )