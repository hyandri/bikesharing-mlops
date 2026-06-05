# dags/common/preprocessing.py
import pandas as pd
import numpy as np
import pyarrow as pa
from sqlalchemy import text
from db_connectors import get_engine, get_redis_connection


# ─────────────────────────────────────────
# 1. EXTRACT
# ─────────────────────────────────────────
def extract_from_star_schema():
    """
    Joins all four star schema tables into a single flat DataFrame.
    Includes weathersit_original and workingday for feature engineering.
    """
    query = text("""
        SELECT
            dd.dteday,
            dd.yr,
            dd.mnth,
            dd.hr,
            dd.weekday,
            dd.holiday,
            dd.workingday,
            dw.weathersit_original,
            ds.season_original,
            f.temp,
            f.atemp,
            f.hum,
            f.windspeed,
            f.cnt
        FROM fact_hourly_rentals f
        JOIN dim_datetime dd ON f.datetime_id   = dd.datetime_id
        JOIN dim_weather  dw ON f.weathersit_id = dw.weathersit_id
        JOIN dim_season   ds ON f.season_id     = ds.season_id
        ORDER BY dd.dteday, dd.hr
    """)

    with get_engine().connect() as conn:
        df = pd.read_sql(query, conn)

    return df


# ─────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────
def engineer_features(df):
    """
    Feature engineering for XGBoost.
    Tree-based models work better with raw integers
    than cyclical encodings.
    """
    # Rush hour flags
    df['is_morning_rush'] = df['hr'].between(7, 9).astype(int)
    df['is_evening_rush'] = df['hr'].between(16, 19).astype(int)

    # Weekend flag
    df['is_weekend'] = (df['weekday'] >= 5).astype(int)

    # Peak season flag (Summer=2, Fall=3)
    df['is_peak_season'] = df['season_original'].isin([2, 3]).astype(int)

    # Interaction features
    df['temp_hum_interaction']  = df['temp'] * (1 - df['hum'])
    df['temp_wind_interaction'] = df['temp'] * (1 - df['windspeed'])

    # Keep hr, mnth, weekday, season_original as raw integers
    # XGBoost handles these better than sin/cos transforms
    # No one-hot encoding needed for tree models

    return df


# ─────────────────────────────────────────
# 3. TEMPORAL SPLIT
# ─────────────────────────────────────────
def temporal_split(df):
    """
    Splits data temporally to avoid leakage.
    80/20 split based on date order.
    Train: Jan 2011 - Aug 7 2012
    Test:  Aug 8 2012 - Dec 31 2012
    """
    df['dteday'] = pd.to_datetime(df['dteday'])

    train = df[df['dteday'] <= '2012-08-07'].copy()
    test  = df[df['dteday'] >  '2012-08-07'].copy()

    for split in [train, test]:
        split.drop(columns=['dteday'], inplace=True)

    TARGET   = 'cnt'
    FEATURES = [c for c in train.columns if c != TARGET]

    return {
        'X_train': train[FEATURES],
        'y_train': train[[TARGET]],
        'X_test':  test[FEATURES],
        'y_test':  test[[TARGET]],
    }


# ─────────────────────────────────────────
# 4. SERIALISE + REDIS
# ─────────────────────────────────────────
def serialise_to_arrow(df):
    """Serialises a DataFrame to PyArrow bytes."""
    table  = pa.Table.from_pandas(df, preserve_index=False)
    sink   = pa.BufferOutputStream()
    writer = pa.ipc.new_stream(sink, table.schema)
    writer.write_table(table)
    writer.close()
    return bytes(sink.getvalue())


def store_splits_in_redis(splits, ttl_seconds=86400):
    """
    Serialises each split and stores in Redis with a 24-hour TTL.
    Returns only the Redis keys.
    """
    r    = get_redis_connection()
    keys = {}

    for name, df in splits.items():
        key       = f'bike_sharing:preprocessing:{name}'
        r.set(key, serialise_to_arrow(df), ex=ttl_seconds)
        keys[name] = key

    return keys


def load_split_from_redis(key):
    """Deserialises a PyArrow bytes object from Redis back to DataFrame."""
    r    = get_redis_connection()
    data = r.get(key)

    if data is None:
        raise ValueError(f"Redis key '{key}' not found or expired.")

    reader = pa.ipc.open_stream(pa.py_buffer(data))
    return reader.read_pandas()