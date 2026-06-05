# dags/bike_sharing_pipeline.py

from datetime import datetime, timedelta
import os
import sys
import pandas as pd

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'common'))

from db_connectors import get_engine, get_redis_connection
from gx_validator import validate_source_csv
from etl_transforms import (
    transform_to_star_schema,
    load_star_schema_to_db,
)

from preprocessing import (
    extract_from_star_schema,
    engineer_features,
    temporal_split,
    store_splits_in_redis,
    serialise_to_arrow,
    load_split_from_redis,
)

from training import (
    train_xgboost,
    register_best_model,
)

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────

DATA_PATH = '/opt/airflow/data/hour.csv'

default_args = {
    'owner': 'hyandri',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# ─────────────────────────────────────────
# INGESTION TASKS
# ─────────────────────────────────────────

def task_validate_source():
    results = validate_source_csv(DATA_PATH)

    if not results['success']:
        raise ValueError(
            f"Data validation failed! "
            f"{results['failed_expectations']} expectation(s) failed."
        )

    return results


def task_extract_and_transform():
    df = pd.read_csv(DATA_PATH)

    return transform_to_star_schema(df)


def task_load_to_olap(ti):
    star_schema = ti.xcom_pull(task_ids='extract_and_transform')

    if not star_schema:
        raise ValueError(
            "No star schema data received from upstream task."
        )

    return load_star_schema_to_db(
        get_engine(),
        star_schema
    )


# ─────────────────────────────────────────
# PREPROCESSING TASKS
# ─────────────────────────────────────────

def task_extract():
    """
    Extracts from ColumnStore,
    serialises to PyArrow,
    stores in Redis.
    Returns only the Redis key.
    """

    df = extract_from_star_schema()

    if df.empty:
        raise ValueError(
            "Extracted DataFrame is empty. "
            "Check star schema tables."
        )

    key = 'bike_sharing:preprocessing:raw'

    r = get_redis_connection()

    r.set(
        key,
        serialise_to_arrow(df),
        ex=86400
    )

    return key

def task_debug_data(ti):
    from preprocessing import load_split_from_redis
    
    keys = ti.xcom_pull(task_ids='engineer_and_split')
    
    X_train = load_split_from_redis(keys['X_train'])
    y_train = load_split_from_redis(keys['y_train'])
    X_test  = load_split_from_redis(keys['X_test'])
    y_test  = load_split_from_redis(keys['y_test'])
    
    print("=== SHAPES ===")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test:  {X_test.shape}")
    print(f"y_test:  {y_test.shape}")
    
    print("\n=== X_train COLUMNS ===")
    print(list(X_train.columns))
    
    print("\n=== y_train SAMPLE ===")
    print(y_train.head(10))
    
    print("\n=== y_train STATS ===")
    print(y_train.describe())
    
    print("\n=== X_train SAMPLE ===")
    print(X_train.head(5).to_string())
    
    print("\n=== X_train NULL CHECK ===")
    print(X_train.isnull().sum())


def task_engineer_and_split(ti):
    """
    Loads raw data from Redis,
    engineers features,
    splits temporally,
    stores splits in Redis.
    Returns dict of Redis keys.
    """

    raw_key = ti.xcom_pull(task_ids='extract')

    df = load_split_from_redis(raw_key)

    df = engineer_features(df)

    splits = temporal_split(df)

    keys = store_splits_in_redis(
        splits,
        ttl_seconds=86400
    )

    return keys


def task_verify_redis_splits(ti):
    keys = ti.xcom_pull(task_ids='engineer_and_split')

    if not keys:
        raise ValueError("No Redis keys received from upstream task.")

    for name, key in keys.items():
        df = load_split_from_redis(key)
        print(f"{name}: {len(df)} rows, {len(df.columns)} cols — key: {key}")

    return keys


# ─────────────────────────────────────────
# TRAINING TASKS
# ─────────────────────────────────────────

def task_train_xgboost(ti):
    """
    Loads splits from Redis, trains XGBoost,
    logs to MLflow. Returns run ID.
    """
    redis_keys = ti.xcom_pull(task_ids='verify_redis_splits')

    if not redis_keys:
        raise ValueError("No Redis keys received from verify_redis_splits.")

    run_id = train_xgboost(redis_keys)

    return run_id


def task_register_model(ti):
    """
    Registers the trained model in MLflow
    and stores run ID in Redis.
    """
    run_id = ti.xcom_pull(task_ids='train_xgboost')

    if not run_id:
        raise ValueError("No run ID received from train_xgboost.")

    return register_best_model(run_id)


# ─────────────────────────────────────────
# DAG DEFINITION
# ─────────────────────────────────────────

with DAG(
    'bike_sharing_full_pipeline',
    default_args=default_args,
    description=(
        'Full Bike Sharing Pipeline: '
        'Validation -> Star Schema -> OLAP -> '
        'Feature Engineering -> Temporal Split -> Redis'
    ),
    schedule_interval=None,
    catchup=False,
    tags=[
        'bike_sharing',
        'etl',
        'preprocessing',
        'redis',
        'olap',
        'feature_engineering',
    ],
) as dag:

    # INGESTION TASKS

    validate_task = PythonOperator(
        task_id='validate_source',
        python_callable=task_validate_source,
    )

    extract_transform_task = PythonOperator(
        task_id='extract_and_transform',
        python_callable=task_extract_and_transform,
    )

    load_olap_task = PythonOperator(
        task_id='load_to_olap',
        python_callable=task_load_to_olap,
    )

    # PREPROCESSING TASKS

    extract_task = PythonOperator(
        task_id='extract',
        python_callable=task_extract,
    )

    engineer_split_task = PythonOperator(
        task_id='engineer_and_split',
        python_callable=task_engineer_and_split,
    )

    verify_task = PythonOperator(
        task_id='verify_redis_splits',
        python_callable=task_verify_redis_splits,
    )

    # TRAINING TASKS

    train_xgboost_task = PythonOperator(
        task_id='train_xgboost',
        python_callable=task_train_xgboost,
    )

    register_model_task = PythonOperator(
        task_id='register_model',
        python_callable=task_register_model,
    )

    # PIPELINE FLOW

(
validate_task
>> extract_transform_task
>> load_olap_task
>> extract_task
>> engineer_split_task
>> verify_task
>> train_xgboost_task
>> register_model_task
    )