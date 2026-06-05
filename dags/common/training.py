# dags/common/training.py
import numpy as np
import mlflow
import mlflow.xgboost
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from db_connectors import get_redis_connection
from preprocessing import load_split_from_redis

MLFLOW_TRACKING_URI = 'http://mlflow_server:5000'
EXPERIMENT_NAME     = 'bike-sharing-demand'
MODEL_NAME          = 'bike-sharing-best-model'
REDIS_RUN_ID_KEY    = 'bike_sharing:training:best_run_id'


def train_xgboost(redis_keys):
    """
    Loads train and test splits from Redis,
    trains XGBoost, evaluates on test set,
    logs everything to MLflow.
    Returns the MLflow run ID.
    """
    # ── Load splits ──────────────────────────────
    X_train = load_split_from_redis(redis_keys['X_train'])
    y_train = load_split_from_redis(redis_keys['y_train']).values.ravel()
    X_test  = load_split_from_redis(redis_keys['X_test'])
    y_test  = load_split_from_redis(redis_keys['y_test']).values.ravel()

    # ── Model parameters ─────────────────────────
    # params = {
    #     'n_estimators':  300,
    #     'learning_rate': 0.02,
    #     'max_depth':     5,
    #     'random_state':  42,
    #     'verbosity':     0,
    # }
    params = {
    'n_estimators':  500,
    'learning_rate': 0.03,
    'max_depth':     8,
    'subsample':     0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'random_state':  42,
    'verbosity':     0,
}   

    # ── MLflow tracking ──────────────────────────
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name='xgboost') as run:

        # Train
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)

        # Predict and clip negatives
        y_pred = np.clip(model.predict(X_test), 0, None)

        # Metrics
        r2   = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)

        print(f"XGBoost — R²: {r2:.4f}  RMSE: {rmse:.2f}  MAE: {mae:.2f}")

        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metrics({'r2': r2, 'rmse': rmse, 'mae': mae})
        mlflow.xgboost.log_model(model, artifact_path='model')

        run_id = run.info.run_id

    return run_id


def register_best_model(run_id):
    """
    Registers the trained XGBoost model in MLflow model registry.
    Stores the run ID in Redis for downstream serving.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_uri = f'runs:/{run_id}/model'

    mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    print(f"Model registered as '{MODEL_NAME}' from run {run_id}")

    # Store run ID in Redis for serving stage
    r = get_redis_connection()
    r.set(REDIS_RUN_ID_KEY, run_id)

    print(f"Run ID stored in Redis under key: {REDIS_RUN_ID_KEY}")

    return {
        'run_id':     run_id,
        'model_name': MODEL_NAME,
        'model_uri':  model_uri,
    }