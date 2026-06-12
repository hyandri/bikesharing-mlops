# app/model.py
import os
import mlflow
import mlflow.xgboost

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow_server:5000')
MODEL_NAME          = 'bike-sharing-best-model'
MODEL_VERSION       = '1'

# Loaded once at startup
_model        = None
_model_name   = MODEL_NAME
_model_version = MODEL_VERSION


def load_model():
    """
    Loads the registered XGBoost model from MLflow
    model registry at application startup.
    """
    global _model

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_uri = f'models:/{MODEL_NAME}/{MODEL_VERSION}'

    print(f"Loading model from: {model_uri}")

    _model = mlflow.xgboost.load_model(model_uri)

    print(f"Model loaded successfully: {MODEL_NAME} v{MODEL_VERSION}")


def get_model():
    """Returns the loaded model instance."""
    if _model is None:
        raise RuntimeError("Model is not loaded. Call load_model() first.")
    return _model


def get_model_info():
    """Returns model name and version for response metadata."""
    return _model_name, _model_version