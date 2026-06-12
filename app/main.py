# app/main.py
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

from model import load_model, get_model, get_model_info
from schemas import PredictionInput, PredictionOutput

LOG_PATH = '/app/logs/predictions.jsonl'


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title='Bike Sharing Demand Prediction API',
    description='Predicts hourly bike rental counts using XGBoost',
    version='1.0.0',
    lifespan=lifespan,
)

FEATURE_ORDER = [
    'yr', 'mnth', 'hr', 'weekday', 'holiday', 'workingday',
    'weathersit_original', 'season_original',
    'temp', 'atemp', 'hum', 'windspeed',
    'is_morning_rush', 'is_evening_rush',
    'is_weekend', 'is_peak_season',
    'temp_hum_interaction', 'temp_wind_interaction',
]


def log_prediction(input_data: dict, predicted_cnt: int):
    """Appends prediction request and result to JSONL log file."""
    record = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'predicted_cnt': predicted_cnt,
        **input_data
    }
    with open(LOG_PATH, 'a') as f:
        f.write(json.dumps(record) + '\n')


@app.get('/health')
def health():
    try:
        get_model()
        model_loaded = True
    except RuntimeError:
        model_loaded = False

    return {
        'status':       'healthy' if model_loaded else 'unhealthy',
        'model_loaded': model_loaded,
        'model_name':   'bike-sharing-best-model',
    }


@app.post('/predict', response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    try:
        model = get_model()
        model_name, model_version = get_model_info()

        features = pd.DataFrame(
            [input_data.model_dump()],
            columns=FEATURE_ORDER
        )

        prediction = model.predict(features)
        predicted_cnt = max(0, int(np.round(prediction[0])))

        # Log prediction to file
        log_prediction(input_data.model_dump(), predicted_cnt)

        return PredictionOutput(
            predicted_cnt=predicted_cnt,
            model_name=model_name,
            model_version=model_version,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))