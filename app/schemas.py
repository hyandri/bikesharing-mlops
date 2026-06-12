# app/schemas.py
from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    yr:                    int   = Field(..., ge=0, le=1, description="Year: 0=2011, 1=2012")
    mnth:                  int   = Field(..., ge=1, le=12)
    hr:                    int   = Field(..., ge=0, le=23)
    weekday:               int   = Field(..., ge=0, le=6)
    holiday:               int   = Field(..., ge=0, le=1)
    workingday:            int   = Field(..., ge=0, le=1)
    weathersit_original:   int   = Field(..., ge=1, le=4)
    season_original:       int   = Field(..., ge=1, le=4)
    temp:                  float = Field(..., ge=0.0, le=1.0)
    atemp:                 float = Field(..., ge=0.0, le=1.0)
    hum:                   float = Field(..., ge=0.0, le=1.0)
    windspeed:             float = Field(..., ge=0.0, le=1.0)
    is_morning_rush:       int   = Field(..., ge=0, le=1)
    is_evening_rush:       int   = Field(..., ge=0, le=1)
    is_weekend:            int   = Field(..., ge=0, le=1)
    is_peak_season:        int   = Field(..., ge=0, le=1)
    temp_hum_interaction:  float
    temp_wind_interaction: float


class PredictionOutput(BaseModel):
    predicted_cnt: int
    model_name:    str
    model_version: str