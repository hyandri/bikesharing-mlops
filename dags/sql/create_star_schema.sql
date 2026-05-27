-- ============================================
-- OLAP Star Schema (ColumnStore)
-- Bike Sharing Dataset
-- ============================================
CREATE DATABASE IF NOT EXISTS bike_sharing;
USE bike_sharing;

DROP TABLE IF EXISTS fact_hourly_rentals;
DROP TABLE IF EXISTS dim_datetime;
DROP TABLE IF EXISTS dim_weather;
DROP TABLE IF EXISTS dim_season;

CREATE TABLE dim_datetime (
    datetime_id INT      NOT NULL,
    instant     INT      NOT NULL,
    dteday      DATE     NOT NULL,
    yr          INT      NOT NULL,
    mnth        INT      NOT NULL,
    hr          INT      NOT NULL,
    weekday     INT      NOT NULL,
    holiday     INT      NOT NULL,
    workingday  INT      NOT NULL
) ENGINE=ColumnStore;

CREATE TABLE dim_weather (
    weathersit_id       INT         NOT NULL,
    weathersit_original INT         NOT NULL,
    weather_desc        VARCHAR(50) NOT NULL
) ENGINE=ColumnStore;

CREATE TABLE dim_season (
    season_id       INT         NOT NULL,
    season_original INT         NOT NULL,
    season_name     VARCHAR(20) NOT NULL
) ENGINE=ColumnStore;

CREATE TABLE fact_hourly_rentals (
    fact_id       INT           NOT NULL,
    datetime_id   INT           NOT NULL,
    weathersit_id INT           NOT NULL,
    season_id     INT           NOT NULL,
    temp          FLOAT         NOT NULL,
    atemp         FLOAT         NOT NULL,
    hum           FLOAT         NOT NULL,
    windspeed     FLOAT         NOT NULL,
    cnt           INT           NOT NULL,
    casual        INT           NOT NULL,
    registered    INT           NOT NULL
) ENGINE=ColumnStore;