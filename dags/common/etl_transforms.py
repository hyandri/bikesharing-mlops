# dags/common/etl_transforms.py
import pandas as pd
from sqlalchemy import text


def transform_to_star_schema(df):
    """
    Transforms raw bike sharing DataFrame into star schema dictionary.
    Weather measures (temp, atemp, hum, windspeed) go in fact table.
    dim_weather holds descriptive attributes only.
    workingday included in dim_datetime.
    """

    # 1. dim_datetime
    dim_datetime = df[[
        'instant', 'dteday', 'yr', 'mnth',
        'hr', 'weekday', 'holiday', 'workingday'
    ]].copy()
    dim_datetime['dteday'] = pd.to_datetime(dim_datetime['dteday'])
    dim_datetime['datetime_id'] = range(1, len(dim_datetime) + 1)

    # 2. dim_weather (descriptive only)
    weather_label_map = {
        1: 'Clear/Few clouds',
        2: 'Mist/Cloudy',
        3: 'Light Snow/Rain',
        4: 'Heavy Rain/Snow'
    }
    dim_weather = df[['weathersit']].drop_duplicates().copy()
    dim_weather['weathersit_id'] = range(1, len(dim_weather) + 1)
    dim_weather['weather_desc'] = dim_weather['weathersit'].map(weather_label_map)
    dim_weather.rename(columns={'weathersit': 'weathersit_original'}, inplace=True)

    # 3. dim_season
    season_label_map = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
    dim_season = df[['season']].drop_duplicates().copy()
    dim_season['season_id'] = range(1, len(dim_season) + 1)
    dim_season['season_name'] = dim_season['season'].map(season_label_map)
    dim_season.rename(columns={'season': 'season_original'}, inplace=True)

    # 4. fact_hourly_rentals
    weather_id_map  = dict(zip(dim_weather['weathersit_original'], dim_weather['weathersit_id']))
    season_id_map   = dict(zip(dim_season['season_original'], dim_season['season_id']))
    datetime_id_map = dict(zip(dim_datetime['instant'], dim_datetime['datetime_id']))

    fact = df[[
        'instant', 'weathersit', 'season',
        'temp', 'atemp', 'hum', 'windspeed',
        'cnt', 'casual', 'registered'
    ]].copy()

    fact['fact_id']       = range(1, len(fact) + 1)
    fact['datetime_id']   = fact['instant'].map(datetime_id_map)
    fact['weathersit_id'] = fact['weathersit'].map(weather_id_map)
    fact['season_id']     = fact['season'].map(season_id_map)

    fact_final = fact[[
        'fact_id', 'datetime_id', 'weathersit_id', 'season_id',
        'temp', 'atemp', 'hum', 'windspeed',
        'cnt', 'casual', 'registered'
    ]]

    return {
        'dim_datetime': dim_datetime[[
            'datetime_id', 'instant', 'dteday', 'yr', 'mnth',
            'hr', 'weekday', 'holiday', 'workingday'
        ]],
        'dim_weather': dim_weather[[
            'weathersit_id', 'weathersit_original', 'weather_desc'
        ]],
        'dim_season': dim_season[[
            'season_id', 'season_original', 'season_name'
        ]],
        'fact_hourly_rentals': fact_final
    }


def load_star_schema_to_db(engine, star_schema_dict):
    """
    Loads star schema into MariaDB ColumnStore.
    Truncates tables first to prevent duplicates on re-run.
    Inserts dimensions first, then fact table.
    """
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE fact_hourly_rentals"))
        conn.execute(text("TRUNCATE TABLE dim_season"))
        conn.execute(text("TRUNCATE TABLE dim_weather"))
        conn.execute(text("TRUNCATE TABLE dim_datetime"))

    star_schema_dict['dim_datetime'].to_sql(
        'dim_datetime', engine, if_exists='append', index=False)
    star_schema_dict['dim_weather'].to_sql(
        'dim_weather', engine, if_exists='append', index=False)
    star_schema_dict['dim_season'].to_sql(
        'dim_season', engine, if_exists='append', index=False)
    star_schema_dict['fact_hourly_rentals'].to_sql(
        'fact_hourly_rentals', engine, if_exists='append', index=False)

    return {
        'dim_datetime_rows': len(star_schema_dict['dim_datetime']),
        'dim_weather_rows':  len(star_schema_dict['dim_weather']),
        'dim_season_rows':   len(star_schema_dict['dim_season']),
        'fact_rows':         len(star_schema_dict['fact_hourly_rentals'])
    }