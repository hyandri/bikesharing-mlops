# dags/common/db_connectors.py
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
import redis

DB_CONFIG = {
    'host': 'mcs1',
    'port': 3306,
    'user': 'mlops_user',
    'password': 'mlops@123.',
    'database': 'bike_sharing'
}

REDIS_CONFIG = {
    'host': 'redis',
    'port': 6379,
    'db': 0
}

def get_engine():
    url = URL.create(
        "mysql+pymysql",
        username=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        database=DB_CONFIG["database"],
    )
    return create_engine(url)

def get_redis_connection():
    return redis.Redis(
        host=REDIS_CONFIG['host'],
        port=REDIS_CONFIG['port'],
        db=REDIS_CONFIG['db'],
        decode_responses=False
    )