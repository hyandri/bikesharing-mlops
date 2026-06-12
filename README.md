# Bike Sharing MLOps Pipeline

A complete end-to-end MLOps pipeline for bike sharing demand prediction, 
built with Apache Airflow, MariaDB ColumnStore, MLflow, Redis, and FastAPI.

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose installed
- Linux/Ubuntu (or WSL on Windows)
- At least 4GB available memory

### Setup and Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/hyandri/bikesharing-mlops.git
   cd airflow-docker
   ```

2. **Set Airflow UID:**
   ```bash
   echo "AIRFLOW_UID=$(id -u)" > .env
   ```

3. **Start all services:**
   ```bash
   bash start.sh
   ```
   This will:
   - Start all Docker containers
   - Trigger the ColumnStore CMAPI to start internal services
   - Create the star schema database and tables
   - Create the database user with correct permissions

4. **Trigger the pipeline:**
   - Access Airflow UI at `http://localhost:8080`
   - Credentials: `airflow` / `airflow`
   - Trigger `bike_sharing_full_pipeline` DAG manually

5. **Restart FastAPI after first pipeline run:**
   ```bash
   docker compose restart fastapi
   ```
   FastAPI loads the model from MLflow on startup. 
   It must be restarted after the first pipeline run 
   registers the model. On subsequent starts this is 
   not needed.

## 📁 Project Structure

```
airflow-docker/
├── dags/
│   ├── bike_sharing_pipeline.py   # Main pipeline DAG
│   └── common/
│       ├── db_connectors.py       # SQLAlchemy engine and Redis connection
│       ├── etl_transforms.py      # Star schema transformation and loading
│       ├── gx_validator.py        # Great Expectations validation
│       ├── preprocessing.py       # Feature engineering, Redis serialisation
│       ├── training.py            # XGBoost training and MLflow logging
│       └── monitoring.py          # Drift detection helpers
├── app/
│   ├── main.py                    # FastAPI application and endpoints
│   ├── model.py                   # MLflow model loading
│   └── schemas.py                 # Pydantic input/output schemas
├── monitoring/
│   ├── monitor.py                 # Standalone drift detection script
│   └── reports/                   # Generated Evidently HTML reports
├── tests/
│   └── test_pipeline.py           # pytest test suite (21 tests)
├── dags/sql/
│   └── create_star_schema.sql     # ColumnStore star schema
├── data/
│   └── hour.csv                   # UCI Bike Sharing dataset
├── logs/
│   └── predictions/
│       └── predictions.jsonl      # FastAPI prediction logs
├── Dockerfile                     # Airflow custom image
├── Dockerfile.fastapi             # FastAPI image
├── docker-compose.yml             # All services definition
└── start.sh                       # Full startup script
```

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Orchestration | Apache Airflow 2.11.0 (CeleryExecutor) |
| OLAP Database | MariaDB ColumnStore |
| Experiment Tracking | MLflow |
| Inter-task Storage | Redis + PyArrow |
| Data Validation | Great Expectations 0.18.14 |
| Model | XGBoost |
| Model Serving | FastAPI + Uvicorn |
| Monitoring | Evidently 0.4.16 |
| Testing | pytest |

## 📊 Pipeline DAG

`bike_sharing_full_pipeline` consists of 8 sequential tasks:

```
validate_source
    >> extract_and_transform
    >> load_to_olap
    >> extract
    >> engineer_and_split
    >> verify_redis_splits
    >> train_xgboost
    >> register_model
```

| Task | Description |
|---|---|
| `validate_source` | Great Expectations validation on source CSV |
| `extract_and_transform` | Transform CSV to star schema in memory |
| `load_to_olap` | Load star schema into ColumnStore |
| `extract` | Extract from ColumnStore, store in Redis |
| `engineer_and_split` | Feature engineering + 80/20 temporal split |
| `verify_redis_splits` | Confirm all splits stored correctly in Redis |
| `train_xgboost` | Train XGBoost, log metrics to MLflow |
| `register_model` | Register best model in MLflow model registry |

## 🌐 Service Ports

| Service | Port | URL |
|---|---|---|
| Airflow | 8080 | http://localhost:8080 |
| MLflow | 5000 | http://localhost:5000 |
| FastAPI | 8000 | http://localhost:8000 |
| MariaDB ColumnStore | 3306 | localhost:3306 |
| Redis | 6379 | Internal only |

## 🤖 Model Performance

XGBoost model evaluated on temporally held-out test set 
(August–December 2012):

| Metric | Value |
|---|---|
| R² | 0.9139 |
| RMSE | 64.64 |
| MAE | 41.49 |

## 🌐 API Usage

**Health check:**
```bash
curl http://localhost:8000/health
```

**Predict:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "yr": 1, "mnth": 6, "hr": 8,
    "weekday": 2, "holiday": 0, "workingday": 1,
    "weathersit_original": 1, "season_original": 2,
    "temp": 0.6, "atemp": 0.58, "hum": 0.4, "windspeed": 0.1,
    "is_morning_rush": 1, "is_evening_rush": 0,
    "is_weekend": 0, "is_peak_season": 1,
    "temp_hum_interaction": 0.36, "temp_wind_interaction": 0.54
  }'
```

Swagger UI available at `http://localhost:8000/docs`

## 📈 Model Monitoring

Monitoring runs as a standalone script — not a DAG component:

```bash
cd monitoring
python monitor.py
```

Generates 6 Evidently HTML reports in `monitoring/reports/`:
- Data drift reports for control, temp_drift, hr_drift batches
- Regression performance reports for each batch

## 🧪 Testing

21 pytest tests covering data ingestion, API prediction, 
and model performance:

```bash
python -m pytest tests/test_pipeline.py -v -s
```

## 🔧 Common Commands

**Restart ColumnStore after system reboot:**
```bash
bash start.sh
```

**Manually trigger CMAPI if ColumnStore is read-only:**
```bash
docker exec mcs1 curl -s -X PUT https://127.0.0.1:8640/cmapi/0.4.0/node/start \
  --header 'Content-Type:application/json' \
  --header 'x-api-key:somekey123' \
  --data '{"timeout":60}' -k
```

**Check all containers:**
```bash
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

**Stop everything:**
```bash
docker compose down
```

**Stop and wipe all data:**
```bash
docker compose down -v
```

## 🐛 Troubleshooting

**DBRM read-only error when creating tables:**
ColumnStore processes did not start. Run the CMAPI trigger above 
and wait 20 seconds before retrying.

**FastAPI returns model not found:**
The pipeline DAG has not been run yet or MLflow was restarted. 
Trigger the full pipeline DAG and then restart FastAPI:
```bash
docker compose restart fastapi
```

**Redis keys expired (preprocessing tests skipped):**
Redis keys have a 24-hour TTL. Re-trigger the pipeline DAG 
to restore them.

**Airflow cannot find module:**
Package missing from Dockerfile. Add it and rebuild:
```bash
docker compose build
docker compose up -d
```

## 📄 Dataset

UCI Bike Sharing Dataset — hourly rental records from 2011-2012.
Place `hour.csv` in the `data/` folder before running the pipeline.

## 👤 Author

hyandri

## 📄 License

Apache License 2.0
