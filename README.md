# Airflow Docker - MLOps Data Pipelines

A Docker-based Apache Airflow project for orchestrating ETL pipelines with machine learning workflows. Includes multiple data ingestion and processing DAGs using MariaDB ColumnStore, Redis, and data validation with Great Expectations.

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose installed
- Linux/Mac (or WSL on Windows)
- At least 4GB available memory

### Setup & Run

1. **Clone and navigate to the project:**
   ```bash
   git clone <your-repo-url>
   cd airflow-docker
   ```

2. **Start the Airflow services:**
   ```bash
   bash start.sh
   ```
   This will:
   - Start Docker containers for Airflow, MariaDB, Redis, and ColumnStore
   - Initialize the database schema
   - Create necessary database users

3. **Access Airflow UI:**
   - URL: `http://localhost:8080`
   - Default credentials: `airflow` / `airflow`

## 📁 Project Structure

```
airflow-docker/
├── dags/                          # Airflow DAG definitions
│   ├── bike_sharing_pipeline.py   # Bike sharing data ETL
│   ├── common/                    # Shared modules
│   │   ├── db_connectors.py       # Database connection utilities
│   │   ├── etl_transforms.py      # ETL transformation functions
│   │   ├── gx_validator.py        # Great Expectations validators
│   │   ├── preprocessing.py       # Data preprocessing functions
│   ├── sql/                       # SQL scripts
│   │   └── create_star_schema.sql # Database schema definitions
├── config/
│   └── airflow.cfg                # Airflow configuration
├── data/
│   └── hour.csv                   # Sample bike sharing dataset
├── docker-compose.yaml            # Docker Compose configuration
├── Dockerfile                     # Custom Airflow image with dependencies
├── start.sh                        # Startup script
└── logs/                          # Airflow execution logs 
```

## 🛠️ Tech Stack

- **Orchestration:** Apache Airflow 2.11.0
- **Database:** MariaDB with ColumnStore for analytics
- **Cache:** Redis
- **Data Validation:** Great Expectations 0.18.14
- **Data Processing:** Pandas, PyArrow
- **Analytics:** ClickHouse (via clickhouse-connect)
- **Notebooks:** Jupyter for exploration

## 📊 Main DAGs

### 1. **Bike Sharing Pipeline** (`bike_sharing_pipeline.py`)
- Ingests bike sharing hourly data
- Transforms to star schema
- Data validation with Great Expectations
- Loads to MariaDB

## 🔧 Configuration

### Environment Variables (`.env`)
The `.env` file contains:
- `AIRFLOW_UID`: User ID for Airflow containers (default: 1000)

### Airflow Configuration (`config/airflow.cfg`)
Customize Airflow settings like:
- Executor type
- Database connections
- DAG defaults
- Email notifications

### Database Schema
Run SQL scripts in `dags/sql/` to initialize database:
```bash
docker exec -i <container_name> mariadb < dags/sql/create_star_schema.sql
```

## 📝 Common Commands

### Airflow CLI
```bash
# Trigger a DAG
docker exec <container_name> airflow dags trigger <dag_id>

# List all DAGs
docker exec <container_name> airflow dags list

# View task logs
docker exec <container_name> airflow tasks logs <dag_id> <task_id> <execution_date>
```

### Docker
```bash
# Stop all services
docker compose down

# View logs
docker compose logs -f <service_name>

# Rebuild images
docker compose build --no-cache
```

## 🧪 Development

### Adding New DAGs
1. Create a Python file in `dags/`
2. Define your DAG using Airflow operators
3. The DAG will auto-discover within 5 minutes (configurable)

### Using Shared Modules
```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'common'))

from db_connectors import get_engine
from etl_transforms import transform_to_star_schema
```

### Running Locally with Jupyter
```bash
jupyter notebook dags/Untitled.ipynb
```

## 📊 Monitoring & Logs

- **Airflow UI:** http://localhost:8080 (DAG status, execution history, logs)
- **Logs Directory:** `logs/` contains detailed task execution logs
- **Docker Logs:** Use `docker compose logs <service>`

## 🔒 Security Notes


Before deploying to production:
- Change default passwords in `.env`
- Use secrets management (Vault, AWS Secrets Manager)
- Enable SSL/TLS
- Configure proper authentication
- Use production-grade database
- Set up monitoring and alerting

## 🐛 Troubleshooting

### MariaDB ColumnStore not starting
```bash
docker exec <container> curl -s -X PUT https://127.0.0.1:8640/cmapi/0.4.0/node/start \
  --header 'Content-Type:application/json' \
  --header 'x-api-key:somekey123' \
  --data '{"timeout":60}' -k
```

### Airflow can't connect to database
- Check `.env` file for `AIRFLOW_UID`
- Verify Docker containers are running: `docker compose ps`
- Check logs: `docker compose logs airflow-webserver`

### Permissions issues
```bash
# Fix volume permissions
chmod -R 777 logs/
```

## 📚 Resources

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [Great Expectations Documentation](https://docs.greatexpectations.io/)
- [MariaDB ColumnStore](https://mariadb.com/docs/analytics/columnstore/)

## 📄 License

This project uses Apache Airflow, which is licensed under the Apache License 2.0. See LICENSE file for details.

## 👤 Author

[Your Name/Organization]

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

**Last Updated:** May 2026
