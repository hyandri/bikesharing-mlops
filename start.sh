#!/bin/bash
docker compose up -d
echo "Waiting for MariaDB to start..."
sleep 30

echo "Starting ColumnStore services..."
docker exec mcs1 curl -s -X PUT https://127.0.0.1:8640/cmapi/0.4.0/node/start \
  --header 'Content-Type:application/json' \
  --header 'x-api-key:somekey123' \
  --data '{"timeout":60}' \
  -k

echo "Waiting for ColumnStore to initialize..."
sleep 20

echo "Creating database and schema..."
docker exec -i mcs1 mariadb -u root -pmlops@123. < dags/sql/create_star_schema.sql

echo "Creating database user..."
docker exec mcs1 mariadb -u root -pmlops@123. -e "
DROP USER IF EXISTS 'mlops_user'@'%';
CREATE USER 'mlops_user'@'%' IDENTIFIED BY 'mlops@123.';
GRANT ALL PRIVILEGES ON bike_sharing.* TO 'mlops_user'@'%';
FLUSH PRIVILEGES;
"

echo "Done. Testing connection..."
docker exec mcs1 mariadb -u root -pmlops@123. -e "SHOW ENGINES;" | grep -i columnstore
echo "All done. Ready to trigger DAG."
