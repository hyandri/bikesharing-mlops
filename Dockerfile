FROM apache/airflow:2.11.0

USER airflow

RUN pip install --no-cache-dir \
    pyarrow \
    redis \
    pandas \
    clickhouse-connect \
    great-expectations==0.18.14
