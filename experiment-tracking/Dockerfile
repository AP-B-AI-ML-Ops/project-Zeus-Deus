FROM python:3.13-slim

RUN pip install -U pip

RUN pip install psycopg2-binary mlflow

CMD ["mlflow", "server", "--backend-store-uri", "postgresql+psycopg2://postgres:postgres@database/mlflow_db", "--default-artifact-root", "/mlflow/mlruns", "--host", "0.0.0.0", "--port", "5000"]