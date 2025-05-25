FROM python:3.11-slim

RUN pip install -U pip setuptools wheel

COPY train-deploy/requirements.txt .
RUN pip install -r requirements.txt

# Prefect
RUN pip install pendulum prefect --upgrade

WORKDIR /app

RUN prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"

CMD ["prefect", "server", "start", "--host", "0.0.0.0", "--port", "4200"]