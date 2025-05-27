# Forest Fires Prediction - MLOps Project

## Project Overview

This project focuses on building an end-to-end MLOps pipeline for wildfire prediction. It leverages machine learning to predict forest fire risks based on various environmental factors.

## Dataset

- **Source**: UCI Machine Learning Repository
- **Name**: Algerian Forest Fires Dataset
- **Link**: [Dataset Information](https://archive.ics.uci.edu/dataset/547/algerian+forest+fires+dataset)

## Tech Stack

- **MLflow**: Experiment tracking and model registry
- **Optuna**: Hyperparameter optimization
- **Prefect**: Workflow orchestration
- **Evidently**: Model monitoring
- **Docker**: Containerization
- **PostgreSQL**: Database management
- **FastAPI**: Web API development

## Project Structure

```
.
├── database/           # Database configuration
├── deploy-batch/       # Batch deployment
├── experiment-tracking/ # MLflow experiment tracking
├── monitoring/         # Model monitoring
├── orchestration/      # Workflow orchestration
├── train-deploy/       # Model training and deployment
├── tests/              # Test files
└── web-api/           # Web API service
```

## Getting Started

### Prerequisites

- Docker
- Docker Compose

### Installation

1. Clone the repository
2. Build and start the services:
   ```bash
   docker compose up -d --build
   ```

## Usage

### Model Training

1. Access the orchestration container:
   ```bash
   docker compose exec -it orchestration /bin/bash
   python flow.py
   ```
2. Monitor training progress at `http://localhost:5000`
3. View experiment runs in MLflow UI at `http://localhost:4200`

### Batch Processing

1. Run the batch service:
   ```bash
   docker compose --profile batch-service up --build batch-service
   ```
2. Monitor batch processing in the Prefect UI at `http://localhost:4200`
3. Predictions are saved in the `batch-data` volume at `/batch-data/output`

### API Testing

Test the prediction API with sample requests:

```bash
docker compose up -d
docker compose --profile test up test-api
```

The test script verifies:

- API health check
- High-risk fire scenario prediction
- Low-risk fire scenario prediction

### Model Monitoring

1. Start the monitoring service:
   ```bash
   docker compose --profile monitoring up monitoring -d
   ```
2. View monitoring reports in the `monitoring/reports` directory
3. Reports include CSV data and visualizations of model metrics

## Monitoring

Model performance and data drift are monitored using Evidently. Reports are generated after each batch prediction run and can be found in the `monitoring/reports` directory.
