# Forest Fires Prediction - MLOps Project

## Project Overview

This project demonstrates an end-to-end MLOps pipeline for wildfire prediction, with a primary focus on implementing and showcasing MLOps best practices and tools. While it includes a wildfire prediction model, the main objective is to illustrate a robust MLOps workflow rather than to create a production-grade prediction system.

The implementation covers the full machine learning lifecycle, including data processing, model training, experiment tracking, model deployment, monitoring, and serving predictions through a web API. This serves as a comprehensive example of how to operationalize machine learning models in a production environment.

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

### Generate Additional Data for Monitoring

To create sufficient data for meaningful monitoring, repeat the model training and batch processing steps:

1. Run the orchestration flow again:

   ```bash
   docker compose exec -it orchestration /bin/bash
   python flow.py
   ```

2. Run the batch service again:
   ```bash
   docker compose --profile batch-service up --build batch-service
   ```

This will generate additional data points that will be used for model monitoring and drift detection.

### API Testing

Test the prediction API with sample requests. The test results will be displayed directly in the terminal:

```bash
docker compose up -d
docker compose --profile test up test-api
```

The test script will output the results of:

- API health check
- High-risk fire scenario prediction
- Low-risk fire scenario prediction

### Model Monitoring

1. Start the monitoring service:
   ```bash
   docker compose --profile monitoring up monitoring -d
   ```
2. View monitoring reports in the `monitoring/reports` directory
3. Reports include:

   - CSV files with detailed metrics
   - PNG visualizations of model metrics

   **Note about the visualization**: The PNG graph might show what appears to be a single line because the batch processing was run back-to-back with minimal time difference between runs. This causes the data points to overlap in the visualization. In a production environment with more time between runs, you would see distinct data points showing the model's performance over time.

## Monitoring

Model performance and data drift are monitored using Evidently. Reports are generated after each batch prediction run and can be found in the `monitoring/reports` directory.
