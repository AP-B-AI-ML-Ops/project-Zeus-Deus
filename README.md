# Wildfire Prediction MLOps Project

## Problem Statement

Wildfires are becoming increasingly frequent and severe due to climate change. Early prediction of high-severity wildfires can help authorities allocate resources more effectively and potentially save lives and property. This project aims to predict the severity of wildfires based on satellite data, weather conditions, and vegetation indices.

## Dataset Description

This project uses three main data sources:

1. **NASA FIRMS (Fire Information for Resource Management System)**: Provides near real-time active fire data from MODIS and VIIRS satellites. The data includes fire locations, brightness, confidence level, and other attributes.

2. **Weather Data**: Historical and current weather conditions including temperature, precipitation, wind speed, and atmospheric pressure at fire locations.

3. **Vegetation Indices**: NDVI (Normalized Difference Vegetation Index) and EVI (Enhanced Vegetation Index) data that indicate vegetation health and density, which are critical factors in fire spread and intensity.

## Architecture

The project follows MLOps best practices with the following components:

### Data Pipeline

- Data ingestion from NASA FIRMS API, weather APIs, and vegetation data sources
- Data preprocessing and feature engineering
- Data versioning and storage

### Model Training

- Experiment tracking with MLflow
- Hyperparameter tuning
- Model versioning and registry

### Model Deployment

- FastAPI service for model serving
- Docker containers for all components
- RESTful API for predictions

### Orchestration

- Prefect for workflow orchestration
- Automated pipeline execution

### Monitoring

- Data drift detection with Evidently
- Model performance monitoring
- Dashboard for visualization

## Project Structure

```
├── data/                  # Data storage
│   ├── raw/               # Raw data from sources
│   │   ├── firms/         # NASA FIRMS data
│   │   ├── weather/       # Weather data
│   │   └── vegetation/    # Vegetation data
│   ├── processed/         # Processed data ready for training
│   └── predictions/       # Model predictions
├── src/                   # Source code
│   ├── data_ingestion/    # Data collection scripts
│   │   └── generate_data.py  # Script to generate synthetic data
│   ├── preprocessing/     # Data preprocessing scripts
│   ├── models/            # Model training and prediction code
│   │   ├── train.py       # Main model training script
│   │   ├── predict.py     # Model prediction script
│   │   └── train_with_real_data.py  # Script to train with generated data
│   ├── api/               # FastAPI service
│   └── utils/             # Utility scripts
│       └── test_prediction.py  # Script to test the prediction API
├── orchestration/         # Prefect workflows
├── monitoring/            # Evidently monitoring dashboards
├── experiment_tracking/   # MLflow configuration
├── models/                # Saved model artifacts
├── docker-compose.yml     # Docker Compose configuration
└── requirements.txt       # Python dependencies
```

## How to Run

### Prerequisites

- Docker and Docker Compose
- API keys for data sources (if using real data)

### Setup

1. Clone the repository
2. Create a `.env` file with `cp .env.example .env`
3. Build and start the services:
   ```bash
   docker compose up -d
   ```

### Accessing Services

- **API**: http://localhost:8000
  - Health check: http://localhost:8000/health
  - Swagger UI: http://localhost:8000/docs
- **MLflow**: http://localhost:5000
- **Monitoring Dashboard**: http://localhost:8050
- **Prefect**: http://localhost:4200

### Utility Scripts

The project includes several utility scripts to help with testing and data generation:

1. **Generate Data** (`src/data_ingestion/generate_data.py`):

   ```bash
   docker compose exec api python -m src.data_ingestion.generate_data
   ```

   Generates synthetic data for training, validation, testing, and production.

2. **Train Model** (`src/models/train_with_real_data.py`):

   ```bash
   docker compose exec api python -m src.models.train_with_real_data
   ```

   Trains a model using the generated data and logs to MLflow.

3. **Test API** (`src/utils/test_prediction.py`):
   ```bash
   python src/utils/test_prediction.py
   ```
   Tests the prediction API with sample data.

### Making Predictions

Send a POST request to the API with fire event data:

```bash
curl -X POST -H "Content-Type: application/json" -d '[{
  "event": {
    "lat": 37.7749,
    "lon": -122.4194,
    "brightness": 320.5,
    "scan": 1.0,
    "track": 1.0,
    "confidence": 95,
    "bright_t31": 290.5,
    "frp": 25.5,
    "daynight": "D"
  },
  "weather": {
    "tavg": 25.5,
    "tmin": 20.0,
    "tmax": 30.0,
    "prcp": 0.0,
    "wspd": 10.2,
    "pres": 1013.2
  },
  "vegetation": {
    "ndvi": 0.65,
    "evi": 0.45
  }
}]' http://localhost:8000/predict
```
