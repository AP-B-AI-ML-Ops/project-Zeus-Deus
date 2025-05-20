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

## Orchestration with Prefect

This project uses Prefect to orchestrate the entire MLOps workflow. Prefect manages data generation, model training, evaluation, and prediction in an automated fashion.

### First-Time Setup

1. Start all services:

   ```bash
   docker compose up -d
   ```

2. Wait for services to initialize (10-15 seconds)

3. **Important:** For first-time setup, create required directories:

   ```bash
   docker exec -it project-api-1 bash -c "mkdir -p data/raw data/processed data/predictions models"
   ```

4. Run the complete MLOps pipeline first to generate data and train the model:

   ```bash
   docker exec -it project-prefect-1 bash -c "prefect deployment run 'Complete MLOps Pipeline/wildfire-prediction-deployment'"
   ```

   Wait for this to complete (check status in Prefect UI). This step is required before making predictions.

   **Note:** The Complete MLOps Pipeline can take several minutes to run completely, especially for the first time. The Data Generation and Model Training steps will finish first, followed by Model Evaluation and Prediction steps. You can monitor the progress in the Prefect UI.

5. After the model is trained, you can run the daily prediction flow:

   ```bash
   docker exec -it project-prefect-1 bash -c "prefect deployment run 'Daily Prediction Flow/daily-prediction-deployment'"
   ```

6. To verify everything is working:

   - Check Prefect UI: http://localhost:4200 for successful flow runs
   - Check MLflow UI: http://localhost:5000 for logged model experiments
   - Examine the generated files:
     ```bash
     docker exec -it project-api-1 bash -c "ls -la data/predictions/"
     ```

7. Access the Prefect UI:
   - Open your browser and navigate to: http://localhost:4200
   - Here you can monitor flow runs, check logs, and manage your workflows

### Scheduled Flows

The following flows are automatically scheduled:

- Daily Prediction Flow: Runs every day at 8:00 AM UTC
- Complete MLOps Pipeline: Runs on the 1st of every month at midnight UTC

### Flow Structure

- **Complete MLOps Pipeline**: End-to-end pipeline for data generation, model training, evaluation, and prediction
- **Daily Prediction Flow**: Generates new data and makes predictions using the latest model

### Model Limitations

This project is designed as an MLOps demonstration and educational tool. The wildfire prediction model:

- Is trained on synthetic data, not real-world historical data
- Uses simplified feature representations
- Is not validated against actual wildfire occurrences

For actual wildfire prediction, this architecture would need to be adapted to use comprehensive real-world data sources, domain expertise, and extensive validation.
