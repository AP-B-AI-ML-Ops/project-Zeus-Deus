import os
import sys
import json
from datetime import datetime
from prefect import flow, task
import mlflow
import logging

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our modules
from src.data_ingestion.nasa_firms_ingest import fetch_nasa_firms_data
from src.data_ingestion.weather_ingest import main as fetch_weather_data
from src.data_ingestion.vegetation_ingest import process_vegetation_data, save_data
from src.preprocessing.merge_all_data import merge_all_data
from src.models.train import preprocess_data, train_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@task(name="Fetch NASA FIRMS Data", retries=3, retry_delay_seconds=60)
def task_fetch_firms_data(start_date, end_date, region=None):
    """Task to fetch NASA FIRMS wildfire data."""
    logger.info(f"Fetching NASA FIRMS data from {start_date} to {end_date}")
    fetch_nasa_firms_data()  # This function doesn't take parameters in the actual implementation
    
    # The function saves to a fixed path, so we'll return that path
    firms_data_path = os.path.join(os.path.dirname(__file__), '../data/nasa_firms_data.json')
    
    logger.info(f"NASA FIRMS data saved to {firms_data_path}")
    return firms_data_path

@task(name="Fetch Weather Data", retries=3, retry_delay_seconds=60)
def task_fetch_weather_data(firms_data_path):
    """Task to fetch weather data for wildfire locations."""
    logger.info(f"Fetching weather data for locations in {firms_data_path}")
    
    # The weather_ingest.main() function reads from a fixed path and saves to a fixed path
    fetch_weather_data()
    
    # Return the path where the data is saved
    weather_data_path = os.path.join(os.path.dirname(__file__), '../data/weather_data.json')
    
    logger.info(f"Saved weather data to {weather_data_path}")
    return weather_data_path

@task(name="Fetch Vegetation Data", retries=3, retry_delay_seconds=60)
def task_fetch_vegetation_data(firms_data_path):
    """Task to fetch vegetation data for wildfire locations."""
    logger.info(f"Fetching vegetation data for locations in {firms_data_path}")
    
    # Process vegetation data
    vegetation_data = process_vegetation_data()
    
    # Save vegetation data
    output_dir = "data/raw/vegetation"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{output_dir}/vegetation_data_{timestamp}.json"
    
    save_data(vegetation_data, output_path)
    
    logger.info(f"Saved vegetation data to {output_path}")
    return output_path

@task(name="Merge Data", retries=2)
def task_merge_data(firms_data_path, weather_data_path, vegetation_data_path):
    """Task to merge all data sources."""
    logger.info(f"Merging data from multiple sources")
    
    # Load data
    with open(firms_data_path, 'r') as f:
        firms_data = json.load(f)
    
    with open(weather_data_path, 'r') as f:
        weather_data = json.load(f)
    
    with open(vegetation_data_path, 'r') as f:
        vegetation_data = json.load(f)
    
    # Merge data
    merged_data = merge_all_data(firms_data, weather_data, vegetation_data)
    
    # Save merged data
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{output_dir}/merged_data_{timestamp}.json"
    
    with open(output_path, 'w') as f:
        json.dump(merged_data, f)
    
    logger.info(f"Saved merged data to {output_path}")
    return output_path

@task(name="Train Model", retries=1)
def task_train_model(merged_data_path):
    """Task to train the wildfire prediction model."""
    logger.info(f"Training model with data from {merged_data_path}")
    
    # Set up MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")  # Updated to use the MLflow service
    mlflow.set_experiment("wildfire-prediction")
    
    # Load merged data
    with open(merged_data_path, 'r') as f:
        merged_data = json.load(f)
    
    # Preprocess data
    df = preprocess_data(merged_data)
    
    # Train model
    model, accuracy, X_test, y_test = train_model(df)
    
    logger.info(f"Model training completed with accuracy: {accuracy:.4f}")
    
    # Return the run_id from the last MLflow run
    run_id = mlflow.search_runs(filter_string="tags.mlflow.runName='random-forest-wildfire'").iloc[0]["run_id"]
    
    return {
        "run_id": run_id,
        "accuracy": accuracy,
        "model_path": "models/wildfire_model_latest.pkl"
    }

@flow(name="Wildfire Prediction Pipeline")  # Removed task_runner parameter
def wildfire_prediction_pipeline(start_date, end_date, region=None):
    """Main workflow for the wildfire prediction pipeline."""
    # Fetch data
    firms_data_path = task_fetch_firms_data(start_date, end_date, region)
    weather_data_path = task_fetch_weather_data(firms_data_path)
    vegetation_data_path = task_fetch_vegetation_data(firms_data_path)
    
    # Merge data
    merged_data_path = task_merge_data(firms_data_path, weather_data_path, vegetation_data_path)
    
    # Train model
    model_info = task_train_model(merged_data_path)
    
    return {
        "firms_data_path": firms_data_path,
        "weather_data_path": weather_data_path,
        "vegetation_data_path": vegetation_data_path,
        "merged_data_path": merged_data_path,
        "model_info": model_info
    }

if __name__ == "__main__":
    # Example usage
    from datetime import date, timedelta
    
    # Get data for the last 30 days
    end_date = date.today()
    start_date = end_date - timedelta(days=30)
    
    # Run the pipeline
    result = wildfire_prediction_pipeline(
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        region="US"  # Example: United States
    )
    
    print(f"Pipeline completed successfully!")
    print(f"Model accuracy: {result['model_info']['accuracy']:.4f}")
    print(f"MLflow run ID: {result['model_info']['run_id']}")