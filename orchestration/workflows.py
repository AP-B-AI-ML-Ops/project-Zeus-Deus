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
from src.data_ingestion.generate_data import generate_event_data, save_data
from src.models.train import preprocess_data, train_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define file paths
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
MODELS_DIR = "models"

@task(name="Generate Training Data", retries=2)
def task_generate_training_data(num_records=100, start_date="2022-01-01", days=180):
    """Task to generate training data."""
    logger.info(f"Generating {num_records} training data records")
    
    # Generate training data
    training_data = generate_event_data(num_records=num_records, start_date=start_date, days=days)
    
    # Save training data
    output_path = f"{PROCESSED_DATA_DIR}/training_data.json"
    save_data(training_data, output_path)
    
    logger.info(f"Saved training data to {output_path}")
    return output_path

@task(name="Generate Validation Data", retries=2)
def task_generate_validation_data(num_records=30, start_date="2022-07-01", days=90):
    """Task to generate validation data."""
    logger.info(f"Generating {num_records} validation data records")
    
    # Generate validation data
    validation_data = generate_event_data(num_records=num_records, start_date=start_date, days=days)
    
    # Save validation data
    output_path = f"{PROCESSED_DATA_DIR}/validation_data.json"
    save_data(validation_data, output_path)
    
    logger.info(f"Saved validation data to {output_path}")
    return output_path

@task(name="Generate Test Data", retries=2)
def task_generate_test_data(num_records=20, start_date="2022-10-01", days=90):
    """Task to generate test data."""
    logger.info(f"Generating {num_records} test data records")
    
    # Generate test data
    test_data = generate_event_data(num_records=num_records, start_date=start_date, days=days)
    
    # Save test data
    output_path = f"{PROCESSED_DATA_DIR}/test_data.json"
    save_data(test_data, output_path)
    
    logger.info(f"Saved test data to {output_path}")
    return output_path

@task(name="Generate Production Data", retries=2)
def task_generate_production_data(num_records=10, start_date="2023-01-01", days=30):
    """Task to generate production data."""
    logger.info(f"Generating {num_records} production data records")
    
    # Generate production data
    production_data = generate_event_data(num_records=num_records, start_date=start_date, days=days)
    
    # Save production data
    output_path = f"{PROCESSED_DATA_DIR}/production_data.json"
    save_data(production_data, output_path)
    
    logger.info(f"Saved production data to {output_path}")
    return output_path

@task(name="Train Model", retries=2)
def task_train_model(training_data_path):
    """Task to train the wildfire prediction model."""
    logger.info(f"Training model with data from {training_data_path}")
    
    # Set up MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("wildfire-prediction")
    
    # Load training data
    with open(training_data_path, 'r') as f:
        training_data = json.load(f)
    
    # Preprocess data
    df = preprocess_data(training_data)
    
    # Train model
    model, accuracy, X_test, y_test = train_model(df)
    
    logger.info(f"Model training completed with accuracy: {accuracy:.4f}")
    
    # Return the run_id from the last MLflow run
    run_id = mlflow.search_runs(filter_string="tags.mlflow.runName='random-forest-wildfire'").iloc[0]["run_id"]
    
    return {
        "run_id": run_id,
        "accuracy": accuracy,
        "model_path": f"{MODELS_DIR}/wildfire_model_latest.pkl"
    }

@flow(name="Data Generation Pipeline")
def data_generation_pipeline(
    training_records=100, 
    validation_records=30, 
    test_records=20, 
    production_records=10
):
    """Pipeline for generating all necessary data."""
    # Create directories
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Generate data
    training_path = task_generate_training_data(num_records=training_records)
    validation_path = task_generate_validation_data(num_records=validation_records)
    test_path = task_generate_test_data(num_records=test_records)
    production_path = task_generate_production_data(num_records=production_records)
    
    return {
        "training_path": training_path,
        "validation_path": validation_path,
        "test_path": test_path,
        "production_path": production_path
    }

@flow(name="Model Training Pipeline")
def model_training_pipeline(training_data_path=f"{PROCESSED_DATA_DIR}/training_data.json"):
    """Pipeline for training the model."""
    # Train model
    model_info = task_train_model(training_data_path)
    
    return model_info

@flow(name="Complete MLOps Pipeline")
def complete_mlops_pipeline(
    training_records=100, 
    validation_records=30, 
    test_records=20, 
    production_records=10
):
    """Complete MLOps pipeline that generates data and trains the model."""
    # Generate data
    data_paths = data_generation_pipeline(
        training_records=training_records,
        validation_records=validation_records,
        test_records=test_records,
        production_records=production_records
    )
    
    # Train model
    model_info = model_training_pipeline(training_data_path=data_paths["training_path"])
    
    return {
        "data_paths": data_paths,
        "model_info": model_info
    }

if __name__ == "__main__":
    # Run the complete pipeline
    result = complete_mlops_pipeline()
    
    print(f"Pipeline completed successfully!")
    print(f"Model accuracy: {result['model_info']['accuracy']:.4f}")
    print(f"MLflow run ID: {result['model_info']['run_id']}")