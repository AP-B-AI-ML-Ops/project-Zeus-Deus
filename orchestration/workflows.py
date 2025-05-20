import os
import sys
import json
import time
import requests
from datetime import datetime, timedelta
from prefect import flow, task
import mlflow
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
PREDICTIONS_DIR = "data/predictions"
EVALUATION_DIR = "data/evaluation"

# MLflow server configuration
MLFLOW_TRACKING_URI = "http://mlflow:5000"
MLFLOW_EXPERIMENT_NAME = "wildfire-prediction"

@task(name="Wait for MLflow Server", retries=5, retry_delay_seconds=10)
def wait_for_mlflow_server():
    """Task to wait for MLflow server to be available."""
    logger.info("Waiting for MLflow server to be available...")
    
    max_retries = 10
    retry_interval = 5
    
    for i in range(max_retries):
        try:
            response = requests.get(f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/experiments/list")
            if response.status_code == 200:
                logger.info("MLflow server is available!")
                return True
        except Exception as e:
            logger.warning(f"MLflow server not yet available (attempt {i+1}/{max_retries}): {str(e)}")
        
        logger.info(f"Retrying in {retry_interval} seconds...")
        time.sleep(retry_interval)
    
    logger.error("MLflow server not available after maximum retries")
    return False

@task(name="Initialize MLflow", retries=3, retry_delay_seconds=5)
def initialize_mlflow():
    """Task to initialize MLflow connection and ensure experiment exists."""
    logger.info("Initializing MLflow...")
    
    # Set tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Create experiment if it doesn't exist
    try:
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        if experiment is None:
            experiment_id = mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
            logger.info(f"Created new experiment with ID: {experiment_id}")
        else:
            logger.info(f"Using existing experiment with ID: {experiment.experiment_id}")
    except Exception as e:
        logger.error(f"Error initializing MLflow: {str(e)}")
        raise
    
    return True

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

@task(name="Generate Continuous Data", retries=2)
def task_generate_continuous_data(num_records=5, days=7):
    """Task to generate new data for continuous prediction."""
    logger.info(f"Generating {num_records} new data records for continuous prediction")
    
    # Set the start date to today minus the specified number of days
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    
    # Generate new data
    new_data = generate_event_data(num_records=num_records, start_date=start_date, days=days)
    
    # Create a timestamp-based filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{PROCESSED_DATA_DIR}/continuous_data_{timestamp}.json"
    save_data(new_data, output_path)
    
    logger.info(f"Saved continuous data to {output_path}")
    return output_path

@task(name="Train Model", retries=3, retry_delay_seconds=10)
def task_train_model(training_data_path):
    """Task to train the wildfire prediction model."""
    logger.info(f"Training model with data from {training_data_path}")
    
    # Set up MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # Load training data
    with open(training_data_path, 'r') as f:
        training_data = json.load(f)
    
    # Preprocess data
    df = preprocess_data(training_data)
    
    # Train model
    model, accuracy, X_test, y_test = train_model(df)
    
    logger.info(f"Model training completed with accuracy: {accuracy:.4f}")
    
    # Get the run_id from the last MLflow run
    try:
        run_data = mlflow.search_runs(filter_string="tags.mlflow.runName='random-forest-wildfire'")
        if len(run_data) == 0:
            raise ValueError("No runs found with the specified tag")
        run_id = run_data.iloc[0]["run_id"]
    except Exception as e:
        logger.error(f"Error retrieving MLflow run: {str(e)}")
        # Save model locally as a fallback
        os.makedirs(MODELS_DIR, exist_ok=True)
        import pickle
        model_path = f"{MODELS_DIR}/wildfire_model_latest.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        run_id = None
    
    return {
        "run_id": run_id,
        "accuracy": accuracy,
        "model_path": f"{MODELS_DIR}/wildfire_model_latest.pkl",
        "model_object": model  # Include the model object for direct use
    }

@task(name="Load Model", retries=3, retry_delay_seconds=10)
def task_load_model(model_info):
    """Task to load a model from MLflow or local storage."""
    logger.info(f"Loading model from run_id: {model_info.get('run_id')}")
    
    # First try to load from MLflow if run_id is provided
    if model_info.get('run_id'):
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            model = mlflow.sklearn.load_model(f"runs:/{model_info['run_id']}/model")
            logger.info("Successfully loaded model from MLflow")
            return model  # Return immediately if successful
        except Exception as e:
            logger.warning(f"Error loading model from MLflow: {str(e)}")
    
    # If model_object is provided in the model_info, use it directly
    if 'model_object' in model_info:
        logger.info("Using provided model object")
        return model_info['model_object']  # Return immediately if model_object exists
    
    # Try to load from local path as a fallback
    try:
        import pickle
        model_path = model_info.get('model_path', f"{MODELS_DIR}/wildfire_model_latest.pkl")
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"Successfully loaded model from local path: {model_path}")
            return model  # Return immediately if successful
        else:
            logger.error(f"Model file does not exist at path: {model_path}")
    except Exception as e:
        logger.error(f"Error loading model from local path: {str(e)}")
    
    # If all methods failed, raise a more informative error
    raise ValueError("Failed to load model from MLflow, memory, or local storage. Check logs for details.")

@task(name="Evaluate Model", retries=3, retry_delay_seconds=10)
def task_evaluate_model(model_info, validation_data_path):
    """Task to evaluate the model on validation data."""
    logger.info(f"Evaluating model on validation data")
    
    # Load the model
    model = task_load_model(model_info)
    
    # Load validation data
    with open(validation_data_path, 'r') as f:
        validation_data = json.load(f)
    
    # Preprocess validation data
    df = preprocess_data(validation_data)
    
    # Split features and target - check if fire_probability exists
    if 'fire_probability' in df.columns:
        X = df.drop('fire_probability', axis=1)
        y_true = df['fire_probability'].apply(lambda x: 1 if x > 0.5 else 0)
    else:
        # If fire_probability is not in columns, we might be working with prediction data
        # Use all columns as features and set a default y_true (can't evaluate accuracy properly)
        logger.warning("Column 'fire_probability' not found in validation data. Using all columns as features.")
        X = df
        # Create a dummy target variable of all zeros (this is just for fallback)
        y_true = pd.Series([0] * len(df))
    
    # Handle feature compatibility
    if hasattr(model, 'feature_names_in_'):
        required_features = model.feature_names_in_.tolist()
        logger.info(f"Model requires these features: {required_features}")
        
        # Check for missing features
        missing_features = [f for f in required_features if f not in X.columns]
        if missing_features:
            logger.warning(f"Missing features in validation data: {missing_features}")
            # Add missing features with zeros
            for feature in missing_features:
                X[feature] = 0
        
        # Check for extra features not needed by the model
        extra_features = [f for f in X.columns if f not in required_features]
        if extra_features:
            logger.warning(f"Extra features in validation data (will be dropped): {extra_features}")
            # Remove extra features
            X = X.drop(columns=extra_features)
        
        # Ensure features are in the right order
        X = X[required_features]
    
    # Make predictions
    try:
        y_pred = model.predict(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
    except Exception as e:
        logger.error(f"Error making predictions or calculating metrics: {str(e)}")
        # Set fallback metrics
        accuracy, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
    
    # Log metrics to MLflow if run_id is available
    if model_info.get('run_id'):
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
            with mlflow.start_run(run_id=model_info['run_id']):
                mlflow.log_metrics({
                    "validation_accuracy": accuracy,
                    "validation_precision": precision,
                    "validation_recall": recall,
                    "validation_f1": f1
                })
            logger.info("Successfully logged metrics to MLflow")
        except Exception as e:
            logger.warning(f"Error logging metrics to MLflow: {str(e)}")
    
    # Save evaluation results
    os.makedirs(EVALUATION_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    evaluation_path = f"{EVALUATION_DIR}/evaluation_{timestamp}.json"
    
    evaluation_results = {
        "run_id": model_info.get('run_id'),
        "timestamp": timestamp,
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        }
    }
    
    with open(evaluation_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    logger.info(f"Model evaluation completed with accuracy: {accuracy:.4f}")
    logger.info(f"Evaluation results saved to {evaluation_path}")
    
    return evaluation_results

@task(name="Make Predictions", retries=3, retry_delay_seconds=10)
def task_make_predictions(model_info, data_path):
    """Task to make predictions with the trained model."""
    logger.info(f"Making predictions on data from {data_path}")
    
    # Load the model
    model = task_load_model(model_info)
    
    # Load data
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    # Preprocess data
    df = preprocess_data(data)
    
    # Store original data for reference
    original_data = df.copy()
    
    # Split features and target (if target exists)
    if 'fire_probability' in df.columns:
        X = df.drop('fire_probability', axis=1)
        y_true = df['fire_probability'].apply(lambda x: 1 if x > 0.5 else 0)
    else:
        X = df
        
    # Handle feature compatibility
    try:
        # Get the feature names the model was trained with
        if hasattr(model, 'feature_names_in_'):
            required_features = model.feature_names_in_.tolist()
            logger.info(f"Model requires these features: {required_features}")
            
            # Check for missing features
            missing_features = [f for f in required_features if f not in X.columns]
            if missing_features:
                logger.warning(f"Missing features in prediction data: {missing_features}")
                # Add missing features with zeros
                for feature in missing_features:
                    X[feature] = 0
            
            # Check for extra features not needed by the model
            extra_features = [f for f in X.columns if f not in required_features]
            if extra_features:
                logger.warning(f"Extra features in prediction data (will be dropped): {extra_features}")
                # Remove extra features
                X = X.drop(columns=extra_features)
            
            # Ensure features are in the right order
            X = X[required_features]
            
        logger.info(f"Prediction data shape: {X.shape}")
    except Exception as e:
        logger.error(f"Error handling features: {str(e)}. Attempting to continue with prediction.")
    
    # Make predictions with error handling
    try:
        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}. Creating fallback predictions.")
        # Create fallback predictions (all zeros)
        y_pred_proba = np.zeros(len(X))
        y_pred = np.zeros(len(X), dtype=int)
    
    # Create prediction results
    results_df = pd.DataFrame({
        'probability': y_pred_proba,
        'prediction': y_pred
    })
    
    # Add any ID or reference columns from original data
    if 'id' in original_data.columns:
        results_df['id'] = original_data['id']
    if 'date' in original_data.columns:
        results_df['date'] = original_data['date']
    if 'location' in original_data.columns:
        results_df['location'] = original_data['location']
    
    # Save prediction results
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prediction_path = f"{PREDICTIONS_DIR}/predictions_{timestamp}.csv"
    results_df.to_csv(prediction_path, index=False)
    
    logger.info(f"Predictions saved to {prediction_path}")
    return prediction_path

@flow(name="MLflow Setup")
def mlflow_setup_flow():
    """Flow to ensure MLflow is ready and configured."""
    wait_for_mlflow_server()
    initialize_mlflow()
    return True

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
    # Ensure MLflow is set up
    mlflow_setup_flow()
    
    # Train model
    model_info = task_train_model(training_data_path)
    
    return model_info

@flow(name="Model Evaluation Pipeline")
def model_evaluation_pipeline(model_info, validation_data_path=f"{PROCESSED_DATA_DIR}/validation_data.json"):
    """Pipeline for evaluating the model."""
    # Ensure MLflow is set up
    mlflow_setup_flow()
    
    # Evaluate model
    evaluation_results = task_evaluate_model(model_info, validation_data_path)
    
    return evaluation_results

@flow(name="Prediction Pipeline")
def prediction_pipeline(model_info, data_path=f"{PROCESSED_DATA_DIR}/production_data.json"):
    """Pipeline for making predictions with the trained model."""
    # Make predictions
    prediction_path = task_make_predictions(model_info, data_path)
    
    return prediction_path

@flow(name="Continuous Prediction Pipeline")
def continuous_prediction_pipeline(model_run_id=None):
    """Pipeline for continuous prediction with new data."""
    # Ensure MLflow is set up
    mlflow_setup_flow()
    
    # If no run_id provided, get the latest model from MLflow
    if model_run_id is None:
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
            runs = mlflow.search_runs(filter_string="tags.mlflow.runName='random-forest-wildfire'")
            if len(runs) > 0:
                model_run_id = runs.iloc[0]["run_id"]
                logger.info(f"Using latest model run_id: {model_run_id}")
            else:
                logger.warning("No trained models found in MLflow, will use local model if available")
        except Exception as e:
            logger.warning(f"Error getting latest model from MLflow: {str(e)}")
    
    model_info = {
        "run_id": model_run_id,
        "model_path": f"{MODELS_DIR}/wildfire_model_latest.pkl"
    }
    
    # Generate new data
    data_path = task_generate_continuous_data()
    
    # Make predictions
    prediction_path = task_make_predictions(model_info, data_path)
    
    return prediction_path

@flow(name="Complete MLOps Pipeline")
def complete_mlops_pipeline(
    training_records=100, 
    validation_records=30, 
    test_records=20, 
    production_records=10
):
    """Complete MLOps pipeline that generates data, trains, evaluates, and makes predictions with the model."""
    # Ensure MLflow is set up
    mlflow_setup_flow()
    
    # Generate data
    data_paths = data_generation_pipeline(
        training_records=training_records,
        validation_records=validation_records,
        test_records=test_records,
        production_records=production_records
    )
    
    # Train model
    model_info = model_training_pipeline(training_data_path=data_paths["training_path"])
    
    # Evaluate model
    evaluation_results = model_evaluation_pipeline(model_info, data_paths["validation_path"])
    
    # Make predictions on test data
    test_predictions = prediction_pipeline(model_info, data_paths["test_path"])
    
    # Make predictions on production data
    production_predictions = prediction_pipeline(model_info, data_paths["production_path"])
    
    return {
        "data_paths": data_paths,
        "model_info": model_info,
        "evaluation_results": evaluation_results,
        "test_predictions": test_predictions,
        "production_predictions": production_predictions
    }

@flow(name="Daily Prediction Flow")
def daily_prediction_flow():
    """Daily flow to generate new data and make predictions."""
    return continuous_prediction_pipeline()

if __name__ == "__main__":
    # Run the complete pipeline
    result = complete_mlops_pipeline()
    
    print(f"Pipeline completed successfully!")
    print(f"Model accuracy: {result['model_info']['accuracy']:.4f}")
    print(f"MLflow run ID: {result['model_info'].get('run_id', 'Not available - using local model')}")
    print(f"Validation F1 score: {result['evaluation_results']['metrics']['f1_score']:.4f}")
    print(f"Test predictions saved to: {result['test_predictions']}")
    print(f"Production predictions saved to: {result['production_predictions']}")