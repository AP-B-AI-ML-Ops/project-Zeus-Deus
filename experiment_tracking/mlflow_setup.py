import os
import mlflow
from mlflow.tracking import MlflowClient

def setup_mlflow(tracking_uri="sqlite:///mlflow.db", experiment_name="wildfire-prediction"):
    """Set up MLflow tracking server."""
    # Set the tracking URI
    mlflow.set_tracking_uri(tracking_uri)
    
    # Create or get the experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        experiment = mlflow.get_experiment(experiment_id)
    
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"MLflow experiment name: {experiment.name}")
    print(f"MLflow experiment ID: {experiment.experiment_id}")
    
    return experiment

def list_runs(experiment_name="wildfire-prediction", max_results=5):
    """List recent runs for an experiment."""
    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment:
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], max_results=max_results)
        print(f"Recent runs for experiment '{experiment_name}':")
        for i, (index, run) in enumerate(runs.iterrows()):
            print(f"Run {i+1}:")
            print(f"  Run ID: {run.run_id}")
            print(f"  Status: {run.status}")
            print(f"  Accuracy: {run.metrics.get('accuracy', 'N/A')}")
            print(f"  Start time: {run.start_time}")
            print()
    else:
        print(f"Experiment '{experiment_name}' not found.")

if __name__ == "__main__":
    # Set up MLflow
    setup_mlflow()
    
    # List recent runs
    list_runs()