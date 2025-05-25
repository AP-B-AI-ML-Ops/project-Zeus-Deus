import os
import pickle
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

HPO_EXPERIMENT_NAME = "forest-fires-hpo"
EXPERIMENT_NAME = "forest-fires-best-models"
RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state', 'n_jobs']

mlflow.set_tracking_uri("http://experiment-tracking:5000")
mlflow.set_experiment(EXPERIMENT_NAME)

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

def train_and_log_model(params):
    # Load data
    X_train = load_pickle("/shared/data/X_train.pkl")
    y_train = load_pickle("/shared/data/y_train.pkl")
    X_val = load_pickle("/shared/data/X_val.pkl")
    y_val = load_pickle("/shared/data/y_val.pkl")
    X_test = load_pickle("/shared/data/X_test.pkl")
    y_test = load_pickle("/shared/data/y_test.pkl")

    with mlflow.start_run():
        # Only use the parameters we care about
        safe_params = {
            'n_estimators': int(params.get('n_estimators', 100)),
            'max_depth': int(params.get('max_depth', None)) if params.get('max_depth') != 'None' else None,
            'min_samples_split': int(params.get('min_samples_split', 2)),
            'min_samples_leaf': int(params.get('min_samples_leaf', 1)),
            'random_state': int(params.get('random_state', 42)),
            'n_jobs': int(params.get('n_jobs', -1))
        }

        print(f"Using params: {safe_params}")  # Debug print
        
        # Train model
        rf = RandomForestRegressor(**safe_params)
        rf.fit(X_train, y_train)
        
        # Predictions
        y_val_pred = rf.predict(X_val)
        y_test_pred = rf.predict(X_test)
        
        # Metrics
        val_rmse = root_mean_squared_error(y_val, y_val_pred)
        test_rmse = root_mean_squared_error(y_test, y_test_pred)
        
        mlflow.log_metric('val_rmse', val_rmse)
        mlflow.log_metric('test_rmse', test_rmse)
        mlflow.sklearn.log_model(rf, artifact_path="model")

def run_register_model(top_n: int):
    client = MlflowClient()

    # Get top N models from HPO experiment
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )
    
    print(f"Training top {len(runs)} models from HPO...")
    for run in runs:
        train_and_log_model(params=run.data.params)

    # Select best model based on test RMSE
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.test_rmse ASC"]
    )[0]

    # Register best model
    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri, name="forest-fires-best-model")
    
    print(f"Best model registered with test RMSE: {best_run.data.metrics['test_rmse']}")

if __name__ == "__main__":
    print("...registering model")
    run_register_model(5)