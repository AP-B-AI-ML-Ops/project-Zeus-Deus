import os
import pickle
import mlflow
import optuna
from optuna.samplers import TPESampler

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

mlflow.set_tracking_uri("http://experiment-tracking:5000")
mlflow.set_experiment("forest-fires-hpo")

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

def objective(trial):
    # Load data
    X_train = load_pickle(os.path.join("/shared/data", "X_train.pkl"))
    y_train = load_pickle(os.path.join("/shared/data", "y_train.pkl"))
    X_val = load_pickle(os.path.join("/shared/data", "X_val.pkl"))
    y_val = load_pickle(os.path.join("/shared/data", "y_val.pkl"))

    # Hyperparameters to tune
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 5, 30)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 4)

    with mlflow.start_run():
        mlflow.sklearn.autolog()
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=0,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)
    return rmse

def run_optimization(data_path: str, num_trials: int):
    # Load data
    X_train = load_pickle(os.path.join(data_path, "X_train.pkl"))
    y_train = load_pickle(os.path.join(data_path, "y_train.pkl"))
    X_val = load_pickle(os.path.join(data_path, "X_val.pkl"))
    y_val = load_pickle(os.path.join(data_path, "y_val.pkl"))

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=num_trials)

    # Train and log best model
    best_params = study.best_params
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1

    print("Best params:", best_params)

    rf = RandomForestRegressor(**best_params)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_val)
    rmse = root_mean_squared_error(y_val, y_pred)

    with mlflow.start_run(run_name="best_model"):
        mlflow.log_params(best_params)
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(rf, "model")

    with open(os.path.join(data_path, "best_model.pkl"), "wb") as f_out:
        pickle.dump(rf, f_out)
    print("Best model saved to", os.path.join(data_path, "best_model.pkl"))

if __name__ == "__main__":
    print("...starting HPO with Optuna")
    run_optimization("/shared/data", 20)
