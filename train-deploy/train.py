import os
import pickle
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

mlflow.set_tracking_uri("http://experiment-tracking:5000")
mlflow.set_experiment("forest-fires-train")


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def run_train(data_path: str):
    
    mlflow.sklearn.autolog()
    X_train = load_pickle(os.path.join(data_path, "X_train.pkl"))
    y_train = load_pickle(os.path.join(data_path, "y_train.pkl"))
    X_val = load_pickle(os.path.join(data_path, "X_val.pkl"))
    y_val = load_pickle(os.path.join(data_path, "y_val.pkl"))

    with mlflow.start_run():

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse)


if __name__ == "__main__":
    print("...training models")
    run_train("./data/") 