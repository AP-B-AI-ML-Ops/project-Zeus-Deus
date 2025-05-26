import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import os
import pickle
import numpy as np

from prefect import flow, task

mlflow.set_tracking_uri("http://experiment-tracking:5000")
client = MlflowClient()
model_name = "forest-fires-best-model"

def get_latest_version(model_name):
    # Gets the latest version of the model from the MLflow Model Registry
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ValueError(f"No versions found for model '{model_name}'")
    latest_version = max(versions, key=lambda v: int(v.version))
    return latest_version.version

@task
def load_model():
    print("...loading model")
    latest_version = get_latest_version(model_name)
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/{latest_version}")
    return model, latest_version

@task
def load_batch_data():
    path = "/shared/data/forest_fires.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Batch data not found at {path}")
    df = pd.read_csv(path)
    return df

@task
def prep_features(df):
    # Same preprocessing as in train-deploy/preprocess.py
    df_processed = df.copy()
    month_dummies = pd.get_dummies(df_processed['month'], prefix='month')
    day_dummies = pd.get_dummies(df_processed['day'], prefix='day')
    df_processed = df_processed.drop(['month', 'day'], axis=1)
    df_processed = pd.concat([df_processed, month_dummies, day_dummies], axis=1)
    X = df_processed.drop(['area'], axis=1)
    return X

@task
def get_feature_names():
    # Loads the feature order as used during training
    path = "/shared/data/feature_names.pkl"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature names not found at {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

@flow
def run_batch():
    print("=== Forest Fires Batch Prediction ===")
    model, version = load_model.submit().result()
    print(f"Model version: {version}")

    df = load_batch_data.submit().result()
    print(f"Loaded batch data: {df.shape}")

    feature_names = get_feature_names.submit().result()
    X = prep_features.submit(df).result()
    X = X[feature_names]  # Ensure correct order

    print("Predicting...")
    y_pred = model.predict(X)
    df["predicted_area_log"] = y_pred
    df["predicted_area"] = np.clip(np.exp(y_pred) - 1, 0, None)

    # Save output
    os.makedirs("/batch-data/output", exist_ok=True)
    output_path = f"/batch-data/output/batch_predictions_v{version}.csv"
    df.to_csv(output_path, index=False)
    print(f"Batch predictions saved to {output_path}")

if __name__ == "__main__":
    run_batch()
