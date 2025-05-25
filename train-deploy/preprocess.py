import pandas as pd
from ucimlrepo import fetch_ucirepo
import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

def load_and_save_dataset():
    print("Fetching dataset...")

    # fetch dataset 
    forest_fires = fetch_ucirepo(id=162) 
    
    # data (as pandas dataframes) 
    X = forest_fires.data.features 
    y = forest_fires.data.targets 
    
    df = pd.concat([X, y], axis=1)

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    os.makedirs("/shared/data", exist_ok=True)

    df.to_csv("/shared/data/forest_fires.csv", index=False)
    print("Dataset saved to /shared/data/forest_fires.csv")

    print("\nDataset Info:")
    print(forest_fires.metadata)
    print("\nVariable Info:")
    print(forest_fires.variables)
    
    return df

def preprocess_data():
    print("Loading data...")
    df = pd.read_csv("/shared/data/forest_fires.csv")

    print("Processing data...")

    # Handle categorical features
    df_processed = df.copy()

    # One-hot encode onth and day
    month_dummies = pd.get_dummies(df_processed['month'], prefix='month')
    day_dummies = pd.get_dummies(df_processed['day'], prefix='day')

    # Drop original catagorical columns and add dummies
    df_processed = df_processed.drop(['month', 'day'], axis=1)
    df_processed = pd.concat([df_processed, month_dummies, day_dummies], axis=1)
    
    # Transform target variable (area) - log transformation as mentioned in paper
    df_processed['area_log'] = np.log(df_processed['area'] + 1)
    
    # Separate features and target
    X = df_processed.drop(['area', 'area_log'], axis=1)
    y = df_processed['area_log']  # Use log-transformed target
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Train/val/test split (60/20/20)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2
    )
    
    print(f"Train set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples") 
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Save splits
    datasets = {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test
    }
    
    for name, data in datasets.items():
        with open(f'/shared/data/{name}.pkl', 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved {name}.pkl")
    
    # Save feature names for later use
    feature_names = X.columns.tolist()
    with open('/shared/data/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    return datasets

if __name__ == "__main__":
    # Load and save raw dataset
    df = load_and_save_dataset()
    
    # Preprocess and create splits
    datasets = preprocess_data()
    
    print("Preprocessing completed successfully!")
    
    os.makedirs("/shared/models", exist_ok=True)
    