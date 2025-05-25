from flask import Flask, request, jsonify
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import pandas as pd
import traceback
import pickle

# MLflow setup
mlflow.set_tracking_uri("http://experiment-tracking:5000")
client = MlflowClient()

model_name = "forest-fires-best-model"
model = None  # Will be loaded dynamically
model_version = "N/A"  # Updated after loading

def load_latest_model():
    """
    Loads the latest registered version of the model from MLflow Model Registry.
    """
    global model, model_version
    try:
        # Find all versions of the model
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise ValueError(f"No versions found for model '{model_name}'")
        
        # Pick the highest version number (latest)
        latest_version = max(versions, key=lambda v: int(v.version))
        model_version = latest_version.version
        
        print(f"🔄 Loading model {model_name}, version {model_version}...")
        model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")
        print("✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        traceback.print_exc()
        model = None
        model_version = "N/A"

# Load the latest model at startup
load_latest_model()

app = Flask("prediction")

with open("/shared/data/feature_names.pkl", "rb") as f:
    expected_features = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        # Return 503 if model is not loaded
        return jsonify({"error": "Model not loaded"}), 503
    
    try:
        data = request.get_json()
        print(f"📥 Received data: {data}")
        df = pd.DataFrame([[data[feat] for feat in expected_features]], columns=expected_features)
        prediction = model.predict(df)
        print(f"📤 Prediction: {prediction[0]}")

        return jsonify({"prediction": float(prediction[0])})
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    # Health endpoint now also returns model version
    return jsonify({
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None,
        "model_version": model_version
    })

@app.route("/reload-model", methods=["POST"])
def reload_model():
    """
    Endpoint to reload the latest model version on demand.
    Useful after a new model is registered.
    """
    try:
        load_latest_model()
        return jsonify({
            "status": "success",
            "model_version": model_version
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("🚀 Starting Flask app...")
    app.run(host="0.0.0.0", port=9696)
