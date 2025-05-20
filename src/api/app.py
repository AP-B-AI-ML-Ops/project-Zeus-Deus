import os
import sys
import json
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import mlflow
import logging
import pickle

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import our modules
from src.models.predict import prepare_input

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Wildfire Prediction API",
    description="API for predicting high severity wildfires",
    version="0.1.0"
)

# Define data models
class EventData(BaseModel):
    lat: float
    lon: float
    brightness: float
    scan: float
    track: float
    confidence: float
    bright_t31: float
    frp: float
    daynight: str

class WeatherData(BaseModel):
    tavg: Optional[float] = None
    tmin: Optional[float] = None
    tmax: Optional[float] = None
    prcp: Optional[float] = None
    wspd: Optional[float] = None
    pres: Optional[float] = None

class VegetationData(BaseModel):
    ndvi: Optional[float] = None
    evi: Optional[float] = None

class WildfireData(BaseModel):
    event: EventData
    weather: Optional[WeatherData] = None
    vegetation: Optional[VegetationData] = None

class PredictionResponse(BaseModel):
    event_id: str
    lat: float
    lon: float
    high_severity_prediction: int
    high_severity_probability: float

# Load the model at startup
@app.on_event("startup")
async def load_model():
    logger.info("Loading model")
    
    try:
        # First try to load the model from the file system
        model_path = "models/wildfire_model_latest.pkl"
        if os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")
            with open(model_path, 'rb') as f:
                app.state.model = pickle.load(f)
            app.state.is_trained = True
            logger.info("Model loaded successfully from file system")
            return
        
        # If file doesn't exist, try MLflow
        # Set up MLflow
        mlflow.set_tracking_uri("http://mlflow:5000")
        
        # Get the latest run
        runs = mlflow.search_runs(filter_string="tags.mlflow.runName='random-forest-wildfire'")
        
        if len(runs) == 0:
            logger.warning("No model runs found in MLflow. Using a default model.")
            # Create a simple default model (Random Forest with default parameters)
            from sklearn.ensemble import RandomForestClassifier
            app.state.model = RandomForestClassifier()
            app.state.is_trained = False
        else:
            # Get the run with the highest accuracy
            best_run = runs.sort_values("metrics.accuracy", ascending=False).iloc[0]
            run_id = best_run.run_id
            
            # Load the model
            logger.info(f"Loading model from run {run_id}")
            model_uri = f"runs:/{run_id}/random_forest_model"
            app.state.model = mlflow.sklearn.load_model(model_uri)
            app.state.is_trained = True
            logger.info("Model loaded successfully from MLflow")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        # Create a simple default model (Random Forest with default parameters)
        from sklearn.ensemble import RandomForestClassifier
        app.state.model = RandomForestClassifier()
        app.state.is_trained = False

# Define API endpoints
@app.post("/predict", response_model=List[PredictionResponse])
async def predict(data: List[WildfireData]):
    """Predict high severity wildfires."""
    if not hasattr(app.state, "model"):
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Check if the model is trained
    if not getattr(app.state, "is_trained", False):
        logger.warning("Using untrained model for predictions")
    
    # Convert to dict for processing
    input_data = [item.dict() for item in data]
    
    # Prepare input data
    df = prepare_input(input_data)
    
    try:
        # Make predictions
        predictions = app.state.model.predict(df)
        probabilities = app.state.model.predict_proba(df)
        
        # Create response
        results = []
        for i, record in enumerate(input_data):
            result = PredictionResponse(
                event_id=f"event_{i}",
                lat=record['event']['lat'],
                lon=record['event']['lon'],
                high_severity_prediction=int(predictions[i]),
                high_severity_probability=float(probabilities[i][1])
            )
            results.append(result)
        
        return results
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise HTTPException(status_code=500, detail=f"Error making predictions: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if not hasattr(app.state, "model"):
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    status = {
        "status": "healthy",
        "model_trained": getattr(app.state, "is_trained", False)
    }
    
    return status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)