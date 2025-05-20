import os
import sys
import json
import requests
import logging
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(filepath):
    """Load data from a JSON file."""
    logger.info(f"Loading data from {filepath}")
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def make_prediction(data, api_url="http://localhost:8000/predict"):
    """Send data to the prediction API."""
    logger.info(f"Sending prediction request to {api_url}")
    response = requests.post(api_url, json=data)
    
    if response.status_code == 200:
        predictions = response.json()
        logger.info(f"Received {len(predictions)} predictions")
        return predictions
    else:
        logger.error(f"Error making prediction: {response.status_code} - {response.text}")
        return None

def main():
    """Main function to test the prediction API."""
    # Load test data
    test_data = load_data("data/processed/test_data.json")
    logger.info(f"Loaded {len(test_data)} test records")
    
    # Take a subset for testing
    test_subset = test_data[:3]
    
    # Make predictions
    predictions = make_prediction(test_subset)
    
    if predictions:
        # Print predictions
        print("\nPredictions:")
        for i, pred in enumerate(predictions):
            print(f"Event {i+1}: {'High' if pred['high_severity_prediction'] == 1 else 'Low'} severity "
                  f"(Probability: {pred['high_severity_probability']:.2f})")
        
        # Save predictions
        output_dir = "data/predictions"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{output_dir}/predictions_{timestamp}.json"
        
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        logger.info(f"Saved predictions to {output_path}")

if __name__ == "__main__":
    main() 