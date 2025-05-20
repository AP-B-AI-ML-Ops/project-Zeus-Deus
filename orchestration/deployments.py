from prefect.deployments import Deployment
from prefect.orion.schemas.schedules import CronSchedule
from workflows import wildfire_prediction_pipeline
import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def create_deployment():
    # Create a deployment for the wildfire prediction pipeline
    deployment = Deployment.build_from_flow(
        flow=wildfire_prediction_pipeline,
        name="wildfire-prediction-daily",
        schedule=CronSchedule(cron="0 0 * * *"),  # Run daily at midnight
        tags=["wildfire", "production"]
    )
    
    # Apply the deployment
    deployment.apply()
    
    print(f"Deployment 'wildfire-prediction-daily' created successfully!")
    print(f"Run 'prefect deployment run wildfire-prediction-pipeline/wildfire-prediction-daily' to start a flow run.")

if __name__ == "__main__":
    create_deployment()