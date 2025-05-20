from prefect import flow, task
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@task(name="Hello Task", retries=2, log_prints=True)
def say_hello(name):
    """Task to say hello."""
    message = f"Hello, {name}!"
    print(message)
    return message

@flow(name="Hello Flow", log_prints=True)
def hello_flow(name="World"):
    """Simple flow that says hello."""
    logger.info("Starting hello flow")
    result = say_hello(name)
    logger.info("Flow completed successfully")
    return result

if __name__ == "__main__":
    # Run the flow directly
    result = hello_flow("Prefect")
    print(f"Flow result: {result}")
    
    # The deployment is now configured in prefect.yaml, no need to deploy manually here
    print("To deploy the flow, use: prefect deployment apply")
    print("To run the deployed flow, use: prefect deployment run 'Hello Flow/hello-flow-deployment'") 