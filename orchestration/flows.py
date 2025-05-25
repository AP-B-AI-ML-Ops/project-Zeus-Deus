from prefect import flow, task
import subprocess

@task(name="preprocess-data")
def preprocess_task():
    subprocess.run(["python", "/train-deploy/preprocess.py"], check=True)

@task(name="train-model")
def train_task():
    subprocess.run(["python", "/train-deploy/train.py"], check=True)

@task(name="hyperparameter-optimization") 
def hpo_task():
    subprocess.run(["python", "/train-deploy/hpo.py"], check=True)

@task(name="register-model")
def register_task():
    subprocess.run(["python", "/train-deploy/register.py"], check=True)

@flow(name="forest-fires-ml-pipeline")
def ml_pipeline():
    """Complete ML pipeline for forest fires prediction"""
    preprocess_task()
    train_task() 
    hpo_task()
    register_task()

if __name__ == "__main__":
    ml_pipeline()