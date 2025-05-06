"""
Module: nasa_firms_ingest.py
Purpose: Download and update wildfire data from NASA FIRMS API.
"""
import os
import logging
import requests

NASA_FIRMS_API_URL = os.getenv("NASA_FIRMS_API_URL")

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def fetch_nasa_firms_data():
    """
    Fetches wildfire data from the NASA FIRMS URL and saves it to the data directory.
    """
    if not NASA_FIRMS_API_URL:
        logging.error("NASA FIRMS API URL not set in environment variables.")
        return
    url = NASA_FIRMS_API_URL
    try:
        response = requests.get(url)
        response.raise_for_status()
        data_path = os.path.join(os.path.dirname(__file__), '../../data/nasa_firms_data.json')
        with open(data_path, 'w') as f:
            f.write(response.text)
        logging.info(f"NASA FIRMS data saved to {data_path}")
    except requests.RequestException as e:
        logging.error(f"Failed to fetch NASA FIRMS data: {e}")

if __name__ == "__main__":
    fetch_nasa_firms_data()
