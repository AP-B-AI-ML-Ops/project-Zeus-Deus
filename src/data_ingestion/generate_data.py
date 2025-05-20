import os
import sys
import json
import random
import numpy as np
from datetime import datetime, timedelta
import logging

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define file paths
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

def generate_event_data(num_records=50, start_date="2023-01-01", days=30):
    """Generate realistic wildfire event data."""
    logger.info(f"Generating {num_records} wildfire event records")
    
    # Parse start date
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    
    # Define some realistic locations for wildfires (lat, lon pairs)
    wildfire_prone_areas = [
        (34.0522, -118.2437),  # Los Angeles
        (37.7749, -122.4194),  # San Francisco
        (39.5501, -119.8483),  # Reno
        (47.6062, -122.3321),  # Seattle
        (33.4484, -112.0740),  # Phoenix
        (36.1699, -115.1398),  # Las Vegas
        (44.0682, -114.7420),  # Idaho
        (44.5588, -72.5778),   # Vermont
        (35.7596, -79.0193),   # North Carolina
        (30.2672, -97.7431),   # Austin
    ]
    
    records = []
    for i in range(num_records):
        # Select a random location
        lat, lon = random.choice(wildfire_prone_areas)
        
        # Add some random variation to the location
        lat += random.uniform(-0.5, 0.5)
        lon += random.uniform(-0.5, 0.5)
        
        # Generate a random date within the specified range
        random_days = random.randint(0, days)
        event_date = start_dt + timedelta(days=random_days)
        date_str = event_date.strftime("%Y-%m-%d")
        
        # Generate realistic fire attributes
        brightness = random.uniform(300, 400)
        confidence = random.randint(50, 100)
        frp = random.uniform(5, 40)  # Fire Radiative Power
        
        # Create the event record
        record = {
            "event": {
                "lat": round(lat, 4),
                "lon": round(lon, 4),
                "date": date_str,
                "brightness": round(brightness, 1),
                "scan": round(random.uniform(0.5, 1.5), 1),
                "track": round(random.uniform(0.5, 1.5), 1),
                "acq_date": date_str,
                "acq_time": f"{random.randint(0, 23):02d}{random.randint(0, 59):02d}",
                "satellite": random.choice(["Terra", "Aqua", "NOAA-20", "Suomi NPP"]),
                "confidence": confidence,
                "version": "1.0",
                "bright_t31": round(brightness - random.uniform(10, 30), 1),
                "frp": round(frp, 1),
                "daynight": "D" if random.random() > 0.3 else "N"
            },
            "weather": generate_weather_data(lat, lon, date_str),
            "vegetation": generate_vegetation_data(lat, lon, date_str)
        }
        
        records.append(record)
    
    return records

def generate_weather_data(lat, lon, date):
    """Generate realistic weather data for a location and date."""
    # Determine season (Northern Hemisphere)
    month = int(date.split("-")[1])
    is_summer = 5 <= month <= 9
    
    # Base temperature depends on latitude and season
    base_temp = 30 - abs(lat) / 3
    if is_summer:
        base_temp += 10
    else:
        base_temp -= 5
    
    # Add some random variation
    tavg = base_temp + random.uniform(-5, 5)
    tmin = tavg - random.uniform(5, 10)
    tmax = tavg + random.uniform(5, 10)
    
    # Precipitation (more likely in winter)
    prcp_prob = 0.2 if is_summer else 0.4
    prcp = random.uniform(0, 20) if random.random() < prcp_prob else 0
    
    # Wind speed and pressure
    wspd = random.uniform(5, 20)
    pres = random.uniform(1000, 1020)
    
    return {
        "tavg": round(tavg, 1),
        "tmin": round(tmin, 1),
        "tmax": round(tmax, 1),
        "prcp": round(prcp, 1),
        "wspd": round(wspd, 1),
        "pres": round(pres, 1)
    }

def generate_vegetation_data(lat, lon, date):
    """Generate realistic vegetation indices for a location and date."""
    # Parse date
    dt = datetime.strptime(date, "%Y-%m-%d")
    month = dt.month
    
    # Base NDVI depends on latitude (higher near equator) and season
    base_ndvi = 0.8 - abs(lat) / 90 * 0.5
    
    # Seasonal adjustment
    season_factor = np.sin((month / 12.0) * 2 * np.pi)
    if lat > 0:  # Northern hemisphere
        seasonal_adjustment = season_factor * 0.2
    else:  # Southern hemisphere
        seasonal_adjustment = -season_factor * 0.2
    
    ndvi = base_ndvi + seasonal_adjustment
    # EVI is typically lower than NDVI
    evi = ndvi * 0.8
    
    # Add some random variation
    ndvi += random.uniform(-0.1, 0.1)
    evi += random.uniform(-0.1, 0.1)
    
    # Ensure values are in realistic ranges
    ndvi = max(0.0, min(0.9, ndvi))
    evi = max(0.0, min(0.7, evi))
    
    return {
        "ndvi": round(ndvi, 2),
        "evi": round(evi, 2)
    }

def save_data(data, filepath):
    """Save data to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved {len(data)} records to {filepath}")

def main():
    """Generate and save data for training, validation, and testing."""
    # Create directories if they don't exist
    os.makedirs(f"{RAW_DATA_DIR}/firms", exist_ok=True)
    os.makedirs(f"{RAW_DATA_DIR}/weather", exist_ok=True)
    os.makedirs(f"{RAW_DATA_DIR}/vegetation", exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Generate training data
    training_data = generate_event_data(num_records=100, start_date="2022-01-01", days=180)
    save_data(training_data, f"{PROCESSED_DATA_DIR}/training_data.json")
    
    # Generate validation data
    validation_data = generate_event_data(num_records=30, start_date="2022-07-01", days=90)
    save_data(validation_data, f"{PROCESSED_DATA_DIR}/validation_data.json")
    
    # Generate test data
    test_data = generate_event_data(num_records=20, start_date="2022-10-01", days=90)
    save_data(test_data, f"{PROCESSED_DATA_DIR}/test_data.json")
    
    # Generate production data (more recent)
    production_data = generate_event_data(num_records=10, start_date="2023-01-01", days=30)
    save_data(production_data, f"{PROCESSED_DATA_DIR}/production_data.json")
    
    # Also save some raw data examples
    firms_sample = generate_event_data(num_records=5, start_date="2022-01-01", days=30)
    save_data(firms_sample, f"{RAW_DATA_DIR}/firms/nasa_firms_data.json")
    
    # Generate weather and vegetation data
    weather_data = []
    vegetation_data = []
    for event in firms_sample:
        lat = event['event']['lat']
        lon = event['event']['lon']
        date = event['event']['date']
        
        weather_data.append({
            "lat": lat,
            "lon": lon,
            "date": date,
            "weather": generate_weather_data(lat, lon, date)
        })
        
        vegetation_data.append({
            "lat": lat,
            "lon": lon,
            "date": date,
            "vegetation": generate_vegetation_data(lat, lon, date)
        })
    
    save_data(weather_data, f"{RAW_DATA_DIR}/weather/weather_data.json")
    save_data(vegetation_data, f"{RAW_DATA_DIR}/vegetation/vegetation_indices.json")
    
    logger.info("Data generation complete")

if __name__ == "__main__":
    main() 