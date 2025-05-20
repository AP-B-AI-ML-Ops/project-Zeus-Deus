import os
import json
import logging
import pandas as pd
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Define file paths
FIRMS_DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/nasa_firms_data.json')
WEATHER_DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/weather_data.json')
VEGETATION_DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/vegetation_indices.json')
FINAL_MERGED_DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/merged_complete.json')

def load_firms_events():
    """Load FIRMS fire events from CSV file"""
    events = []
    try:
        with open(FIRMS_DATA_PATH, 'r') as f:
            lines = f.readlines()
            header = lines[0].strip().split(',')
            for line in lines[1:]:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue
                lat = float(parts[0])
                lon = float(parts[1])
                acq_date = parts[5]  # 'YYYY-MM-DD'
                acq_time = parts[6]  # 'HHMM'
                event = {"lat": lat, "lon": lon, "date": acq_date, "time": acq_time}
                for i, col in enumerate(header):
                    if i < len(parts):
                        event[col] = parts[i]
                events.append(event)
        logging.info(f"Loaded {len(events)} fire events")
    except Exception as e:
        logging.error(f"Error loading NASA FIRMS data: {e}")
    return events

def load_weather_data():
    """Load weather data"""
    try:
        with open(WEATHER_DATA_PATH, 'r') as f:
            data = json.load(f)
            logging.info(f"Loaded weather data with {len(data)} records")
            return data
    except Exception as e:
        logging.error(f"Error loading weather data: {e}")
        return []

def load_vegetation_data():
    """Load vegetation indices data"""
    try:
        with open(VEGETATION_DATA_PATH, 'r') as f:
            data = json.load(f)
            logging.info(f"Loaded vegetation data with {len(data)} records")
            return data
    except Exception as e:
        logging.error(f"Error loading vegetation data: {e}")
        return []

def merge_all_data(firms_events, weather_data, vegetation_data):
    """Merge all three datasets into one comprehensive dataset"""
    merged = []
    
    # Build weather data lookup by (lat, lon, date)
    weather_lookup = {}
    for entry in weather_data:
        event = entry.get("event", {})
        if not event:
            continue
        # Handle date as string or datetime object
        date_str = str(event.get("date", "")).split(" ")[0]  # Extract YYYY-MM-DD
        lat = round(float(event.get("lat", 0)), 2)
        lon = round(float(event.get("lon", 0)), 2)
        key = (lat, lon, date_str)
        weather_lookup[key] = entry.get("weather")
    
    # Build vegetation data lookup by (lat, lon, date)
    veg_lookup = {}
    for item in vegetation_data:
        lat = round(float(item.get("latitude", 0)), 2)
        lon = round(float(item.get("longitude", 0)), 2)
        date = item.get("date", "")
        key = (lat, lon, date)
        veg_lookup[key] = {
            "ndvi": item.get("ndvi"),
            "evi": item.get("evi")
        }
    
    # Process each fire event
    for event in firms_events:
        lat = round(float(event.get("lat", 0)), 2)
        lon = round(float(event.get("lon", 0)), 2)
        date = event.get("date", "")
        key = (lat, lon, date)
        
        # Get corresponding weather and vegetation data
        weather = weather_lookup.get(key)
        vegetation = veg_lookup.get(key)
        
        # Create comprehensive record
        merged.append({
            "event": event,
            "weather": weather,
            "vegetation": vegetation
        })
    
    logging.info(f"Created comprehensive dataset with {len(merged)} records")
    return merged

def main():
    """Main function to merge all datasets"""
    # Load individual datasets
    firms_events = load_firms_events()
    weather_data = load_weather_data()
    vegetation_data = load_vegetation_data()
    
    if not firms_events:
        logging.error("No FIRMS data available. Exiting.")
        return
    
    # Merge all data
    comprehensive_data = merge_all_data(firms_events, weather_data, vegetation_data)
    
    # Save final merged dataset
    with open(FINAL_MERGED_DATA_PATH, 'w') as f:
        json.dump(comprehensive_data, f, default=str)
    logging.info(f"Final comprehensive dataset saved to {FINAL_MERGED_DATA_PATH}")

if __name__ == "__main__":
    main()
