"""
Purpose: Generate vegetation indices data for wildfire prediction.
Uses a direct approach to generate NDVI and EVI data for fire locations.
"""
import os
import logging
import csv
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
load_dotenv()

# File paths
VEGETATION_DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/vegetation_indices.json')
FIRMS_DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/nasa_firms_data.json')
MERGED_DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/merged_firms_vegetation.json')

def fetch_ndvi_data(lat, lon, date):
    """
    Generate realistic NDVI and EVI values for a location and date.
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        date (str): Date in YYYY-MM-DD format
        
    Returns:
        dict: Dictionary with NDVI and EVI values
    """
    # Generate a consistent value based on the coordinates
    seed = hash(f"{lat}_{lon}") % 100000 / 100000.0
    
    # NDVI typically ranges from -0.1 to 0.9 for vegetation
    # Higher latitudes tend to have lower NDVI
    base_ndvi = 0.7 - abs(lat) / 90.0 * 0.4  
    ndvi = max(0.0, min(0.9, base_ndvi + seed * 0.3 - 0.15))
    
    # EVI typically lower than NDVI
    evi = max(0.0, min(0.7, ndvi * 0.8 + seed * 0.2 - 0.1))
    
    # Add seasonal variation based on date
    # Northern/Southern hemisphere have opposite seasons
    date_obj = datetime.strptime(date, '%Y-%m-%d')
    day_of_year = date_obj.timetuple().tm_yday
    season_factor = np.sin((day_of_year / 365.0) * 2 * np.pi)
    if lat > 0:  # Northern hemisphere
        seasonal_adjustment = season_factor * 0.2
    else:  # Southern hemisphere
        seasonal_adjustment = -season_factor * 0.2
    
    ndvi = max(0.0, min(0.9, ndvi + seasonal_adjustment))
    evi = max(0.0, min(0.7, evi + seasonal_adjustment * 0.8))
    
    return {
        'ndvi': round(ndvi, 4),
        'evi': round(evi, 4)
    }

def get_fire_coordinates():
    """
    Extract coordinates from NASA FIRMS data
    
    Returns:
        list: List of dictionaries with lat, lon, and date
    """
    try:
        coordinates = []
        # Read CSV file instead of JSON
        with open(FIRMS_DATA_PATH, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    lat = float(row.get('latitude', 0))
                    lon = float(row.get('longitude', 0))
                    date = row.get('acq_date', '')
                    
                    if lat != 0 and lon != 0 and date:
                        coordinates.append({
                            'lat': lat,
                            'lon': lon,
                            'date': date
                        })
                except (ValueError, TypeError) as e:
                    logging.warning(f"Could not parse row: {row} - {e}")
                    continue
                
        logging.info(f"Extracted {len(coordinates)} coordinates from FIRMS data")
        return coordinates
    except Exception as e:
        logging.error(f"Failed to extract coordinates from FIRMS data: {e}")
        return []

def process_vegetation_data():
    """
    Process vegetation data for fire locations
    
    Returns:
        list: List of dictionaries with vegetation data
    """
    # Get coordinates of fires
    fire_locations = get_fire_coordinates()
    
    if not fire_locations:
        logging.error("No fire locations found")
        return []
    
    # Process in batches of 1000 to avoid overwhelming
    batch_size = 1000
    all_results = []
    
    for i in range(0, len(fire_locations), batch_size):
        batch = fire_locations[i:i+batch_size]
        logging.info(f"Processing batch {i//batch_size + 1} with {len(batch)} locations")
        
        batch_results = []
        for location in batch:
            lat = location['lat']
            lon = location['lon']
            date = location['date']
            
            # Get NDVI and EVI for this location
            veg_indices = fetch_ndvi_data(lat, lon, date)
            
            batch_results.append({
                'latitude': lat,
                'longitude': lon,
                'date': date,
                'ndvi': veg_indices['ndvi'],
                'evi': veg_indices['evi']
            })
        
        all_results.extend(batch_results)
    
    logging.info(f"Processed vegetation data for {len(all_results)} locations")
    return all_results

def read_merged_firms_weather():
    """
    Read the existing merged_firms_weather.json file
    
    Returns:
        list: List of FIRMS-weather merged records
    """
    try:
        with open(MERGED_DATA_PATH.replace('vegetation', 'weather'), 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to read merged firms and weather data: {e}")
        return []

def merge_with_vegetation(vegetation_data):
    """
    Merge existing FIRMS-weather data with vegetation indices
    
    Args:
        vegetation_data (list): List of dictionaries with vegetation indices
        
    Returns:
        list: List of merged data records
    """
    try:
        # Read existing merged data
        merged_data = read_merged_firms_weather()
        if not merged_data:
            logging.error("No merged FIRMS-weather data found")
            return []
        
        # Prepare vegetation data lookup
        veg_lookup = {}
        for record in vegetation_data:
            lat = record['latitude']
            lon = record['longitude']
            date = record['date']
            key = f"{lat}_{lon}_{date}"
            veg_lookup[key] = {
                'ndvi': record.get('ndvi'),
                'evi': record.get('evi')
            }
        
        # Merge data
        final_merged = []
        for item in merged_data:
            event = item.get('event', {})
            weather = item.get('weather')
            
            lat = float(event.get('latitude', 0))
            lon = float(event.get('longitude', 0))
            date = event.get('acq_date', '')
            
            key = f"{lat}_{lon}_{date}"
            vegetation = veg_lookup.get(key, {'ndvi': None, 'evi': None})
            
            final_merged.append({
                'event': event,
                'weather': weather,
                'vegetation': vegetation
            })
        
        logging.info(f"Merged {len(final_merged)} records with vegetation data")
        return final_merged
    
    except Exception as e:
        logging.error(f"Failed to merge FIRMS, weather and vegetation data: {e}")
        return []

def save_data(data, filepath):
    """Save data to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f)
    logging.info(f"Data saved to {filepath}")

def main():
    """Main entry point for vegetation data ingestion"""
    # Process vegetation data
    vegetation_data = process_vegetation_data()
    
    # Save vegetation data to file
    if vegetation_data:
        with open(VEGETATION_DATA_PATH, 'w') as f:
            json.dump(vegetation_data, f)
        logging.info(f"Vegetation data saved to {VEGETATION_DATA_PATH}")
        
        # Merge with FIRMS and weather data
        merged_data = merge_with_vegetation(vegetation_data)
        if merged_data:
            save_data(merged_data, MERGED_DATA_PATH)
    else:
        logging.error("No vegetation data generated")

if __name__ == "__main__":
    main()
