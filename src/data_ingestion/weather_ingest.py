import os
import json
import logging
from datetime import datetime
from meteostat import Point, Daily
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# File paths
FIRMS_DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/nasa_firms_data.json')
WEATHER_DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/weather_data.json')


def read_firms_events():
    """
    Reads NASA FIRMS fire events and returns a list of dicts with lat, lon, and date.
    """
    events = []
    with open(FIRMS_DATA_PATH, 'r') as f:
        lines = f.readlines()
        header = lines[0].strip().split(',')
        for line in lines[1:]:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            lat = float(parts[0])
            lon = float(parts[1])
            acq_date = parts[5]  # e.g., '2025-05-05'
            try:
                dt = datetime.strptime(acq_date, "%Y-%m-%d")
                events.append({"lat": lat, "lon": lon, "date": dt})
            except Exception as e:
                logging.warning(f"Could not parse date for line: {line.strip()} - {e}")
    return events


def fetch_weather_for_event(event):
    """
    Fetches daily weather data for a single event (lat, lon, date) using Meteostat.
    """
    location = Point(event["lat"], event["lon"])
    try:
        data = Daily(location, event["date"], event["date"])
        data = data.fetch()
        if not data.empty:
            # Convert DataFrame row to dict
            weather = data.iloc[0].to_dict()
            return weather
        else:
            logging.warning(f"No weather data found for {event}")
            return None
    except Exception as e:
        logging.error(f"Failed to fetch weather for {event}: {e}")
        return None


def main():
    events = read_firms_events()
    logging.info(f"Found {len(events)} fire events.")
    weather_data = []
    for event in tqdm(events, desc="Fetching weather data"):
        weather = fetch_weather_for_event(event)
        if weather:
            weather_data.append({"event": event, "weather": weather})
    # Save weather data
    with open(WEATHER_DATA_PATH, 'w') as f:
        json.dump(weather_data, f, default=str)
    logging.info(f"Weather data saved to {WEATHER_DATA_PATH}")

if __name__ == "__main__":
    main() 