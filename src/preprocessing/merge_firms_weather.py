import os
import json
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

FIRMS_DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/nasa_firms_data.json')
WEATHER_DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/weather_data.json')
MERGED_DATA_PATH = os.path.join(os.path.dirname(__file__), '../../data/merged_firms_weather.json')


def load_firms_events():
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
                    event[col] = parts[i] if i < len(parts) else None
                events.append(event)
    except Exception as e:
        logging.error(f"Error loading NASA FIRMS data: {e}")
    return events


def load_weather_data():
    try:
        with open(WEATHER_DATA_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading weather data: {e}")
        return []


def merge_events_with_weather(firms_events, weather_data):
    merged = []
    # Build a lookup for weather data by (lat, lon, date)
    weather_lookup = {}
    for entry in weather_data:
        event = entry["event"]
        # Normalize date to just YYYY-MM-DD
        weather_date = str(event["date"]).split(" ")[0]
        key = (round(float(event["lat"]), 2), round(float(event["lon"]), 2), weather_date)
        weather_lookup[key] = entry["weather"]
    for event in firms_events:
        fire_date = str(event["date"]).split(" ")[0]
        key = (round(float(event["lat"]), 2), round(float(event["lon"]), 2), fire_date)
        weather = weather_lookup.get(key)
        merged.append({"event": event, "weather": weather})
    return merged


def main():
    firms_events = load_firms_events()
    weather_data = load_weather_data()
    if not firms_events or not weather_data:
        logging.error("Missing data for merging. Exiting.")
        return
    merged = merge_events_with_weather(firms_events, weather_data)
    with open(MERGED_DATA_PATH, 'w') as f:
        json.dump(merged, f, default=str)
    logging.info(f"Merged data saved to {MERGED_DATA_PATH}")

if __name__ == "__main__":
    main() 