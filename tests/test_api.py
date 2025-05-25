import requests
import json

def test_api():
    url = "http://web-api:9696"
    
    print("🔍 Testing API...")
    
    # Test health
    try:
        health = requests.get(f"{url}/health")
        print("✅ Health:", health.json())
    except Exception as e:
        print("❌ Health check failed:", e)
        return
    
    # Test prediction - High risk scenario (Summer, dry)
    print("\n🔥 Testing HIGH RISK scenario (Summer, dry conditions)...")
    high_risk_data = {
        "X": 7,
        "Y": 5,
        "FFMC": 95.0, "DMC": 45.0, "DC": 300.0, "ISI": 12.0,
        "temp": 35.0, "RH": 25.0, "wind": 8.0, "rain": 0.0,
        "month_aug": 1, "month_jan": 0, "month_feb": 0, "month_mar": 0, 
        "month_apr": 0, "month_may": 0, "month_jun": 0, "month_jul": 0, 
        "month_sep": 0, "month_oct": 0, "month_nov": 0, "month_dec": 0,
        "day_sun": 1, "day_mon": 0, "day_tue": 0, "day_wed": 0, 
        "day_thu": 0, "day_fri": 0, "day_sat": 0
    }
    
    try:
        prediction = requests.post(f"{url}/predict", json=high_risk_data)
        result = prediction.json()
        print(f"📊 High Risk Prediction: {result['prediction']:.4f}")
    except Exception as e:
        print("❌ High risk prediction failed:", e)
    
    # Test prediction - Low risk scenario (Winter, wet)
    print("\n❄️ Testing LOW RISK scenario (Winter, wet conditions)...")
    low_risk_data = {
        "X": 7,
        "Y": 5,
        "FFMC": 60.0, "DMC": 5.0, "DC": 50.0, "ISI": 2.0,
        "temp": 10.0, "RH": 85.0, "wind": 2.0, "rain": 5.0,
        "month_jan": 1, "month_feb": 0, "month_mar": 0, "month_apr": 0, 
        "month_may": 0, "month_jun": 0, "month_jul": 0, "month_aug": 0, 
        "month_sep": 0, "month_oct": 0, "month_nov": 0, "month_dec": 0,
        "day_wed": 1, "day_mon": 0, "day_tue": 0, "day_thu": 0, 
        "day_fri": 0, "day_sat": 0, "day_sun": 0
    }
    
    try:
        prediction = requests.post(f"{url}/predict", json=low_risk_data)
        result = prediction.json()
        print(f"📊 Low Risk Prediction: {result['prediction']:.4f}")
    except Exception as e:
        print("❌ Low risk prediction failed:", e)

    print("\n🎉 Testing completed!")

if __name__ == "__main__":
    test_api()
