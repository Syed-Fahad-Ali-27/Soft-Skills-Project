import requests, json, os, time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("STIB_API_KEY")

WAITING_TIME_URL = "https://api-management-opendata-production.azure-api.net/api/datasets/stibmivb/rt/WaitingTimes/"
VEHICLE_POS_URL  = "https://api-management-opendata-production.azure-api.net/api/datasets/stibmivb/rt/VehiclePositions/[?select][&where][&group_by][&order_by][&limit][&offset]"

def fetch_waiting_times(limit=100):
    params = {"limit": limit, "apikey": API_KEY}
    r = requests.get(WAITING_TIME_URL, params=params)
    return r.json().get("results", [])

def save_snapshot(data, folder="data/raw/realtime"):
    os.makedirs(folder, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"{folder}/snapshot_{ts}.json"
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"Saved {len(data)} records → {path}")

if __name__ == "__main__":
    INTERVAL = 30  # seconds

    while True:
        try:
            data = fetch_waiting_times()
            save_snapshot(data)
        except Exception as e:
            print(f"Error: {e}")

        time.sleep(INTERVAL)