import requests, json, os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("STIB_API_KEY")

WAITING_TIME_URL = "https://data.stib-mivb.brussels/api/explore/v2.1/catalog/datasets/waiting-time-rt-production/records"
VEHICLE_POS_URL  = "https://data.stib-mivb.brussels/api/explore/v2.1/catalog/datasets/vehicle-position-rt-production/records"

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
    data = fetch_waiting_times()
    save_snapshot(data)