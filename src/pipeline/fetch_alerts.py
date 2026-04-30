import requests, os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("STIB_API_KEY")

ALERT_URL = "https://api-management-opendata-production.azure-api.net/api/datasets/stibmivb/rt/TravellersInformation/[?select][&where][&group_by][&order_by][&limit][&offset]"

def fetch_alerts():
    params = {"limit": 100, "apikey": API_KEY}
    r = requests.get(ALERT_URL, params=params)
    return r.json().get("results", [])