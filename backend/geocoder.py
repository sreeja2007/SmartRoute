import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENCAGE_API_KEY")

def geocode(address):
    url = "https://api.opencagedata.com/geocode/v1/json"
    
    params = {
        "q": address,
        "key": API_KEY
    }

    if not API_KEY:
        return None, None

    try:
        response = requests.get(url, params=params, timeout=2)
        data = response.json()
    except Exception:
        return None, None

    if data["results"]:
        location = data["results"][0]["geometry"]
        return location["lat"], location["lng"]

    return None, None