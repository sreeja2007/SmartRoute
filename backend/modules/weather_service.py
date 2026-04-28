import httpx
import os
import math

OPENWEATHER_API_KEY = os.getenv("OWM_API_KEY", "")
BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

async def get_weather(lat: float, lon: float) -> dict:
    async with httpx.AsyncClient() as client:
        params = {
            "lat": lat,
            "lon": lon,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric"
        }
        response = await client.get(BASE_URL, params=params)
        if response.status_code == 200:
            return response.json()
        return {}

def calculate_weather_score(weather_data: dict) -> float:
    """
    Calculates a weather severity score from 0 to 100.
    > 50 triggers a reroute.
    Factors:
    - Wind speed > 10 m/s
    - Rain/Snow > 5 mm/h
    - Thunderstorms/Tornadoes
    """
    if not weather_data:
        return 0.0
    
    score = 0.0
    
    # Wind
    wind_speed = weather_data.get("wind", {}).get("speed", 0)
    if wind_speed > 15:
        score += 50
    elif wind_speed > 10:
        score += 30
        
    # Weather conditions (IDs)
    # https://openweathermap.org/weather-conditions
    # 2xx: Thunderstorm
    # 5xx: Rain
    # 6xx: Snow
    # 7xx: Atmosphere (mist, smoke, dust, tornado)
    weather_info = weather_data.get("weather", [])
    for condition in weather_info:
        cid = condition.get("id", 800)
        if cid >= 200 and cid < 300:
            score += 80 # Thunderstorm
        elif cid == 781:
            score += 100 # Tornado
        elif cid >= 500 and cid < 600:
            # Rain
            rain_vol = weather_data.get("rain", {}).get("1h", 0)
            if rain_vol > 10:
                score += 50
            elif rain_vol > 5:
                score += 20
        elif cid >= 600 and cid < 700:
            # Snow
            snow_vol = weather_data.get("snow", {}).get("1h", 0)
            if snow_vol > 5:
                score += 40
                
    return min(100.0, score)
