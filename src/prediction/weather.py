"""
CloudBurst — Weather Data Fetchers
═══════════════════════════════════
Functions to retrieve live weather from WeatherAPI + Open-Meteo
consensus, forecast data, elevation, and historical CSVs.
"""

import requests
import pandas as pd
from config import WEATHER_API_KEY, DATA_DIR


# ═══════════════════════════════════════════════════════════
# HISTORICAL DATA
# ═══════════════════════════════════════════════════════════

def load_historical_cloudbursts() -> pd.DataFrame:
    """Load the Uttarakhand cloudburst CSV for map overlays."""
    csv_path = DATA_DIR / "uttarakhand_cloudburst_detected.csv"
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception as e:
            print("Error loading historical data:", e)
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════
# ELEVATION
# ═══════════════════════════════════════════════════════════

def get_elevation(lat: float, lon: float) -> float:
    """Query Open-Meteo elevation API."""
    try:
        r = requests.get(
            f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}",
            timeout=5,
        )
        if r.status_code == 200:
            return r.json().get("elevation", [0])[0]
    except Exception:
        pass
    return 0.0


# ═══════════════════════════════════════════════════════════
# LIVE WEATHER (Primary + Consensus)
# ═══════════════════════════════════════════════════════════

def get_weather(city: str) -> dict:
    """Fetch current weather via WeatherAPI with Open-Meteo consensus blend.

    Returns a dict with keys:
        precip, temp, humidity, pressure, wind,
        pressure_delta, humidity_delta,
        wind_dir, feelslike, vis, uv, cloud, condition,
        lat, lon, elevation, name, country, consensus
    or None on failure.
    """
    try:
        # ── Primary (WeatherAPI) ────────────────────────────
        r1 = requests.get(
            f"https://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={city}",
            timeout=8,
        )
        if r1.status_code != 200:
            print(f"WeatherAPI request failed [{r1.status_code}]: {r1.text[:200]}")
            return None

        d1   = r1.json()
        cur1 = d1["current"]
        loc  = d1["location"]
        lat  = loc["lat"]
        lon  = loc["lon"]

        var_temp   = cur1["temp_c"]
        var_precip = cur1["precip_mm"]
        var_humid  = cur1["humidity"]
        var_press  = cur1["pressure_mb"]
        var_wind   = cur1["wind_kph"] / 3.6  # kph → m/s

        pressure_delta = 0.0
        humidity_delta = 0.0
        consensus_active = False

        # ── Secondary (Open-Meteo consensus) ────────────────
        try:
            r2 = requests.get(
                f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}"
                f"&current=temperature_2m,relative_humidity_2m,precipitation,"
                f"surface_pressure,wind_speed_10m"
                f"&past_hours=3"
                f"&hourly=surface_pressure,relative_humidity_2m",
                timeout=5,
            )
            if r2.status_code == 200:
                data2 = r2.json()
                cur2  = data2.get("current", {})

                # Trend Intelligence
                hourly = data2.get("hourly", {})
                if hourly and "surface_pressure" in hourly and hourly["surface_pressure"]:
                    hist_pressure = hourly["surface_pressure"][0]
                    pressure_delta = round(
                        cur2.get("surface_pressure", var_press) - hist_pressure, 2
                    )
                if hourly and "relative_humidity_2m" in hourly and hourly["relative_humidity_2m"]:
                    hist_humidity = hourly["relative_humidity_2m"][0]
                    humidity_delta = round(
                        cur2.get("relative_humidity_2m", var_humid) - hist_humidity, 2
                    )

                # 80/20 Blend for humidity, pressure, wind
                var_humid = (var_humid * 0.20) + (
                    cur2.get("relative_humidity_2m", var_humid) * 0.80
                )
                var_press = (var_press * 0.20) + (
                    cur2.get("surface_pressure", var_press) * 0.80
                )
                w2 = cur2.get("wind_speed_10m", var_wind * 3.6) / 3.6
                var_wind = (var_wind * 0.20) + (w2 * 0.80)

                consensus_active = True
        except Exception as e:
            print("Open-Meteo Backup Failed:", e)

        return {
            "precip": round(var_precip, 2),
            "temp": round(var_temp, 1),
            "humidity": round(var_humid, 1),
            "pressure": round(var_press, 1),
            "wind": round(var_wind, 2),
            "pressure_delta": pressure_delta,
            "humidity_delta": humidity_delta,
            "wind_dir": cur1.get("wind_dir", "\u2014"),
            "feelslike": cur1.get("feelslike_c", var_temp),
            "vis": cur1.get("vis_km", "\u2014"),
            "uv": cur1.get("uv", "\u2014"),
            "cloud": cur1.get("cloud", "\u2014"),
            "condition": cur1["condition"]["text"],
            "lat": lat,
            "lon": lon,
            "elevation": get_elevation(lat, lon),
            "name": loc["name"],
            "country": loc["country"],
            "consensus": consensus_active,
        }

    except Exception as e:
        print("Primary Weather API Error:", e)
        return None


# ═══════════════════════════════════════════════════════════
# FORECAST
# ═══════════════════════════════════════════════════════════

def get_forecast(city: str) -> list:
    """Fetch 3-day hourly forecast (precip, humidity)."""
    try:
        r = requests.get(
            f"https://api.weatherapi.com/v1/forecast.json"
            f"?key={WEATHER_API_KEY}&q={city}&days=3",
            timeout=8,
        )
        if r.status_code != 200:
            return []
        hours = []
        for day in r.json().get("forecast", {}).get("forecastday", []):
            for h in day.get("hour", []):
                hours.append({
                    "time": h["time"][-5:],
                    "precip": h["precip_mm"],
                    "humidity": h["humidity"],
                })
        return hours
    except Exception as e:
        print("Forecast Error:", e)
        return []


