"""
CloudBurst — Prediction Engine (Watchlist & Single-point)
═══════════════════════════════════════════════════════════
Reusable prediction logic for the main dashboard and
multi-location watchlist surveillance.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional

from config import risk_label, calculate_confidence
from weather import get_weather
from models import (
    build_dataframe,
    ensemble_predict,
    load_all_models,
)

# ── Globals (lazy-loaded after Streamlit cache) ────────
_rf   = None
_lstm = None
_lstm_scaler = None
_xgb  = None
_stack = None


def set_models(rf, lstm, scaler, xgb, stack):
    global _rf, _lstm, _lstm_scaler, _xgb, _stack
    _rf = rf
    _lstm = lstm
    _lstm_scaler = scaler
    _xgb = xgb
    _stack = stack


def predict_for_location(
    name: str,
    lat: float,
    lon: float,
    now_dt: datetime = None,
) -> Optional[dict]:
    """Compute the hybrid cloudburst risk score for any (lat, lon) coordinate.

    Uses the same ensemble logic as the main prediction pipeline.
    Designed for single predictions, watchlist, and heatmap use.

    Returns dict with keys:
        name, lat, lon, probability, temp, humidity, precip,
        risk, wx, confidence_score, confidence_label, confidence_color
    or None if weather fetch fails.
    """
    if now_dt is None:
        now_dt = datetime.now()

    query = f"{lat},{lon}"
    wx = get_weather(query)
    if wx is None:
        return None

    base = wx["precip"]
    lag1 = lag2 = lag3 = 0.0  # No stored sequence for watchlist

    X = build_dataframe(
        lat, lon, base,
        now_dt.year, now_dt.month, now_dt.day, now_dt.hour,
        lag1, lag2, lag3,
        wx["humidity"], wx["pressure"], wx["wind"],
        wx["temp"], wx["pressure_delta"], wx["humidity_delta"],
    )

    result = ensemble_predict(
        X, _rf, _lstm, _lstm_scaler, _xgb, _stack,
        history=[lag3, lag2, lag1, base],
    )

    prob = result["probability"]
    conf_score = result["conf_score"]
    conf_label = result["conf_label"]
    conf_color = result["conf_color"]

    # Elevation-based rule boost (re-applied since ensemble_predict
    # doesn't have elevation context)
    elev = wx.get("elevation", 0) or 0
    rule_boost = 0
    if 1000 <= elev <= 3000:
        rule_boost = 12
    elif elev > 3000:
        rule_boost = 5
    prob = min(prob + (rule_boost * 0.4), 100.0)

    if base > 150 and wx["humidity"] > 90:
        prob = max(prob, 90.0)

    return {
        "name": name,
        "lat": lat,
        "lon": lon,
        "probability": round(prob, 1),
        "temp": wx["temp"],
        "humidity": wx["humidity"],
        "precip": base,
        "risk": risk_label(prob),
        "wx": wx,
        "confidence_score": conf_score,
        "confidence_label": conf_label,
        "confidence_color": conf_color,
    }

