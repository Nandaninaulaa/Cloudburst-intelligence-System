"""
CloudBurst — Model Loading & Ensemble Prediction
═══════════════════════════════════════════════════
Loads RF, XGBoost, Stacking, and LSTM models.
Provides the ensemble-based cloudburst prediction pipeline.
"""

import os
import time
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional, Tuple

from config import MODEL_DIR, calculate_confidence

# ── Lazy-load flags ─────────────────────────────────────
_rf_model: Optional[object] = None
_lstm_model: Optional[object] = None
_lstm_scaler: Optional[object] = None
_xgb_model: Optional[object] = None
_stacking_model: Optional[object] = None


# ═══════════════════════════════════════════════════════════
# MODEL LOADING (cached via @st.cache_resource in app.py)
# ═══════════════════════════════════════════════════════════

def load_rf() -> object:
    global _rf_model
    if _rf_model is None:
        path = MODEL_DIR / "rf_cloudburst_model.pkl"
        _rf_model = joblib.load(path)
    return _rf_model


def load_xgb() -> Optional[object]:
    global _xgb_model
    if _xgb_model is not None:
        return _xgb_model
    path = MODEL_DIR / "xgb_cloudburst_model.pkl"
    if path.exists():
        try:
            _xgb_model = joblib.load(path)
        except Exception as e:
            print(f"XGB Load Error: {e}")
    return _xgb_model


def load_stacking() -> Optional[object]:
    global _stacking_model
    if _stacking_model is not None:
        return _stacking_model
    path = MODEL_DIR / "stacking_cloudburst_model.pkl"
    if path.exists():
        try:
            _stacking_model = joblib.load(path)
        except Exception as e:
            print(f"Stacking Load Error: {e}")
    return _stacking_model


def load_lstm() -> Tuple[Optional[object], Optional[object]]:
    global _lstm_model, _lstm_scaler
    if _lstm_model is not None:
        return _lstm_model, _lstm_scaler
    lstm_path   = MODEL_DIR / "cloudburst_lstm_model.keras"
    scaler_path = MODEL_DIR / "lstm_scaler.pkl"
    if lstm_path.exists():
        try:
            from tensorflow.keras.models import load_model
            import tensorflow as tf
            tf.get_logger().setLevel("ERROR")
            _lstm_model  = load_model(lstm_path)
            _lstm_scaler = joblib.load(scaler_path)
        except Exception as e:
            print(f"LSTM Load Error: {e}")
    return _lstm_model, _lstm_scaler


def load_all_models():
    """Load all models and return (rf, lstm, lstm_scaler, xgb, stacking)."""
    rf   = load_rf()
    xgb  = load_xgb()
    stack = load_stacking()
    lstm, scaler = load_lstm()
    return rf, lstm, scaler, xgb, stack


# ═══════════════════════════════════════════════════════════
# PREDICTION ENGINE
# ═══════════════════════════════════════════════════════════

def build_dataframe(
    lat: float, lon: float, base: float,
    year: int, month: int, day: int, hour: int,
    lag1: float, lag2: float, lag3: float,
    humidity: float, pressure: float, wind: float,
    temp: float, pressure_delta: float, humidity_delta: float,
) -> pd.DataFrame:
    """Build the feature DataFrame in the order the models expect."""
    return pd.DataFrame(
        [[lat, lon, base, year, month, day, hour,
          lag1, lag2, lag3, humidity, pressure, wind, temp,
          pressure_delta, humidity_delta]],
        columns=[
            "Latitude","Longitude","precip_hour","year","month","day","hour",
            "rain_lag1","rain_lag2","rain_lag3","humidity","pressure","wind",
            "temp","pressure_delta","humidity_delta",
        ],
    )


def ensemble_predict(
    X: pd.DataFrame,
    rf_model: object,
    lstm_model: object = None,
    lstm_scaler: object = None,
    xgb_model: object = None,
    stacking_model: object = None,
    history: list = None,
    sidebar_placeholder = None,
) -> dict:
    """
    Run the full ensemble prediction pipeline.

    Returns a dict with:
        rf_prob, xgb_prob, lstm_prob, stacking_prob,
        ml_prob, probability (hybrid), rule_risk,
        conf_score, conf_label, conf_color,
        latency_ms, model_size_mb
    """
    base = X["precip_hour"].iloc[0]
    lag1 = X["rain_lag1"].iloc[0]
    lag2 = X["rain_lag2"].iloc[0]
    lag3 = X["rain_lag3"].iloc[0]

    # RF
    rf_prob = rf_model.predict_proba(X)[0][1] * 100

    # XGB
    if xgb_model is not None:
        xgb_prob = xgb_model.predict_proba(X)[0][1] * 100
    else:
        xgb_prob = rf_prob

    # LSTM
    lstm_prob = None
    if lstm_model is not None and lstm_scaler is not None:
        lstm_inputs = np.array([[lag3, lag2, lag1, base]])
        lstm_scaled = lstm_scaler.transform(lstm_inputs)
        lstm_3d     = lstm_scaled.reshape(1, 4, 1)
        lstm_prob   = float(lstm_model.predict(lstm_3d, verbose=0)[0][0]) * 100

    # Stacking
    t0 = time.time()
    stacking_prob = None
    if stacking_model is not None:
        stacking_prob = stacking_model.predict_proba(X)[0][1] * 100
    latency_ms = (time.time() - t0) * 1000

    model_size_mb = 0.0
    try:
        model_size_mb = os.path.getsize(MODEL_DIR / "stacking_cloudburst_model.pkl") / (1024 * 1024)
    except Exception:
        pass

    # ── ML Ensemble Weighted Average ────────────────────
    if lstm_prob is not None:
        if stacking_prob is not None:
            ml_prob = 0.50 * lstm_prob + 0.50 * stacking_prob
            _msg = (f"Meta-Learner Ensemble Active "
                    f"(Stacking: {stacking_prob:.0f}% | Attn-LSTM: {lstm_prob:.0f}%)")
        elif xgb_model is not None:
            ml_prob = 0.40 * lstm_prob + 0.30 * xgb_prob + 0.30 * rf_prob
            _msg = (f"Hybrid Ensemble Active "
                    f"(LSTM: {lstm_prob:.0f}% | XGB: {xgb_prob:.0f}% | RF: {rf_prob:.0f}%)")
        else:
            ml_prob = 0.60 * lstm_prob + 0.40 * rf_prob
            _msg = f"LSTM Ensemble Active (LSTM: {lstm_prob:.0f}% | RF: {rf_prob:.0f}%)"
    else:
        if stacking_prob is not None:
            ml_prob = stacking_prob
            _msg = f"Stacking Meta-Learner Active (Stacking: {stacking_prob:.0f}%)"
        elif xgb_model is not None:
            ml_prob = 0.50 * xgb_prob + 0.50 * rf_prob
            _msg = f"RF + XGB Active (XGB: {xgb_prob:.0f}% | RF: {rf_prob:.0f}%)"
        else:
            ml_prob = rf_prob
            _msg = "LSTM Model Training... Using RF mode"

    if sidebar_placeholder is not None:
        sidebar_placeholder.markdown(
            f"<div style='font-size:0.75rem; color:#059669; padding-top:10px;'>{_msg}</div>",
            unsafe_allow_html=True,
        )

    if sidebar_placeholder is not None:
        inference_info = (
            f"<div style='font-size:0.75rem; color:#64748b; padding-top:15px; "
            f"border-top:1px solid #e2e8f0; margin-top:15px;'>"
            f"\u23f1 Inference Time: {latency_ms:.1f} ms | "
            f"\U0001f4e6 Model Size: {model_size_mb:.1f} MB</div>"
        )
        sidebar_placeholder.markdown(inference_info, unsafe_allow_html=True)

    # ── Rule-Based Risk ─────────────────────────────────
    rule_risk = 0
    humidity_val = X["humidity"].iloc[0]
    pressure_val = X["pressure"].iloc[0]
    wind_val     = X["wind"].iloc[0]
    temp_val     = X["temp"].iloc[0]
    p_delta      = X["pressure_delta"].iloc[0]
    h_delta      = X["humidity_delta"].iloc[0]

    if base > 150: rule_risk += 50
    elif base > 100: rule_risk += 30
    elif base > 50: rule_risk += 15
    if humidity_val > 95: rule_risk += 15
    elif humidity_val > 85: rule_risk += 10
    if pressure_val < 1000: rule_risk += 10
    if wind_val > 8: rule_risk += 5
    rule_risk = min(rule_risk, 100)

    # ── Hybrid Final ────────────────────────────────────
    probability = 0.6 * ml_prob + 0.4 * rule_risk
    if base > 150 and humidity_val > 90:
        probability = max(probability, 90.0)

    conf_score, conf_label, conf_color = calculate_confidence(
        rf_prob,
        xgb_prob if xgb_model is not None else None,
        stacking_prob,
        lstm_prob if (lstm_model is not None and lstm_scaler is not None) else None,
    )

    return {
        "rf_prob": rf_prob,
        "xgb_prob": xgb_prob,
        "lstm_prob": lstm_prob,
        "stacking_prob": stacking_prob,
        "ml_prob": ml_prob,
        "probability": round(probability, 1),
        "rule_risk": rule_risk,
        "conf_score": conf_score,
        "conf_label": conf_label,
        "conf_color": conf_color,
        "latency_ms": latency_ms,
        "model_size_mb": model_size_mb,
        "ensemble_msg": _msg,
    }


def predict_lead_time(
    forecast: list,
    now: object,
    lat: float, lon: float,
    base: float, lag1: float, lag2: float, lag3: float,
    humidity: float, pressure: float, wind: float, temp: float,
    rf_model: object, lstm_model: object = None,
    lstm_scaler: object = None, stacking_model: object = None,
    xgb_model: object = None,
) -> int:
    """Check forecast hours to see when risk exceeds 75%.

    Returns the number of hours ahead (lead_time), or None.
    """
    if not forecast or stacking_model is None:
        return None

    for i, f_hr in enumerate(forecast[:12]):
        f_precip = f_hr.get("precip", 0)
        f_humid  = f_hr.get("humidity", humidity)
        future_X = build_dataframe(
            lat, lon, f_precip,
            now.year, now.month, now.day, now.hour + i + 1,
            base, lag1, lag2, f_humid,
            pressure - 2, wind, temp, -2.0, 5.0,
        )
        if lstm_model is not None and lstm_scaler is not None:
            f_lstm_inputs = np.array([[lag2, lag1, base, f_precip]])
            f_lstm_scaled = lstm_scaler.transform(f_lstm_inputs)
            f_lstm_3d = f_lstm_scaled.reshape(1, 4, 1)
            f_lstm_prob = lstm_model.predict(f_lstm_3d, verbose=0)[0][0]
        else:
            f_lstm_prob = rf_model.predict_proba(future_X)[0][1]

        f_prob = stacking_model.predict_proba(future_X)[0][1] * 100
        if f_prob > 75:
            return i + 1
    return None

