"""
CloudBurst — Configuration & Shared Helpers
═══════════════════════════════════════════════
All constants, environment setup, CSS theme,
helper functions (risk colors, badges, etc.)
and the SCENARIOS dictionary.
"""

import os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# ── Base paths ───────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent          # src/prediction/
MODEL_DIR = BASE_DIR.parent.parent / "model"         # cloudburst/model/
DATA_DIR  = BASE_DIR.parent.parent / "data"          # cloudburst/data/

# ── Load .env from model/ (fallback: root) ──────────────
_env_file = MODEL_DIR / ".env"
if not _env_file.exists():
    _env_file = BASE_DIR.parent.parent / ".env"
load_dotenv(dotenv_path=_env_file)

# ── API keys & credentials ──────────────────────────────
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")
SENDER_EMAIL    = os.getenv("SENDER_EMAIL", "")
SENDER_PASS     = os.getenv("SENDER_PASS", "")
TWILIO_SID      = os.getenv("TWILIO_SID", "")
TWILIO_AUTH     = os.getenv("TWILIO_AUTH", "")
TWILIO_PHONE    = os.getenv("TWILIO_PHONE", "")

# ── Feature names (RF / SHAP) ───────────────────────────
FEATURE_NAMES = [
    "Current Rainfall", "Rainfall \u20131 hr", "Rainfall \u20132 hrs",
    "Rainfall \u20133 hrs", "Year", "Month", "Day", "Hour",
    "Latitude", "Longitude",
]

FEATURE_NAMES_RF = [
    "Latitude", "Longitude", "Precip (1hr)", "Year", "Month",
    "Day", "Hour", "Lag 1hr", "Lag 2hr", "Lag 3hr",
    "Humidity", "Pressure", "Wind", "Temp",
    "Pressure Drop", "Humidity Spike",
]

# ── Scenarios ────────────────────────────────────────────
SCENARIOS = {
    "\U0001f534  Kedarnath 2013 \u2014 Extreme Cloudburst": {
        "history": [10, 25, 60, 180], "humidity": 98, "temp": 10,
        "pressure": 995, "wind": 10, "pressure_delta": -9.2,
        "humidity_delta": 22.5, "lat": 30.7346, "lon": 79.0669,
        "elevation": 3583, "wind_dir": "SW", "feelslike": 7,
        "vis": 0.5, "uv": 1, "cloud": 100, "condition": "Heavy rain",
        "desc": "One of India's deadliest cloudburst events \u2014 Uttarakhand floods.",
        "region": "Kedarnath, Uttarakhand",
    },
    "\U0001f7e0  Simulated Cloudburst \u2014 Rapid Spike": {
        "history": [5, 20, 45, 150], "humidity": 95, "temp": 12,
        "pressure": 998, "wind": 8, "pressure_delta": -7.5,
        "humidity_delta": 18.0, "lat": 30.0, "lon": 79.0,
        "elevation": 1500, "wind_dir": "W", "feelslike": 10,
        "vis": 1, "uv": 1, "cloud": 95, "condition": "Torrential rain",
        "desc": "Rainfall spikes 5\u2192150 mm in 3 hours \u2014 classic cloudburst signature.",
        "region": "Simulated, Himalayan foothills",
    },
    "\U0001f7e1  Sustained Heavy Rainfall": {
        "history": [15, 30, 55, 80], "humidity": 88, "temp": 18,
        "pressure": 1005, "wind": 6, "pressure_delta": -2.0,
        "humidity_delta": 5.0, "lat": 29.9, "lon": 78.8,
        "elevation": 450, "wind_dir": "SE", "feelslike": 17,
        "vis": 3, "uv": 2, "cloud": 80, "condition": "Heavy rain",
        "desc": "Continuous heavy rainfall over 4 hours \u2014 monitoring zone.",
        "region": "Haridwar, Uttarakhand",
    },
    "\U0001f7e2  Normal / Clear Conditions": {
        "history": [2, 3, 4, 5], "humidity": 60, "temp": 25,
        "pressure": 1013, "wind": 2, "pressure_delta": 0.5,
        "humidity_delta": -2.0, "lat": 28.6, "lon": 77.2,
        "elevation": 216, "wind_dir": "N", "feelslike": 25,
        "vis": 10, "uv": 5, "cloud": 20, "condition": "Partly cloudy",
        "desc": "Stable atmospheric conditions. No imminent risk.",
        "region": "Delhi, India",
    },
    "\U0001f535  Monsoon Onset \u2014 Pre-cloudburst Watch": {
        "history": [8, 18, 35, 70], "humidity": 92, "temp": 22,
        "pressure": 1001, "wind": 12, "pressure_delta": -4.0,
        "humidity_delta": 12.0, "lat": 32.2, "lon": 77.1,
        "elevation": 2050, "wind_dir": "SW", "feelslike": 23,
        "vis": 2, "uv": 1, "cloud": 90, "condition": "Overcast, moderate rain",
        "desc": "Rapidly rising humidity and pressure drop \u2014 monsoon onset watch.",
        "region": "Kullu, Himachal Pradesh",
    },
}


# ═══════════════════════════════════════════════════════════
# RISK HELPERS
# ═══════════════════════════════════════════════════════════

def risk_color(p: float) -> str:
    if p > 85: return "#dc2626"
    if p > 65: return "#ea580c"
    if p > 40: return "#d97706"
    return "#059669"


def risk_label(p: float) -> str:
    if p > 85: return "SEVERE"
    if p > 65: return "HIGH"
    if p > 40: return "MODERATE"
    return "LOW"


def risk_badge(p: float) -> str:
    lbl = risk_label(p)
    bg  = {"SEVERE":"#fee2e2","HIGH":"#fff7ed",
           "MODERATE":"#fef3c7","LOW":"#d1fae5"}[lbl]
    col = risk_color(p)
    return (
        f'<span style="background:{bg};color:{col};font-size:0.68rem;'
        f'font-weight:700;letter-spacing:0.1em;padding:0.2rem 0.65rem;'
        f'border-radius:99px;">{lbl}</span>'
    )


def bar_gradient(p: float) -> str:
    if p > 85: return "linear-gradient(90deg,#ea580c,#dc2626)"
    if p > 65: return "linear-gradient(90deg,#d97706,#ea580c)"
    if p > 40: return "linear-gradient(90deg,#059669,#d97706)"
    return "linear-gradient(90deg,#2563eb,#059669)"


def calculate_confidence(
    rf_p: float,
    xgb_p: float = None,
    stack_p: float = None,
    lstm_p: float = None,
):
    """Ensemble confidence score based on prediction spread."""
    preds = []
    if rf_p is not None: preds.append(rf_p)
    if xgb_p is not None: preds.append(xgb_p)
    if stack_p is not None: preds.append(stack_p)
    if lstm_p is not None: preds.append(lstm_p)

    if len(preds) > 1:
        std_val = float(np.std(preds))
        conf = max(10.0, 100.0 - (std_val * 2.5))
    else:
        conf = 85.0

    if conf > 80:
        label = "High"
        color = "#059669"
    elif conf > 50:
        label = "Moderate"
        color = "#d97706"
    else:
        label = "Low"
        color = "#dc2626"

    return round(conf, 1), label, color


# ═══════════════════════════════════════════════════════════
# GLOBAL CSS
# ═══════════════════════════════════════════════════════════

GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Fraunces:ital,wght@0,700;1,400&display=swap');

:root {
    --primary: #2563eb;
    --sidebar: #1e3a5f;
    --bg-light: #f4f7fc;
    --card: #ffffff;
    --text-main: #0f172a;
    --text-muted: #64748b;
    --border: #e2e8f0;
}

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif !important;
    color: var(--text-main) !important;
}

.stApp {
    background: var(--bg-light) !important;
    background-image:
        radial-gradient(circle at 15% 50%, rgba(37,99,235,0.04) 0%, transparent 50%),
        radial-gradient(circle at 85% 30%, rgba(14,165,233,0.04) 0%, transparent 50%) !important;
}

[data-testid="stSidebar"] {
    background: var(--sidebar) !important;
    box-shadow: 2px 0 15px rgba(0,0,0,0.05) !important;
    border-right: none !important;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
[data-testid="stSidebar"] label { color: #94a3b8 !important; }

div[style*="background:#fff"],
div[style*="background:#ffffff"],
div[style*="background-color:#fff"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-main) !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.02), 0 1px 3px rgba(0,0,0,0.02) !important;
    transition: transform 0.3s cubic-bezier(0.175,0.885,0.32,1.275), box-shadow 0.3s ease !important;
}

div[style*="background:#fff"]:hover {
    transform: translateY(-4px) !important;
    box-shadow: 0 12px 24px rgba(37,99,235,0.08), 0 4px 8px rgba(37,99,235,0.04) !important;
    border-color: rgba(37,99,235,0.3) !important;
}

div[style*="color:#0f172a"],
div[style*="color:#334155"],
h3, h4 { color: #1e293b !important; }
div[style*="color:#475569"] { color: #475569 !important; }

[data-baseweb="tab-list"] {
    background: #ffffff !important;
    border-radius: 12px; padding: 6px; gap: 4px;
    border: 1px solid var(--border);
    box-shadow: 0 2px 5px rgba(0,0,0,0.02) !important;
}
[data-baseweb="tab"] {
    border-radius: 8px !important;
    padding: 0.5rem 1.4rem !important;
    color: var(--text-muted) !important;
    border: none !important;
    background: transparent !important;
    transition: all 0.25s ease !important;
    font-weight: 500 !important;
}
[aria-selected="true"] {
    background: #eff6ff !important;
    color: var(--primary) !important;
    border: 1px solid #bfdbfe !important;
    box-shadow: 0 2px 4px rgba(37,99,235,0.05) !important;
    font-weight: 600 !important;
}

.stButton > button {
    background: var(--primary) !important;
    color: white !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    border-radius: 8px !important;
    border: none !important;
    box-shadow: 0 2px 8px rgba(37,99,235,0.2) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 15px rgba(37,99,235,0.35) !important;
    background: #1d4ed8 !important;
}

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-thumb { background: #94a3b8; border-radius: 6px; }
</style>
"""

