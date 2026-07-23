"""
CloudBurst — Persistent SQLite Database Module
═══════════════════════════════════════════════
Handles prediction logging, watchlist management,
and data persistence for the Streamlit dashboard.

Exported Functions
──────────────────
init_db()                         — Create tables if not exist
save_prediction(...)              — Log a prediction entry
get_recent_logs(limit=20)         — Fetch recent predictions
clear_logs()                      — Delete all prediction logs
add_watchlist_location(...)       — Add a location to monitor
remove_watchlist_location(name)   — Remove a location by name
get_watchlist()                   — List all monitored locations
"""

import sqlite3
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

# ── Database file lives next to this module ──────────────
DB_DIR = Path(__file__).resolve().parent
DB_PATH = DB_DIR / "cloudburst_logs.db"


# ═══════════════════════════════════════════════════════════
# CONNECTION HELPERS
# ═══════════════════════════════════════════════════════════

def _get_conn() -> sqlite3.Connection:
    """Open (or create) the database and return a connection."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


# ═══════════════════════════════════════════════════════════
# TABLE SCHEMA
# ═══════════════════════════════════════════════════════════

def init_db() -> None:
    """Create tables if they do not already exist."""
    conn = _get_conn()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS predictions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                time        TEXT    NOT NULL DEFAULT (datetime('now','localtime')),
                location    TEXT    NOT NULL,
                rainfall    TEXT    NOT NULL,
                risk        TEXT    NOT NULL,
                status      TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS watchlist (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                name    TEXT    NOT NULL UNIQUE,
                lat     REAL    NOT NULL,
                lon     REAL    NOT NULL,
                added   TEXT    NOT NULL DEFAULT (datetime('now','localtime'))
            );

            CREATE INDEX IF NOT EXISTS idx_pred_time
                ON predictions(time DESC);
        """)
        conn.commit()
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════
# PREDICTION LOGGING
# ═══════════════════════════════════════════════════════════

def save_prediction(
    location: str,
    rainfall: str,
    risk: str,
    status: str,
) -> None:
    """Insert a new prediction record into the database.

    Parameters
    ----------
    location : str   — e.g. "Dehradun, India"
    rainfall : str   — e.g. "45.2 mm"
    risk     : str   — e.g. "72.3%"
    status   : str   — e.g. "HIGH" or "LOW"
    """
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO predictions (location, rainfall, risk, status) "
            "VALUES (?, ?, ?, ?)",
            (location, rainfall, risk, status),
        )
        conn.commit()
    finally:
        conn.close()


def get_recent_logs(limit: int = 20) -> List[Dict]:
    """Return the *limit* most recent predictions as dicts."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT time, location, rainfall, risk, status "
            "FROM predictions ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def clear_logs() -> None:
    """Delete every prediction record."""
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM predictions")
        conn.commit()
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════
# WATCHLIST MANAGEMENT
# ═══════════════════════════════════════════════════════════

def add_watchlist_location(
    name: str,
    lat: float,
    lon: float,
) -> bool:
    """Add a location to the monitoring watchlist.

    Returns True if inserted, False if the name already exists.
    """
    conn = _get_conn()
    try:
        conn.execute(
            "INSERT INTO watchlist (name, lat, lon) VALUES (?, ?, ?)",
            (name, lat, lon),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def remove_watchlist_location(name: str) -> None:
    """Remove a location from the watchlist by name."""
    conn = _get_conn()
    try:
        conn.execute("DELETE FROM watchlist WHERE name = ?", (name,))
        conn.commit()
    finally:
        conn.close()


def get_watchlist() -> List[Dict]:
    """Return every location currently on the watchlist."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT name, lat, lon FROM watchlist ORDER BY added ASC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()

