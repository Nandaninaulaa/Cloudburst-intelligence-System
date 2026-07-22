# cloudburst_detection_ready.py

import re
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Optional plotting/map
try:
    import folium
    FOLIUM_AVAILABLE = True
except Exception:
    FOLIUM_AVAILABLE = False

# ---------- CONFIG ----------
INPUT_CSV = "uttarakhand_precip_final.csv"   # your extracted CSV
OUTPUT_EVENTS_CSV = "detected_cloudbursts.csv"
CLOUDMAP_HTML = "cloudburst_map.html"

# Detection thresholds (you can tune these)
HOURLY_THRESH = 50.0      # mm in 1 hour (set 100 for IMD strict)
THREE_HOUR_THRESH = 100.0 # mm in 3 hours
SPIKE_FACTOR = 4.0        # precip_hour > mean(prev_3_hours) * SPIKE_FACTOR

# ---------- HELPERS ----------
def parse_datetime_from_filename(fname):
    """
    Extract start date and start time from filename pattern like:
    ...3IMERG.20210101-S000000-E002959...
    Returns pandas.Timestamp (naive).
    """
    m = re.search(r'(\d{8})-S(\d{6})', fname)
    if not m:
        # fallback: extract YYYYMMDD only (set time 00:00)
        m2 = re.search(r'(\d{8})', fname)
        if m2:
            return pd.to_datetime(m2.group(1), format="%Y%m%d")
        return None
    date_part = m.group(1)
    time_part = m.group(2)
    ts = pd.to_datetime(date_part + time_part, format="%Y%m%d%H%M%S")
    return ts

# ---------- LOAD ----------
print("Loading CSV (this may take a bit)...")
df = pd.read_csv(INPUT_CSV)

# Expect columns: Date (filename), Latitude, Longitude, Precipitation
if 'Date' not in df.columns or 'Latitude' not in df.columns or 'Longitude' not in df.columns or 'Precipitation' not in df.columns:
    raise RuntimeError("CSV does not contain required columns: Date, Latitude, Longitude, Precipitation")

# ---------- PARSE TIMESTAMPS ----------
print("Parsing timestamps from filenames...")
df['timestamp'] = df['Date'].astype(str).apply(parse_datetime_from_filename)
# drop rows with no parsed timestamp
df = df.dropna(subset=['timestamp'])
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Floor to 30-min resolution if needed (IMERG is 30-min). Ensure consistent frequency
# We'll floor to minute 0/30 then later floor to hour for hourly sums.
df['ts_30min'] = df['timestamp'].dt.floor('30min')

# ---------- AGGREGATE to hourly per grid cell ----------
print("Aggregating to hourly sums per grid cell...")
# floor to hour
df['ts_hour'] = df['ts_30min'].dt.floor('H')

# group by lat/lon/hour and sum precipitation (this sums half-hourly values belonging to that hour)
hourly = df.groupby(['Latitude','Longitude','ts_hour'])['Precipitation'].sum().reset_index()
hourly = hourly.rename(columns={'ts_hour':'datetime','Precipitation':'precip_hour'})

# ---------- Build time-series per grid cell and compute 3-hour rolling ----------
print("Computing 3-hour rolling sums per grid cell (this may take a while)...")
events = []  # collect detected events rows

# We'll iterate per (lat,lon) group to compute rolling sums reliably
grouped = hourly.groupby(['Latitude','Longitude'])

for (lat, lon), g in tqdm(grouped, total=grouped.ngroups):
    # ensure sorted by datetime
    g = g.sort_values('datetime').set_index('datetime')
    # reindex to continuous hourly index (fills missing hours with 0)
    idx = pd.date_range(start=g.index.min(), end=g.index.max(), freq='H')
    s = g['precip_hour'].reindex(idx, fill_value=0.0)
    # compute 3-hour rolling sum (window=3 hours including current)
    roll3 = s.rolling(window=3, min_periods=1).sum()
    # compute mean of previous 3 hours (exclude current) for spike calc
    prev3_mean = s.shift(1).rolling(window=3, min_periods=1).mean().fillna(0.0)

    # detection rules
    mask_hourly = s >= HOURLY_THRESH
    mask_3h = roll3 >= THREE_HOUR_THRESH
    mask_spike = (s > prev3_mean * SPIKE_FACTOR) & (prev3_mean > 0)  # require previous mean >0 to avoid false spike

    # combine masks and save detected rows
    combined_mask = mask_hourly | mask_3h | mask_spike
    if combined_mask.any():
        detected_times = idx[combined_mask]
        for dt in detected_times:
            events.append({
                'Latitude': lat,
                'Longitude': lon,
                'datetime': dt,
                'precip_hour': float(s.loc[dt]),
                'precip_3h': float(roll3.loc[dt]),
                'prev3_mean': float(prev3_mean.loc[dt]),
                'rule_hourly': bool(mask_hourly.loc[dt]),
                'rule_3h': bool(mask_3h.loc[dt]),
                'rule_spike': bool(mask_spike.loc[dt])
            })

# ---------- SAVE RESULTS ----------
events_df = pd.DataFrame(events)
if events_df.empty:
    print("No cloudburst-like events detected with current thresholds.")
else:
    events_df = events_df.sort_values(['datetime','Latitude','Longitude'])
    events_df.to_csv(OUTPUT_EVENTS_CSV, index=False)
    print(f"Detected events saved to: {OUTPUT_EVENTS_CSV}")
    print(events_df.head())

# ---------- OPTIONAL: create a quick folium map of detections ----------
if FOLIUM_AVAILABLE and not events_df.empty:
    print("Creating folium map (cloudburst_map.html)...")
    # center map at mean coord
    center_lat = events_df['Latitude'].mean()
    center_lon = events_df['Longitude'].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=8)
    for _, r in events_df.iterrows():
        popup = f"{r['datetime']}<br>precip_hour={r['precip_hour']:.2f}<br>precip_3h={r['precip_3h']:.2f}"
        folium.CircleMarker(location=[r['Latitude'], r['Longitude']],
                            radius=4,
                            color='red',
                            fill=True,
                            fill_opacity=0.7,
                            popup=popup).add_to(m)
    m.save(CLOUDMAP_HTML)
    print(f"Map saved to {CLOUDMAP_HTML}. Open it in your browser to view detections.")
else:
    if not FOLIUM_AVAILABLE:
        print("Folium not installed — map not created. Install folium if you want a map.")
