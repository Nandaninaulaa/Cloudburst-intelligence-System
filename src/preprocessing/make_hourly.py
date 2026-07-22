import pandas as pd
import re

INPUT = "uttarakhand_precip_final.csv"     # your raw file
OUTPUT = "uttarakhand_hourly.csv"

def parse_datetime_from_filename(fname):
    m = re.search(r'(\d{8})-S(\d{6})', str(fname))
    if m:
        return pd.to_datetime(m.group(1)+m.group(2), format="%Y%m%d%H%M%S")
    m2 = re.search(r'(\d{8})', str(fname))
    if m2:
        return pd.to_datetime(m2.group(1), format="%Y%m%d")
    return None

df = pd.read_csv(INPUT)

df["timestamp"] = df["Date"].apply(parse_datetime_from_filename)
df = df.dropna(subset=["timestamp"])
df["timestamp"] = pd.to_datetime(df["timestamp"])

df["ts_30min"] = df["timestamp"].dt.floor("30min")
df["ts_hour"] = df["ts_30min"].dt.floor("H")

hourly = df.groupby(["Latitude","Longitude","ts_hour"])["Precipitation"].sum().reset_index()
hourly = hourly.rename(columns={"ts_hour":"datetime", "Precipitation":"precip_hour"})

hourly.to_csv(OUTPUT, index=False)

print("DONE — 'uttarakhand_hourly.csv' CREATED.")
