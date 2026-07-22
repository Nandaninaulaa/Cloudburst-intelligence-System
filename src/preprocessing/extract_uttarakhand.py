import h5py
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# === Path to your downloaded folder ===
folder = r"C:\Users\bohra\Downloads\GPM_3IMERGHH_07-20251111_162614"
output_csv = "uttarakhand_precip_final.csv"

# === Uttarakhand bounding box ===
lat_min, lat_max = 28.5, 31.5
lon_min, lon_max = 77.0, 81.0

data_rows = []

# === Loop through all HDF5 files ===
for file in tqdm([f for f in os.listdir(folder) if f.endswith(".HDF5")]):
    path = os.path.join(folder, file)
    with h5py.File(path, 'r') as f:
        lat = f["Grid/lat"][:]
        lon = f["Grid/lon"][:]
        precip = f["Grid/precipitation"][:]  # ✅ correct dataset key

        # Make sure lon and lat are increasing arrays
        lat = np.array(lat)
        lon = np.array(lon)

        # Create masks for Uttarakhand region
        lat_mask = (lat >= lat_min) & (lat <= lat_max)
        lon_mask = (lon >= lon_min) & (lon <= lon_max)

        # Skip if no data in range
        if not np.any(lat_mask) or not np.any(lon_mask):
            continue

        # Extract subset
        sub_data = precip[0, lon_mask, :][:, lat_mask]  # (lon, lat)
        avg_precip = np.nanmean(sub_data)

        data_rows.append({
            "File": file,
            "AvgPrecip": avg_precip
        })

# === Save output ===
if data_rows:
    df = pd.DataFrame(data_rows)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved {len(df)} rows to {output_csv}")
else:
    print("⚠️ No data found in Uttarakhand region — try slightly expanding bounding box.")
