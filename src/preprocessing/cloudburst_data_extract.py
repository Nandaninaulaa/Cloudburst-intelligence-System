import h5py, os
import numpy as np
import pandas as pd
from tqdm import tqdm

# === path to your data folder ===
data_dir = r"C:\Users\bohra\Downloads\GPM_3IMERGHH_07-20251111_162614"  # change if needed

# === Uttarakhand lat/lon boundaries ===
LAT_MIN, LAT_MAX = 28.7, 31.5
LON_MIN, LON_MAX = 77.0, 81.0

all_records = []

files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".HDF5")])

for fpath in tqdm(files[:1000]):  # test first 1000; remove slice later
    try:
        with h5py.File(fpath, "r") as f:
            rain = f["Grid/precipitationCal"][:]
            lat  = f["Grid/lat"][:]
            lon  = f["Grid/lon"][:]
        # Filter indices for Uttarakhand
        lat_mask = (lat >= LAT_MIN) & (lat <= LAT_MAX)
        lon_mask = (lon >= LON_MIN) & (lon <= LON_MAX)
        rain_utt = rain[:, lat_mask, :][:, :, lon_mask]
        mean_rain = np.nanmean(rain_utt)
        # Extract date/time from filename (e.g. 3B-HHR.MS.MRG.3IMERG.20250713-S233000-E235959.HDF5)
        name = os.path.basename(fpath)
        timestamp = name.split('.')[4][1:]  # e.g., 20250713-S233000
        all_records.append({'file': name, 'rainfall_mm_hr': mean_rain, 'timestamp': timestamp})
    except Exception as e:
        print("Error reading", fpath, e)

# Convert to DataFrame
df = pd.DataFrame(all_records)
df.to_csv("uttarakhand_halfhourly.csv", index=False)
print("✅ Saved:", len(df), "records")
