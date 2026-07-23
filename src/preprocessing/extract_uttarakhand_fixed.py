import h5py
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# === CONFIG ===
data_dir = r"C:\Users\bohra\Downloads\GPM_3IMERGHH_07-20251111_162614" # your folder path
output_csv = "uttarakhand_precip_final.csv"

# Uttarakhand bounding box (approx)
lat_min, lat_max = 28.5, 31.5
lon_min, lon_max = 77.5, 81.5

all_data = []

for filename in tqdm(os.listdir(data_dir)):
    if not filename.endswith(".HDF5") and not filename.endswith(".h5"):
        continue

    file_path = os.path.join(data_dir, filename)

    with h5py.File(file_path, 'r') as f:
        grid = f['Grid']
        lat = grid['lat'][:]
        lon = grid['lon'][:]
        precip = grid['precipitation'][0, :, :]  # shape (3600, 1800)

        # Create lon-lat grid
        lon_grid, lat_grid = np.meshgrid(lon, lat)
        precip = precip.T  # align dimensions

        # Apply geographic mask for Uttarakhand
        mask = (lat_grid >= lat_min) & (lat_grid <= lat_max) & \
               (lon_grid >= lon_min) & (lon_grid <= lon_max)

        sub_lat = lat_grid[mask]
        sub_lon = lon_grid[mask]
        sub_precip = precip[mask]

        if sub_precip.size > 0:
            date = os.path.splitext(filename)[0]  # extract from file name
            for la, lo, pr in zip(sub_lat, sub_lon, sub_precip):
                all_data.append((date, la, lo, pr))

# Save all extracted data
if all_data:
    df = pd.DataFrame(all_data, columns=["Date", "Latitude", "Longitude", "Precipitation"])
    df.to_csv(output_csv, index=False)
    print(f"✅ Extracted {len(df)} data points and saved to {output_csv}")
else:
    print("⚠️ Still no data found — check data folder or bounding box.")
