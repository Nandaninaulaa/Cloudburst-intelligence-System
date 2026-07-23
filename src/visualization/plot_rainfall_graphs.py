import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========== STEP 1: Load dataset ==========
df = pd.read_csv("uttarakhand_precip_final.csv")

print("Loaded:", df.shape)

# Convert Date
df['Raw_Date'] = df['Date'].str.extract(r'3IMERG\.(\d{8})')
df['Date'] = pd.to_datetime(df['Raw_Date'], format="%Y%m%d")
df.drop(columns=['Raw_Date'], inplace=True)

# ========== STEP 2: Daily Rainfall ==========
daily_rain = df.groupby("Date")['Precipitation'].sum()

plt.figure(figsize=(12,5))
plt.plot(daily_rain.index, daily_rain.values)
plt.title("Daily Rainfall - Uttarakhand")
plt.xlabel("Date")
plt.ylabel("Rainfall (mm)")
plt.grid(True)
plt.tight_layout()
plt.savefig("daily_rainfall_plot.png")
plt.close()

print("Saved: daily_rainfall_plot.png")

# ========== STEP 3: Hourly Grid Heatmap ==========
# Mean rainfall for each grid cell (lat-lon)
grid = df.groupby(['Latitude', 'Longitude'])['Precipitation'].mean().unstack()

plt.figure(figsize=(10,7))
plt.imshow(grid, aspect='auto')
plt.title("Rainfall Heatmap (Lat vs Lon)")
plt.xlabel("Longitude Index")
plt.ylabel("Latitude Index")
plt.colorbar(label="Rainfall (mm)")
plt.tight_layout()
plt.savefig("rainfall_heatmap.png")
plt.close()

print("Saved: rainfall_heatmap.png")

# ========== STEP 4: Rainfall Histogram ==========
plt.figure(figsize=(8,5))
plt.hist(df["Precipitation"], bins=50)
plt.title("Rainfall Distribution")
plt.xlabel("Rainfall (mm)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.savefig("rainfall_histogram.png")
plt.close()

print("Saved: rainfall_histogram.png")

print("\nAll graphs generated successfully!")
