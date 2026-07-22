import pandas as pd
import numpy as np

# ===== STEP 1: Load Uttarakhand rainfall data =====
df = pd.read_csv("uttarakhand_precip_final.csv")

print("Dataset loaded:", df.shape)
print(df.head())

# ===== STEP 2: Extract and convert date =====
df['Raw_Date'] = df['Date'].str.extract(r'3IMERG\.(\d{8})')  # extract YYYYMMDD

df['Date'] = pd.to_datetime(df['Raw_Date'], format='%Y%m%d')
df.drop(columns=['Raw_Date'], inplace=True)

print("\nConverted dates:")
print(df.head())

# ===== STEP 3: Filter invalid rainfall =====
df = df[df['Precipitation'] >= 0]

# ===== STEP 4: Create hourly rainfall sum =====
hourly = df.groupby(['Date', 'Latitude', 'Longitude'])['Precipitation'].sum().reset_index()

print("\nHourly Data:")
print(hourly.head())

# ===== STEP 5: Detect cloudburst =====
# IMD threshold: 100 mm/hour continuous intense rainfall
cloudburst_events = hourly[hourly['Precipitation'] >= 100]

print("\nCloudburst Events Detected:")
print(cloudburst_events)

# ===== STEP 6: Save results =====
cloudburst_events.to_csv("uttarakhand_cloudburst_detected.csv", index=False)

print("\nCloudburst detection completed!")
print("Saved file: uttarakhand_cloudburst_detected.csv")
