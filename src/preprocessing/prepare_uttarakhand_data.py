import pandas as pd
import numpy as np

# --- STEP 1: Load your extracted rainfall file ---
df = pd.read_csv("uttarakhand_precip_final.csv")

print("Loaded data:")
print(df.head())

# --- STEP 2: Extract date from filename (first column) ---
# Example filename: 3B-HHR.MS.MRG.3IMERG.20210101-S003000
df['Raw_Date'] = df['Date'].str.extract(r'3IMERG\.(\d{8})')

# Convert to datetime
df['Date'] = pd.to_datetime(df['Raw_Date'], format='%Y%m%d')

# Drop unused column
df = df.drop(columns=['Raw_Date'])

print("\nAfter converting date:")
print(df.head())

# --- STEP 3: Clean precipitation ---
df = df[df['Precipitation'] >= 0]   # remove invalid values

# --- STEP 4: Group hourly rainfall ---
hourly = df.groupby(['Date', 'Latitude', 'Longitude'])['Precipitation'].sum().reset_index()

print("\nHourly rainfall:")
print(hourly.head())

# --- STEP 5: Detect cloudburst (IMD threshold 100 mm/hr) ---
cloudburst = hourly[hourly['Precipitation'] > 100]

print("\nDetected cloudburst events:")
print(cloudburst)

# --- STEP 6: Prepare for LSTM ---
# Region-wide daily rainfall
daily = df.groupby('Date')['Precipitation'].sum().reset_index()

# Create sequences of last 24 hours
def create_sequences(data, seq_len=24):
    x, y = [], []
    for i in range(len(data) - seq_len):
        x.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(x), np.array(y)

daily_values = daily['Precipitation'].values
X, y = create_sequences(daily_values)

print("\nShapes for LSTM:")
print("X:", X.shape)
print("y:", y.shape)

# Save cleaned files
hourly.to_csv("uttarakhand_hourly_cleaned.csv", index=False)
daily.to_csv("uttarakhand_daily_cleaned.csv", index=False)

print("\nSaved cleaned datasets successfully.")
