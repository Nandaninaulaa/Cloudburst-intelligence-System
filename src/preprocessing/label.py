import pandas as pd

# Load the hourly file created earlier
hourly = pd.read_csv("uttarakhand_hourly.csv")

# Convert datetime column
hourly["datetime"] = pd.to_datetime(hourly["datetime"])

# Sort by time (important)
hourly = hourly.sort_values("datetime")

# Define cloudburst threshold (100 mm/hr or your chosen value)
THRESHOLD = 100

# Create label: 1 → cloudburst, 0 → normal
hourly["label"] = (hourly["precip_hour"] >= THRESHOLD).astype(int)

# Save labeled dataset
hourly.to_csv("ml_dataset.csv", index=False)

print("DONE — Labels created and saved to ml_dataset.csv")

