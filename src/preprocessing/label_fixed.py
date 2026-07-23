import pandas as pd

# Load hourly file
df = pd.read_csv("uttarakhand_hourly.csv")

# Cloudburst threshold
CLOUDBURST_THRESHOLD = 100  # mm/hour

# Create label
df["label"] = (df["precip_hour"] >= CLOUDBURST_THRESHOLD).astype(int)

# Save final ML dataset
df.to_csv("ml_dataset.csv", index=False)

print("DONE — Labels fixed and saved to ml_dataset.csv")
