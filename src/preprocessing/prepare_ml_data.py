import pandas as pd

df = pd.read_csv("data/ml_ready.csv")

# sort by time
df = df.sort_values(by=["year","month","day","hour"])

# create lag rainfall features
df["rain_lag1"] = df["precip_hour"].shift(1)
df["rain_lag2"] = df["precip_hour"].shift(2)
df["rain_lag3"] = df["precip_hour"].shift(3)

# remove missing rows
df = df.dropna()

# save new dataset
df.to_csv("data/ml_ready_lagged.csv", index=False)

print("Lag features added successfully")
