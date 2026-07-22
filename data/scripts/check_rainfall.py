import pandas as pd

df = pd.read_csv("uttarakhand_hourly.csv")

print("Min rainfall:", df["precip_hour"].min())
print("Max rainfall:", df["precip_hour"].max())
print(df.head())
