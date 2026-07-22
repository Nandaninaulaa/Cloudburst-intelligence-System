import pandas as pd

df = pd.read_csv("ml_dataset.csv")

print(df["label"].value_counts())
print(df[df["label"] == 1].head())
