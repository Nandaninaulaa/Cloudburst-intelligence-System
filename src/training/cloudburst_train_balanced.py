import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ========================
# STEP 1: LOAD DATA
# ========================
df = pd.read_csv(r"C:\Users\bohra\OneDrive\Desktop\cloudburst\ml_ready.csv")

print("\n=== ORIGINAL LABEL COUNTS ===")
print(df['label'].value_counts())

# ========================
# STEP 2: BALANCE DATA
# ========================

# Extract cloudburst rows (label = 1)
df_cloudburst = df[df['label'] == 1]

# Extract SAME number of non-cloudburst rows
df_normal = df[df['label'] == 0].sample(n=len(df_cloudburst), random_state=42)

# Combine balanced dataset
df_balanced = pd.concat([df_cloudburst, df_normal], axis=0)

print("\n=== BALANCED LABEL COUNTS ===")
print(df_balanced['label'].value_counts())

# ========================
# STEP 3: SELECT FEATURES & LABEL
# ========================
X = df_balanced[['Latitude', 'Longitude', 'precip_hour', 'year', 'month', 'day', 'hour']]
y = df_balanced['label']

# ========================
# STEP 4: TRAIN/TEST SPLIT
# ========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========================
# STEP 5: TRAIN MODEL
# ========================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ========================
# STEP 6: EVALUATE MODEL
# ========================
pred = model.predict(X_test)

print("\n=== MODEL ACCURACY ===")
print("Accuracy:", accuracy_score(y_test, pred))

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, pred))

# ========================
# STEP 7: SAVE MODEL
# ========================
joblib.dump(model, "cloudburst_model.pkl")
print("\nModel saved as cloudburst_model.pkl")

# ========================
# STEP 8: TEST WITH CUSTOM INPUT
# ========================
new_data = pd.DataFrame([{
    "Latitude": 30.12,
    "Longitude": 79.43,
    "precip_hour": 65,     # example rainfall
    "year": 2022,
    "month": 8,
    "day": 10,
    "hour": 14
}])

prediction = model.predict(new_data)
print("\n=== PREDICTION FOR NEW DATA ===")
print("Cloudburst (1=Yes, 0=No):", prediction[0])
