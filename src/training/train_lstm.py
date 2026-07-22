import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from sklearn.metrics import precision_score, recall_score, f1_score
import json
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Attention, GlobalAveragePooling1D
from sklearn.metrics import roc_auc_score, roc_curve
import os

print("1. Loading Dataset...")
df = pd.read_csv(r"C:\Users\nanda\Downloads\front end\cloudburst\data\ml_ready_lagged.csv")

# Address extreme class imbalance to ensure robust models
df_0 = df[df['label'] == 0]
if len(df_0) > 20000:
    df_0 = df_0.sample(n=20000, random_state=42)
df_1 = df[df['label'] == 1]
# Oversample positives to give models enough data to learn
df_1 = df_1.sample(n=2000, replace=True, random_state=42)

df = pd.concat([df_0, df_1]).sample(frac=1, random_state=42).reset_index(drop=True)
    
print(f"Dataset Size: {len(df)} rows")
print("Class Distribution:\n", df['label'].value_counts())

print("\n2. Preprocessing Data...")
# Extract the temporal sequence in chronological order
sequence_cols = ['rain_lag3', 'rain_lag2', 'rain_lag1', 'precip_hour']
X_temporal = df[sequence_cols].values
y = df['label'].values

# Scale the data - LSTMs require data to be between 0 and 1
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_temporal)

# Reshape from 2D (samples, features) to 3D (samples, time_steps, features)
X_3d = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

X_train, X_test, y_train, y_test = train_test_split(X_3d, y, test_size=0.2, random_state=42, stratify=y)

# ---------------------------------------------------------
# REAL-WORLD NOISE INJECTION (TEST SET ONLY)
# We train on perfect data, but evaluate on noisy data
# ---------------------------------------------------------
import numpy as np
X_test_noisy = X_test.copy()
np.random.seed(42)

# X_test_noisy shape is (samples, 4, 1) corresponding to ['rain_lag3', 'rain_lag2', 'rain_lag1', 'precip_hour']
# We add noise to precip_hour (index 3) and rain_lag1 (index 2)
# Since it's scaled (MinMaxScaler), noise should be scaled down. 15 on a ~150 scale is roughly 0.1
X_test_noisy[:, 3, 0] = np.clip(X_test_noisy[:, 3, 0] + np.random.normal(0, 0.1, len(X_test_noisy)), 0, 1)
X_test_noisy[:, 2, 0] = np.clip(X_test_noisy[:, 2, 0] + np.random.normal(0, 0.05, len(X_test_noisy)), 0, 1)

y_test_noisy = y_test.copy()
flip_mask = np.random.rand(len(y_test_noisy)) < 0.015
y_test_noisy[flip_mask] = 1 - y_test_noisy[flip_mask]

# Save the scaler so we can re-scale live data in Streamlit seamlessly
joblib.dump(scaler, "lstm_scaler.pkl")
print("Scaler saved as lstm_scaler.pkl")

print("\n3. Building Attention-LSTM Neural Network...")
inputs = Input(shape=(X_train.shape[1], 1))
lstm_out = LSTM(64, return_sequences=True)(inputs)
query = Dense(64)(lstm_out)
value = Dense(64)(lstm_out)
attn_out = Attention()([query, value])
pooled = GlobalAveragePooling1D()(attn_out)
dropout = Dropout(0.2)(pooled)
outputs = Dense(1, activation='sigmoid')(dropout)
model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Calculate penalties to handle class imbalance securely mapping 0 -> x, 1 -> y
class_weights_array = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
weights_dict = {0: class_weights_array[0], 1: class_weights_array[1]}

print("\n4. Training LSTM...")
# Fit the model (this will take a few minutes for 800k training rows)
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=512, # Large batch size to leverage hardware speed
    validation_split=0.1,
    class_weight=weights_dict,
    verbose=1
)

# Save the training history for Loss Curves
history_dict = history.history
for key in history_dict:
    history_dict[key] = [float(x) for x in history_dict[key]]
with open(r"C:\Users\nanda\Downloads\front end\cloudburst\data/preprocessing/model/lstm_history.json", "w") as f:
    json.dump(history_dict, f)
print("LSTM Training History saved to lstm_history.json")

# Evaluate
print("\n5. Benchmarking Model...")

def get_metrics(y_true, y_pred_prob):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Sample 50 points for ROC curve to save space in JSON
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    indices = np.linspace(0, len(fpr)-1, min(50, len(fpr)), dtype=int)
    roc_points = [{"fpr": round(float(fpr[i]), 3), "tpr": round(float(tpr[i]), 3)} for i in indices]
    
    return {
        "accuracy": round(accuracy_score(y_true, y_pred) * 100, 1),
        "precision": round(precision_score(y_true, y_pred, zero_division=0) * 100, 1),
        "recall": round(recall_score(y_true, y_pred, zero_division=0) * 100, 1),
        "f1": round(f1_score(y_true, y_pred, zero_division=0) * 100, 1),
        "auc": round(roc_auc_score(y_true, y_pred_prob) * 100, 1),
        "cm": confusion_matrix(y_true, y_pred).tolist(),
        "roc_curve": roc_points
    }

# Evaluate Clean
y_pred_prob_clean = model.predict(X_test, verbose=0)
clean_metrics = get_metrics(y_test, y_pred_prob_clean)

# Evaluate Noisy
y_pred_prob_noisy = model.predict(X_test_noisy, verbose=0)
noisy_metrics = get_metrics(y_test_noisy, y_pred_prob_noisy)
print("\n========== LSTM RESULTS ==========")

print("\n--- CLEAN TEST SET ---")
print(f"Accuracy : {clean_metrics['accuracy']}%")
print(f"Precision: {clean_metrics['precision']}%")
print(f"Recall   : {clean_metrics['recall']}%")
print(f"F1 Score : {clean_metrics['f1']}%")
print(f"AUC      : {clean_metrics['auc']}%")
print("Confusion Matrix:")
print(clean_metrics['cm'])

print("\n--- NOISY TEST SET ---")
print(f"Accuracy : {noisy_metrics['accuracy']}%")
print(f"Precision: {noisy_metrics['precision']}%")
print(f"Recall   : {noisy_metrics['recall']}%")
print(f"F1 Score : {noisy_metrics['f1']}%")
print(f"AUC      : {noisy_metrics['auc']}%")
print("Confusion Matrix:")
print(noisy_metrics['cm'])

# Ablation Study
X_test_ablation = X_test.copy()

# Remove rain_lag1 feature
X_test_ablation[:, 2, 0] = 0

ablation_pred_prob = model.predict(X_test_ablation, verbose=0).flatten()

ablation_auc = round(
    roc_auc_score(y_test, ablation_pred_prob) * 100,
    1
)

print("\nAblation AUC:", ablation_auc)

output_json = {
    "lstm": {
        "clean": clean_metrics,
        "noisy": noisy_metrics,
        "ablation_auc_without_lag1": ablation_auc
    }
}

import json
with open(r"C:\Users\nanda\Downloads\front end\cloudburst\data/preprocessing/model/metrics_lstm.json", "w") as f:
    json.dump(output_json, f)
print("Nested Metrics saved successfully to data/preprocessing/model/metrics_lstm.json")

# Save the Keras Architecture
model.save("cloudburst_lstm_model.keras")
print("\nModel saved successfully as cloudburst_lstm_model.keras!")
