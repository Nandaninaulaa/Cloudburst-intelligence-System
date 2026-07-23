import tkinter as tk
from tkinter import messagebox
import numpy as np
import joblib

# Load trained model
model = joblib.load("rf_cloudburst_model.pkl")

# Prediction function
def predict_cloudburst():
    try:
        lat = float(entry_lat.get())
        lon = float(entry_lon.get())
        rain = float(entry_rain.get())
        year = int(entry_year.get())
        month = int(entry_month.get())
        day = int(entry_day.get())
        hour = int(entry_hour.get())

        # Prepare input in same order as training
        X = np.array([[lat, lon, rain, year, month, day, hour]])

        # Probability prediction
        prob = model.predict_proba(X)[0][1]

        # Risk levels
        if prob > 0.7:
            result = f"HIGH RISK \nCloudburst Probability: {prob*100:.2f}%"
            result_label.config(text=result, fg="red")

        elif prob > 0.4:
            result = f"MODERATE RISK\nCloudburst Probability: {prob*100:.2f}%"
            result_label.config(text=result, fg="orange")

        else:
            result = f"LOW RISK\nCloudburst Probability: {prob*100:.2f}%"
            result_label.config(text=result, fg="green")

    except Exception as e:
        messagebox.showerror("Input Error", "Please enter valid values")

# GUI window
root = tk.Tk()
root.title("Cloudburst Prediction System")
root.geometry("400x450")

# Labels and input fields
tk.Label(root, text="Latitude").pack()
entry_lat = tk.Entry(root)
entry_lat.pack()

tk.Label(root, text="Longitude").pack()
entry_lon = tk.Entry(root)
entry_lon.pack()

tk.Label(root, text="Rainfall (mm/hr)").pack()
entry_rain = tk.Entry(root)
entry_rain.pack()

tk.Label(root, text="Year").pack()
entry_year = tk.Entry(root)
entry_year.pack()

tk.Label(root, text="Month").pack()
entry_month = tk.Entry(root)
entry_month.pack()

tk.Label(root, text="Day").pack()
entry_day = tk.Entry(root)
entry_day.pack()

tk.Label(root, text="Hour").pack()
entry_hour = tk.Entry(root)
entry_hour.pack()

# Predict button
predict_btn = tk.Button(root, text="Predict", command=predict_cloudburst)
predict_btn.pack(pady=10)

# Result label
result_label = tk.Label(root, text="", font=("Arial", 12, "bold"))
result_label.pack(pady=20)

# Run GUI
root.mainloop()
