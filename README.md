# 🌩️ CloudBurst Intelligence System

CloudBurst is an **AI-powered Cloudburst Detection and Early Warning System** designed to predict extreme rainfall events in mountainous regions using a **hybrid Machine Learning approach** and a **5-Agent Agentic AI architecture**.

The system combines **Random Forest**, **XGBoost**, **Stacking Ensemble**, and **LSTM** models with real-time weather data, disaster intelligence, and automated alert generation to assist in disaster preparedness and emergency response.

---

# 🚀 Features

## 🌧️ Hybrid AI Prediction Engine

A multi-model prediction pipeline that combines:

* 🌲 Random Forest
* ⚡ XGBoost
* 🧠 Stacking Ensemble
* 🔄 LSTM Neural Network

The models are trained on historical precipitation data and evaluated using standard classification metrics to improve prediction reliability.

---

## 🤖 5-Agent AI Architecture

CloudBurst employs a collaborative **multi-agent system**, where each agent performs a dedicated role in the prediction and decision-making process.

### 🌦️ Weather Monitoring Agent

* Collects and monitors weather information.
* Tracks rainfall intensity and abnormal weather patterns.
* Generates weather summaries.

### ⚠️ Risk Assessment Agent

* Analyzes machine learning predictions.
* Estimates cloudburst probability.
* Classifies risk levels (Low, Medium, High, Extreme).

### 🚨 Emergency Response Agent

* Suggests emergency actions.
* Provides evacuation recommendations.
* Assists disaster management authorities.

### 📢 Alert Generation Agent

* Generates warning notifications.
* Categorizes alerts based on severity.
* Supports SMS and Email notifications.

### 📊 Decision Support Agent

* Combines outputs from all agents.
* Produces an overall disaster situation report.
* Supports informed decision-making during emergencies.

---

## 🌍 Consensus Weather Data

The system integrates weather information from multiple sources, including:

* WeatherAPI
* Open-Meteo

An intelligent weighted consensus algorithm improves reliability before predictions are generated.

---

## 📊 Data Processing Pipeline

The preprocessing workflow includes:

* Dataset extraction
* Rainfall cleaning
* Hourly rainfall generation
* Feature engineering
* Cloudburst labeling
* Machine learning dataset preparation

---

## 📈 Interactive Visualizations

Generate rich visual insights including:

* 🌧️ Rainfall Heatmaps
* 📉 Rainfall Histograms
* 📊 Daily Rainfall Trends
* 📍 Regional Rainfall Analysis
* 📈 Model Evaluation Metrics

---

## 🗄️ Persistent Logging

Prediction history is stored using **SQLite**, allowing users to:

* View previous predictions
* Analyze historical events
* Monitor system performance over time

---

## 📲 Multi-Channel Alert System

CloudBurst automatically sends notifications through:

* 📧 Email (SMTP)
* 📱 SMS (Twilio)

Alerts are generated whenever high-risk cloudburst conditions are detected.

---

## 🖥️ Interactive Dashboard

The Streamlit dashboard provides:

* 🌦️ Live weather information
* 📊 Cloudburst prediction results
* 📈 Rainfall visualizations
* 🤖 AI-generated recommendations
* 🚨 Emergency alerts
* 📋 Prediction history

---

# 🛠️ Tech Stack

### Programming

* 🐍 Python

### Machine Learning

* Scikit-learn
* TensorFlow / Keras
* XGBoost

### Data Processing

* Pandas
* NumPy

### Visualization

* Streamlit
* Matplotlib
* Plotly

### Database

* SQLite

### APIs

* WeatherAPI
* Open-Meteo
* Twilio
* SMTP Email

### AI

* Multi-Agent AI System (5 Agents)

---

# 📂 Project Structure

```text
cloudburst/
│
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── data/
│   ├── datasets/
│   └── scripts/
│
├── models/
│   ├── Random Forest
│   ├── XGBoost
│   ├── Stacking Ensemble
│   └── LSTM
│
└── src/
    ├── preprocessing/
    ├── training/
    ├── prediction/
    └── visualization/
```

---

# 🛠️ Local Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Nandaninaulaa/cloudburst-intelligence-system.git

cd cloudburst-intelligence-system
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv .venv
```

**Windows**

```bash
.venv\Scripts\activate
```

**Linux/macOS**

```bash
source .venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure Environment Variables

Create a `.env` file using `.env.example`.

```env
WEATHER_API_KEY=your_key
SENDER_EMAIL=your_email
SENDER_PASS=your_app_password

TWILIO_SID=your_sid
TWILIO_AUTH=your_auth
TWILIO_PHONE=your_phone
```

### 5️⃣ Run the Application

```bash
streamlit run src/prediction/app.py
```

---

# ☁️ Deployment (Streamlit Cloud)

Deploy CloudBurst easily using **Streamlit Cloud**.

### Step 1

Push your project to GitHub.

> ⚠️ Never commit your `.env` file.

### Step 2

Create a new Streamlit Cloud application linked to your repository.

### Step 3

Add your secrets in **Settings → Secrets**.

```toml
WEATHER_API_KEY="your_key"
SENDER_EMAIL="your_email"
SENDER_PASS="your_password"

TWILIO_SID="your_sid"
TWILIO_AUTH="your_auth"
TWILIO_PHONE="your_phone"
```

### Step 4

Deploy the application.

Streamlit automatically installs all packages from `requirements.txt`.

---

# 📊 Models Implemented

| Model                | Purpose                          |
| -------------------- | -------------------------------- |
| 🌲 Random Forest     | Cloudburst classification        |
| ⚡ XGBoost            | Gradient boosting prediction     |
| 🧠 Stacking Ensemble | Improved prediction accuracy     |
| 🔄 LSTM              | Time-series rainfall forecasting |

---

## 📁 Dataset

Large training datasets are not included in this repository due to GitHub size limits.

To run the project:

1. Download the datasets from the provided link.
2. Place them inside the `data/` directory.

A small sample dataset is included for testing.

---

# 🔮 Future Improvements

* 🛰️ Satellite imagery integration
* 🌍 IMD and NASA GPM live data support
* 🗺️ GIS-based hazard mapping
* 📱 Mobile application
* ☁️ Cloud deployment
* 🔔 Push notification support
* 🤖 Enhanced multi-agent collaboration
* 📈 Real-time disaster monitoring dashboard

---

# 👩‍💻 Author

## **Nandani Naula**

🎓 **B.Tech Computer Science & Engineering (Artificial Intelligence & Machine Learning)**

💻 **AI/ML Engineer • Full Stack Developer • Agentic AI • Deep Learning • Disaster Intelligence**

🔗 **GitHub:** https://github.com/Nandaninaulaa

---

# ⭐ Support

If you found this project useful, consider giving it a **⭐ Star** on GitHub. Your support helps improve the project and encourages future development.
