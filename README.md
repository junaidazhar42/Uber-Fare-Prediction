---
title: NYC Fare Estimator
emoji: 🚖
colorFrom: yellow
colorTo: gray
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: true
---

# 🚖 NYC Uber Fare Estimator

A machine learning web application that predicts Uber ride fares in New York City from natural-language location inputs. Type any address or landmark, pick your travel time, number of passengers and get an instant fare estimate — no coordinates required.

**[▶ Try the live demo](https://huggingface.co/spaces/junaidazhar/Uber-Fare-Prediction)**

---

## 📌 Project Overview

Ride-hailing fare prediction is a classic spatial-temporal regression problem. The goal of this project was to build an end-to-end ML pipeline — from raw data to a deployed, interactive application — that estimates Uber fares in NYC based on trip distance, time of day, and passenger count.

The project covers the full data science workflow:

- Data cleaning and exploratory data analysis
- Feature engineering (haversine distance, rush hour flags, time features)
- Model training and benchmarking across three algorithms
- Hyperparameter tuning with cross-validated grid search
- Deployment as an interactive web application

---

## 🛠️ Tech Stack & Model Details

### Machine Learning
| Component | Detail                                                     |
|---|------------------------------------------------------------|
| Dataset | [NYC Uber Fares (Kaggle)](https://www.kaggle.com/datasets/yasserh/uber-fares-dataset)                                |
| Target | `fare_amount` (USD)                                        |
| Key feature | Haversine distance between pickup and dropoff              |
| Models trained | Linear Regression, Random Forest, XGBoost                  |
| Best model | Random Forest Regressor (tuned via GridSearchCV)           |
| Tuning strategy | 3-fold cross-validation · `neg_mean_squared_error` scoring |

### Feature Engineering
- **Haversine distance** — great-circle distance between pickup and dropoff coordinates (km)
- **Time features** — hour, weekday, month, year extracted from `pickup_datetime`
- **Rush hour flag** — binary feature for weekday 7–9 AM and 4–7 PM windows

### Application
| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Geocoding | OpenStreetMap Nominatim API (free, no key needed) |
| Map rendering | Folium + streamlit-folium |
| Model serving | joblib |

---

## 🚀 How to Run Locally

**1. Clone the repository**
```bash
git clone https://huggingface.co/spaces/junaidazhar/Uber-Fare-Prediction/tree/main
cd nyc-fare-estimator
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add the trained model**

Run the notebook and export the best model:
```python
import joblib
joblib.dump(best_rf, "model.pkl")
```
Place `model.pkl` in the same directory as `app.py`.

**4. Launch the app**
```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`.

### Requirements
```
streamlit
numpy
joblib
scikit-learn
requests
folium
streamlit-folium
```

---

## 📁 Project Structure

```
nyc-fare-estimator/
├── streamlit_app.py     # Streamlit application
├── model.pkl            # Trained Random Forest model
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## 📊 How It Works

1. User types a pickup and dropoff location (any NYC address or landmark)
2. The app geocodes both locations via the Nominatim API, biased toward NYC
3. Haversine distance is computed from the resolved coordinates
4. Time features and a rush hour flag are derived from the selected date and time
5. Features are passed to the trained Random Forest model
6. The estimated fare and a route map are displayed

---

*Built as a portfolio project · NYC Uber Fares Prediction using Spatial Temporal Modeling · OpenStreetMap geocoding*