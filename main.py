from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import tensorflow as tf
import numpy as np
import pandas as pd
from weather_api import get_lat_lon, get_monthly_weather, get_three_month_average


# Load model, encoder, and scaler
model = tf.keras.models.load_model(r"C:\Users\tarus\minor_project\Crop_Recommendation\models\ann_model.keras")
# model = joblib.load(r"C:\Users\tarus\minor_project\Crop_Recommendation\models\gb_model.pkl")
encoder = joblib.load(r"C:\Users\tarus\minor_project\Crop_Recommendation\models\label_encoder.pkl")
scaler = joblib.load(r"C:\Users\tarus\minor_project\Crop_Recommendation\models\scaler.pkl")

app = FastAPI(title="Agrocast Crop Recommendation")

# Allow frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Agrocast crop recommendation API is working!"}

@app.post("/recommend_crop")
def recommend_crop(N: float, P: float, K: float, ph: float, city: str, month: str):
    lat, lon = get_lat_lon(city)

    if not lat:
        return {"error": "Invalid city name!"}

    monthly_data = get_monthly_weather(lat, lon)
    if monthly_data is not None:
        temp, rain, humidity = get_three_month_average(monthly_data, month)

        # prepare input
        features = np.array([[N, P, K, temp, humidity, ph, rain]])
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)

        # top 3 recommendations
        top3_index = np.argsort(prediction, axis=1)[:, -3:][:, ::-1]
        top3_crops = np.array(encoder.classes_)[top3_index][0]
        top3_scores = prediction[0][top3_index][0]

        return {
            "weather": {
                "temperature": float(temp),
                "humidity": float(humidity),
                "rainfall": float(rain)
            },
            "top_3_crops": [
                {"crop": crop, "confidence": float(score)}
                for crop, score in zip(top3_crops, top3_scores)
            ]
        }