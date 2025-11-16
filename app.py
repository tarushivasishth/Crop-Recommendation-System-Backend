import streamlit as st
import tensorflow as tf
import joblib
import numpy as np
from weather_api import get_lat_lon, get_monthly_weather, get_three_month_average

# Load model, encoder, and scaler
model = tf.keras.models.load_model(r"C:\Users\tarus\minor_project\Crop_Recommendation\models\ann_model.keras")
# model = joblib.load(r"C:\Users\tarus\minor_project\Crop_Recommendation\models\gb_model.pkl")
encoder = joblib.load(r"C:\Users\tarus\minor_project\Crop_Recommendation\models\label_encoder.pkl")
scaler = joblib.load(r"C:\Users\tarus\minor_project\Crop_Recommendation\models\scaler.pkl")

st.title("🌱 Agrocast Crop Recommendation System")

# Initialize session state
for key in ['N', 'P', 'K', 'ph', 'city', 'month', 'temp', 'rain', 'humidity']:
    if key not in st.session_state:
        st.session_state[key] = None

# Inputs
st.subheader("Soil Information")
st.session_state['N'] = st.number_input("Nitrogen (N)", 0, 200, 50)
st.session_state['P'] = st.number_input("Phosphorus (P)", 0, 200, 50)
st.session_state['K'] = st.number_input("Potassium (K)", 0, 200, 50)
st.session_state['ph'] = st.number_input("pH Value", 0.0, 14.0, 6.5)

st.subheader("Location Information")
st.session_state['city'] = st.text_input("City Name", "Delhi")
st.session_state['month'] = st.selectbox(
    "Month to sow crop",
    ("January","February","March","April","May","June",
     "July","August","September","October","November","December")
)

# Fetch weather
if st.button("Fetch Weather Data"):
    lat, lon = get_lat_lon(st.session_state['city'])
    if lat and lon:
        monthly_data = get_monthly_weather(lat, lon)
        if monthly_data is not None:
            temp, rain, humidity = get_three_month_average(monthly_data, st.session_state['month'])
            st.session_state['temp'], st.session_state['rain'], st.session_state['humidity'] = temp, rain, humidity
            st.success(f"🌡 Temperature: {temp:.2f} °C")
            st.success(f"💧 Humidity: {humidity:.2f} %")
            st.success(f"🌧 Rainfall: {rain:.2f} mm")
        else:
            st.error("Could not fetch weather data. Try a nearby city.")
    else:
        st.error("Invalid city name!")

# Recommend
if st.button("Recommend Crop"):
    if None in [st.session_state['temp'], st.session_state['rain'], st.session_state['humidity']]:
        st.error("Please fetch weather data first!")
    else:
        features = np.array([[st.session_state['N'], st.session_state['P'], st.session_state['K'],
                              st.session_state['temp'], st.session_state['humidity'],
                              st.session_state['ph'], st.session_state['rain']]])

        # ✅ Scale input using the same scaler as training
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)
        top3_indices = np.argsort(prediction, axis=1)[:, -3:][:, ::-1]
        top3_crops = np.array(encoder.classes_)[top3_indices][0]
        top3_scores = prediction[0][top3_indices][0]

        if np.max(prediction) > 0.8:
            st.success(f"Recommended Crop: {top3_crops[0]} (Confidence: {top3_scores[0]*100:.2f}%)")
        else:
            st.subheader("Top 3 Recommended Crops with Confidence Scores:")
            for crop, score in zip(top3_crops, top3_scores):
                st.write(f"🌾 {crop} — Confidence: {score*100:.2f}%")
