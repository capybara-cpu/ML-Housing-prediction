# app.py

import streamlit as st
import pickle
import numpy as np

# Load the trained model
@st.cache_data
def load_model():
    with open("model.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

# Streamlit UI
st.title("🏠 California Housing Price Predictor")
st.markdown("Enter the input features to estimate the median house value (in $100,000s).")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        MedInc = st.number_input("Median Income (0.00 - 20.00)", 0.00, 20.00, value=3.87, step=0.01)
        AveRooms = st.number_input("Average Rooms (0.50 - 15.00)", 0.50, 15.00, value=6.0, step=0.1)
        Population = st.number_input("Population (100 - 35000)", 100, 35000, value=1000, step=1)
        Latitude = st.number_input("Latitude (32.00 - 42.00)", 32.00, 42.00, value=34.0, step=0.01)

    with col2:
        HouseAge = st.number_input("House Age (1 - 52)", 1, 52, value=20, step=1)
        AveBedrms = st.number_input("Average Bedrooms (0.50 - 5.00)", 0.50, 5.00, value=1.0, step=0.1)
        AveOccup = st.number_input("Average Occupancy (0.50 - 10.00)", 0.50, 10.00, value=3.0, step=0.1)
        Longitude = st.number_input("Longitude (-124.35 to -114.31)", -124.35, -114.31, value=-120.0, step=0.01)

    submit = st.form_submit_button("Predict")

    if submit:
        # Prepare input and predict
        input_data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
        prediction = model.predict(input_data)[0]
        st.success(f"🏡 Estimated Median House Price: **${prediction * 100000:.2f}**")

        st.caption("Model: Linear Regression | Trained on California Housing Dataset")

