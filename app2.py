import streamlit as st
import pandas as pd
import joblib
from geopy.distance import geodesic

# Load model and encoders
model = joblib.load("fraud_detection_model.jb")
encoders = joblib.load("label_encoder.jb")

def haversine(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")
st.title("🕵️‍♂️ Credit Card Fraud Detection")

# Initialize session state for input fields
if "form_data" not in st.session_state:
    st.session_state.form_data = {
        "merchant": "",
        "category": "",
        "amt": 0.0,
        "lat": 0.0,
        "long": 0.0,
        "merch_lat": 0.0,
        "merch_long": 0.0,
        "hour": 12,
        "day": 1,
        "month": 1,
        "gender": "Male",
        "cc_num": ""
    }

def load_sample(sample_type):
    if sample_type == "sample1":
        st.session_state.form_data = {
            "merchant": "fraud_Rippin, Kub and Mann",
            "category": "misc_net",
            "amt": 4.97,
            "lat": 36.0788,
            "long": -81.1781,
            "merch_lat": 36.011293,
            "merch_long": -82.048315,
            "hour": 12,
            "day": 1,
            "month": 1,
            "gender": "Female",
            "cc_num": "2703186189652095"
        }
    elif sample_type == "sample2":
        st.session_state.form_data = {
            "merchant": "fraud_Rutherford-Mertz",
            "category": "grocery_pos",
            "amt": 281.06,
            "lat": 35.9946,
            "long": -81.7266,
            "merch_lat": 36.430124,
            "merch_long": -81.179483,
            "hour": 12,
            "day": 1,
            "month": 1,
            "gender": "Male",
            "cc_num": "4613314721966"
        }

# Sidebar buttons for Samples and Reset
with st.sidebar:
    st.markdown("### Quick Actions")
    if st.button("Sample 1"):
        load_sample("sample1")
    if st.button("Sample 2"):
        load_sample("sample2")
    if st.button("Reset"):
        st.session_state.form_data = {
            "merchant": "",
            "category": "",
            "amt": 0.0,
            "lat": 0.0,
            "long": 0.0,
            "merch_lat": 0.0,
            "merch_long": 0.0,
            "hour": 12,
            "day": 1,
            "month": 1,
            "gender": "Male",
            "cc_num": ""
        }

# Main form inputs
st.subheader("Enter Transaction Details")
merchant = st.text_input("Merchant Name", value=st.session_state.form_data["merchant"], key="merchant")
category = st.text_input("Transaction Category", value=st.session_state.form_data["category"], key="category")
amt = st.number_input("Transaction Amount ($)", value=st.session_state.form_data["amt"], min_value=0.0, format="%.2f", key="amt")
lat = st.number_input("User Latitude", value=st.session_state.form_data["lat"], format="%.6f", key="lat")
long = st.number_input("User Longitude", value=st.session_state.form_data["long"], format="%.6f", key="long")
merch_lat = st.number_input("Merchant Latitude", value=st.session_state.form_data["merch_lat"], format="%.6f", key="merch_lat")
merch_long = st.number_input("Merchant Longitude", value=st.session_state.form_data["merch_long"], format="%.6f", key="merch_long")
hour = st.slider("Hour of Transaction", 0, 23, value=st.session_state.form_data["hour"], key="hour")
day = st.slider("Day of Month", 1, 31, value=st.session_state.form_data["day"], key="day")
month = st.slider("Month", 1, 12, value=st.session_state.form_data["month"], key="month")
gender = st.selectbox("Cardholder Gender", ["Male", "Female"], index=0 if st.session_state.form_data["gender"] == "Male" else 1, key="gender")
cc_num = st.text_input("Credit Card Number", value=st.session_state.form_data["cc_num"], key="cc_num")

# Predict
if st.button("Check For Fraud"):
    if merchant and category and cc_num:
        distance = haversine(lat, long, merch_lat, merch_long)

        input_df = pd.DataFrame([[merchant, category, amt, cc_num, hour, day, month, gender, distance]],
                                columns=['merchant', 'category', 'amt', 'cc_num', 'hour', 'day', 'month', 'gender', 'distance'])

        for col in ['merchant', 'category', 'gender']:
            try:
                input_df[col] = encoders[col].transform(input_df[col])
            except ValueError:
                input_df[col] = -1

        input_df['cc_num'] = input_df['cc_num'].apply(lambda x: hash(x) % (10 ** 2))

        prob = model.predict_proba(input_df)[0][1]
        prediction = 1 if prob >= 0.97 else 0
        label = "🚨 Fraudulent Transaction" if prediction == 1 else "✅ Legitimate Transaction"

        st.subheader(f"Prediction: {label}")
    else:
        st.error("⚠️ Please fill all required fields.")


