import streamlit as st
import pandas as pd
import joblib
from geopy.distance import geodesic

# Load saved model and encoders
model = joblib.load("fraud_detection_model.jb")
encoders = joblib.load("label_encoder.jb")

def haversine(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

st.title("🕵️‍♂️ Credit Card Fraud Detection")
st.markdown("Enter transaction details below to predict if it's **fraudulent** or **legitimate**.")

merchant = st.text_input("Merchant Name")
category = st.text_input("Transaction Category")
amt = st.number_input("Transaction Amount ($)", min_value=0.0, format="%.2f")
lat = st.number_input("User Latitude", format="%.6f")
long = st.number_input("User Longitude", format="%.6f")
merch_lat = st.number_input("Merchant Latitude", format="%.6f")
merch_long = st.number_input("Merchant Longitude", format="%.6f")
hour = st.slider("Hour of Transaction", 0, 23, 12)
day = st.slider("Day of Month", 1, 31, 1)
month = st.slider("Month", 1, 12, 1)
gender = st.selectbox("Cardholder Gender", ["Male", "Female"])
cc_num = st.text_input("Credit Card Number")

distance = haversine(lat, long, merch_lat, merch_long)

if st.button("Check For Fraud"):
    if merchant and category and cc_num:
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

