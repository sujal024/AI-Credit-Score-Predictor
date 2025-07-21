import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model/model.pkl")

st.title("🧠 AI Credit Score Predictor")

st.write("Enter alternative data points to get a credit score prediction.")

# Collect input from user
monthly_topups = st.slider("Monthly Top-ups", 1, 30, 10)
sms_sent = st.slider("SMS Sent per Month", 0, 100, 25)
on_time_payments = st.slider("On-time Payments", 0, 12, 10)
ecommerce_transactions = st.slider("E-commerce Transactions", 0, 50, 10)
social_posts = st.slider("Social Media Posts (Per Month)", 0, 100, 12)

# Feature calculation
payment_regular = on_time_payments 
digital_score = (ecommerce_transactions + social_posts + on_time_payments + monthly_topups) / 4

# Prepare input
input_df = pd.DataFrame([[monthly_topups, sms_sent,
                         payment_regular, digital_score]],
                        columns=['monthly_topups', 'sms_sent',
                                 'payment_regular', 'digital_activity_score'])

# Predict
if st.button("Predict Creditworthiness"):
    pred = model.predict(input_df)[0]
    score = model.predict_proba(input_df)[0][1]
    st.success(f"Prediction: {'Creditworthy ✅' if pred == 1 else 'Not Creditworthy ❌'}")
    st.info(f"Confidence Score: {round(score * 100, 2)}%")
