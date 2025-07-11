import streamlit as st
import numpy as np
import pandas as pd
from src.predict import load_model, predict_default
from src.risk_analytics import risk_segmentation

MODEL_PATH = 'models/loan_default_model.pkl'

st.title('AI-powered Loan Default Predictor')

# Example input fields (customize as per your dataset)
age = st.number_input('Age', min_value=18, max_value=100, value=30)
income = st.number_input('Annual Income', min_value=0, value=50000)
loan_amount = st.number_input('Loan Amount', min_value=0, value=10000)
credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=650)
# Add more fields as per your features

input_features = [age, income, loan_amount, credit_score]  # Update order as per model

if st.button('Predict Default Risk'):
    try:
        model = load_model(MODEL_PATH)
        prediction, probability = predict_default(model, input_features)
        risk = risk_segmentation([probability])[0]
        st.write(f'**Default Prediction:** {"Default" if prediction == 1 else "No Default"}')
        st.write(f'**Default Probability:** {probability:.2f}')
        st.write(f'**Risk Segment:** {risk}')
    except Exception as e:
        st.error(f'Error: {e}')

st.info('Note: This is a demo. Please train the model and adjust input fields as per your dataset.') 