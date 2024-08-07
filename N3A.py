import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def app():
    st.title("N3A Predictions")

    # Load the models and scaler
    with open('N3A_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    models = {}
    for col in ['S1', 'S2', 'S3', 'S4', 'S5']:
        with open(f'N3A_model_pred_{col}.pkl', 'rb') as f:
            models[col] = pickle.load(f)

    # Input date
    date_input = st.date_input("Select a date")

    if date_input:
        # Create features from the input date
        date = pd.to_datetime(date_input)
        features = {
            'day_of_week': date.dayofweek,
            'month': date.month,
            'day_of_month': date.day,
            'is_weekend': int(date.dayofweek in [5, 6])
        }

        # Placeholder for lag features
        for i in range(1, 4):
            for col in ['S1', 'S2', 'S3', 'S4', 'S5']:
                features[f'{col}_lag_{i}'] = 0  # Default value, adjust as needed

        # Convert features to DataFrame and scale
        feature_df = pd.DataFrame([features])
        X_scaled = scaler.transform(feature_df)

        # Make predictions
        predictions = {}
        for col in ['S1', 'S2', 'S3', 'S4', 'S5']:
            model = models[col]
            predictions[col] = model.predict(X_scaled)[0]

        # Display predictions
        st.write("Predictions:")
        for col in ['S1', 'S2', 'S3', 'S4', 'S5']:
            st.write(f"{col}: {predictions[col]:.2f}")

