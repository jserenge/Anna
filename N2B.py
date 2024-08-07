import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
from io import BytesIO
import numpy as np
import xlsxwriter

# Load the saved models and scaler
with open('N2B_model_pred.pkl', 'rb') as f:
    model_pred = pickle.load(f)
with open('N2B_model_lower.pkl', 'rb') as f:
    model_lower = pickle.load(f)
with open('N2B_model_upper.pkl', 'rb') as f:
    model_upper = pickle.load(f)
with open('N2B_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Function to create features
def create_features(date):
    """Create time series features based on the date."""
    features = {
        'day_of_week': date.weekday(),
        'month': date.month,
        'Year': date.year,
        'is_weekend': int(date.weekday() in [5, 6])
    }
    # Use placeholders for lag features
    for i in range(1, 4):
        features[f'S_lag_{i}'] = np.nan  # Placeholder values
    return features

# Function to make predictions for a given date
def predict_for_date(date, scaler, model_lower, model_pred, model_upper):
    features = create_features(date)
    
    # Convert to DataFrame with the correct feature order
    feature_names = ['Year', 'month', 'day_of_week', 'is_weekend'] + [f'S_lag_{i}' for i in range(1, 4)]
    input_df = pd.DataFrame([features], columns=feature_names)
    
    # Scale the features
    input_scaled = scaler.transform(input_df)
    
    # Make predictions
    lower_pred = model_lower.predict(input_scaled)[0]
    pred = model_pred.predict(input_scaled)[0]
    upper_pred = model_upper.predict(input_scaled)[0]

    return lower_pred, pred, upper_pred

# Function to convert DataFrame to Excel
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Predictions')
    writer.close()  # Use close instead of save
    processed_data = output.getvalue()
    return processed_data

# Streamlit app
def app():
    st.title('N2B Prediction')

    date_input = st.date_input('Select a date', value=datetime.today())
    
    if st.button('Predict'):
        try:
            # Predict value
            lower, pred, upper = predict_for_date(date_input, scaler, model_lower, model_pred, model_upper)

            # Display the predicted values
            st.subheader('Predicted S:')
            st.write(f"Lower bound (0.5%): {lower:.2f}")
            st.write(f"Predicted S: {pred:.2f}")
            st.write(f"Upper bound (99.5%): {upper:.2f}")

            # Prepare predictions for download
            predictions_df = pd.DataFrame({
                'Lower Bound': [lower],
                'Predicted': [pred],
                'Upper Bound': [upper]
            })

            # Convert predictions to Excel
            excel_data = to_excel(predictions_df)

            # Download button for Excel
            st.download_button(
                label='Download Predictions as Excel',
                data=excel_data,
                file_name='predicted_s.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

        except Exception as e:
            st.error(f"An error occurred: {e}")

    # Add some styling
    st.markdown("""
    <style>
    .stButton>button {
        color: #ffffff;
        background-color: #4CAF50;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Run the app
if __name__ == '__main__':
    app()
