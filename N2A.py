import streamlit as st
import pandas as pd
import pickle
from datetime import datetime
from io import BytesIO
import numpy as np

def create_features(date):
    """Create time series features based on the date."""
    features = {
        'Year': date.year,
        'Month': date.month,
        'Day': date.day,
        'DayOfWeek': date.weekday(),
    }
    return features

def app():
    # Load the scaler
    with open('N2A_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Load the models for S1 to S5
    models = {}
    for s in range(1, 6):
        with open(f'N2A_S{s}_model.pkl', 'rb') as f:
            models[f'S{s}'] = pickle.load(f)

    # Create a Streamlit app
    st.title('S1 to S5 Prediction App')

    # Add explanation
    st.write('This app predicts S1 to S5 based on a given date. Please select a date below.')

    # Date input
    selected_date = st.date_input("Select a date")

    # Create a button to trigger the prediction
    if st.button('Make Prediction'):
        try:
            # Create features
            features = create_features(selected_date)

            # Convert to DataFrame
            input_data = pd.DataFrame([features])

            # Scale the input data
            input_scaled = scaler.transform(input_data)

            # Make predictions for S1 to S5
            predictions = {}
            upper_bounds = {}
            lower_bounds = {}
            for s in range(1, 6):
                model_info = models[f'S{s}']
                model = model_info['model']
                predictions[f'S{s}_pred'] = model.predict(input_scaled)[0]
                upper_bounds[f'S{s}_upper'] = model_info['upper_bound'][0]
                lower_bounds[f'S{s}_lower'] = model_info['lower_bound'][0]

            # Display the predicted values and bounds
            st.subheader('Predicted S1 to S5 with Bounds:')
            for s in range(1, 6):
                st.write(f"### S{s}")
                st.write(f"Predicted S: {predictions[f'S{s}_pred']:.2f}")
                st.write(f"Upper Bound: {upper_bounds[f'S{s}_upper']:.2f}")
                st.write(f"Lower Bound: {lower_bounds[f'S{s}_lower']:.2f}")

            # Prepare predictions for download
            predictions_df = pd.DataFrame({
                'Pred_S1': [predictions['S1_pred']],
                'Pred_S2': [predictions['S2_pred']],
                'Pred_S3': [predictions['S3_pred']],
                'Pred_S4': [predictions['S4_pred']],
                'Pred_S5': [predictions['S5_pred']],
                'Upper_S1': [upper_bounds['S1_upper']],
                'Upper_S2': [upper_bounds['S2_upper']],
                'Upper_S3': [upper_bounds['S3_upper']],
                'Upper_S4': [upper_bounds['S4_upper']],
                'Upper_S5': [upper_bounds['S5_upper']],
                'Lower_S1': [lower_bounds['S1_lower']],
                'Lower_S2': [lower_bounds['S2_lower']],
                'Lower_S3': [lower_bounds['S3_lower']],
                'Lower_S4': [lower_bounds['S4_lower']],
                'Lower_S5': [lower_bounds['S5_lower']]
            })

            # Function to convert DataFrame to Excel
            def to_excel(df):
                output = BytesIO()
                writer = pd.ExcelWriter(output, engine='xlsxwriter')
                df.to_excel(writer, index=False, sheet_name='Predictions')
                writer.close()  # Use close instead of save
                processed_data = output.getvalue()
                return processed_data

            # Convert predictions to Excel
            excel_data = to_excel(predictions_df)

            # Download button for Excel
            st.download_button(
                label='Download Predictions as Excel',
                data=excel_data,
                file_name='predicted_s1_to_s5_with_bounds.xlsx',
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
