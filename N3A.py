import streamlit as st
import pickle
import pandas as pd
from io import BytesIO

def app():
    st.title("N3A Predictions")

    # Load the scaler
    with open('N3A_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Load the models and bounds
    models = {}
    upper_bounds = {}
    lower_bounds = {}
    for col in ['S1', 'S2', 'S3', 'S4', 'S5']:
        with open(f'N3A_model_pred_{col}.pkl', 'rb') as f:
            models[col] = pickle.load(f)
        with open(f'N3A_model_upper_{col}.pkl', 'rb') as f:
            upper_bounds[col] = pickle.load(f)
        with open(f'N3A_model_lower_{col}.pkl', 'rb') as f:
            lower_bounds[col] = pickle.load(f)

    # Date input
    date_input = st.date_input("Select a date")

    if st.button('Make Prediction'):
        if date_input:
            try:
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
                    prediction = model.predict(X_scaled)[0]
                    predictions[col] = prediction

                # Display predictions with bounds
                st.subheader("Predicted S1 to S5 with Bounds:")
                for col in ['S1', 'S2', 'S3', 'S4', 'S5']:
                    st.write(f"### {col}")
                    st.write(f"Predicted {col}: {predictions[col]:.2f}")
                    st.write(f"Upper Bound: {upper_bounds[col][0]:.2f}")
                    st.write(f"Lower Bound: {lower_bounds[col][0]:.2f}")

                # Prepare predictions for download
                predictions_df = pd.DataFrame({
                    'Pred_S1': [predictions['S1']],
                    'Upper_S1': [upper_bounds['S1'][0]],
                    'Lower_S1': [lower_bounds['S1'][0]],
                    'Pred_S2': [predictions['S2']],
                    'Upper_S2': [upper_bounds['S2'][0]],
                    'Lower_S2': [lower_bounds['S2'][0]],
                    'Pred_S3': [predictions['S3']],
                    'Upper_S3': [upper_bounds['S3'][0]],
                    'Lower_S3': [lower_bounds['S3'][0]],
                    'Pred_S4': [predictions['S4']],
                    'Upper_S4': [upper_bounds['S4'][0]],
                    'Lower_S4': [lower_bounds['S4'][0]],
                    'Pred_S5': [predictions['S5']],
                    'Upper_S5': [upper_bounds['S5'][0]],
                    'Lower_S5': [lower_bounds['S5'][0]]
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
                    file_name='N3A_predictions_with_bounds.xlsx',
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
