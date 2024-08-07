import streamlit as st
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime

def create_date_features(df):
    """Create date features from the date column."""
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    return df

def fit_and_predict(df, selected_feature, num_days):
    """Fit a SARIMAX model and make forecasts."""
    df = df.rename(columns={'Date': 'ds', selected_feature: 'y'})
    
    # Fit the SARIMAX model
    model = SARIMAX(df['y'], 
                    order=(1, 1, 1),  # Change these orders as necessary
                    seasonal_order=(1, 1, 1, 12))  # Change seasonal order as needed
    model_fit = model.fit(disp=False)
    
    # Generate future dates
    future_dates = pd.date_range(start=df['ds'].max(), periods=num_days + 1, closed='right')
    
    # Forecast
    forecast = model_fit.get_forecast(steps=num_days)
    forecast_df = forecast.summary_frame()
    forecast_df['ds'] = future_dates
    forecast_df.rename(columns={'mean': 'yhat', 'mean_ci_lower': 'yhat_lower', 'mean_ci_upper': 'yhat_upper'}, inplace=True)
    
    return forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

def app():
    st.title('Forecasts App with SARIMAX')
    st.write('Upload your data and specify the forecast settings below.')

    # File upload
    uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])
    
    if uploaded_file:
        try:
            # Read the uploaded file
            df = pd.read_excel(uploaded_file)
            
            # Show the uploaded data
            st.subheader("Uploaded Data")
            st.dataframe(df)

            # Feature selection
            selected_feature = st.selectbox("Select the feature to forecast", df.columns)

            # Create date features
            df = create_date_features(df)

            # Slider input
            num_days = st.slider("Number of days to forecast", min_value=1, max_value=30, value=7)
            
            if st.button('Generate Forecast'):
                with st.spinner('Generating forecast...'):
                    try:
                        forecast_df = fit_and_predict(df, selected_feature, num_days)

                        st.subheader('Forecast Results:')
                        st.dataframe(forecast_df)

                    except Exception as e:
                        st.error(f"An error occurred: {e}")

        except Exception as e:
            st.error(f"An error occurred while reading the file: {e}")

if __name__ == '__main__':
    app()
