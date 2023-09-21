import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pickle
from statsmodels.tsa.arima.model import ARIMA


st.set_page_config(page_title="Oil Price Prediction", page_icon=":smiley:", layout="centered")
st.markdown(
    """
    <style>
    .main {
        background-image: url('https://www.fool.com.au/wp-content/uploads/2018/11/oil-price-increase.jpg');
        background-size: cover;
        background-position: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Load the ARMA model from a pickle file
with open("D:/Oil Project/arima_model.pkl", "rb") as file:
    model_fit = pickle.load(file)

def main():
    st.markdown(
        f'<h1 style="color: black;">Oil Price Prediction</h1>',
        unsafe_allow_html=True
    )
        
    # Forecasting
    start_date = st.date_input('Enter the start date:')
    end_date = st.date_input('Enter the end date:')
    forecast_button = st.button('Forecast')
    
    if forecast_button:
        # Convert start and end dates to pandas datetime
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Generate dates between start and end date
        forecast_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Make the forecast using the ARMA model
        forecast = model_fit.forecast(steps=len(forecast_dates))
        
        # Create a DataFrame with forecasted values and dates
        forecast_data = pd.DataFrame({'Date': forecast_dates, 'Price': forecast})
        
        forecast_data = forecast_data.reset_index(drop=True)
        
        # Display the forecasted values with dates
        st.write(forecast_data)
        
              
        # Plot the predicted values
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(forecast_data['Date'], forecast_data['Price'], color='orange', label='Predicted Values')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        
        # Display the plot
        st.pyplot(fig)
        
        csv_exp = forecast_data.to_csv(index=False)
        st.download_button(
            label='Download Forecasted Data (CSV)',
            data=csv_exp,
            file_name='forecast_data.csv',
            mime='text/csv'
        )

if __name__ == '__main__':
    main()








