import yfinance as yf
import streamlit as st
from dotenv import load_dotenv
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import google.generativeai as genai
from io import BytesIO

# Load environment variables from .env file
load_dotenv()

# Configure the Google Gemini model with API key
api_key = os.getenv('GENAI_API_KEY')
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-pro")

# Function to fetch stock data from Yahoo Finance using yfinance
def fetch_yfinance_stock_data(symbol):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="1y")  # Fetching 1 year of historical data
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol} from Yahoo Finance: {str(e)}")
        return None

# Function to interact with Gemini AI for stock-related queries
def get_gemini_response(symbol):
    prompt_template = (
        "You are an intelligent assistant with expertise in stock market analysis. "
        "Provide detailed insights or analysis on the following stock:\n\n"
        "Stock: {}\n"
        "Analysis:"
    )

    response = model.generate_content(prompt_template.format(symbol), stream=True)
    full_text = ""
    for chunk in response:
        full_text += chunk.text
    return full_text

# Streamlit app setup
st.set_page_config(page_title="Stock Data Dashboard", layout="wide")
st.title("Stock Data Dashboard")

# Streamlit inputs
symbols = st.multiselect("Select Stock Symbols", ["RELIANCE.BO", "TCS.BO", "HDFCBANK.BO", "INFY.BO", "ICICIBANK.BO", 
    "HINDUNILVR.BO", "BHARTIARTL.BO", "ITC.BO", "KOTAKBANK.BO", "SBI.BO",
    "LTI.BO", "WIPRO.BO", "HCLTECH.BO", "M&M.BO", "ADANIGREEN.BO", "NTPC.BO",
    "POWERGRID.BO", "ONGC.BO", "BAJFINANCE.BO", "JSWSTEEL.BO", "HDFC.BO",
    "M&MFIN.BO", "SBILIFE.BO", "CIPLA.BO", "DRREDDY.BO", "SUNPHARMA.BO", "TSLA"], default=["RELIANCE.BO"])

if st.button("Fetch Data"):
    if symbols:
        with st.spinner("Fetching stock data..."):  # Loading spinner
            all_figures = []
            all_data = {}  # To store data for CSV download
            for symbol in symbols:
                data = fetch_yfinance_stock_data(symbol)
                if data is not None and not data.empty:
                    all_data[symbol] = data  # Store data for download
                    st.subheader(f"Stock Data for {symbol}")

                    # Display raw data
                    st.write("**Displaying last 5 records:**")
                    st.write(data.tail())

                    # Create a column layout for visualizations
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        # Stock Price Chart with Trendline
                        st.subheader(f"Stock Price Chart for {symbol}")
                        fig, ax = plt.subplots(figsize=(4, 3))  # Smaller figure size
                        data['Close'].plot(ax=ax, title=f"Closing Prices for {symbol}", legend=True)

                        # Adding a trendline
                        x = np.arange(len(data))
                        z = np.polyfit(x, data['Close'], 1)  # Linear fit
                        p = np.poly1d(z)
                        ax.plot(data.index, p(x), color='red', linestyle='--', label='Trendline')

                        plt.xlabel("Date")
                        plt.ylabel("Price (USD)")
                        plt.legend()
                        st.pyplot(fig)
                        all_figures.append(fig)

                    with col2:
                        # Volume Traded Chart
                        st.subheader(f"Volume Traded Chart for {symbol}")
                        fig, ax = plt.subplots(figsize=(4, 3))  # Smaller figure size
                        data['Volume'].plot(ax=ax, color='orange', title=f"Volume Traded for {symbol}", legend=True)
                        plt.xlabel("Date")
                        plt.ylabel("Volume")
                        st.pyplot(fig)
                        all_figures.append(fig)

                    with col3:
                        # Moving Average Chart
                        st.subheader(f"Moving Average Chart for {symbol}")
                        fig, ax = plt.subplots(figsize=(4, 3))  # Smaller figure size
                        data['Close'].rolling(window=30).mean().plot(ax=ax, color='blue', label='30-Day Moving Average')
                        data['Close'].plot(ax=ax, title=f"Closing Prices and Moving Average for {symbol}", legend=True)
                        plt.xlabel("Date")
                        plt.ylabel("Price (USD)")
                        plt.legend()
                        st.pyplot(fig)
                        all_figures.append(fig)

                    # Fetch and display Gemini response
                    gemini_response = get_gemini_response(symbol)
                    st.subheader(f"Gemini Response for {symbol}")
                    st.write(gemini_response)

                    # Future Trend Analysis
                    st.subheader(f"Future Trend Analysis for {symbol}")
                    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=30)
                    future_prices = data['Close'].iloc[-1] * (1 + np.random.normal(0, 0.05, size=30)).cumprod()  # Simulated prices
                    future_data = pd.Series(future_prices, index=future_dates)

                    # Plot future trend
                    fig, ax = plt.subplots(figsize=(3, 2))  # Smaller figure size
                    data['Close'].plot(ax=ax, label='Historical Prices', legend=True, fontsize=8)
                    future_data.plot(ax=ax, label='Projected Future Prices', color='green', linestyle='--', legend=True, fontsize=8)

                    plt.title(f"Projected Future Prices for {symbol}", fontsize=10)
                    plt.xlabel("Date", fontsize=8)
                    plt.ylabel("Price (USD)", fontsize=8)
                    plt.legend(fontsize=8)
                    st.pyplot(fig)
                    all_figures.append(fig)

                    # CSV Download
                    csv = data.to_csv().encode()
                    st.download_button(f"Download {symbol} Data as CSV", csv, f"{symbol}_data.csv", "text/csv")

                else:
                    st.error(f"Failed to fetch data for {symbol}")

    else:
        st.error("Please select at least one stock symbol.")
