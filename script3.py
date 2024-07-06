import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf

st.title("Stock Price Prediction with LSTM")

# Upload a dataset
st.header("Upload a CSV file with historical stock prices")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

# Stock selection from yfinance
st.header("Select a Stock from Yahoo Finance")
ticker_symbol = st.text_input("Enter the stock symbol (e.g., AAPL for Apple):")

if ticker_symbol:
    # Fetch stock data from yfinance
    stock_data = yf.Ticker(ticker_symbol)
    data = stock_data.history(period="5y")

    st.subheader("Fetched Data from Yahoo Finance")
    st.write(data)

    # Predict stock prices
    st.header("Predict Stock Prices")

    # Extract the 'Close' prices
    dataset = data[['Close']]

    # Normalize the dataset using Min-Max scaling
    # scaler = MinMaxScaler()
    # dataset['Close'] = scaler.fit_transform(dataset['Close'].values.reshape(-1, 1))
# Normalize the dataset using Min-Max scaling
    # Normalize the dataset using Min-Max scaling
    # Normalize the dataset using Min-Max scaling
    scaler = MinMaxScaler()
    dataset_copy = dataset.copy()  # Make a copy of the DataFrame
    
    dataset_copy.loc[:, 'Close'] = scaler.fit_transform(dataset_copy['Close'].values.reshape(-1, 1))

# Use dataset_copy for subsequent operations

    dataset.loc[:, 'Close'] = scaler.fit_transform(dataset['Close'].values.reshape(-1, 1))


    # Create a function to prepare data for LSTM
    def create_dataset(data, time_steps=1):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i:(i + time_steps), 0])
            y.append(data[i + time_steps, 0])
        return np.array(X), np.array(y)

    # Choose the number of time steps (e.g., 5 days)
    time_steps = st.number_input("Number of Time Steps", min_value=1, max_value=30, value=5)

    if st.button("Train LSTM Model"):
        # Create the training dataset
        X_train, y_train = create_dataset(dataset.values, time_steps)

        # Reshape the data for LSTM input
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

        # Create an LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))

        # Compile the model
        model.compile(loss='mean_squared_error', optimizer='adam')

        # Train the model
        model.fit(X_train, y_train, epochs=10, batch_size=1)

        st.success("LSTM Model Trained")

    # Make predictions
    if st.button("Make Predictions"):
        if 'model' in globals():
            last_days = dataset[-time_steps:].values
            last_days = last_days.reshape(1, -1, 1)
            next_day_price = model.predict(last_days)

            # Rescale the prediction back to the original scale
            next_day_price = scaler.inverse_transform(next_day_price.reshape(-1, 1))

            st.subheader("Predicted Stock Price for the Next Day")
            st.write(next_day_price[0][0])
        else:
            st.error("Please train the model first.")

    st.header("Open VS Close")
    line_fig = go.Figure()
    line_fig.add_trace(go.Scatter(x=data.index, y=data['Open'], mode='lines', name='Open Price'))
    line_fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
    st.plotly_chart(line_fig, use_container_width=True)

    # Interactive scatter plot for "High" and "Low" columns
    st.header("High VS Low")
    high_low_fig = px.line(data, x=data.index, y=['High', 'Low'], labels={'x': 'Date', 'value': 'Price'})
    st.plotly_chart(high_low_fig, use_container_width=True)

    # Interactive historical stock price plot
    st.header("Historical Stock Prices")
    fig = go.Figure(data=[go.Candlestick(x=data.index,
                                         open=data['Open'],
                                         high=data['High'],
                                         low=data['Low'],
                                         close=data['Close'])])

    # Set y-axis interval to 20 values
    fig.update_yaxes(range=[100, max(data['High'])], dtick=10)

    st.plotly_chart(fig, use_container_width=True)

# Handle uploaded file (optional feature, added for completeness)
if uploaded_file:
    uploaded_data = pd.read_csv(uploaded_file)
    st.subheader("Uploaded CSV File Data")
    st.write(uploaded_data)
