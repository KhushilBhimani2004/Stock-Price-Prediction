import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import plotly.graph_objects as go
import yfinance as yf

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Stock Price Prediction",
    layout="wide"
)

st.title("📈 Stock Price Prediction with LSTM")

# ---------------------------------------------------------
# STOCK INPUT
# ---------------------------------------------------------
st.header("Select a Stock from Yahoo Finance")

ticker_symbol = st.text_input(
    "Enter the stock symbol (e.g., AAPL for Apple):"
).upper()

# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
if "model" not in st.session_state:
    st.session_state.model = None

# ---------------------------------------------------------
# FETCH DATA
# ---------------------------------------------------------
if ticker_symbol:

    try:

        with st.spinner("Fetching stock data..."):

            # MORE STABLE METHOD
            data = yf.download(
                ticker_symbol,
                period="5y",
                interval="1d",
                progress=False,
                threads=False
            )

        # ---------------------------------------------------------
        # CHECK EMPTY DATA
        # ---------------------------------------------------------
        if data.empty:
            st.error(
                "❌ No data found. Please check stock symbol."
            )
            st.stop()

        # ---------------------------------------------------------
        # FIX MULTIINDEX ISSUE
        # ---------------------------------------------------------
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        # ---------------------------------------------------------
        # DISPLAY DATA
        # ---------------------------------------------------------
        st.success("✅ Data fetched successfully")

        st.subheader("Fetched Stock Data")

        st.dataframe(data)

        # ---------------------------------------------------------
        # DATASET PREPARATION
        # ---------------------------------------------------------
        dataset = data[['Close']].copy()

        scaler = MinMaxScaler(feature_range=(0, 1))

        dataset_scaled = scaler.fit_transform(dataset)

        # ---------------------------------------------------------
        # CREATE DATASET FUNCTION
        # ---------------------------------------------------------
        def create_dataset(data, time_steps=1):

            X = []
            y = []

            for i in range(len(data) - time_steps):

                X.append(data[i:(i + time_steps), 0])

                y.append(data[i + time_steps, 0])

            return np.array(X), np.array(y)

        # ---------------------------------------------------------
        # TIME STEPS
        # ---------------------------------------------------------
        time_steps = st.number_input(
            "Number of Time Steps",
            min_value=1,
            max_value=30,
            value=5
        )

        # ---------------------------------------------------------
        # TRAIN MODEL
        # ---------------------------------------------------------
        st.header("Train LSTM Model")

        if st.button("Train LSTM Model"):

            with st.spinner("Training model..."):

                # CREATE DATASET
                X_train, y_train = create_dataset(
                    dataset_scaled,
                    time_steps
                )

                # RESHAPE
                X_train = X_train.reshape(
                    X_train.shape[0],
                    X_train.shape[1],
                    1
                )

                # MODEL
                model = Sequential()

                model.add(
                    LSTM(
                        units=50,
                        return_sequences=True,
                        input_shape=(X_train.shape[1], 1)
                    )
                )

                model.add(LSTM(units=50))

                model.add(Dense(1))

                # COMPILE
                model.compile(
                    optimizer='adam',
                    loss='mean_squared_error'
                )

                # TRAIN
                model.fit(
                    X_train,
                    y_train,
                    epochs=10,
                    batch_size=1,
                    verbose=1
                )

                # SAVE MODEL
                st.session_state.model = model

            st.success("✅ LSTM Model Trained Successfully")

        # ---------------------------------------------------------
        # PREDICTION
        # ---------------------------------------------------------
        st.header("Predict Stock Price")

        if st.button("Make Predictions"):

            if st.session_state.model is None:

                st.error("❌ Please train the model first.")

            else:

                with st.spinner("Predicting..."):

                    # LAST DAYS
                    last_days = dataset_scaled[-time_steps:]

                    # RESHAPE
                    last_days = last_days.reshape(
                        1,
                        time_steps,
                        1
                    )

                    # PREDICT
                    predicted_price = (
                        st.session_state.model.predict(last_days)
                    )

                    # INVERSE TRANSFORM
                    predicted_price = scaler.inverse_transform(
                        predicted_price
                    )

                st.subheader(
                    "📌 Predicted Next Day Closing Price"
                )

                st.success(
                    f"${predicted_price[0][0]:.2f}"
                )

        # ---------------------------------------------------------
        # OPEN VS CLOSE
        # ---------------------------------------------------------
        st.header("📊 Open vs Close")

        fig1 = go.Figure()

        fig1.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Open'],
                mode='lines',
                name='Open Price'
            )
        )

        fig1.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Close'],
                mode='lines',
                name='Close Price'
            )
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

        # ---------------------------------------------------------
        # HIGH VS LOW
        # ---------------------------------------------------------
        st.header("📊 High vs Low")

        fig2 = go.Figure()

        fig2.add_trace(
            go.Scatter(
                x=data.index,
                y=data['High'],
                mode='lines',
                name='High Price'
            )
        )

        fig2.add_trace(
            go.Scatter(
                x=data.index,
                y=data['Low'],
                mode='lines',
                name='Low Price'
            )
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        # ---------------------------------------------------------
        # CANDLESTICK
        # ---------------------------------------------------------
        st.header("📉 Historical Stock Prices")

        fig3 = go.Figure(
            data=[
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close']
                )
            ]
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

    except Exception as e:

        st.error(f"❌ Error: {str(e)}")


# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# import plotly.graph_objects as go
# import plotly.express as px
# import yfinance as yf

# st.title("Stock Price Prediction with LSTM")

# # Upload a dataset
# st.header("Upload a CSV file with historical stock prices")
# uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

# # Stock selection from yfinance
# st.header("Select a Stock from Yahoo Finance")
# ticker_symbol = st.text_input("Enter the stock symbol (e.g., AAPL for Apple):")

# model = Sequential()

# if ticker_symbol:
#     try:
#         # Fetch stock data from yfinance
#         stock_data = yf.Ticker(ticker_symbol)
#         data = stock_data.history(period="5Y")

#         st.subheader("Fetched Data from Yahoo Finance")
#         st.write(data)

#         # Predict stock prices
#         st.header("Predict Stock Prices")

#         # Extract the 'Close' prices
#         dataset = data[['Close']]

#         # Normalize the dataset using Min-Max scaling
#         scaler = MinMaxScaler()
#         dataset['Close'] = scaler.fit_transform(dataset['Close'].values.reshape(-1, 1))

#         # Create a function to prepare data for LSTM
#         def create_dataset(data, time_steps=1):
#             X, y = [], []
#             for i in range(len(data) - time_steps):
#                 X.append(data[i:(i + time_steps), 0])
#                 y.append(data[i + time_steps, 0])
#             return np.array(X), np.array(y)

#         # Choose the number of time steps (e.g., 5 days)
#         time_steps = st.number_input("Number of Time Steps", min_value=1, max_value=30, value=5)

#         if st.button("Train LSTM Model"):
#             # Create the training dataset
#             X_train, y_train = create_dataset(dataset.values, time_steps)

#             # Reshape the data for LSTM input
#             X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

#             # Clear existing model state to ensure fresh start
#             model = Sequential()

#             # Create an LSTM model
#             model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
#             model.add(LSTM(units=50))
#             model.add(Dense(1))

#             # Compile the model
#             model.compile(loss='mean_squared_error', optimizer='adam')

#             # Train the model
#             model.fit(X_train, y_train, epochs=10, batch_size=1)

#             st.success("LSTM Model Trained")

#         # Make predictions
#         if st.button("Make Predictions"):
#             # Prepare data for prediction
#             last_days = dataset[-time_steps:].values
#             last_days = last_days.reshape(1, time_steps, 1)
#             next_day_price = model.predict(last_days)

#             # Rescale the prediction back to the original scale
#             next_day_price = scaler.inverse_transform(next_day_price.reshape(-1, 1))

#             st.subheader("Predicted Stock Price for the Next Day")
#             st.write(next_day_price[0][0])

#     except Exception as e:
#         st.error(f"Error fetching data: {str(e)}")

# # Display stock price charts
# if 'data' in locals():
#     st.header("Open VS Close")
#     line_fig = go.Figure()
#     line_fig.add_trace(go.Scatter(x=data.index, y=data['Open'], mode='lines', name='Open Price'))
#     line_fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))
#     st.plotly_chart(line_fig, use_container_width=True)

#     # Interactive scatter plot for "High" and "Low" columns
#     st.header("High VS Low")
#     high_low_fig = px.line(data, x=data.index, y=['High', 'Low'], labels={'x': 'Date', 'value': 'Price'})
#     st.plotly_chart(high_low_fig, use_container_width=True)

#     # Interactive historical stock price plot
#     st.header("Historical Stock Prices")
#     fig = go.Figure(data=[go.Candlestick(x=data.index,
#                                          open=data['Open'],
#                                          high=data['High'],
#                                          low=data['Low'],
#                                          close=data['Close'])])

#     # Set y-axis interval to 20 values
#     fig.update_yaxes(range=[100, max(data['High'])], dtick=10)

#     st.plotly_chart(fig, use_container_width=True)
