import streamlit as st
from datetime import date
# from datetime import datetime
# from datetime import timedelta
import datetime


import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

import numpy as np
import pandas as pd

START = "2019-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Stock Prediction App')

stocks = ('GOOG', 'AAPL', 'MSFT', 'NTAP','JPM','NFLX')
selected_stock = st.selectbox('Select dataset for prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 10)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

st.subheader('Stock prediction using Meta Prophet')
# Predict forecast with Prophet.
df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Forecast data')
st.write(forecast.tail())
    
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)

#LSTM model
st.subheader('Stock prediction using LSTM')

model_load_state = st.text('Loading model...')

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Convert the 'y' values to a 1D NumPy array
y = df_train['y'].values

# Normalize the data using Min-Max scaling
scaler = MinMaxScaler()
y_scaled = scaler.fit_transform(y.reshape(-1, 1))

# Split the data into training and testing sets
train_size = int(len(y_scaled) * 0.8)
train_data, test_data = y_scaled[:train_size], y_scaled[train_size:]

# Define a function to create time series data
def create_time_series_data(dataset, time_steps=1):
    X, Y = [], []
    for i in range(len(dataset) - time_steps):
        X.append(dataset[i:(i + time_steps), 0])
        Y.append(dataset[i + time_steps, 0])
    return np.array(X), np.array(Y)

# Set the number of time steps (you can experiment with this)
time_steps = 30

# Create time series data
X_train, y_train = create_time_series_data(train_data, time_steps)
X_test, y_test = create_time_series_data(test_data, time_steps)

# Reshape the data for LSTM input (samples, time steps, features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(loss='mse',
              optimizer=Adam(learning_rate=0.001),
              metrics=['mean_absolute_error'])

# Train the model
model.fit(X_train, y_train, batch_size=1, epochs=5)

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Inverse transform the scaled predictions to the original scale
train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)

# Calculate RMSE for train and test predictions
train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

# Create a DataFrame for the predictions
train_predict_df = df_train.iloc[time_steps:train_size].copy()
train_predict_df['Predictions'] = train_predictions

test_predict_df = df_train.iloc[train_size+time_steps:].copy()
test_predict_df['Predictions'] = test_predictions

model_load_state.text('Loading model... done!')


# Plot the training and test predictions
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_train['ds'], y=df_train['y'], mode='lines', name='Actual Close Price'))
fig.add_trace(go.Scatter(x=train_predict_df['ds'], y=train_predict_df['y'], mode='lines', name='Training Predictions'))
fig.add_trace(go.Scatter(x=test_predict_df['ds'], y=test_predict_df['y'], mode='lines', name='Test Predictions'))
fig.update_layout(
    title=f'LSTM Model - Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}',
    xaxis_title='Date',
    yaxis_title='Close Price USD ($)'
)
st.plotly_chart(fig)

#Recursive prediction using deep copy
# from copy import deepcopy

# # Make recursive predictions
# recursive_predictions = []

# # Start with the last sequence from the training data
# current_sequence = X_train[-1:]

# for i in range(period):
#     # Predict the next value
#     next_value = model.predict(current_sequence)[0][0]
    
#     # Append the predicted value to the result list
#     recursive_predictions.append(next_value)
    
#     # Create a deep copy of the current sequence to avoid modifying the original sequence
#     new_sequence = deepcopy(current_sequence)
    
#     # Update the new sequence by removing the first element and adding the predicted value
#     new_sequence = np.append(new_sequence[0][1:], [[next_value]], axis=0)
#     new_sequence = new_sequence.reshape(1, time_steps, 1)
    
#     # Set the current sequence to the new sequence
#     current_sequence = new_sequence

# # Inverse transform the scaled predictions to the original scale
# recursive_predictions = scaler.inverse_transform(np.array(recursive_predictions).reshape(-1, 1))

# # Create a DataFrame for the recursive predictions
# recursive_predict_dates = pd.date_range(start=df_train['ds'].iloc[-1], periods=len(recursive_predictions))

# recursive_predict_df = pd.DataFrame({'ds': recursive_predict_dates, 'Predictions': recursive_predictions.flatten()})


# # Plot the training, test, and recursive predictions
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=df_train['ds'], y=df_train['y'], mode='lines', name='Actual Close Price'))
# fig.add_trace(go.Scatter(x=train_predict_df['ds'], y=train_predict_df['y'], mode='lines', name='Training Predictions'))
# fig.add_trace(go.Scatter(x=test_predict_df['ds'], y=test_predict_df['y'], mode='lines', name='Test Predictions'))
# fig.add_trace(go.Scatter(x=recursive_predict_df['ds'], y=recursive_predict_df['Predictions'], mode='lines', name='Recursive Predictions'))
# fig.update_layout(
#     title=f'LSTM Model with Recursive Prediction',
#     xaxis_title='Date',
#     yaxis_title='Close Price USD ($)'
# )
# st.plotly_chart(fig)