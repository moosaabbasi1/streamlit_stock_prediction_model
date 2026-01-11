
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os
import plotly.graph_objects as go


st.write("Current Directory:", os.getcwd())
st.write("Files in Directory:", os.listdir())


#  App Configuration 
st.set_page_config(page_title="Stock Price Prediction", layout="wide")
plt.style.use("fivethirtyeight")

#  App Title 
st.title(" Stock Price Prediction Model")
st.subheader("Programming for AI Lab Project")
st.subheader("Made by")
st.markdown("""       
            
              Moosa Abbasi
            
               Abdul Ahad
            
               Hasnian Raees Abbasi""")

st.markdown("""
This web application predicts stock prices using a **Deep Learning (LSTM) model**.
Historical stock data is fetched from **Yahoo Finance**, and predictions are visualized
against real market prices.
""")

#  Stock Selection 
stock_ticker = st.selectbox(
    "Select Stock Ticker",
    ['AAPL', 'POWERGRID.NS', 'ETH-USD', 'BTC-USD', 'INTC', 'NVDA']
)

start_date = "2015-01-01"
end_date = "2026-01-01"

#  Load Stock Data 
@st.cache_data
def load_stock_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end)

df = load_stock_data(stock_ticker, start_date, end_date)

# Error handling if data is empty
if df.empty:
    st.error(f"No data found for '{stock_ticker}'. Please try another ticker.")
    st.stop()

#  Data Overview 
st.subheader("shape of the Data :")
st.write(df.shape)

st.subheader("Stock Data Summary (2010 - 2027)")
st.write(df.describe())
#columns
st.subheader("avaliable columns ")
st.write(df.columns)

#Candelstick 
if isinstance(df.columns , pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.reset_index()

fig = go.Figure(data = [go.Candlestick(x = df.index ,
                                       open = df['Open'],
                                     high = df['High'],
                                        low = df['Low'],
                                       close = df['Close'])])
fig.update_layout(xaxis_rangeslider_visible = False , title =f"{stock_ticker} Candel stick chart")
st.plotly_chart(fig , use_container_width = True)
    
#  Closing Price Chart 
st.subheader(" Closing Price vs Time")
fig1 = plt.figure(figsize=(12, 6))
plt.plot(df['Close'])
plt.xlabel("Time")
plt.ylabel("Closing Price")
st.pyplot(fig1)

#  Opening Price Chart 
st.subheader(" Opeing Price vs Time")
fig2 = plt.figure(figsize=(12, 6))
plt.plot(df['Open'])
plt.xlabel("Time")
plt.ylabel("opening Price")
st.pyplot(fig2)


#  Load Trained Model 
model_path = "stock_dl_model.h5"

if not os.path.exists(model_path):
    st.error("Model file not found. Please ensure 'stock_dl_model.h5' is in the project directory.")
    st.stop()

model = load_model(model_path)

#  Data Preparation 
data = df[['Close']]
train_size = int(len(data) * 0.70)

data_training = data[:train_size]
data_testing = data[train_size:]

scaler = MinMaxScaler(feature_range=(0, 1))
data_training_scaled = scaler.fit_transform(data_training)

# Use last 100 days from training + testing data
past_100_days = data_training.tail(100)
final_data = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scaler.transform(final_data)

#  Create Test Dataset 
x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100:i])
    y_test.append(input_data[i, 0])

x_test = np.array(x_test)
y_test = np.array(y_test)

#  Prediction 
y_predicted = model.predict(x_test)
y_predicted = scaler.inverse_transform(y_predicted)
y_test = scaler.inverse_transform(y.test.reshape(-1,1))

# Reverse scaling

#scale_factor = 1 / scaler.scale_[0]
#y_predicted = y_predicted * scale_factor
#y_test = y_test * scale_factor

#  Prediction Visualization 
st.subheader(" Predicted Price vs Original Price")
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, label="Original Price", color="blue")
plt.plot(y_predicted, label="Predicted Price", color="red")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig2)

#Footer 
st.markdown("---")
st.markdown("**Developed for Programming for AI Lab**")
