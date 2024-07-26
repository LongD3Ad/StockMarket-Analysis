import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas_datareader import data as pdr #not exactly useful..is depriciated and has errors.
import datetime
from keras.models import load_model
import streamlit as st

import yfinance as yf
yf.pdr_override()

start= datetime.datetime(2000, 1, 1)
end =datetime.datetime(2024, 2, 28)

st.title('STOCK TREND PREDICTIONS')

user_input = st.text_input('Enter the stock ticker', '^NSEI')
df = yf.download(user_input, start=start, end=end)

#describing the data

st.subheader('Data from 2000-2024')
st.write(df.describe())


#Visualizations

st.subheader('Closing price VS Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing price VS Time Chart for 20 Moving average')
mvavg = df.Close.rolling(20).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(mvavg, 'r')
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing price VS Time Chart for 20 & 50 Moving average')
mvavg = df.Close.rolling(20).mean()
mvavg1 = df.Close.rolling(50).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(mvavg, 'r')
plt.plot(mvavg1, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

dtrain = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
dtest = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

dtarray = scaler.fit_transform(dtrain)


#Load the model

model = load_model('keras_model.keras')


#Testing the model

past_100_days = dtrain.tail(100)
final_df = pd.concat([past_100_days, dtest], ignore_index=True)
input_data = scaler.fit_transform(final_df)


x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)

#plotting

scaler = scaler.scale_
scale_factor = 1/scaler[0]

y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor


#final graphs
st.subheader('Predictions VS Originals')
fig2 = plt.figure(figsize = (12,6))
plt.plot(y_test, 'b', label = 'Original PRICE')
plt.plot(y_predicted, 'r', label = 'Predicted PRICE')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


    
    
