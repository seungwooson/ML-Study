"""
Created on Tue Jul 18 06:57:15 2017

@author: Seung Woo Son
"""
import math
import tensorflow
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.graphics.api as smg
from patsy import dmatrices

filename = "./LSTM/bitcoin_ticker.csv"

df = pd.read_csv(filename)
df = df[df["market"]=="bitstamp"]
df = df[df["rpt_key"]=="btc_usd"]
df = df[["date_id", "datetime_id", "rpt_key", "last"]]
df = df.pivot_table(index="datetime_id", columns="rpt_key")
ts = pd.Series(df["last", "btc_usd"], index=df.index)
back_to_frame = pd.Series.to_frame(ts)
back_to_frame = back_to_frame.astype('float64')
testset = back_to_frame.values
testset
# writer = pd.ExcelWriter('changedUrl.xlsx')
# ts.to_excel(writer, 'Sheet1')
# writer.save()


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=54):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)
# fix random seed for reproducibility
np.random.seed(7)
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
testset = scaler.fit_transform(testset)
testset
# split into train and test sets
train_size = int(len(testset) * 0.67)
test_size = len(testset) - train_size
train, test = testset[0:train_size,:], testset[train_size:len(testset),:]
train.shape
test.shape
print(test.shape)
# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
trainX.shape
trainY.shape
print(trainX)
print(trainY)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=20, batch_size=1, verbose=2)
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = np.empty_like(testset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(testset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(testset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(testset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
