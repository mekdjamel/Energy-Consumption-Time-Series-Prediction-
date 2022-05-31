import pickle
from keras.models import Model
from keras.layers import LSTM
from keras.layers import Dense, Input
import math

# load data
train_opsd_daily_Consumption, test_opsd_daily_Consumption = pickle.load(open('Train-Test_opsd_daily_Consumption.pkl', "rb"))

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def create_lookback_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)
# Normalize the opsd_daily_Consumption (rescaling of the data from the original range so that all values are within the range of 0 and 1.)
scaler = MinMaxScaler(feature_range=(0, 1))
train_opsd_daily_Consumption = scaler.fit_transform(train_opsd_daily_Consumption)
test_opsd_daily_Consumption = scaler.fit_transform(test_opsd_daily_Consumption)

# take the previous lookback values to predict the current value
trainX_opsd_daily_Consumption, trainY_opsd_daily_Consumption = create_lookback_dataset(train_opsd_daily_Consumption, 7)
testX_opsd_daily_Consumption, testY_opsd_daily_Consumption = create_lookback_dataset(test_opsd_daily_Consumption, 7)


trainX_opsd_daily_Consumption = np.reshape(trainX_opsd_daily_Consumption, (trainX_opsd_daily_Consumption.shape[0], trainX_opsd_daily_Consumption.shape[1], 1))
testX_opsd_daily_Consumption = np.reshape(testX_opsd_daily_Consumption, (testX_opsd_daily_Consumption.shape[0], testX_opsd_daily_Consumption.shape[1], 1))


""""""""""""""""'MODEL PREDICTION'"""""""""""""""
input = Input(shape=(trainX_opsd_daily_Consumption.shape[1], trainX_opsd_daily_Consumption.shape[2]), name='history_Finput')
out = LSTM(64, activation='tanh')(input)
out = Dense(64, activation='relu')(out)
main_output = Dense(1, activation='relu', name='main_output')(out)

model = Model(inputs=input, outputs=main_output)
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(trainX_opsd_daily_Consumption, trainY_opsd_daily_Consumption, epochs=100,
                        batch_size=128,
                        validation_split=0.2,
                        verbose=2,
                        shuffle=False)


testPredict = model.predict(testX_opsd_daily_Consumption)
testPredict = scaler.inverse_transform(testPredict)
testY_opsd_daily_Consumption = scaler.inverse_transform([testY_opsd_daily_Consumption])
RMSE = math.sqrt(mean_squared_error(testY_opsd_daily_Consumption[0], testPredict[:, 0]))
print('Test Score RMSE: ', RMSE)
