
import pandas as pd
import numpy as np
import tensorflow as tf
from preprocessing import DataProcessing
from datetime import datetime, timedelta
import os.path
import time



process = DataProcessing("ETH.csv", 0.3)
process.gen_test(21)
process.gen_train(21)

process.X_train = process.X_train[0:4074]
X_train = process.X_train.reshape((398, 21, 1))
Y_train = process.Y_train
X_test = process.X_test.reshape((965,21,1))
Y_test = process.Y_test

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(40, input_shape=(21, 1), return_sequences=True))
model.add(tf.keras.layers.LSTM(40))
model.add(tf.keras.layers.Dense(1, activation='linear'))

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
np.random.seed(1337)
model.compile(optimizer=opt, loss="mean_squared_error")
model.fit(X_train, Y_train, epochs=100, batch_size=16)

if os.path.isfile('Models/eth_train_modelLSTM.h5') is False:
    model.save('Models/eth_train_modelLSTM.h5')

print(model.evaluate(X_test, Y_test))

start = '2021-02-08'
end = '2021-02-28'

while True:
    data = pd.read_csv("ETH-USD.csv", index_col=0)
    data = data.loc[start:end]
    crypto = data['Adj Close']
    X_predict = np.array(crypto).reshape((1, 21, 1))
    output = model.predict(X_predict,verbose=0)
    start = (datetime.strptime(start, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    end = (datetime.strptime(end, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    print("Predicted price for " + str(end) + " is " + str(output))
    time.sleep(5)