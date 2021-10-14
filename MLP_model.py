import tensorflow as tf
import numpy as np
from preprocessing import DataProcessing
import pandas as pd
import time
import os.path
from datetime import datetime, timedelta


process = DataProcessing("ETH.csv", 0.3)
process.gen_test(21)
process.gen_train(21)

X_train = process.X_train
Y_train = process.Y_train
X_test = process.X_test
Y_test = process.Y_test

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(1, activation=tf.nn.relu))

opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
np.random.seed(1337)
model.compile(optimizer=opt, loss="mean_squared_error")
model.fit(X_train, Y_train, validation_split=0.1,batch_size=10, epochs=10000)

if os.path.isfile('Models/eth_train_model2.h5') is False:
    model.save('Models/eth_train_model2.h5')


start = '2021-02-08'
end = '2021-02-28'



while True:
    data = pd.read_csv("ETH-USD.csv", index_col=0)
    data = data.loc[start:end]
    crypto = data['Adj Close']
    X_predict = np.array(crypto).reshape((1, 21))
    output = model.predict(X_predict,verbose=0)
    start = (datetime.strptime(start, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    end = (datetime.strptime(end, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    print("Predicted price for " + str(end) + " is " + str(output))
    time.sleep(5)
