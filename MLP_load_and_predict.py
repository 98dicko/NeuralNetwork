import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model

MLP_loaded = load_model('Models/eth_train_model2.h5')


start = '2021-02-08'
end = '2021-02-28'



loaded = load_model('Models/lstm.h5')


start = '2021-02-08'
end = '2021-02-28'

while True:
    data = pd.read_csv("ETH-USD.csv", index_col=0)
    data = data.loc[start:end]
    crypto = data['Adj Close']
    X_predict = np.array(crypto).reshape((1, 21, 1))
    output = loaded.predict(X_predict,verbose=0)
    start = (datetime.strptime(start, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    end = (datetime.strptime(end, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    print("Predicted price for " + str(end) + " is " + str(output))
    time.sleep(5)
