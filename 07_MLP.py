import warnings
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Silence warnings
warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv('03_TSLA.csv')
df["Close"].mean()
cl = df.Close.astype('float32')
train = cl[0:int(len(cl) * 0.80)]
scl = MinMaxScaler()

# Scale the data
scl.fit(train.values.reshape(-1, 1))
cl = scl.transform(cl.values.reshape(-1, 1))

# Create a function to process the data into lb observations look back slices and create the train test dataset (80-20)
def processData(data, lb):
    X, Y = [], []
    for i in range(len(data) - lb - 1):
        X.append(data[i:(i + lb), 0])
        Y.append(data[(i + lb), 0])
    return np.array(X), np.array(Y)

lb = 10
X, y = processData(cl, lb)
X_train, X_test = X[:int(X.shape[0] * 0.80)], X[int(X.shape[0] * 0.80):]
y_train, y_test = y[:int(y.shape[0] * 0.80)], y[int(y.shape[0] * 0.80):]
print(X_train.shape[0], X_train.shape[1])
print(X_test.shape[0], X_test.shape[1])
print(y_train.shape[0])
print(y_test.shape[0])

########################################################################################################################
################################################## MLP forecasting #####################################################
########################################################################################################################

# Create model
model = Sequential()
model.add(Dense(256, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(1))

# Optimizers and loss
#model.compile(optimizer='adam', loss='mse')
#model.compile(optimizer='RMSprop', loss='mse')
model.compile(optimizer='SGD', loss='mse')

# Fit model with history to check for overfitting
history = model.fit(X_train, y_train, epochs=400, validation_data=(X_test, y_test), shuffle=False)
model.summary()

# Calculate RMSE
testPredict = model.predict(X_test)
testPredict = scl.inverse_transform(testPredict)
testY = scl.inverse_transform([y_test])
RMSE = sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('RMSE = {}'.format(RMSE))

########################################################################################################################
#################################################### Create plot #######################################################
########################################################################################################################

plt.style.use('fivethirtyeight')
plt.figure(figsize=(12,8))
Xt = model.predict(X_test)
plt.plot(scl.inverse_transform(y_test.reshape(-1, 1)), linestyle='dashed', marker='o', color='blue', label='Πραγματική Τιμή')
plt.plot(scl.inverse_transform(Xt), color='red', label='Προβλεφθείσα Τιμή')
plt.title("Πρόβλεψη τιμής μετοχής TSLA με MLP", fontweight ='bold')
plt.xlabel('Ημέρες', fontweight ='bold')
plt.ylabel('Τιμή Κλεισίματος σε $', fontweight ='bold')
plt.legend()
plt.show()
