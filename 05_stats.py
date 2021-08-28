import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# Load data
df = pd.read_csv('03_TSLA.csv')
df = df[["Date", "Close"]]
print('===========================================================================')
print(df.describe())
print('===========================================================================')

# Convert to time series:
ts = df['Close']
# Log transform time series
ts_log=np.log(ts)

########################################################################################################################
#################################################### Create plots ######################################################
########################################################################################################################

decomposition = seasonal_decompose(ts_log, period=30)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
# Customize subplots
plt.subplot(411)
plt.plot(ts, label='Παρατηρήσεις')
plt.subplot(412)
plt.plot(trend, label='Τάση')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Εποχικότητα')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Yπόλοιπα')
plt.legend(loc='best')
plt.tight_layout()
# Show plot
plt.show()

########################################################################################################################
######################################################## ADF ###########################################################
########################################################################################################################

print(
    'Results of Dickey-Fuller Test:')
dftest = adfuller(ts_log, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['ADF Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
for key, value in dftest[4].items():
    dfoutput['Critical Value (%s)' % key] = value
print(dfoutput)