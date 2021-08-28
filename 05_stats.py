import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# Load data
df = pd.read_csv('01_AMD.csv', parse_dates=[0])
df = df[["Date", "Close"]]
con = df['Date']
df['Date']=pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Convert to time series:
ts = df['Close']

########################################################################################################################
#################################################### Create plots ######################################################
########################################################################################################################

decomposition = seasonal_decompose(ts, model='multiplicative', freq = 2)
observed = decomposition.observed
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
# Customize subplots
plt.subplot(411)
plt.plot(observed, label='Παρατηρήσεις')
plt.legend(loc='best')
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
dftest = adfuller(ts, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['ADF Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
for key, value in dftest[4].items():
    dfoutput['Critical Value (%s)' % key] = value
print(dfoutput)