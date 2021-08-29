import warnings
import matplotlib.pyplot as plt
import pandas as pd
from neuralprophet import NeuralProphet

# Silence warnings
warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv('01_AMD.csv')

# Select only date and close price
df = df[["Date", "Close"]]

# Rename the columns to ds (timestamp) and y (observed values)
df.columns = ['ds', 'y']
print(df.tail())

########################################################################################################################
################################################# Prophet Forecasting ##################################################
########################################################################################################################

# Create model
m = NeuralProphet(batch_size=64, num_hidden_layers=1, learning_rate=1.0,
                  epochs=100, n_lags=10, n_changepoints=10)
m.fit(df, freq='d')

# Make prediction
future = m.make_future_dataframe(df, periods=1, n_historic_predictions=len(df))
forecast = m.predict(future)
print(forecast.tail())

########################################################################################################################
#################################################### Create plot #######################################################
########################################################################################################################

plt.style.use('fivethirtyeight')
plt.figure(figsize=(12,8))
plt.plot(forecast['y'], linestyle='dashed', marker='o', color='blue', label='Πραγματική Τιμή')
plt.plot(forecast['yhat1'], color='red', label='Προβλεφθείσα Τιμή')
plt.title("Πρόβλεψη τιμής μετοχής AMD με NeuralProphet", fontweight ='bold')
plt.xlabel('Ημέρες', fontweight ='bold')
plt.ylabel('Τιμή Κλεισίματος σε $', fontweight ='bold')
plt.legend()
plt.show()

########################################################################################################################
################################################### Export to excel ####################################################
########################################################################################################################
# create excel writer object
writer = pd.ExcelWriter('output.xlsx')
# write dataframe to excel
forecast.to_excel(writer)
# save the excel
writer.save()
print('DataFrame is written successfully to Excel File.')
