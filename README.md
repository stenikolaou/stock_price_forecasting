# Stock price forecasting 

This is a python ML project for stock price forecasting using *streamlit* and *NeuralProphet*. 

Before choosing to apply NeuralProphet, three artificial neural network models were tested and more specifically:

* LSTM (Long-short term memory)
* MLP (Multilayer perceptron)
* NeuralProphet

## Model testing

As mentioned above, before applying NeuralProphet, three ANN models were tested. Before applying the model each timeseries used for testing was plotted and decomposed. Moreover, the Augmented Dickey-Fuller test was applied to check for stationarity. 

![](https://github.com/stenikolaou/stock_price_forecasting/blob/master/images/area_chart.png)

*NVDA stock closing price area chart 01/01/2011 - 31/12/2020*

![](https://github.com/stenikolaou/stock_price_forecasting/blob/master/images/decompose.png)

*NVDA stock timeseries decomposition diagram to check for trend, seasonality and residuals*

![](https://github.com/stenikolaou/stock_price_forecasting/blob/master/images/lstm.png)

*LSTM model applied with 200 epochs and ADAM optimizer applied to NVDA stock price timeseries*

## Streamlit

Using streamlit, a simple web app was created. The user can choose a stock through a combobox and set the desired forecasting period (in days) through a slider. After choosing a stock, he gets information about the stock price for the last five days,as well as an interactive chart.

![](https://github.com/stenikolaou/stock_price_forecasting/blob/master/images/streamlit_main.png)

After choosing the desired stock and setting the prediction days a forecast is produced and the user can see a table containing the last five values of the forecast dataset, as well as a plot. 

![](https://github.com/stenikolaou/stock_price_forecasting/blob/master/images/streamlit_forecast.png)


