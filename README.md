# Stock price forecasting 

This is a python ML project for stock price forecasting using *streamlit* and *NeuralProphet*. 

Before choosing to apply NeuralProphet, three artificial neural networks models were tested and more specifically:

* LSTM (Long-short term memory)
* MLP (Multilayer perceptron)
* NeuralProphet

## Model testing

As mentioned above, before applying NeuralProphet, three ANN models were tested. Before applying the model each timeseries used for testing was plotted and decomposed. Moreover, the Augmented Dickey-Fuller test was applied to check for stationarity. 

*AMD stock closing price area chart 01/01/2011 - 31/12/2020*

![](https://github.com/stenikolaou/stock_price_forecasting/blob/master/images/area_chart.png)

*AMD stock timeseries decomposition diagram to check for trend, seasonality and residuals*

![](https://github.com/stenikolaou/stock_price_forecasting/blob/master/images/decompose.png)


