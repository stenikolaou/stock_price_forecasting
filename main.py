import streamlit as st
from datetime import date
import yfinance as yf
from plotly import graph_objs as go
from neuralprophet import NeuralProphet
import pandas as pd

# Define start date and current date
start = '2015-01-01'
today = date.today().strftime("%Y-%m-%d")
# Set app title
st.title("Εφαρμογή πρόβλεψης μελλοντικών τιμών στο χρηματιστήριο")
# Create a combobox with desired stocks
stocks = ("AMZN", "FB", "GOOG", "MSFT", "AAPL", "AMD", "NVDA", "PYPL", "TSLA", "SHOP")
selected_stocks = st.selectbox("Επιλέξτε μετοχή για πρόβλεψη", stocks)
# Create a slider for desired prediction years
n_days = st.slider("Ημέρες πρόβλεψης:", 1, 365)
period = n_days
# Create a function to load data from Yahoo finance
@st.cache
def load_data(ticker):
    data = yf.download(ticker, start, today)
    data.reset_index(inplace=True)
    return data
# Create load messages
data_load_state = st.text("Αναμονή φόρτωσης δεδομένων...")
data = load_data(selected_stocks)
data_load_state.text("Επιτυχής φόρτωση δεδομένων!")
# Show last 5 days data
st.subheader('Δεδομένα τελευταίας εβδομάδας')
st.write(data.tail())
# Plot original data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Τιμή κλεισίματος'))
    fig.layout.update(title_text="Διάγραμμα τιμής κλεισίματος", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

########################################################################################################################
################################################# Prophet Forecasting ##################################################
########################################################################################################################

# Select only date and close price
df = data[["Date", "Close"]]

# Rename the columns to ds (timestamp) and y (observed values)
df.rename(columns={"Date": "ds", "Close": "y"}, inplace=True)

# Create model
m = NeuralProphet(batch_size=64, num_hidden_layers=1, learning_rate=1.0,
                  epochs=20, n_forecasts=period, n_lags=10, n_changepoints=40)
m.fit(df, freq='d')

# Make prediction
future = m.make_future_dataframe(df, n_historic_predictions=len(df))
forecast = m.predict(future)
print(forecast.tail())

# Show forecast data
st.subheader('Forecast data')
forecast['forecast'] = forecast['trend'] + forecast['season_yearly'] + forecast['season_weekly']
forecast['ds'] = pd.to_datetime(forecast['ds'], errors='coerce')
forecast['ds'].dt.strftime("YYYY-MM-DD")
simple_f = forecast[['ds', 'forecast']]
st.write(simple_f.tail())

########################################################################################################################
#################################################### Create plot #######################################################
########################################################################################################################
# Draw plot
data_load_state = st.text("Αναμονή φόρτωσης δεδομένων...")
st.subheader('Πρόβλεψη μελλοντικής τιμής')
fig1 = m.plot(forecast)
st.plotly_chart(fig1)
data_load_state.text("Επιτυχής φόρτωση δεδομένων!")



