### Assignment 1

# Making sure I'm ready for everything...
import pandas as pd
import numpy as np
import scipy
import plotly.express as px
import plotly.graph_objects as go
from plotly import tools
import plotly.offline as py
import plotly.graph_objs as go
from datetime import datetime
import statsmodels
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from pygam import LinearGAM, s, f, l
from prophet import Prophet

# Data Import & Framing
train = pd.read_csv("https://github.com/dustywhite7/econ8310-assignment1/raw/main/assignment_data_train.csv")
train['Timestamp'] = pd.to_datetime(train['Timestamp'])

# Just Exploring...
px.line(train, x='Timestamp', y='trips')

# I've tried; (1) Seasonal-Dampened Trend ES, (2) VAR (failed to run since each time components are non-stationary), and GAMs
# Ended up using GAM

## Generalized Additive Models (GAMs) with Prophet
# The reason I chosed Prophet: 
# Given dataset dosen't include any exogenous variables (all time components) & thus Prophet is simpler in this case 
train_p = train[['Timestamp', 'trips']] 
train_p.columns = ['ds', 'y']
train_p.head()

# Creating and fitting model
model = Prophet(daily_seasonality=True) # Didn't consider changepoint_prior_scale (let it use default = 0.05); 
# After some research, I've got to know that, Prophet automatically adds yearly and weekly seasonality, but not the daily
# So I set up daily seasonality equals true  
modelFit = model.fit(train_p)

# Creating an empty dataframe with dates for future periods (744) 
# At this point, didn't feel like having to use test.csv
future = modelFit.make_future_dataframe(periods=744, freq = 'h')

# Filling in dataframe wtih forecasts of `y` for the future periods
forecast = modelFit.predict(future)
forecast.head()

# Plotting the model together with the forecast
fig = modelFit.plot(forecast)
fig.show()

# Plotting the components of the forecast
fig = modelFit.plot_components(forecast)
fig.show()

# Raw forecast numbers & Creating a vector
pred = forecast['yhat'].iloc[-744:].to_numpy()