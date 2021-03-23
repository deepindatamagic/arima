# -*- coding: utf-8 -*-
# forecasts_arima.py
# Santos Borom
# License: Creative Commons Zero v1.0

import numpy as np 
import pandas as pd
import plotly.express as px
from plotly import graph_objects
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error 
from statsmodels.tools.eval_measures import rmse
from statsmodels.tsa.arima_model import ARMA
from pmdarima.arima import ADFTest
from statsmodels.tsa.seasonal import seasonal_decompose 
from pmdarima import auto_arima 
from skits.preprocessing import (ReversibleImputer,
                                 DifferenceTransformer)
from skits.pipeline import ForecasterPipeline
from sklearn.preprocessing import StandardScaler

# """**Load Dataset**"""

data = pd.read_csv('stocks.csv', index_col=0)

print(data)

print(data.Open)

# convert to datetime format
data['Date'] = pd.to_datetime(data['Date'])

print(data.Date)

# sort by date
data.sort_values(by='Date', inplace=True, ascending=True)

print(data)

y = data.Open
fig = px.line(data, x=data.index, y='Open', title='Google Stock Prices')
fig.show()

# """**ARIMA**

# * p is the auto-regressive part of the model, incorporating the effect of the past.

# * d is the integrated part of the model. This includes terms in the model that incorporate the amount of differencing: the number of past times to subtract from the present time.

# * q is the moving average part of the model. This allows us to set the error of our model as a linear combination of the error values observed at previous time points in the past.

# * When dealing with seasonal effects, we make use of the seasonal ARIMA, which is denoted as ARIMA(p,d,q)(P,D,Q)s. Here, (p, d, q) are the non-seasonal parameters described above, while (P, D, Q) follow the same definition but are applied to the seasonal component of the time series. The term s is the periodicity of the time series (4 for quarterly periods, 12 for yearly periods, etc.).

# **Model**

# * When looking to fit time series data with a seasonal ARIMA model, our first goal is to find the values of ARIMA(p,d,q)(P,D,Q)s that optimize a metric of interest.
# """

# Fit auto_arima function to  dataset 
stepwise_fit = auto_arima(data['Open'], start_p = 1, start_q = 1, 
                          max_p = 3, max_q = 3, m = 12, 
                          start_P = 0, seasonal = True, 
                          d = None, D = 1, trace = True, 
                          error_action ='ignore',   # we don't want to know if an order does not work 
                          suppress_warnings = True,  # we don't want convergence warnings 
                          stepwise = True)           # set to stepwise

stepwise_fit.summary()

# Split data into train / test sets 
train = data.iloc[:len(data)-12] 
test = data.iloc[len(data)-12:] # set one year(12 months) for testing

print(test)

print(train)

from statsmodels.tsa.statespace.sarimax import SARIMAX 
  
model = SARIMAX(train['Open'],  
                order = (0, 1, 1),  
                seasonal_order =(2, 1, 1, 12)) 
  
result = model.fit() 
result.summary()

start = len(train) 
end = len(train) + len(test) - 1
  
# Predictions for one-year against the test set 
predictions = result.predict(start, end, typ = 'levels').rename("Predictions")

print(predictions)

print(predictions.index)

# Calculate root mean squared error 
rmse(test["Open"], predictions)

# Calculate mean squared error 
mean_squared_error(test["Open"], predictions)

# Train the model on the full dataset 
model = model = SARIMAX(data['Open'],  
                        order = (0, 1, 1),  
                        seasonal_order =(2, 1, 1, 12)) 
result = model.fit() 
  
# Forecast for the next 3 years 
forecast = result.predict(start = len(data),  
                          end = (len(data)-1) + 3 * 12,  
                          typ = 'levels').rename('Forecast')

print(forecast)

forecast.to_csv('forecasts_arima.csv')