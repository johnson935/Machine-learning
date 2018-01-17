#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 18:05:58 2018
predicts the price of bitcoin in GBP
@author: johuson
"""

import sys
path = '/Library/Frameworks/Python.framework/Versions/3.6/lib/python3.6/site-packages'
if path not in sys.path:
    sys.path.insert(0, path)

        
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import quandl, math, datetime
import numpy as np
#surppot vector machines,svm
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression
import pickle

style.use('ggplot')

df = quandl.get('LOCALBTC/GBP')
print(len(df))
df = df[['24h Average','12h Average','Last','Volume (BTC)']]
df['Day_PCT'] = (df['24h Average'] - df['12h Average']) / df['Last'] * 100
df['PCT_change'] = (df['12h Average'] - df['24h Average']) / df['24h Average'] * 100
#creating new data columns
df = df[['24h Average','Day_PCT','PCT_change','Volume (BTC)']]


#dont delete data
#fillna fills the cells with -99999 instead of na,
df.fillna(-99999, inplace=True)
#rounding to nearest whole number
forecast_out = int(math.ceil(0.1*len(df)))
#shifting column data up to stock price 1 day in advance adjusted price
df['label'] = df['24h Average'].shift(-forecast_out)
#dropping row


X = np.array(df.drop(['label'],1))
#have to scale new values with other values
#helps with training but increases processing time
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
df.dropna(inplace=True)
Y = np.array(df['label'])

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X,Y,test_size = 0.5)
clf = LinearRegression(n_jobs=1)
clf.fit(X_train, Y_train)
with open('Bitcoin.pickle','wb') as f:
    pickle.dump(clf, f)
    
pickle_in = open('Bitcoin.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test,Y_test)

forecast_set =  clf.predict(X_lately)
#puts NAN in the column
df['Forecast'] = np.nan
print(forecast_set, accuracy, forecast_out)

#the very last date
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
#number of seconds in a day
one_day = 86400
#next unix = next day
#naming the future date
next_unix = last_unix + one_day

# populating the column with new dates and the predicted values
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    #df.loc is the index and the index is the date setting the row values the 
    #columns are set to NAN apart from the last column forecast which is assigned
    #the value of i 
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
    


#dataframe plot, plots against the index date equivalent to plt.plot(df['Adj. Close'])
df['24h Average'].plot()
df['Forecast'].plot()
plt.legend(loc=1)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show(block=True)
plt.figure(1)

