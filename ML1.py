#This code uses linear regression for stock prices using quandl more information
# can be obtained from the qundl website
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
# pickle module saves the trained data into a pickle file so we do not have to train
# the data every time we run the program
style.use('ggplot')
# df = dataframe
df = quandl.get('WIKI/GOOGL')
#using dictionaries
#accessing columns using df[['','']]
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100
#creating new data columns
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

            
forecast_col = 'Adj. Close'
#dont delete data
#fillna fills the cells with -99999 instead of na,
df.fillna(-99999, inplace=True)
#rounding to nearest whole number
forecast_out = int(math.ceil(0.01*len(df)))
#shifting column data up to stock price 1 day in advance adjusted price
df['label'] = df[forecast_col].shift(-forecast_out)
#dropping row


#df.drop excludes that column
X = np.array(df.drop(['label'],1))
#have to scale new values with other values
#helps with training but increases processing time
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
df.dropna(inplace=True)
Y = np.array(df['label'])

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X,Y,test_size = 0.2)
clf = LinearRegression(n_jobs=1)
clf.fit(X_train, Y_train)
with open('linearregression.pickle','wb') as f:
    pickle.dump(clf, f)
    
pickle_in = open('linearregression.pickle', 'rb')
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
    

print(df.tail())
#dataframe plot, plots against the index date equivalent to plt.plot(df['Adj. Close'])
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show(block=True)
plt.figure(1)

#counts from the last item to the number forecast_out as we are using a negative
#digit
#print(Y[-forecast_out:])
