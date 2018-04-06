
'''
This project explores the use of linear regression to predict
stock prices based on historical data from Quandl.
'''

import pandas as pd
import quandl
import os, math, datetime, time, pickle
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')


# configure auth
quandl.ApiConfig.api_key = os.environ.get('auth_token')


# get dataset
df = quandl.get('WIKI/GOOGL')

# trim dataset to useful data
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

# create relationships between columns to use as features
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Low'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Close']) / df['Adj. Open'] * 100.0

# new dataframe containing only useful data including derived relationships
df = df [['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]


# define the predicted output
forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True) # replace NA data with massive outlier instead of removing whole row of data if NA

forecast_out = int(math.ceil(0.01*len(df))) # number of days into future to predict, using 10% of total dataset

df['label'] = df[forecast_col].shift(-forecast_out) # shift all data dependent on number of days out predicted
df.dropna(inplace=True) # drop any predicted rows so dont use linear regression on predicted data points, only actual data values


# linear regression model
X = np.array(df.drop(['label'], 1)) # features
X = preprocessing.scale(X)
X_lately = X[-forecast_out:] # X values to predict against
X = X[:-forecast_out]

y = np.array(df['label']) # labels
df.dropna(inplace=True)
y = y[:-forecast_out]


# seperate testing data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2) # shuffle and split data x% testing to avoid bias

# classifier
clf = LinearRegression(n_jobs=-1) # swap classifier with other models, e.g. svm.SVR(kernel='poly') for polynomial, n_jobs allows cpu multithreading
clf.fit(X_train, y_train) # train on training data


# save trained model into pickle
with open('sklearn-linear-regression.pickle', 'wb') as f:
    pickle.dump(clf, f)


# load in trained model
pickle_in = open('sklearn-linear-regression.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test) # predict on testing data

forecast_set = clf.predict(X_lately) # create forecast set
df['Forecast'] = np.nan


# create date values for graph
last_date = df.iloc[-1].name
last_unix = last_date.to_datetime()
last_unix = time.mktime(last_unix.timetuple())
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns-1))] + [i]


# plot
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
