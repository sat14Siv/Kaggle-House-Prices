# -*- coding: utf-8 -*-
"""
Created on Wed May 30 12:22:38 2018

@author: Sateesh
"""
import pandas as pd
from sklearn.model_selection import train_test_split

def get_mae(pricePredictions, test_y):
    from sklearn.metrics import mean_absolute_error
    error = mean_absolute_error(pricePredictions, test_y)
    return error

data = pd.read_csv('train.csv')
X = data[data.columns[:-1]]
y = data[data.columns[-1]]
train_X, test_X, train_y, test_y = train_test_split(X, y)

feature_columns = test_X.columns
feature_columns = feature_columns[1:]
in_train_X = train_X[feature_columns]
in_test_X = test_X[feature_columns]

enc_train_X = pd.get_dummies(in_train_X)
enc_test_X = pd.get_dummies(in_test_X)
fin_train_X, fin_test_X = enc_train_X.align(enc_test_X, join = 'inner',axis = 1)

if sum(fin_train_X.columns == fin_test_X.columns) == len(fin_train_X.columns):
    from sklearn.preprocessing import Imputer
    imputer = Imputer(strategy = 'mean')
    fin_train_X = pd.DataFrame(imputer.fit_transform(fin_train_X), columns = fin_train_X.columns)
    fin_test_X = pd.DataFrame(imputer.transform(fin_test_X), columns = fin_train_X.columns)
    
    from xgboost import XGBRegressor
    
    # Choose either by early_stopping rounds, or examining the error plots of the eval_set
    '''my_model = XGBRegressor(n_estimators=10000, learning_rate = 0.01)
    my_model.fit(fin_train_X, train_y, early_stopping_rounds=15, eval_metric = 'mae',  eval_set=[(fin_test_X, test_y)], verbose=True)'''
    
    my_model = XGBRegressor(n_estimators=4000, learning_rate = 0.03)
    eval_set = [(fin_train_X, train_y), (fin_test_X, test_y)]
    my_model.fit(fin_train_X, train_y, eval_metric = 'mae',  eval_set=eval_set, verbose=True)
    
    results = my_model.evals_result()
    
    pricePredictions = my_model.predict(fin_test_X)
    error = get_mae(pricePredictions, test_y)
#%% Check how the error is changing with epoch on the training data vs the test data and decide on the ideal n_estimators
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.plot(results['validation_0']['mae'], label = 'train' )
ax.plot(results['validation_1']['mae'], label = 'test' )
plt.ylim((10000,18000))
plt.show()