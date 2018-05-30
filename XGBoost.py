# -*- coding: utf-8 -*-
"""
Created on Wed May 30 12:02:29 2018

@author: Sateesh
"""
#%%
import pandas as pd 

def submitPrediction(Id, pricePredictions):
    submission = pd.DataFrame({'Id':Id, 'SalePrice':pricePredictions})
    submission.to_csv('satxg_house.csv', index = False)
    return submission

def get_mae(pricePredictions, test_y):
    from sklearn.metrics import mean_absolute_error
    error = mean_absolute_error(pricePredictions, test_y)
    return error

# Preparing the test and training sets.  XGBoost can handle only numeric data.  It can also fill in missing values.  But first I will use Imputer() to fill in the values and in a later case let XGBoost handle the missing values.

#%% 1) Remove Categorical Features and Use Imputer

def case1(test_data, train_data):
    Id = test_data['Id']
    feature_columns = [feature for feature in test_data.columns
                          if test_data[feature].dtype != 'object']
    feature_columns = feature_columns[1:]
    train_X = train_data[feature_columns]
    train_y = train_data['SalePrice']

    test_X = test_data[feature_columns]

    from sklearn.preprocessing import Imputer

    impute = Imputer(strategy = 'mean')
    train_X = impute.fit_transform(train_X)
    test_X = impute.transform(test_X)

    from xgboost import XGBRegressor

    model = XGBRegressor()
    model.fit(train_X, train_y)
    pricePredictions = model.predict(test_X)

    submitted = submitPrediction(Id, pricePredictions)
    return submitted

#%% 2) Remove Categorical Features and let XGBoost handle the Missing Values

def case2(test_data, train_data):
    Id = test_data['Id']
    feature_columns = [feature for feature in test_data.columns
                          if test_data[feature].dtype != 'object']
    feature_columns = feature_columns[1:]
    train_X = train_data[feature_columns]
    train_y = train_data['SalePrice']

    test_X = test_data[feature_columns]

    from xgboost import XGBRegressor

    model = XGBRegressor()
    model.fit(train_X, train_y)
    pricePredictions = model.predict(test_X)

    submitted = submitPrediction(Id, pricePredictions)
    return submitted
#%% 3) One hot Encoding of Categorical Features  and Impute Missing Values:

def case3(test_data, train_data):
    Id = test_data['Id']
    feature_columns = test_data.columns
    feature_columns = feature_columns[1:]
    in_train_X = train_data[feature_columns]
    train_y = train_data['SalePrice']
    in_test_X = test_data[feature_columns]

    enc_train_X = pd.get_dummies(in_train_X)
    enc_test_X = pd.get_dummies(in_test_X)
    train_X, test_X = enc_train_X.align(enc_test_X, join = 'inner',axis = 1)
    
    if sum(train_X.columns == test_X.columns) == len(train_X.columns):
        from sklearn.preprocessing import Imputer
        imputer = Imputer(strategy = 'mean')
        train_X = pd.DataFrame(imputer.fit_transform(train_X), columns = train_X.columns)
        test_X = pd.DataFrame(imputer.transform(test_X), columns = train_X.columns)
        
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators = 500, learning_rate = 0.1)
        model.fit(train_X, train_y)
        pricePredictions = model.predict(test_X)

        submitted= submitPrediction(Id, pricePredictions)
        
    else:
        print('failed')
    return submitted
#%% 4) One Hot Encoding and let XGBoost deal with Missing Values
    
def case4(test_data, train_data):
    Id = test_data['Id']
    feature_columns = test_data.columns
    feature_columns = feature_columns[1:]
    in_train_X = train_data[feature_columns]
    train_y = train_data['SalePrice']
    in_test_X = test_data[feature_columns]

    enc_train_X = pd.get_dummies(in_train_X)
    enc_test_X = pd.get_dummies(in_test_X)
    train_X, test_X = enc_train_X.align(enc_test_X, join = 'inner',axis = 1)
    
    if sum(train_X.columns == test_X.columns) == len(train_X.columns):
        from xgboost import XGBRegressor
        model = XGBRegressor()
        model.fit(train_X, train_y)
        pricePredictions = model.predict(test_X)

        submitted = submitPrediction(Id, pricePredictions)
        
    else:
        print('failed')  
    return submitted
#%% 5) Using Early Stopping Rounds
    
def case5(test_data, train_data):
    Id = test_data['Id']
    feature_columns = test_data.columns
    feature_columns = feature_columns[1:]
    in_train_X = train_data[feature_columns]
    train_y = train_data['SalePrice']
    in_test_X = test_data[feature_columns]

    enc_train_X = pd.get_dummies(in_train_X)
    enc_test_X = pd.get_dummies(in_test_X)
    train_X, test_X = enc_train_X.align(enc_test_X, join = 'inner',axis = 1)
    
    if sum(train_X.columns == test_X.columns) == len(train_X.columns):
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators = 4000, learning_rate = 0.01)
        model.fit(train_X, train_y)
        pricePredictions = model.predict(test_X)

        submitted = submitPrediction(Id, pricePredictions)
        
    else:
        print('failed')  
    return submitted
    
#%%
if __name__ == '__main__':
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    submitted = case5(test_data, train_data)
