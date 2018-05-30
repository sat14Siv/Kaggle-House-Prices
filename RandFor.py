# -*- coding: utf-8 -*-
"""
Created on Fri May 25 18:14:35 2018

@author: Sateesh
"""
#%%
import pandas as pd


def ourChoice(train_data, test_data):
    feature_columns = ['Id','LotArea','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
    train_X = train_data[feature_columns]
    train_y = train_data['SalePrice']
    test_X = test_data[feature_columns]
    
    return train_X, train_y, test_X
    
def deleteNullFeatures(train_data, test_data):
    # Retaining features with only numeric data and no null values
    column_null_count_train = train_data.isnull().sum()
    column_null_count_test = test_data.isnull().sum()
    feature_columns = [feature for feature in train_data.columns[:-1]
                        if ((train_data.dtypes[feature] != 'object') and (column_null_count_train[feature] == 0) and (column_null_count_test[feature] == 0))]
    
    train_X = train_data[feature_columns]
    train_y = train_data['SalePrice']
    test_X = test_data[feature_columns]
    
    return train_X, train_y, test_X

def imputeData(train_data, test_data):
    from sklearn.preprocessing import Imputer
    
    # Imputer works on numerical data. Remove the object type features first
    feature_columns = [feature for feature in test_data.columns
                       if test_data[feature].dtype !=  'object']
    train_X = train_data[feature_columns]
    train_y = train_data['SalePrice']
    test_X = test_data[feature_columns]
    
    imputer = Imputer()
    train_X = imputer.fit_transform(train_X)
    test_X = imputer.transform(test_X)
    
    # Outputs are numpy arrays. Converting to Dataframe
    train_X = pd.DataFrame(train_X)
    test_X = pd.DataFrame(test_X)
    
    return train_X, train_y, test_X 

def RandomForestPredictor(train_data, test_data):
    from sklearn.ensemble import RandomForestRegressor
    
    # Model fitting 
    iowa_model = RandomForestRegressor()
    iowa_model.fit(train_X, train_y)
    
    pricePredictions = iowa_model.predict(test_X)
    return pricePredictions

def submit(pricePredictions, Id):
    submission = pd.DataFrame({'Id':Id, 'SalePrice':pricePredictions})
    submission.to_csv('satsiv_HousePrice.csv', index = False)
    
if __name__ == "__main__":
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    Id = test_data['Id']
    
    '''**Model Choice Made here**'''
    train_X, train_y, test_X = imputeData(train_data, test_data)
    pricePredictions = RandomForestPredictor(train_data, test_data)
    
    submit(pricePredictions,Id)

    
