# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 17:46:52 2018

@author: Chetan
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset=pd.read_csv('combined.csv')
X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2, random_state=0)

# Applying LinearRegression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred= regressor.predict(X_test)

#OLS Regression summary
import statsmodels.formula.api as sm
X= np.append(arr=np.ones((9568,1)).astype(int), values=X, axis=1)
X_opt=X[:,[0,1,2,3,4]]
regressor_OLS=sm.OLS(endog=y, exog= X_opt).fit()
regressor_OLS.summary()

# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(X)

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(X)
