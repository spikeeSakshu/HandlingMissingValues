# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 19:22:42 2019

@author: Spikee
"""
import numpy as np 
import pandas as pd

data = pd.read_csv('pima-indians-diabetes.csv')

data['Skin'].replace(0, np.nan, inplace=True)

data_without_missing_values = data.dropna()

data_with_missing_values = data[data.isnull().any(axis=1)]

X_train = data_without_missing_values.drop(['Skin'], axis=1)

y_train = data_without_missing_values['Skin']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.25, random_state = 10)
X_train.shape

from sklearn.linear_model import BayesianRidge
model = BayesianRidge(compute_score = True, fit_intercept= True, tol= 10)
model = model.fit(X_train, y_train)

print("Results For Bayesian Ridge")
scoreBeforeFilling = model.score(X_test, y_test)
print("\nScore", scoreBeforeFilling*100)

X_test_nan = data_with_missing_values.drop(['Skin'], axis=1)
y_test_nan = data_with_missing_values['Skin']

predicted = model.predict(X_test_nan)

predicted =predicted.astype(int)

# =============================================================================
# #
# =============================================================================
data_with_missing_values['Skin'] = predicted

data_after_filling = pd.concat([data_without_missing_values, data_with_missing_values], ignore_index = True)

from sklearn.utils import shuffle
data_after_filling = shuffle(data_after_filling, random_state=10)

X_train = data_after_filling.drop(['Skin'], axis=1)
y_train = data_after_filling['Skin']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.25, random_state = 10)


from sklearn.linear_model import BayesianRidge
model = BayesianRidge(compute_score = True, fit_intercept= True, tol= 10)
model = model.fit(X_train, y_train)

print("Results For Bayesian Ridge")
scoreAfterFilling = model.score(X_test, y_test)
print("\nScore", scoreAfterFilling*100)






# =============================================================================
# 
# =============================================================================
from sklearn.linear_model import LinearRegression
modelReg = LinearRegression(fit_intercept= True)
modelReg = modelReg.fit(X_train, y_train)

scoreReg = modelReg.score(X_test, y_test)
print("\nScore", scoreReg*100)

predicted = modelReg.predict(X_test)

from sklearn.neighbors import KNeighborsRegressor
modelKNN = KNeighborsRegressor(n_neighbors = 5, algorithm='kd_tree')
modelKNN = modelKNN.fit(X_train, y_train)

scoreKNN = modelKNN.score(X_test, y_test)
print("\nScore", scoreKNN*100)


from sklearn.neural_network import MLPRegressor

modelMLP = MLPRegressor(random_state=0)
modelMLP = modelMLP.fit(X_train, y_train)

scoreMLP = modelMLP.score(X_test, y_test)
print("\nScore", scoreMLP*100)