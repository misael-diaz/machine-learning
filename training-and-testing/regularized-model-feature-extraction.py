#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Learning Exercises                                  April 28, 2023

source: regularized-model-feature-extraction.py
author: @misael-diaz

Synopsis:
Uses Lasso to obtain a reduced model in terms of the number of features that it needs
to make reasonable predictions of the target values.


Copyright (c) 2023 Misael Diaz-Maldonado
This file is released under the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.


References:
[0] A Gilat and V Subramanian, Numerical Methods for Engineers and
    Scientists: An Introduction with Applications using MATLAB
[1] R Johansson, Numerical Python: Scientific Computing and Data
    Science Applications with NumPy, SciPy, and Matplotlib, 2nd edition
[2] https://www.w3schools.com/python/python_ml_getting_started.asp
[3] https://towardsdatascience.com/random-forest-in-python-24d0893d51c0

"""


import pandas as pd
from numpy import sum
from numpy import sqrt
from numpy import mean
from numpy import array
from numpy import arange
from numpy import zeros_like
from numpy import zeros
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from matplotlib import pyplot as plt



def dataset():
  ''' returns all the features X and the target y '''
  data = pd.read_csv('datasets/temps.csv')
  data = pd.get_dummies(data)
  dataframe = data
  y = data['actual'].values

  # inserts a days identifier (new feature) to evaluate its importance
  records = y.size
  days = arange(records)
  data.insert(0, 'days', days, True)

  data = data.drop(labels = ['actual'], axis = 1)

  X = data.values

  features = list(data.columns)

  return (X, y, features, dataframe)


def split(X, y):

  return train_test_split(X, y, test_size = 0.2, random_state = 0)


def isSelectedFeature(X, X_trans, feature, features):
  '''
  Synopsis:
  Searches linearly for the given feature.
  '''

  idx = features.index(feature)
  keys = X[:, idx]

  rows, cols = X_trans.shape
  for col in arange(cols):

    equals = X_trans[:, col] == keys

    if sum(equals) == rows:
      return True

  return False


def getSelectedFeatures(X, X_trans, features):
  '''
  Synopsis:
  Gets the features selected by the regularized model.
  The time complexity of the algorithm is O(N * N), where N is the number of features.
  '''

  mask = zeros( len(features) )
  for n, feature in enumerate(features):

    if isSelectedFeature(X, X_trans, feature, features):
      mask[n] = 1
    else:
      mask[n] = 0

  mask = (mask == 1)
  selected_features = array(features)[mask]

  return selected_features


def featureSelection():
  '''
  Synopsis:
  Obtains a linear model that relates the features X with the target y. Returns the
  model score and the Root Mean Squared Errors with respect to the degree of the
  polynomial features of the model.
  '''

  X, y, features, data = dataset()

  X_train, X_test, y_train, y_test = split(X, y)

  sc = StandardScaler(copy = True, with_mean = True, with_std = True)
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)

  estimator = LassoCV(eps = 2**-16, n_alphas = 256, alphas = None,
                      fit_intercept = True, precompute = 'auto', max_iter = 2**16,
                      tol = 2**-16, copy_X = True, cv = 5, verbose = False,
                      n_jobs = 1, positive = False, random_state = None,
                      selection = 'cyclic')

  sfm = SelectFromModel(estimator, threshold = None, prefit = False, norm_order = 1,
                        max_features = None, importance_getter = 'auto')

  sfm.fit(X_train, y_train)
  X_trans = sfm.transform(X_train)

  num_selected_features = X_trans.shape[1]
  selected_features = getSelectedFeatures(X_train, X_trans, features)

  print(f'number of selected features {num_selected_features}')
  print('selected features:')
  print(selected_features)

  model = LinearRegression (fit_intercept = True, copy_X = True,
                            n_jobs = 1, positive = False)

  model.fit(X_trans, y_train)
  score_train = model.score(X_trans, y_train)
  score_test = model.score(sfm.transform(X_test), y_test)
  print(f'model score on the train set: {score_train}')
  print(f'model score on the test set: {score_test}')

  y_pred = model.predict( sfm.transform(X_test) )
  error = mean( abs(y_pred - y_test) )


  ''' plots the temperature evolution in a year '''


  time = arange(y.size)
  y_pred = model.predict( sfm.transform( sc.transform(X) ) )

  plt.close('all')
  plt.ion()
  fig, ax = plt.subplots()
  ax.scatter(time, y)
  ax.scatter(time, y_pred, color = 'red')
  ax.set_title('temperature evolution')
  ax.set_xlabel('time (days)')
  ax.set_ylabel('Temperature (degrees Fahrenheit)')


  avg = data['average'].values
  actual = data['actual'].values
  baseline_error = mean( abs(avg - actual) )
  percentage = abs(error - baseline_error) / ( (error + baseline_error) / 2)
  percentage *= 100

  log = (
      f'model error:    {error:.2f} degrees\n'
      f'baseline error: {baseline_error:.2f} degrees\n'
      f'model / baseline error: {error / baseline_error}\n'
      f'percent difference: {percentage:.2f} %\n'
  )

  print(log)

  return


featureSelection()


'''
COMMENTS:
The reduced model (in terms of the number of features) exhibits satisfactory scores and
its error is less than the baseline error.

Interestingly, the only time frame in the selected features is the month identifier.
It seems that by knowing the most recent temperatures `temp_1` and `temp_2` the model
along other selected features is able to predict the `actual` temperature (the target).
'''
