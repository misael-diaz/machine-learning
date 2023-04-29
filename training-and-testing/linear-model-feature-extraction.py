#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Learning Exercises                                  April 28, 2023

source: linear-model-feature-extraction.py
author: @misael-diaz

Synopsis:
Finds the linear model that best predicts the dataset. Considers the effect of extracting
unimportant features from the dataset on the model score.


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
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt


def dataset():
  ''' returns all the features X and the target y '''
  data = pd.read_csv('datasets/temps.csv')
  data = pd.get_dummies(data)
  y = data['actual'].values
  data = data.drop (labels = ['actual'], axis = 1)
  X = data.values
  features = list(data.columns)
  return (X, y, features)


def extractFeatures():
  ''' returns the most important features X and the target y '''

  data = pd.read_csv('datasets/temps.csv')
  data = pd.get_dummies(data)
  y = data['actual'].values
  data = data.drop (labels = [
                      'day',
                      'month',
                      'year',
                      'friend',
                      'temp_2',
                      'week_Mon',
                      'week_Tues',
                      'week_Wed',
                      'week_Thurs',
                      'week_Fri',
                      'week_Sat',
                      'week_Sun',
                      'actual'
                      ],
                    axis = 1)
  X = data.values
  features = list(data.columns)
  return (X, y, features)


def split(X, y):

  return train_test_split(X, y, test_size = 0.2, random_state = 0)


def linearRegression(X, y):
  '''
  Synopsis:
  Obtains a linear model that relates the features X with the target y. Returns the
  model score and the Root Mean Squared Errors with respect to the degree of the
  polynomial features of the model.
  '''

  X = X.copy()
  y = y.copy()

  X_train, X_test, y_train, y_test = split(X, y)

  sc = StandardScaler(copy = True, with_mean = True, with_std = True)
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)

  rmses = []
  scores = []
  degree_min, degree_max = 1, 4
  degrees = arange(degree_min, degree_max + 1)
  for degree in degrees:

    pf = PolynomialFeatures(degree = degree, interaction_only = False,
                            include_bias = True)
    polX_train = pf.fit_transform(X_train)
    polX_test = pf.transform(X_test)

    model = LinearRegression (fit_intercept = True, copy_X = True,
                              n_jobs = 1, positive = False)

    model.fit(polX_train, y_train)
    y_pred = model.predict(polX_test)

    score_train = model.score(polX_train, y_train)
    score_test = model.score(polX_test, y_test)

    rmse = sqrt( sum( (y_pred - y_test)**2 ) )

    rmses.append(rmse)
    scores.append([score_train, score_test])

  rmses = array(rmses)
  scores = array(scores)

  return (degrees, rmses, scores)


def plots():
  '''
  Synopsis:
  Plots the scores and the Root Means Squared Errors RMSEs with respect to the degree of
  the polynomial features of the linear model. Compares the linear model that considers
  all features with its counterpart that only considers the most important features.
  '''

  plt.close('all')
  plt.ion()
  fig, ax = plt.subplots()

  # scores
  X, y, _ = dataset()
  degrees, rmses_allFeatures, scores_allFeatures = linearRegression(X, y)
  scores_train, scores_test = scores_allFeatures.transpose()
  ax.plot(degrees, scores_test, color = 'black', label = 'all-features' )

  X, y, _ = extractFeatures()
  degrees, rmses, scores = linearRegression(X, y)
  scores_train, scores_test = scores.transpose()
  ax.plot(degrees, scores_test, color = 'red', label = 'important-features')

  ax.set_xlabel('degree')
  ax.set_ylabel('score')
  ax.set_title('test scores')

  # RMSEs
  fig, ax = plt.subplots()
  ax.plot(degrees, rmses_allFeatures, color = 'black', label = 'all-features' )
  ax.plot(degrees, rmses, color = 'red', label = 'important-features' )
  ax.set_xlabel('degree')
  ax.set_ylabel('RMSE')
  ax.set_title('Root Mean Squared Errors')

  return


def error():
  ''' determines the error of the best fit linear model '''

  X, y, features = extractFeatures()
  X_train, X_test, y_train, y_test = split(X, y)

  sc = StandardScaler(copy = True, with_mean = True, with_std = True)
  X_train = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)

  pf = PolynomialFeatures(degree = 2, interaction_only = False, include_bias = True)
  X_train = pf.fit_transform(X_train)
  X_test = pf.transform(X_test)

  model = LinearRegression (fit_intercept = True, copy_X = True,
                            n_jobs = 1, positive = False)

  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  error = mean( abs(y_pred - y_test) )

  ''' plots the temperature evolution in a year '''

  time = arange(y.size)
  x = sc.transform(X)
  x = pf.transform(x)
  y_pred = model.predict(x)

  plt.close('all')
  plt.ion()
  fig, ax = plt.subplots()
  ax.scatter(time, y)
  ax.scatter(time, y_pred, color = 'red')
  ax.set_title('temperature evolution')
  ax.set_xlabel('time (days)')
  ax.set_ylabel('Temperature (degrees Fahrenheit)')

  ''' obtains the model and baseline errors '''

  csv = 'datasets/temps.csv'
  data = pd.read_csv(csv)
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


# main
# plots()   # shows that the linear feature extracted model has better scores
error()     # logs the model error and plots the temperature evolution


'''
COMMENTS:
The linear model that considers only the most important features is superior to its
counterpart which considers all features. For this reason the model trained on all the
features is not considered further.

The error of the linear model with polynomial features of second-degree is less than the
baseline error.
'''
