#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Learning Exercises                                  April 28, 2023

source: random-forest.py
author: @misael-diaz

Synopsis:
Determines the ability of the considered models to predict future values.


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
from numpy import array
from numpy import arange
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from matplotlib import pyplot as plt


def randomForestRegression():
  ''' uses Random Forests to determine the least important features '''

  data = pd.read_csv('datasets/temps.csv')

  # applies OneHotEncoding to the dataframe (days of the week -> zeros or ones)
  data = pd.get_dummies(data)

  # gets the target
  y = data['actual'].values

  # drops target
  data = data.drop (labels = ['actual'], axis = 1)

  # gets the features
  X = data.values

  features = list(data.columns)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                      random_state = 0)

  model = RandomForestRegressor(n_estimators = 1024, random_state = 0)
  model.fit(X_train, y_train)
  score_train = model.score(X_train, y_train)
  score_test = model.score(X_test, y_test)

  scores = [score_train, score_test]

  importances = model.feature_importances_
  ranked = [ (feature, importance) for feature, importance in zip(features, importances)]
  ranked = sorted(ranked, key = lambda t: t[1], reverse = True)

  for feature, importance in ranked:
    print(f'feature: {feature:16s} importance: {importance:.2f}')

  return (model, scores)


def score_linearRegressions(X_train, X_test, y_train, y_test):
  ''' uses the traditional linear regression to model the dataset '''

  sc = StandardScaler(copy = True, with_mean = True, with_std = True)
  stdX_train = sc.fit_transform(X_train)
  stdX_test = sc.transform(X_test)

  scores = []
  degree_min, degree_max = 1, 4
  degrees = arange(degree_min, degree_max + 1)
  for degree in degrees:

    pf = PolynomialFeatures(degree = degree, interaction_only = False,
                            include_bias = True)
    polX_train = pf.fit_transform(stdX_train)
    polX_test = pf.transform(stdX_test)

    model = LinearRegression (fit_intercept = True, copy_X = True,
                              n_jobs = -1, positive = False)

    model.fit(polX_train, y_train)
    score_train = model.score(polX_train, y_train)
    score_test = model.score(polX_test, y_test)

    scores.append([score_train, score_test])

  return scores


def score_regularizedRegressions(X_train, X_test, y_train, y_test):
  ''' uses the regularized linear (LassoCV) regression to model the dataset '''

  sc = StandardScaler(copy = True, with_mean = True, with_std = True)
  stdX_train = sc.fit_transform(X_train)
  stdX_test = sc.transform(X_test)

  scores = []
  degree_min, degree_max = 1, 4
  degrees = arange(degree_min, degree_max + 1)
  for degree in degrees:

    pf = PolynomialFeatures(degree = degree, interaction_only = False,
                            include_bias = True)
    polX_train = pf.fit_transform(stdX_train)
    polX_test = pf.transform(stdX_test)

    model = LassoCV(eps = 2**-16, n_alphas = 256, alphas = None, fit_intercept = True,
                    precompute = 'auto', max_iter = 2**24, tol = 2**-16, copy_X = True,
                    cv = 128, verbose = False, n_jobs = -1, positive = False,
                    random_state = None, selection = 'cyclic')

    model.fit(polX_train, y_train)
    score_train = model.score(polX_train, y_train)
    score_test = model.score(polX_test, y_test)

    scores.append([score_train, score_test])

  return scores


data = pd.read_csv('datasets/temps.csv')

# applies OneHotEncoding to the dataframe (days of the week -> zeros or ones)
data = pd.get_dummies(data)

# gets the target values, the actual temperature
y = data['actual'].values

# drops target and irrelevant features from the dataset (based on reference[3])
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

# gets the features
X = data.values

features = list(data.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 0)

model, scores = randomForestRegression()
lin_scores = score_linearRegressions(X_train, X_test, y_train, y_test)
reg_scores = score_regularizedRegressions(X_train, X_test, y_train, y_test)


baseline_pred = X_test[:, features.index('average')]
baseline_errs = abs(baseline_pred - y_test)
print(f'averaged baseline error: {baseline_errs.mean()} degrees')


'''
COMMENTS:
Random forests were used to both determine unimportant features and to model the dataset.
The random forest model yields a score of about 0.8 and so the linear and regularized
models. However, the computational cost of the regularized (LassoCV) model is much
higher than that of its counterparts.

It is worth mentioning that the best fit was obtained for a linear regression model
(without regularization) with the features of a second-degree polynomial, which yields
about a 0.84 score. The score was possible after extracting the unimportant features.
'''
