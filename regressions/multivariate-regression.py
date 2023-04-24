#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Learning Exercises                                      April 24, 2023

source: multivariate-regresion.py
author: @misael-diaz

Synopsis:
Assess the ability of a multivariate regression model to predict future values.


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
[3] https://www.analyticsvidhya.com/blog/2021/10/implementing-artificial-neural-networkclassification-in-python-from-scratch/?
[4] https://towardsdatascience.com/machine-learning-with-python-easy-and-robust-method-to-fit-nonlinear-data-19e8a1ddbd49

"""


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline
from matplotlib import pyplot as plt
from numpy import arange
from numpy import array
from numpy import sqrt
from numpy import sum


def encodeGender(genders):

  for i in range(genders.size):

    if genders[i] == 'Male':
      genders[i] = 0
    else:
      genders[i] = 1

  return


def decodeGender(genders):

  for i in range(genders.size):

    if genders[i] == 0:
      genders[i] = 'Male'
    else:
      genders[i] = 'Female'

  return


def encodeCountry(countries):

  for i in range(countries.size):

    if countries[i] == 'France':
      countries[i] = 0
    elif countries[i] == 'Germany':
      countries[i] = 1
    else:
      countries[i] = 2

  return


def decodeCountry(countries):

  for i in range(countries.size):

    if countries[i] == 0:
      countries[i] = 'France'
    elif countries[i] == 1:
      countries[i] = 'Germany'
    else:
      countries[i] = 'Spain'

  return


csvfile = 'Churn_Modelling.csv'
data = pd.read_csv(csvfile)

# excludes useless data (first three columns) and the outcome (last column), total 10 cols
X = data.iloc[:, 3:13].values
# gets the outcome (last column) if the user `exited` the bank or not
y = data.iloc[:, -1].values

# encodes non-numeric data (gender and geographical location or country)
encodeCountry(X[:, 1])
encodeGender(X[:, 2])

# splits into train and test sets
tt = traintest = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train, X_test, y_train, y_test = tt

# defines the min and max polynomial degrees to consider for the model
min_degree, max_degree = 1, 6
degrees = arange(min_degree, max_degree + 1)
rmses = []  # inits the Root Mean Squared Errors RMSEs list
scores = [] # inits the R-squared scores
for degree in degrees:

  model = make_pipeline(StandardScaler(),
                        PolynomialFeatures(degree, interaction_only = False),
                        LinearRegression())

  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  score = model.score(X_test, y_test)
  rmse = sqrt( sum( (y_pred - y_test)**2 ) )

  rmses.append(rmse)
  scores.append(score)

rmses = array(rmses)
scores = array(scores)

plt.close('all')
plt.ion()
fig, ax = plt.subplots()
ax.plot(degrees, scores)
ax.set_title('R-squared scores')

fig, ax = plt.subplots()
ax.plot(degrees, rmses)
ax.set_title('RMSE')

"""
COMMENTS:
Owing to the poor R-squared scores of the regression models, we conclude that the models
are unsuitable for conducting future predictions.
"""
