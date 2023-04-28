#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Learning Exercises                                      April 27, 2023

source: regresion.py
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
[5] https://towardsdatascience.com/simple-and-multiple-linear-regression-in-python-c928425168f9
[6] https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6
[7] https://medium.com/@yanhann10/a-brief-view-of-machine-learning-pipeline-in-python-5f50b941fca8

"""


import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold
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

tt = traintest = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train, X_test, y_train, y_test = tt

scores = []
degree_min, degree_max = 1, 5
degrees = arange(degree_min, degree_max + 1)
for degree in degrees:

  print('standarizing ...')
  standardScaler = StandardScaler(copy = True, with_mean = True, with_std = True)
  stdX_train = standardScaler.fit_transform(X_train)
  stdX_test = standardScaler.transform(X_test)

  print('generating polynomial features ...')
  polynomialFeatures = PolynomialFeatures(degree = degree, interaction_only = False,
                                          include_bias = True)
  polX_train = polynomialFeatures.fit_transform(stdX_train)
  polX_test = polynomialFeatures.transform(stdX_test)

  print('regularizing ...')
  model = LassoCV(eps = 2**-16, n_alphas = 256, alphas = None, fit_intercept = True,
                  precompute = 'auto', max_iter = 2**16, tol = 2**-16, copy_X = True,
                  cv = 24, verbose = False, n_jobs = 1, positive = False,
                  random_state = None, selection = 'cyclic')


  print('fitting ...')
  model.fit(polX_train, y_train)
  print('scoring ...')
  score_train = model.score(polX_train, y_train)
  score_test = model.score(polX_test, y_test)

  scores.append([score_train, score_test])

"""
COMMENTS:
It is computationally expensive to consider up to fifth degree polynomial. This is why
the scores are stored in a HDF5 file.

The second-order degree polynomial fitting describes the training set satisfactorily
(about 0.8 score) but does not peform well on the test set (about 0.3 score).

The considered higher order polynomials do not even describe very well the training
set (scores about 0.25). Hence, even with regularization the model predictabilities
are unsatisfactory.
"""
