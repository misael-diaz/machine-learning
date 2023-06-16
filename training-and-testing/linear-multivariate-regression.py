#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Learning Exercises                                      April 27, 2023

source: linear-multivariate-regresion.py
author: @misael-diaz

Synopsis:
Assess the ability of a linear multivariate models to predict future values.


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
[3] https://towardsdatascience.com/simple-and-multiple-linear-regression-in-python-c928425168f9
[4] https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6

"""


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from numpy.random import normal
from numpy.random import seed
from numpy import linspace
from numpy import newaxis
from numpy import hstack

def dataset():
  ''' generates the dataset '''

  # defines limits for the independent variables or features
  x1_min, x1_max = -4, 4
  x2_min, x2_max = -8, 8

  N = 512
  # generates 1D uniformly spaced values in [min, max] for the features
  x1 = linspace(x1_min, x1_max, N)
  x2 = linspace(x2_min, x2_max, N)

  # concatenates x1 and x2 into a matrix of expected shape (rows: records, cols: features)
  x1 = x1[newaxis, :].transpose()
  x2 = x2[newaxis, :].transpose()
  X = hstack([x1, x2])

  # creates a linear relationship between the features X and the dependent variable y
  y = x1 + x2

  # adds a small noise
  seed(0)
  y = normal(loc = y, scale = 1.5)

  return (X, y)


# splits the data into train and test sets
X, y = dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 0)

# obtains a linear model
model = LinearRegression (fit_intercept = True, copy_X = True, n_jobs = None)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(f'linear model score: {score}')

# obtains a regularized linear model
model = LassoCV(eps = 2**-16, n_alphas = 256, alphas = None, fit_intercept = True,
                precompute = 'auto', max_iter = 1024, tol = 2**-16,
                copy_X = True, cv = 5, verbose = False, n_jobs = None,
                positive = False, random_state = None, selection = 'cyclic')

model.fit(X_train, y_train.ravel())
score = model.score(X_test, y_test)
print(f'regularized linear model score: {score}')
