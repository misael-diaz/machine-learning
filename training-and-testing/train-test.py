#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Learning Exercises                                      March 06, 2023

source: train-test.py
author: @misael-diaz

Synopsis:
Determines the ability of the linear models of the training and testing sets to predict
future values.

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

"""


from numpy import ones_like
from numpy.random import seed, normal
from scipy.stats import linregress
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt


STRIDE = 4


def data():
    # creates the data set
    seed(0x2AFFFFFF)
    x = normal(3, 1, 256)
    y = normal(150, 40, 256) / x
    X = 1/x
    return (X, y)


def test():
    # creates the testing set
    X, y = data()
    test_X = X[::STRIDE]
    test_y = y[::STRIDE]
    return (test_X, test_y)


def train():
    # creates the training set
    X, y = data()

    mask = ones_like(X)
    mask = (mask == 1)
    mask[::STRIDE] = False

    train_X = X[mask]
    train_y = y[mask]
    return (train_X, train_y)


def linearFit(X, y):
    slope, intercept, r, p, stderr = linregress(X, y)
    return (slope, intercept, r, p, stderr)


def linearModel(X, y):
    slope, intercept, r, p, stderr = linearFit(X, y)
    return (lambda x: slope * x + intercept)



X, y = data()
train_X, train_y = train()
test_X, test_y = test()


plt.close('all')
plt.ion()
fig, ax = plt.subplots()
ax.scatter(X, y)
ax.set_title('original set')


fig, ax = plt.subplots()
ax.scatter(test_X, test_y)
ax.set_title('testing set')


fig, ax = plt.subplots()
ax.scatter(train_X, train_y)
ax.set_title('training set')


# determines the usefulness of the linear model for predicting future values
slope, intercept, r, p, stderr = linearFit(train_X, train_y)

model = linearModel(train_X, train_y)
r2 = r2_score(train_y, model(train_X))

# expected output R**2 ~ 0.8 (good enough)
out = (
    f'Training Set:\n'
    f'R**2: {r**2} from the linear regression\n'
    f'R**2: {r**2} by the r2-score of sklean\n'
)

print(out)


# determines the usefulness of the linear model for predicting future values
slope, intercept, r, p, stderr = linearFit(test_X, test_y)

model = linearModel(test_X, test_y)
r2 = r2_score(test_y, model(test_X))

# expected output R**2 ~ 0.8 (good enough)
out = (
    f'Testing Set:\n'
    f'R**2: {r**2} from the linear regression\n'
    f'R**2: {r**2} by the r2-score of sklean\n'
)

print(out)
