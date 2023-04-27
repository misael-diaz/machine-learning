#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Learning Exercises                                      April 27, 2023

source: polynomial-multivariate-regression.py
author: @misael-diaz

Synopsis:
Assesses the ability of a polynomial model to fit nonlinear data.


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

"""


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from numpy.random import normal
from numpy import linspace
from numpy import newaxis
from numpy import hstack


''' generates the dataset '''


# defines limits for the independent variables or features
x1_min, x1_max = -4, 4
x2_min, x2_max = -8, 8

N = 256
# generates 1D uniformly spaced values in [min, max] for the features
x1 = linspace(x1_min, x1_max, N)
x2 = linspace(x2_min, x2_max, N)

# concatenates x1 and x2 into a matrix of expected shape (rows: records, cols: features)
x1 = x1[newaxis, :].transpose()
x2 = x2[newaxis, :].transpose()
X = hstack([x1, x2])

# creates a polynomial relationship between the features X and the dependent variable y
y = 1 + x1 + x2 + x1 * x2 + x1**2 + x2**2
# adds a small noise
y = normal(loc = y, scale = 1.5)

''' linear model  '''

# obtains a linear regression model
model = LinearRegression (fit_intercept = True, normalize = True,
                          copy_X = True, n_jobs = None)
model.fit(X, y)
score = model.score(X, y)
print(f'linear model score: {score}')

''' polynomial model '''

# standarizes features
sc = StandardScaler(copy = True, with_mean = True, with_std = True)
X_std = sc.fit_transform(X)

# gets the polynomial features from X by assuming a second-degree polynomial relationship
pf = PolynomialFeatures(degree = 2, interaction_only = False, include_bias = True)
X_pol = pf.fit_transform(X_std)

model = LinearRegression (fit_intercept = True, normalize = False,
                          copy_X = True, n_jobs = None)

model.fit(X_pol, y)
score = model.score(X_pol, y)
print(f'polynomial model score: {score}')

'''
COMMENTS:
Shows that the polynomial model captures the inherent polynomial relationship of the
dependent variable y with the independent variables x1 and x2. On the other hand, the
linear model does not fit the dataset as expected, and this is evidenced by its poor
score.

The features X could have overwritten upon standarization and polynomial feature
generation via

```
sc = StandardScaler(copy = True, with_mean = True, with_std = True)
X = sc.fit_transform(X)
pf = PolynomialFeatures(degree = 2, interaction_only = False, include_bias = True)
X = pf.fit_transform(X)
```

but decided not to do so in this example.

'''
