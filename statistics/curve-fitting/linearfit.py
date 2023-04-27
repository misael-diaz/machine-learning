#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Computational Solutions                                      April 28, 2022
IST 4360
Prof. M. Diaz-Maldonado


Synopsis:
Applies the Least Squares Method to obtain the ``best fit'' of a dataset.


Copyright (c) 2022 Misael Diaz-Maldonado
This file is released under the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.


References:
[0] B Munson, D Young, T Okiishi, and W Huebsch, Fundamentals of
    Fluid Mechanics, 8th edition
[1] A Gilat and V Subramanian, Numerical Methods for Engineers and
    Scientists: An Introduction with Applications using MATLAB
[2] R Johansson, Numerical Python: Scientific Computing and Data
    Science Applications with NumPy, SciPy, and Matplotlib, 2nd edition
"""


from matplotlib import pyplot as plt
from numpy.random import normal
from numpy.linalg import det
from numpy import linspace
from numpy import array
from numpy import sqrt
from numpy import sum


def LinearFit(xi, yi):
    """ Obtains the linear fit parameters of the dataset xi, yi via Least Squares """

    # solves the system of linear equations by Crammer's rule
    d = det([
        [(xi**2).sum(), xi.sum()],
        [     xi.sum(), xi.size]
    ])


    D1 = det([
        [(xi * yi).sum(), xi.sum()],
        [       yi.sum(), xi.size]
    ])


    D2 = det([
        [(xi**2).sum(), (xi * yi).sum()],
        [     xi.sum(), yi.sum()]
    ])

    # calculates the slope and intercept, respectively
    s, i = (slope, intercept) = (D1 / d, D2 / d)

    return (s, i, lambda x: i + s * x)


def Pearson(xi, yi):
    """ Returns the Pearson's coefficient of correlation, R**2 """

    Dx, Dy = ( xi - xi.mean() ) / std(xi), ( yi - yi.mean() ) / std(yi)
    R, _, _ = LinearFit(Dx, Dy)

    return R**2


def std(x):
    """ Returns the standard deviation """

    if (x.size == 1):
        raise RuntimeError('std(): single element array')

    return sqrt( ( sum( (x.mean() - x)**2 ) ) / (x.size - 1) )




""" Dataset """
xi = linspace(-8, 8, 256)
yi = normal(loc = xi, scale = 0.25)

# applies the least squares method to obtain the ``best fit'', y = f(x)
slope, intercept, f = LinearFit(xi, yi)


""" Plots the dataset and the ``best fit'' for visualization """
x = linspace(xi.min(), xi.max(), 256)

plt.close('all')
plt.ion()
fig, ax = plt.subplots()
# plots the dataset
ax.scatter(xi, yi)
# plots the ``best fit'' y = f(x)
ax.plot(x, f(x), color='red', linestyle='--')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Least Squares')


""" Display info about the fitting parameters and ``goodness of fit'' """
output = (
    f'\nLinear Fit via Least Squares:\n'
    f'Slope: {slope:.12f}\n'
    f'Intercept: {intercept:.12f}\n'
)
print(output)
print(f"Pearson's correlation coefficient, R^2: {Pearson(xi, yi):.12f}")
