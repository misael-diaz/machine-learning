#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Learning                                      May 24, 2023

source: plots.py
author: @misael-diaz

Synopsis:
Plots the actual and the ANN predicted temperatures.

Copyright (c) 2023 Misael Diaz-Maldonado
This file is released under the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.


References:
[0] A Gilat and V Subramanian, Numerical Methods for Engineers and
    Scientists: An Introduction with Applications using MATLAB
[1] R Johansson, Numerical Python: Scientific Computing and Data
    Science Applications with NumPy, SciPy, and Matplotlib, 2nd edition
[3] https://towardsdatascience.com/step-by-step-guide-to-building-your-own-neural-network-from-scratch-df64b1c5ab6e
[4] https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
"""

from numpy import arange
from numpy import loadtxt
from matplotlib import pyplot as plt

y, y_pred = loadtxt('results.txt').transpose()
t = arange(y.size)

plt.close('all')
plt.ion()
fig, ax = plt.subplots()
ax.scatter(t, y, color = 'black', label = 'actual')
ax.scatter(t, y_pred, color = 'red', label = 'predicted')
ax.set_xlabel('time (days)')
ax.set_ylabel('non-dimensional temperature')
