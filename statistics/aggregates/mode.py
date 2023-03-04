#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Learning Exercises                                      March 04, 2023

source: mode.py
author: @misael-diaz

Synopsis:
Uses the facilities of Scientific Python to calculate the mode of an array of numbers.


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

from scipy import stats
from data import speeds

# prints the median speed on the console
mode, count = stats.mode(speeds, axis=None)
mode = mode[0]

print(f'mode: {mode} km / h')


'''
Example from the documentation of scipy.stats.mode():

mode, count = stats.mode([
    [6, 8, 3, 0],
    [3, 2, 1, 7],
    [8, 1, 8, 4],
    [5, 3, 0, 5],
    [4, 7, 5, 9]
])

output:
mode, count = (array([[3, 1, 0, 0]]), array([[1, 1, 1, 1]]))

Despite that numpy arrays have a C-like memory layout, the method stats.mode()
does it work along the columns of the array. Bearing this in mind it is easy to
see that the elements of mode correspond to the smallest value of each column;
this is the case because the method selects the smallest value when there are 
multiple instances. (Note that the columns of the array are comprised by distinct
values.)
'''

# Note: scipy submodules need to be imported explicitly (it is not possible to invoke
# scipy.stats.mode() directly, for some of the scipy submodules do not load automatically)
