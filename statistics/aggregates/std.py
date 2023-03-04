#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Learning Exercises                                      March 04, 2023

source: std.py
author: @misael-diaz

Synopsis:
Uses the facilities of Numerical Python to calculate the standard deviation of an array of
numbers.


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

import numpy
from data import speeds

# prints the standard deviation of the speed array on the console
print(f'standard-deviation: {numpy.std(speeds)} km / h')

'''
Notes borrowed from the documentation of numpy.std() regarding its implementation:

The average squared deviation is normally calculated as
``x.sum() / N``, where ``N = len(x)``.  If, however, `ddof` is specified,
the divisor ``N - ddof`` is used instead. In standard statistical
practice, ``ddof=1`` provides an unbiased estimator of the variance
of the infinite population. ``ddof=0`` provides a maximum likelihood
estimate of the variance for normally distributed variables. The
standard deviation computed in this function is the square root of
the estimated variance, so even with ``ddof=1``, it will not be an
unbiased estimate of the standard deviation per se.
'''
