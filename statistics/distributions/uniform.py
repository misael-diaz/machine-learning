#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Learning Exercises                                      March 04, 2023

source: uniform.py
author: @misael-diaz

Synopsis:
Plots the probability density function of uniformly distributed data.


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
from matplotlib import pyplot as plt

data = numpy.random.rand(1048576)   # Note: 2**21 = 1048576

plt.close('all')
plt.ion()
plt.hist(data, density=True)
