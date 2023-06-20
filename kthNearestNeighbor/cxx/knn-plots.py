#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Learning Exercises                                      June 20, 2023

source: knn-plots.py
author: @misael-diaz

Synopsis:
Plots the Kth Nearest Neighbor modeling results of the Advertising dataset.

Copyright (c) 2023 Misael Diaz-Maldonado
This file is released under the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

References:
[0] A Gilat and V Subramanian, Numerical Methods for Engineers and
    Scientists: An Introduction with Applications using MATLAB
[1] R Johansson, Numerical Python: Scientific Computing and Data
    Science Applications with NumPy, SciPy, and Matplotlib, 2nd edition
[2] https://www.statlearning.com/resources-second-edition
"""

import pandas as pd
from numpy import loadtxt
from matplotlib import pyplot as plt

df = pd.read_csv('datasets/Advertising.csv')
dataset = df[['TV', 'sales']]
xi, yi = dataset.to_numpy().transpose()

x, y = loadtxt('results/1stNearestNeighbors.txt').transpose()

plt.close('all')
plt.ion()
fig, ax = plt.subplots()
ax.plot(x, y)
ax.plot(xi, yi, linestyle='', markersize=12, marker='s', color='black')
