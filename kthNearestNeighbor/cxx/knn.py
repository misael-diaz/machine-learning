#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Learning Exercises                                      June 20, 2023

source: knn.py
author: @misael-diaz

Synopsis:
Tests the implementation of the Kth Nearest Neighbor KNN algorithm.
Generates a KNN plot of the Advertising dataset (from ref[2]).

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
[3] https://github.com/chcomin/ctypes-numpy-example/blob/master/simplest/script.py
"""


import ctypes
import pandas as pd
from numpy import sort
from numpy import array
from numpy import zeros
from numpy import ctypeslib
from matplotlib import pyplot as plt


def isSorted(dataset):

  cols = dataset.shape[1]
  for i in range(cols - 1):
    if dataset[0, i + 1] < dataset[0, i]:
      return False

  return True


def distances(target, dataset):
  # sorts the output `y' with respect to the distance from the `target'

  dist = dataset.copy()
  dist[0, :] = (dist[0, :] - target)**2
  dist = dist.transpose().tolist()
  dist = sorted(dist, key = lambda elem: elem[0])
  dist = array(dist).transpose()

  return dist


def knn(K, target, dataset):
  # forwards the task to the C++ implementation of the KNN algorithm 

  if not isSorted(dataset):
    raise ValueError('KNN(): expects a dataset sorted by the features')

  dataset = dataset.flatten()   # flattens the 2 x `N' dataset into a 2 * `N' dataset
  numel = dataset.size          # total number of elements in the flattened dataset
  size = numel // 2             # the size of any of the columns, `N', of the dataset

  CWD = '.'
  lknn = ctypeslib.load_library('libknn.so', CWD)

  # defines the types of the parameters of the C++ implementation of the KNN algorithm
  ret = zeros([1, 2])
  lknn.KNN (ctypes.c_int(K),
            ctypes.c_double(target),
            ctypes.c_int(size),
            dataset.ctypes.data_as( ctypes.POINTER(ctypes.c_double) ),
            ret.ctypes.data_as( ctypes.POINTER(ctypes.c_double) ) )

  return ret.flatten()


def assertions():
  # tests the output of the KNN algorithm against the expected values

  df = pd.read_csv('datasets/Advertising.csv')
  dataset = df[['TV', 'sales']]
  dset = dataset.sort_values(by = ['TV'])
  dset = dset.to_numpy().transpose()

  pyknn = lambda kth, dist: dist[:, kth - 1]

  X = dset[0, :]
  cols = dset.shape[1]
  # for-each possible kth
  for k in range(cols):

    kth = (k + 1)
    # for-each X in the dataset
    for x_target in X:

      X_kth, y_kth = knn(kth, x_target, dset)

      dist = distances(x_target, dset)

      # there might be a duplicate kth nearest neighbor at the left or right of the target
      if kth == 1:

        this_dist, this_y = pyknn(kth, dist)
        next_dist, next_y = pyknn(kth + 1, dist)

        if this_dist == next_dist:
          assert y_kth == this_y or y_kth == next_y
        else:
          assert y_kth == this_y

      elif kth == cols:

        prev_dist, prev_y = pyknn(kth - 1, dist)
        this_dist, this_y = pyknn(kth, dist)

        if prev_dist == this_dist:
          assert y_kth == prev_y or y_kth == this_y
        else:
          assert y_kth == this_y

      else:

        prev_dist, prev_y = pyknn(kth - 1, dist)
        this_dist, this_y = pyknn(kth, dist)
        next_dist, next_y = pyknn(kth + 1, dist)

        if prev_dist == this_dist:
          assert y_kth == prev_y or y_kth == this_y
        elif this_dist == next_dist:
          assert y_kth == this_y or y_kth == next_y
        else:
          assert y_kth == this_y

  return


# The assertions fail when there are inconsistent entries in the dataset;
# that is, different responses for the same input (or feature). The original
# dataset (ref[2]) has been edited slightly to eliminate those ``inconsistencies''.
# These ``inconsistencies'' arise because just one feature is being used whereas
# the output really derives from multiple features.
assertions()


df = pd.read_csv('datasets/Advertising.csv')
dataset = df[['TV', 'sales']]
dset = dataset.sort_values(by = ['TV'])
dset = dset.to_numpy().transpose()

kth = 8
X, y = dset.copy()
for i, x_target in enumerate(X):
  _, y[i] = knn(kth, x_target, dset)

xi, yi = dset

plt.close('all')
plt.ion()
fig, ax = plt.subplots()
ax.plot(xi, yi, linestyle='', markersize=12, marker='s', color='red', label='dataset')
ax.plot(X, y, linestyle='-', color='black', label=f'{kth}th nearest neighbor')
ax.legend()
