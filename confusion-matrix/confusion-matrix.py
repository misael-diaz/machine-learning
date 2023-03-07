#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Learning Exercises                                      March 07, 2023

source: confusion-matrix.py
author: @misael-diaz

Synopsis:
Uses scikit-learn metrics to assess the predictive ability of a model. 

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
[3] https://mathworld.wolfram.com/BinomialDistribution.html

"""

from numpy.random import binomial
from sklearn import metrics
from matplotlib import pyplot as plt


'''
samples 1024 elements from the parameterized binomial distribution to generate the
`actual' data and the `predicted' data yielded by some model
'''


# binomial distribution parameters: one trial and probability of success of 0.9
actual = binomial(1, 0.9, size=1024)
predicted = binomial(1, 0.9, size=1024)


# Metrics:


# plots the confusion matrix
cmat = metrics.confusion_matrix(actual, predicted)
cmatDisplay = metrics.ConfusionMatrixDisplay(
    confusion_matrix = cmat, display_labels = [False, True]
)

plt.close('all')
plt.ion()
cmatDisplay.plot()


# Accuracy:
# The number of correct predictions with respect to the total number of predictions:
#
# (True Positives + True Negatives) / Total Number of Predictions
#


accuracy = metrics.accuracy_score(actual, predicted)


# Precision:
# The number of True Positives with respect to the actual number of Positives:
#
# True Positives / (True Positives + False Positives)
#


precision = metrics.precision_score(actual, predicted)


# Sensitivity (or Recall):
# A measure of the ability of the model to predict actual Positivies.
# The number of True Positives with respect to the actual number of Positives:
#
# True Positives / (True Positives + False Negatives)
#


sensitivity = metrics.recall_score(actual, predicted)


# Specificity:
# A measure of the ability of the model to predict actual Negatives.
# The number of True Positives with respect to the actual number of Positives:
#
# True Negative / (True Negative + False Positive)
#


specificity = metrics.recall_score(actual, predicted, pos_label=0)


# F-score:
# a harmonic mean of the model precision and sensitivity
#


f1_score = metrics.f1_score(actual, predicted)


modelMetrics = (
    f'Metrics:\n'
    f'accuracy:    {accuracy:.15f}\n'
    f'precision:   {precision:.15f}\n'
    f'sensitivity: {sensitivity:.15f}\n'
    f'specificity: {specificity:.15f}\n'
    f'F-score:     {f1_score:.15f}\n'
)

print(modelMetrics)
