#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Learning Exercises                                      April 22, 2023

source: artificial-neural-network-ann-exercise.py
author: @misael-diaz

Synopsis:
Uses the facilities provided by tensorflow and sklearn to create an
Artificial Neural Network ANN.

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

"""


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def encodeGender(genders):

  for i in range(genders.size):

    if genders[i] == 'Male':
      genders[i] = 0
    else:
      genders[i] = 1

  return


def decodeGender(genders):

  for i in range(genders.size):

    if genders[i] == 0:
      genders[i] = 'Male'
    else:
      genders[i] = 'Female'

  return


def encodeCountry(countries):

  for i in range(countries.size):

    if countries[i] == 'France':
      countries[i] = 0
    elif countries[i] == 'Germany':
      countries[i] = 1
    else:
      countries[i] = 2

  return


def decodeCountry(countries):

  for i in range(countries.size):

    if countries[i] == 0:
      countries[i] = 'France'
    elif countries[i] == 1:
      countries[i] = 'Germany'
    else:
      countries[i] = 'Spain'

  return


csvfile = 'Churn_Modelling.csv'
data = pd.read_csv(csvfile)
# excludes useless data (first three columns) and the outcome (last column), total 10 cols
X = data.iloc[:, 3:13].values
# gets the outcome (last column) if the user `exited` the bank or not
Y = data.iloc[:, -1].values

# encodes non-numeric data (gender and geographical location or country)
encodeCountry(X[:, 1])
encodeGender(X[:, 2])

# splits into train and test sets
tt = traintest = train_test_split(X, Y, test_size = 0.2,random_state = 0)
X_train, X_test, Y_train, Y_test = tt

# normalizes the train and test sets
standardScaler = StandardScaler()
X_train = standardScaler.fit_transform(X_train)
X_test = standardScaler.transform(X_test)

# inits artificial neural network ann
ann = tf.keras.models.Sequential()

# adds first hidden layer (from ref[3]: units: #neurons, relu: rectified linear unit)
ann.add( tf.keras.layers.Dense(units = 6, activation = "relu") )
# adds second hidden layer
ann.add( tf.keras.layers.Dense(units = 6, activation = "relu") )
# adds output layer
ann.add( tf.keras.layers.Dense(units = 1, activation = "sigmoid") )

ann.compile(optimizer = "adam",loss = "binary_crossentropy", metrics = ['accuracy'])
ann.fit(X_train, Y_train, batch_size = 32, epochs = 100)

# predicts whether a future customer will exit
customerInfo = [600, 'France', 'Female', 42, 2, 0, 1, 1, 1, 101000]
# encodes Country and Gender according to mapping and actual values
customerInfo[1] = 0
customerInfo[2] = 1

d = standardScaler.transform([customerInfo])
outcome = (ann.predict(d) > 0.5)[0][0]
print(f'\ncustomer exits (based on the given info): {outcome}\n')

"""
COMMENTS:
In constrast to the procedure given in reference [3], the encoding of textual data
was done via user-defined functions. This had little effect on the ability of the ANN
to predict the outcome, for the accuracy varies about 85%.
"""
