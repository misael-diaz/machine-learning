#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Machine Learning Exercises                                      March 07, 2023

source: decision-tree.py
author: @misael-diaz

Synopsis:
Uses the facilities provided by scikit-learn `sklearn' to build a Decision Tree for
predicting an outcome from given data.

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

import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt


df = pandas.read_csv('data.csv')
goMapping = {'YES': 1, 'NO': 0}
nationalityMapping = {'UK': 0, 'USA': 1, 'N': 2}
df['Go'] = df['Go'].map(goMapping)
df['Nationality'] = df['Nationality'].map(nationalityMapping)

features = ['Age', 'Experience', 'Rank', 'Nationality']
X = df[features]
y = df['Go']

dtree = DecisionTreeClassifier()
dtree.fit(X, y)

plt.close('all')
plt.ion()
tree.plot_tree(dtree, feature_names=features)
plt.savefig('decision-tree.png', DPI=300)

"""
COMMENTS:

The `gini' a value which ranges [0, 0.5] tells how the sample is splitted at that level.
If gini is close to 0.5 the sample is evenly splitted or nearly so. If gini is zero then
there is no splitting, all the candidates go in one direction or another.

Note that the root of the decision tree has a gini of nearly 0.5, which makes the Rank
a deciding feature and this is why it has been placed at the root of the tree. Note that
the other features have lower gini values. Thus, from the get-go we can split the sample
in half or nearly so just based on the Rank!

The `value' array tells us how many candidates got NO and YES, respectively, at that
particular level. Note that at the root node sample = 13 and the value = [6, 7] and that
the sum of the values yield the sample size 13 (as it should be).

The `sample' value tells how many candidates are considered at that particular level.
It is important to note that the sum of the `samples' of the children of a parent node
must be equal to the sample size of the parent. For instance, the children nodes at the
second level of the tree have sample sizes of 5 and 8 for a total of 13, which happens
to be the sample size of the parent node (the root).

You can interpret the decision tree as you normally do for the decision tree of an
algorithm.
"""
