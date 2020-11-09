# -*- coding: utf-8 -*-
"""
@author: rapha
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from utils import accuracy, evaluation


# load data from original source
cleveland= np.genfromtxt("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
                         delimiter = ",")

hungarian = np.genfromtxt("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data",
                         delimiter = ",")

## data cleansing and preparation for training on cleveland dataset
# find missing values in cleveland dataset
missing_values_c = np.argwhere(np.isnan(cleveland))

# missing values in six instances, removing instances from dataset
missing_rows_c = missing_values_c[:,0]
cleveland_rem = np.delete(cleveland, obj = missing_rows_c, axis = 0)

#check if there are still missing values in the dataset
(np.argwhere(np.isnan(cleveland_rem))).size == 0

# Split data into features (x) and target (y)
clev_x = cleveland_rem[:,0:13]
clev_y = cleveland_rem[:,13]

# make target data binary
# presence of CHD: 1 (orig: 1,2,3,4), absence: 0
clev_y = clev_y > 0


## split dataset into test and training data
x_train, x_test, y_train, y_test = train_test_split(clev_x, clev_y, test_size = .25)

## Train a Support vector machine for cleveland training and cleveland evaluation data
# Defining the classifier
classifier = SVC(kernel="linear")

# Training the classifier
classifier.fit(x_train, y_train)

# Testing the classifier
y_pred = classifier.predict(x_test)

# evaluate model
test_correct, test_sens, test_spec, test_acc = evaluation(y_test, y_pred)
print("The percentage of correctly predicted targets is {}".format(test_correct))
print("The test sensitivity is {}".format(test_sens))
print("The test specificity is {}".format(test_spec))
print("The test accuracy is {}".format(test_acc))



## data cleansing and preparation for training on hungarian dataset
# rows 10 to 12 missing for every entry in hungarian dataset
# training a classifier without variables 10 to 12
# removing them from hungarian and cleveland dataset for training
cleveland_alltraining = cleveland_rem[:,np.r_[0:10,13]]
hungarian_alltesting = hungarian[:,np.r_[0:10,13]]

# find missing values in hungarian dataset
missing_values_h = np.argwhere(np.isnan(hungarian_alltesting))

# remove rows with further missing values from hungarian dataset
missing_rows_h = missing_values_h[:,0]
hungarian_atest_rem = np.delete(hungarian_alltesting, obj = missing_rows_h, axis = 0)

# split data into features (x) and target (y) for train (cleveland) and test (hungarian) dataset
clev_atrain_x = cleveland_alltraining[:,0:10]
clev_atrain_y = cleveland_alltraining[:,10]

hungarian_atest_x = hungarian_atest_rem[:,0:10]
hungarian_atest_y = hungarian_atest_rem[:,10]

# make target data binary
# presence of CHD: 1 (orig: 1,2,3,4), absence: 0
clev_atrain_y = clev_atrain_y > 0
hungarian_atest_y = hungarian_atest_y > 0

## Train a Support vector machine for cleveland training and hungarian evaluation data
# Defining the classifier
classifier2 = SVC(kernel="linear")

# Training the classifier
classifier2.fit(clev_atrain_x, clev_atrain_y)

# Testing the classifier
y_pred_hung = classifier2.predict(hungarian_atest_x)

# evaluate model trained on cleveland dataset and tested on hungarian dataset
test_correct, test_sens, test_spec, test_acc = evaluation(hungarian_atest_y, y_pred_hung)
print("The percentage of correctly predicted targets is {}".format(test_correct))
print("The test sensitivity is {}".format(test_sens))
print("The test specificity is {}".format(test_spec))
print("The test accuracy is {}".format(test_acc))