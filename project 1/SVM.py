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


## data cleansing and preparation for training

# find missing values in cleveland dataset
missing_values = np.argwhere(np.isnan(cleveland))

# missing values in six instances, removing instances from dataset
missing_rows = missing_values[:,0]
cleveland_rem = np.delete(cleveland, obj = missing_rows, axis = 0)

#check if there are still missing values in the dataset
missing_values = (np.argwhere(np.isnan(cleveland_rem)))

# Split data into features (x) and target (y)
clev_x = cleveland_rem[:,0:13]
clev_y = cleveland_rem[:,13]

# make target data binary
# presence of CHD: 1 (orig: 1,2,3,4), absence: 0
clev_y = clev_y > 0



## split dataset into test and training data
x_train, x_test, y_train, y_test = train_test_split(clev_x, clev_y, test_size = .25)


## Train a Support vector machine
# Defining the classifier
classifier = SVC(kernel="linear")

# Training the classifier
classifier.fit(x_train, y_train)

# Testing the classifier
y_pred = classifier.predict(x_test)

# evaluate model 
train_acc = accuracy(y_train, classifier.predict(x_train))

test_correct, test_sens, test_spec, test_acc = evaluation(y_test, y_pred)
print("The percentage of correctly predicted targets is {}".format(test_correct))
print("The test sensitivity is {}".format(test_sens))
print("The test specificity is {}".format(test_spec))
print("The test accuracy is {}".format(test_acc))



