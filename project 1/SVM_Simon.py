# -*- coding: utf-8 -*-
"""
@author: Simon Ruber
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from utils import accuracy, evaluation, dataFrameFromURLOrFile


#Load data files
cleveland_df = dataFrameFromURLOrFile(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
    "./data/cleveland.csv",
    ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
)

hungarian_df = dataFrameFromURLOrFile(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data",
    "./data/hungarian.csv",
    ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
)

print("datasets loaded!")

#Scan for missing values
def removeMissingValues(df):
    print("\n###\n###\nClean missing values\n###\n###\n")
    print("Missing Values within the dataset: " + str(df.isnull().sum().sum()))
    null_data = df[df.isnull().any(axis=1)]
    #print(null_data)
    #print("Dropped rows containing missing values!")
    return df.dropna()


cleveland_df = removeMissingValues(cleveland_df)

#Encode Dataset
print("\n###\n###\nEncode data\n###\n###\n")
encode_cols= ["restecg","cp", "slope","thal"]
print("discrete features: "+ str(encode_cols))
enc = OneHotEncoder(sparse=False)
cleveland_df_enc = pd.DataFrame(enc.fit_transform(cleveland_df[encode_cols]), index=cleveland_df.index)
cleveland_df_enc.columns = enc.get_feature_names(encode_cols)
print("Dataset encoded!")

cleveland_df_cleaned = cleveland_df[["age", "sex", "trestbps", "chol", "fbs", "thalach", "exang", "oldpeak", "ca"]].copy()
cleveland_df_cleaned["target"] = np.where(cleveland_df['target'] > 1,1,0)
cleveland_df_cleaned = cleveland_df_cleaned.join(cleveland_df_enc)


#Assign X and y
y = cleveland_df_cleaned["target"]
X = cleveland_df_cleaned.drop("target", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25)

#Train Classifier
svc_classifier = SVC(kernel="linear")
svc_classifier.fit(X_train, y_train)

y_pred = svc_classifier.predict(X_test)


test_correct, test_sens, test_spec, test_acc = evaluation(y_test, y_pred)
print("The percentage of correctly predicted targets is {}".format(test_correct))
print("The test sensitivity is {}".format(test_sens))
print("The test specificity is {}".format(test_spec))
print("The test accuracy is {}".format(test_acc))

