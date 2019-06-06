#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys

from sklearn.metrics import accuracy_score
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.tree import DecisionTreeClassifier


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels

def preprocess_emails(feature_percentile):
    print("Feature percentile is: ", feature_percentile)
    features_train, features_test, labels_train, labels_test = preprocess(feature_percentile=feature_percentile)

    print("features_train: ", len(features_train))
    print("labels_train: ", len(labels_train))

    print("features_train: ", len(features_train))
    print("features_test: ", len(features_test))

    print("Number of features in data: ", len(features_train[0]))

preprocess_emails(10)
preprocess_emails(1)


# clf = DecisionTreeClassifier(min_samples_split=40).fit(features_train, labels_train)
# pred = clf.predict(features_test)

# accuracy = accuracy_score(pred, labels_test)
# print(accuracy)



