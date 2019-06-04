#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC


features_train, features_test, labels_train, labels_test = preprocess()


# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]

def execute(kernel_type, c_value):
    print("We are using ", kernel_type, ", and C val is: ", c_value)
    clf = SVC(kernel=kernel_type,
              C=c_value)

    time1 = time()
    clf.fit(features_train, labels_train)
    time2 = time()

    print ("The time to train is: ", time2-time1)

    time1 = time()
    pred = clf.predict(features_test)
    time2 = time()


    print ("The time to predict is: ", time2-time1)



    from sklearn.metrics import accuracy_score
    acc = accuracy_score(pred, labels_test)

    print("The accuracy is ", acc)

    return clf

# execute(kernel_type="rbf", c_value=1)
# execute(kernel_type="rbf", c_value=10)
# execute(kernel_type="rbf", c_value=100)
# execute(kernel_type="rbf", c_value=1000)
clf = execute(kernel_type="rbf", c_value=10000)
