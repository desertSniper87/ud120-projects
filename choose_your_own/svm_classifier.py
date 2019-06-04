import sys

from time import time

from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import copy
import numpy as np
import pylab as pl

import scikitplot as skplt


features_train, labels_train, features_test, labels_test = makeTerrainData()

features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]
########################## SVM #################################
### we handle the import statement and SVC creation for you here
from sklearn.svm import SVC
clf = SVC(kernel="rbf", C=1000)


#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data
time1 = time()
clf.fit(features_train, labels_train)
time2 = time()

print ("The time to train is: ", time2-time1)

#### store your predictions in a list named pred

time1 = time()
pred = clf.predict(features_test)
time2 = time()


print ("The time to predict is: ", time2-time1)



from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

print("The accuracy is ", acc)

def submitAccuracy():
    return acc
