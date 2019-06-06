import sys

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()

clf_2 = DecisionTreeClassifier(min_samples_split=2).fit(features_train, labels_train)
pred = clf_2.predict(features_test)
acc_min_samples_split_2 = accuracy_score(pred, labels_test)

clf_50 = DecisionTreeClassifier(min_samples_split=50).fit(features_train, labels_train)
pred = clf_50.predict(features_test)
acc_min_samples_split_50 = accuracy_score(pred, labels_test)

### be sure to compute the accuracy on the test set
def submitAccuracies():
  return {"acc_min_samples_split_2":round(acc_min_samples_split_2,3),
          "acc_min_samples_split_50":round(acc_min_samples_split_50,3)}

print(submitAccuracies())
