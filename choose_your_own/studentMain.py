#!/usr/bin/python

""" lecture and example code for decision tree unit """

from class_vis import prettyPicture, output_image
from prep_terrain_data import makeTerrainData

from choose_your_own.classifyDT import classify

from sklearn.metrics import accuracy_score

features_train, labels_train, features_test, labels_test = makeTerrainData()



### the classify() function in classifyDT is where the magic
### happens--fill in this function in the file 'classifyDT.py'!
clf = classify(features_train, labels_train)
pred = clf.predict(features_test)


acc = accuracy_score(pred, labels_test)
print("The accuracy is ", acc)


def submitAccuracies():
  return {"acc":round(acc,3)}


#### grader code, do not modify below this line

prettyPicture(clf, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())
