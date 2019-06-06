from sklearn.tree import DecisionTreeClassifier

def classify(features_train, labels_train):

    clf = DecisionTreeClassifier().fit(features_train, labels_train)
    return clf