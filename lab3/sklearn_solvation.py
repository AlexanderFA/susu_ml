import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

# import some data to play with
iris = datasets.load_iris()

features = iris.data  # X
labels = iris.target  # y

logreg = LogisticRegression(C=1e5)
test_size = 0.2
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3)
logreg.fit(features_train, labels_train)

labels_predicted = logreg.predict(features_test)
accuracy = np.mean(labels_predicted == labels_test)
print("Точность предсказаний: {:.2f}%\n".format(accuracy * 100))

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, stratify=labels)
# logreg = LogisticRegression(C=1e5)
logreg.fit(features_train, labels_train)

labels_predicted = logreg.predict(features_test)
accuracy = np.mean(labels_predicted == labels_test)
print("Точность предсказаний: {:.2f}%\n".format(accuracy * 100))

