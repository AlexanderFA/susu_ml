import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import svm
import mglearn

data = loadmat('ex4data3.mat')
features = data["X"]
labels = data["y"].ravel()
features_test = data["Xval"]
labels_test = data["yval"].ravel()

mglearn.discrete_scatter(features[:, 0], features[:, 1], labels)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("Признак 1")
plt.ylabel("Признак 2")
plt.show()
print("X.shape: {}".format(features.shape))  # (51, 2) 51 элемент, каждый имеет 2 признака

C_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
gamma_values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]

# implementing grid search
params = {}
max_score = 0
for C in C_values:
  for gamma in gamma_values:
    svm_classifier = svm.SVC(kernel='rbf', C=C, gamma=gamma)
    svm_classifier.fit(features, labels)
    score = svm_classifier.score(features_test, labels_test)
    if score > max_score:
        max_score = score
        params['C'] = C
        params['gamma'] = gamma

print(max_score, params)

print(f"При параметрах C = {params['C']} иgamma = {params['gamma']} достигнута наилучшая точность {max_score}\n")

svm_classifier = svm.SVC(kernel='rbf', C=params['C'], gamma=params['gamma'])
svm_classifier.fit(features, labels)

# Строим кривую, разделяющую два класса
mglearn.plots.plot_2d_separator(svm_classifier, features, eps=.05)
# Строим исходный набор данных
mglearn.discrete_scatter(features[:, 0], features[:, 1], labels)
plt.show()
