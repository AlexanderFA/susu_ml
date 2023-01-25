import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import svm
import mglearn

data = loadmat('ex4data1.mat')
features = data["X"]
labels = data["y"].ravel()

mglearn.discrete_scatter(features[:, 0], features[:, 1], labels)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("Признак 1")
plt.ylabel("Признак 2")
plt.show()
print("X.shape: {}".format(features.shape))  # (51, 2) 51 элемент, каждый имеет 2 признака

# Большое значение C соответствует малому значению 𝜆. Регуляризации нет, модель склонна к переобучению и "ловит" выбросы
sv_classifier = svm.LinearSVC(C=1, loss='hinge', max_iter=10000)
sv_classifier.fit(features, labels)
score = sv_classifier.score(features, labels)
print(score)
mglearn.plots.plot_2d_separator(sv_classifier, features, eps=.5)
mglearn.discrete_scatter(features[:, 0], features[:, 1], labels)
plt.show()
