import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import svm
import mglearn
import sys

data = loadmat('ex4data2.mat')
features = data["X"]
labels = data["y"].ravel()

mglearn.discrete_scatter(features[:, 0], features[:, 1], labels)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("Признак 1")
plt.ylabel("Признак 2")
plt.show()
print("X.shape: {}".format(features.shape))  # (51, 2) 51 элемент, каждый имеет 2 признака

# берем С=100 и RBF ядро с параметром y=10
# можно улучшить точность за счет увеличения гаммы (например 50) и С
# при С = 100 и gamma = 50, точность 0.9976825028968713
svm_classifier = svm.SVC(kernel='rbf', C=100, gamma=10)
svm_classifier.fit(features, labels)
print(svm_classifier.score(features, labels))

# Строим кривую, разделяющую два класса
mglearn.plots.plot_2d_separator(svm_classifier, features, eps=.05)
# Строим исходный набор данных
mglearn.discrete_scatter(features[:, 0], features[:, 1], labels)
plt.show()

# sys.exit()
