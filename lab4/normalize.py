import sys

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

cancer = load_breast_cancer()
features_train, features_test, labels_train, labels_test = train_test_split(
    cancer.data,
    cancer.target,
    # stratify=None,
    # stratify=cancer.target,
    # random_state=33
)

svc = svm.SVC(C=10)  # при С в районе 10 максимальная точность на тестовом наборе
svc.fit(features_train, labels_train)

# Правильность на обучающем наборе: 0.93
# Правильность на тестовом наборе: 0.87
print("Правильность на обучающем наборе: {:.2f}".format(svc.score(features_train, labels_train)))
print("Правильность на тестовом наборе: {:.2f}".format(svc.score(features_test, labels_test)))

# чтобы визуально показать насколько в каждом векторе признаков
# различаются по величине отобразим график
plt.plot(features_train.max(axis=0), '^', label="максимальный признак")
plt.plot(features_train.min(axis=0), 'o', label="минимальный признак")
plt.legend(loc=4)
plt.xlabel("Индекс признака")
plt.ylabel("Величина признака")
plt.yscale("log")
plt.show()

# scaler = MinMaxScaler()
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))  # эти параметры и так по умолчанию
scaler.fit(features_train)
features_train_scaled = scaler.transform(features_train)
features_test_scaled = scaler.transform(features_test)

# print("not-transformed shape: %s" % (features_train.shape,))
# print("transformed shape: %s" % (features_train_scaled.shape,))
# print("per-feature minimum before scaling:\n %s" % features_train.min(axis=0))
# print("per-feature maximum before scaling:\n %s" % features_train.max(axis=0))
# print("per-feature minimum after scaling:\n %s" % features_train_scaled.min(axis=0))
# print("per-feature maximum after scaling:\n %s" % features_train_scaled.max(axis=0))
# print(features_train_scaled)

# теперь видим на графике, что значения выровнены
plt.plot(features_train_scaled.max(axis=0), '^', label="максимальный признак")
plt.plot(features_train_scaled.min(axis=0), 'o', label="минимальный признак")
plt.title('все признаки принадлежат диапазону [0, 1]')
plt.legend(loc=4)
plt.xlabel("Индекс признака")
plt.ylabel("Величина признака")
plt.ylim(bottom=-0.03, top=1.03)
plt.legend(loc='center right')
plt.show()

svc.fit(features_train_scaled, labels_train)

# Правильность на обучающем наборе: 0.98
# Правильность на тестовом наборе: 0.98
print("Правильность на обучающем наборе: {:.2f}".format(svc.score(features_train_scaled, labels_train)))
print("Правильность на тестовом наборе: {:.2f}".format(svc.score(features_test_scaled, labels_test)))

fig, axes = plt.subplots(2, 1, figsize=(8, 15))
axes[0].scatter(features_train[:, 0], features_train[:, 1], c='b', label="training set", s=60)
axes[0].scatter(features_test[:, 0], features_test[:, 1], marker='^', c='r', label="test set", s=60)
axes[0].legend(loc='upper left')
axes[0].set_title("Исходные данные")

axes[1].scatter(features_train_scaled[:, 0], features_train_scaled[:, 1], c='b', label="training set", s=60)
axes[1].scatter(features_test_scaled[:, 0], features_test_scaled[:, 1], marker='^', c='r', label="test set", s=60)
axes[1].set_title("Отмасштабированные данные")
plt.show()

# имплементация minMaxScaler
def custom_scaler(features):
    min_vals = features.min(axis=0)
    max_vals = features.max(axis=0)
    scaled_features = (features - min_vals) / (max_vals - min_vals)

    return scaled_features


features_train_custom_scaled = custom_scaler(features_train)

features_train_scaled_r = np.round(features_train_scaled, 10)
features_train_custom_scaled_r = np.round(features_train_custom_scaled, 3)
# A = features_train_scaled_r[:2, :5]
# B = features_train_custom_scaled_r[:2, :5]
# print(A)
# print(B)


def calc_accuracy(matrix1, matrix2):
    # Посчитаем количество элементов в матрице matrix1
    num_elements = matrix1.shape[0] * matrix1.shape[1]

    # Посчитаем сумму абсолютных разниц между каждым элементом той и другой матрицы
    sum_differences = np.sum(np.abs(matrix1 - matrix2))

    return 1 - (sum_differences / num_elements)


accuracy = calc_accuracy(features_train_scaled_r, features_train_custom_scaled_r)
print("Совпадение результата работы кастомной функции масштабирвания: {:.2f}%\n".format(accuracy * 100))


