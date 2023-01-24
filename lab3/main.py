import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import sys


iris = load_iris()

features = iris.data  # X
labels = iris.target  # y
labels_names = ['I.setosa', 'I.versicolor', 'I.virginica']
colors = ['purple', 'teal', 'orange']

print(f"Классы датасета: {labels}")  # не перемешаны

# соберем данные признаков для каждого типа цветка - x - это длина чашелистника а y - его ширина
for i in range(len(colors)):
    px = features[:, 0][labels == i]
    py = features[:, 1][labels == i]
    plt.scatter(px, py, c=colors[i])
plt.legend(labels_names, loc=4)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.show()

# то же самое - но для других признаков - длина и ширина лепестка
for i in range(len(colors)):
    px = features[:, 2][labels == i]
    py = features[:, 3][labels == i]
    plt.scatter(px, py, c=colors[i])
plt.legend(labels_names, loc=0)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.67, random_state=33, stratify=labels)

# тоже самое что _, counts = np.unique(labels_test, return_counts=True)
print(f"Количество строк в y_train по классам: {np.bincount(labels_train)}")
print(f"Количество строк в y_test по классам: {np.bincount(labels_test)}")
print(f"Всего строк в датасете: {np.size(labels)}")

# sys.exit()

# создадим тестовый набор по правилам лабораторной работы (каждый 15й начиная с первого элемента и включая его)
dataset = features[:150:15]  # 10 строк
# dataset = features[:9:3]
output = labels[:150:15]


# функция, вычисляющая евклидово расстояние между двумя векторами euclidean_distance(row1, row2).
# def euclidean_distance(row1, row2):
#     distance = 0
#     for i in range(len(row1)):
#         distance += (row1[i] - row2[i]) ** 2
#
#     return distance ** 0.5  # тоже самое что math.sqrt()

# вариант через numpy
def euclidean_distance(row1, row2):
    return np.sqrt(np.sum((row1 - row2) ** 2))

# print(dataset[5])
# print(dataset)
# print(output)


vector_compare = dataset[5]
# пример расчета евклидового расстояния между вектором 0 и 5 из свойств датасета
distance = euclidean_distance(vector_compare, dataset[0])
# print(distance)

# применим функцию к каждому элементу к тестовому набору
result = np.apply_along_axis(euclidean_distance, 1, dataset, vector_compare)
# result = np.apply_along_axis(euclidean_distance, 0, dataset, example_instance)
print(result)
