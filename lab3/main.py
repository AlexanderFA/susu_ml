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

print(f"Классы датасета: {labels}\n")  # не перемешаны

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

# создадим тестовый набор по правилам лабораторной работы (каждый 15й начиная с первого элемента и включая его)
dataset = features[:150:15]  # 10 строк
# dataset = features[:9:3]
output = labels[:150:15]


# функция, вычисляющая евклидово расстояние между двумя векторами euclidean_distance(row1, row2). (вариант через numpy)
def euclidean_distance(row1, row2):
    return np.sqrt(np.sum((row1 - row2) ** 2))


vector_compare = dataset[5]
label_compare = output[5]
# пример расчета евклидового расстояния между вектором 0 и 5 из свойств датасета
distance = euclidean_distance(vector_compare, dataset[0])
# print(distance)

# применим функцию к каждому элементу к тестовому набору
result = np.apply_along_axis(euclidean_distance, 1, dataset, vector_compare)
# result = np.apply_along_axis(euclidean_distance, 0, dataset, example_instance)
print(f"евклидово расстояние между вектором dataset[5]:\n{result}\n")


# функция которая находит в train_set выборке k = num_neighbors ближайших соседей
# (в смысле близости евклидова расстояния) к данному (test_row)
def get_neighbors(train_set, train_labels, test_row, num_neighbors):
    distances = np.zeros(len(train_set))
    for i, train_row in enumerate(train_set):
        dist = euclidean_distance(test_row, train_row)
        distances[i] = dist
    sorted_distances_indices = np.argsort(distances)

    nearest_neighbors = list()
    for i in range(num_neighbors):
        index = sorted_distances_indices[i]
        neighbor_label = train_labels[index]
        neighbor_features = train_set[index]
        neighbor_distance = distances[index]

        nearest_neighbors.append([neighbor_features, neighbor_distance, neighbor_label])

    return np.array(nearest_neighbors, dtype=object)


print(f"ближайшие 3 соседа для dataset[5]:\n{get_neighbors(dataset, output, vector_compare, 3)}\n")


def predict_classification(train_set, train_labels, test_row, num_neighbors):
    neighbors = get_neighbors(train_set, train_labels, test_row, num_neighbors)
    classes = neighbors[:, -1].astype(int)

    return np.bincount(classes).argmax()


prediction = predict_classification(dataset, output, vector_compare, 3)
print('Expected %d, Got %d.\n' % (label_compare, prediction))


def k_nearest_neighbors(train_set, train_labels, test_set, num_neighbors):
    # мы объявим такую функцию чтобы было удобно использовать  apply_along_axis (array_mep)
    def call_predict_classification(test_row):
        return predict_classification(train_set, train_labels, test_row, num_neighbors)

    predictions = np.apply_along_axis(call_predict_classification, axis=1, arr=test_set)

    return predictions


# prediction_array = k_nearest_neighbors(features, labels, dataset, 3)
# print(prediction_array)

# features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.67, random_state=33, stratify=labels)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)

# почти тоже самое что _, counts = np.unique(labels_test, return_counts=True)
print(f"Количество строк в y_train по классам: {np.bincount(labels_train)}")
print(f"Количество строк в y_test по классам: {np.bincount(labels_test)}")
print(f"Всего строк в датасете: {np.size(labels)}")


prediction_array = k_nearest_neighbors(features_train, labels_train, features_test, 3)
print(prediction_array)
print(labels_test)


sys.exit()
