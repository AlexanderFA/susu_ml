# here is classic functions
def euclidean_distance(row1, row2):
    distance = 0
    for i in range(len(row1) - 1):  # -1 если в последнем элементе вектора находится класс ириса, иначе не надо
        distance += (row1[i] - row2[i]) ** 2

    return distance ** 0.5  # тоже самое что math.sqrt()


def get_neighbors(train_set, labels, test_row, num_neighbors):  # функция без учета классов
    distances = list()
    for train_row in train_set:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda x: x[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])

    return neighbors
