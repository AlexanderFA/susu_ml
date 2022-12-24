import numpy as np


class Helper:
    @staticmethod
    def prepare_data(data):
        n = len(data.columns) - 1

        data.insert(0, 'Ones', 1)  # Добавим нулевую колонку с единичками
        x = data.iloc[:, 0:n + 1]  # 0:2 или 0 и 1
        y = data.iloc[:, n + 1:n + 2]  # 2:3 или 2

        x = np.matrix(x.values)
        y = np.matrix(y.values)
        # theta = np.matrix(np.zeros((n + 1, 1)))
        theta = np.zeros(n + 1)

        return x, y, theta

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-1 * z))

    @staticmethod
    def compute_cost(theta, x, y):
        m = len(x)
        probability = Helper.calc_probability(x, theta)

        one = np.multiply(y, np.log(probability))
        two = np.multiply((1 - y), np.log(1 - probability))

        return -(1 / m) * (one + two).sum()  # J(θ)

    @staticmethod
    def calc_probability(x, theta):
        matrix = np.matrix(theta).T
        x_theta = x * np.matrix(theta).T
        # x_theta = np.dot(x, theta)
        probability = Helper.sigmoid(x_theta)

        return probability

    @staticmethod
    def gradient_func(theta, x, y):
        probability = Helper.calc_probability(x, theta)
        error = probability - y

        m = len(x)
        n = len(theta)
        res = np.zeros(n)

        for j in range(n):
            koef = x[:, j]
            res[j] = (1 / m) * np.sum(np.multiply(error, koef))

        return res


    @staticmethod
    def predict(theta, x):
        probability = Helper.calc_probability(x, theta)
        # print(probability)
        return [x >= 0.5 for x in probability]
