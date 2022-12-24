import numpy as np
import pandas as pd
from sklearn import preprocessing

class Helper:
    # DAMAGE_PER_HIT = 20
    # __hp = 100

    @staticmethod
    def prepare_data(data, n):
        # Добавим нулевую колонку с единичками
        data.insert(0, 'Ones', 1)
        # Заберем первую колонку из единиц1 и 2ю колонку (популяция) и положим в x
        # Заберем третью колонку (профит) и положим в y
        x = data.iloc[:, 0:n + 1]  # 0:2 или 0 и 1
        # в случае с единичной x будет содержать колонку единиц и значение 1го признака
        # в случае с множественной регрессией x содержать единицы и все остальные признаки
        # y всегда содержит некий рассчитываемый результат
        y = data.iloc[:, n + 1:n + 2]  # 2:3 или 2

        # Convert to matrices and initialize parameters theta to 0s.
        # Theta is a vector [n + 1 x 1] and Theta Transpose is a vector [1 x n+1],
        # where n is the number of features.
        x = np.matrix(x.values)
        y = np.matrix(y.values)
        theta = np.matrix(np.zeros((n + 1, 1)))
        # theta = np.matrix(np.full((n + 1, 1), 3))
        # На выходе получаем 3 матрицы x [1, data[i][profit]], y [data[i][population]], theta [[0.], [0.]]

        # Check the dimensions of the matrices.
        # x.shape, y.shape, theta.shape

        return x, y, theta

    @staticmethod
    # Create a function to compute cost. (Lost function)
    def compute_cost(x, y, theta):
        """
        Compute the cost function.
        Args:
            x: a m by n+1 matrix
            y: a m by 1 vector
            theta: a n+1 by 1 vector
        Returns:
            cost: float
        """
        m = len(x)
        x_theta = x * theta
        x_theta_minus_y = x_theta - y
        transposed = x_theta_minus_y.transpose()
        cost = (transposed * x_theta_minus_y).item() / (2 * m)
        # просто возведение в квадрат без транспонирования дает такой же результат
        # cost = np.sum(np.square(x_theta_minus_y)) / (2 * m)
        # print((transposed * x_theta_minus_y).item())  # 6222.1103722264
        # print(np.sum(np.square(x_theta_minus_y)))     # 6222.110372226401

        return cost

    # Create a function to implement gradient descent.
    @staticmethod
    def gradient_descent(x, y, theta, alpha, iterations):
        """
        Implement gradient descent.
        Args:
            x: a m by n+1 matrix
            theta: a n+1 by 1 vector
        Return:
            theta: a n+1 by 1 vector
            J_vals: a #iterations by 1 vector
        """
        m = len(x)
        j_vals = []

        for i in range(iterations):
            error = (x * theta) - y
            for j in range(len(theta.flat)):
                koef = x[:, j]  # когда j = 0, то всегда равен 1 (массив единиц)
                theta.T[0, j] = theta.T[0, j] - (alpha / m) * np.sum(np.multiply(error, koef))
            j_vals.append(Helper.compute_cost(x, y, theta))

        return theta, j_vals

    @staticmethod
    def normalize(data, result_column_name):
        data1_clone = data

        scaler = preprocessing.MinMaxScaler()
        names = data.columns
        new_array = scaler.fit_transform(data)
        data1 = pd.DataFrame(new_array, columns=names)
        data1[result_column_name] = data1_clone[result_column_name]

        return data1

    # def __init__(self, name):
    #     self.__name = name
    #     # self.__hp = self.HP_INIT
