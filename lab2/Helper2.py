import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class Helper2:
    @staticmethod
    def prepare_data(data):
        # data.insert(0, 'Ones', 1)  # За нас это сделает полиномфичерс
        x = data.iloc[:, :-1]  # все кроме последней
        y = data.is_accepted.values  # вернет array
        # y = data.iloc[:, -1]  # последняя, вернет матрицу (pandas series)

        pf = PolynomialFeatures(degree=6, include_bias=True)
        x_poly = pf.fit_transform(x)

        n = x_poly.shape[1]  # 28 features
        theta = np.zeros(n)

        return x_poly, y, theta

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-1 * z))

    @staticmethod
    def calc_probability(x, theta):
        # matrix = np.matrix(theta).T
        x_theta = x * np.matrix(theta).T
        # x_theta = np.dot(x, theta)
        probability = Helper2.sigmoid(x_theta)

        return probability

    @staticmethod
    def compute_cost(theta, x, y, lam):
        m = len(x)
        probability = Helper2.calc_probability(x, theta)
        one = y * (np.log(probability))
        two = ((1 - y) * (np.log(1 - probability)))
        slag = lam * np.sum(theta ** 2) / (2 * m)
        # print(slag)
        goal_func = -(1 / m) * (one + two).sum()

        return goal_func + slag

    @staticmethod
    def compute_gradient(theta, x, y, lam):
        probability = Helper2.calc_probability(x, theta)
        error = probability - y

        m = len(x)
        n = len(theta)
        res = np.zeros(n)

        for j in range(n):
            koef = x[:, j]

            if j == 0:
                slag = 0
            else:
                slag = (lam / m) * np.sum(theta[j])

            res[j] = (1 / m) * np.sum(np.multiply(error, koef)) + slag

        return res
