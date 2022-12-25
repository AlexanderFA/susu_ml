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

        return x_poly, y, theta, pf

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-1 * z))

    @staticmethod
    def calc_probability(x, theta):
        # x_theta = x * np.matrix(theta).T
        x_theta = np.dot(x, theta)
        probability = Helper2.sigmoid(x_theta)

        return probability

    @staticmethod
    def compute_cost(theta, x, y, lam):
        m = len(x)
        probability = Helper2.calc_probability(x, theta)
        one = y * (np.log(probability))
        two = ((1 - y) * (np.log(1 - probability)))
        slag = lam * np.sum(theta ** 2) / (2 * m)  # lam / 2 / len(x) * np.sum(np.square(theta))
        # print(slag)
        goal_func = -(1 / m) * (one + two).sum()

        return goal_func + slag

    @staticmethod
    def compute_gradient(theta, x, y, lam):
        m = len(x)
        probability = Helper2.calc_probability(x, theta)
        error = probability - y

        tmp = error[:, np.newaxis] * x  # ошибку умножаем для каождого признака (столбца)
        gradient = (1 / m) * np.sum(tmp, axis=0)  # считаем сумму по каждому столбцу (признаку)

        return gradient + (lam / m) * theta

    @staticmethod
    def predict(theta, x):
        probability = Helper2.calc_probability(x, theta)
        # print(probability)
        return [x >= 0.5 for x in probability]
