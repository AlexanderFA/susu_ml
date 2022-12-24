import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class Helper2:
    @staticmethod
    def prepare_data(data):
        # data.insert(0, 'Ones', 1)  # За нас это сделает полиномфичерс
        x = data.iloc[:, :-1]  # все кроме последней
        y = data.iloc[:, -1]  # последняя

        x = np.matrix(x.values)
        y = np.matrix(y.values)
        # theta = np.matrix(np.zeros((n + 1, 1)))

        pf = PolynomialFeatures(degree=6, include_bias=True)
        x_poly = pf.fit_transform(x)

        n = x_poly.shape[1]  # 28 features
        theta = np.zeros(n)

        return x, y, x_poly, theta

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
        # one = np.multiply(y, np.log(probability))
        # two = np.multiply((1 - y), np.log(1 - probability))

        slag = lam * np.sum(theta ** 2) / (2 * m)

        goal_func = -(1 / m) * (one + two).sum()

        # test = theta[1:]
        return goal_func + slag
        # one = np.multiply(y, np.log(probability))
        # two = np.multiply((1 - y), np.log(1 - probability))
        #
        # slag = lam * np.sum(theta[1:] ** 2) / 2 * m
        #
        # return -(1 / m) * (one + two).sum() + slag  # J

    @staticmethod
    def compute_gradient(theta, x, y, lam):
        # probability = Helper2.calc_probability(x, theta)
        # error = probability - y
        #
        # m = len(x)
        # n = len(theta)
        # res = np.zeros(n)

        # gradient_in = (1 / m) * np.sum(np.multiply(error, x.transpose()))
        #
        # res[0] = gradient_in[0, 0]  # theta(0)
        # res[1:] = gradient_in[1:, 0] + (lam * theta / m)  # theta(j) ; j>0
        # res = res.flatten()
        #
        # return res
        # for j in range(n):
        #     koef = x[:, j]
        #
        #     if j == 0:
        #         slag = 0
        #     else:
        #         slag = (lam / m) * np.sum(theta[j])
        #
        #     res[j] = (1 / m) * np.sum(np.multiply(error, koef))
        #
        # return res

        # gradient = np.zeros(n)
        #
        # # gradient_in = np.dot(x.transpose(), error) / m
        # gradient_in = (1 / m) * np.sum(np.multiply(error, x))
        #
        # gradient[0] = gradient_in[0, 0]  # theta(0)
        # # gradient[1:] = gradient_in[1:, 0] + (lam * theta[1:, ] / m).reshape(n - 1, )  # theta(j) ; j>0
        # gradient[1:] = gradient_in[1:, 0] + (lam * theta[1:, ] / m)
        # # gradient = gradient.flatten()
        # #
        # return gradient
        n = len(x)

        gradient = np.zeros(n).reshape(n, )
        theta = theta.reshape(n, 1)
        infunc1 = sigmoid(lr_hypothesis(x, theta)) - y
        gradient_in = np.dot(x.transpose(), infunc1) / m
        gradient[0] = gradient_in[0, 0]  # theta(0)
        gradient[1:] = gradient_in[1:, 0] + (lambda_ * theta[1:, ] / m).reshape(n - 1, )  # theta(j) ; j>0
        gradient = gradient.flatten()
        return gradient

    # probability = Helper.calc_probability(x, theta)
    # error = probability - y
    #
    # m = len(x)
    # n = len(theta)
    # res = np.zeros(n)
    #
    # for j in range(n):
    #     koef = x[:, j]
    #     res[j] = (1 / m) * np.sum(np.multiply(error, koef))
    #
    # return res
