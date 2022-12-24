import numpy as np
import matplotlib.pyplot as plt

class ChartBuilder:
    @staticmethod
    def build_j_vals_chart(j_vals):  # график изменения значения целевой функции от номера итерации
        xs = np.arange(0, len(j_vals))
        plt.plot(xs, j_vals)
        plt.xlabel('x - axis')
        plt.ylabel('y - axis')
        plt.title('My first graph!')
        plt.show()

    @staticmethod
    def build_prediction_chart(theta, data1):
        # Plot the Fit Line
        theta_f = list(theta.flat)
        xs = np.arange(0, 26)
        ys = theta_f[0] + theta_f[1] * xs

        plt.figure(figsize=(12, 8))
        plt.xlabel('Population of City in 10,000s')
        plt.ylabel('Profit in $10,000s')
        plt.grid()
        # plt.plot(data1.population, data1.profit, '.', label='Training Data')
        plt.plot(xs, ys, 'b-', label='Linear Regression: h(x) = %0.2f + %0.2fx' % (theta[0], theta[1]))
        plt.legend(loc=4)

        # Predict the profit for population of 35000 and 70000.
        prediction1 = (theta_f[0] + theta_f[1] * 3.5)
        prediction2 = (theta_f[0] + theta_f[1] * 7)
        print('Предсказания: ', [prediction1 * 10000, prediction2 * 10000])
        data1.loc[len(data1)] = [1, 3.5, prediction1]
        data1.loc[len(data1)] = [1, 7, prediction2]
        plt.plot(data1.population, data1.profit, '.', label='Training Data')
        plt.show()
