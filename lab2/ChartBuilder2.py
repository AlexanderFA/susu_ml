from matplotlib import pyplot as plt
import numpy as np
from Helper2 import Helper2
from sklearn.preprocessing import PolynomialFeatures


class ChartBuilder2:
    @staticmethod
    def build_training_data_chart(accepted_df, rejected_df):
        plt.plot(accepted_df['test1'], accepted_df['test2'], '*', label='Accepted')
        plt.plot(rejected_df['test1'], rejected_df['test2'], 'x', label='Rejected')
        plt.legend(loc=3)
        plt.grid()
        plt.xlabel('Microchip Test 1')
        plt.ylabel('Microchip Test 2')

        plt.show()

    @staticmethod
    def build_training_3d(data):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data.test1, data.test2, data.is_accepted)

        plt.show()

    @staticmethod
    def build_boundary_plot(data, accepted_df, rejected_df, theta, x, pf):
        plt.plot(accepted_df['test1'], accepted_df['test2'], '*', label='Accepted')
        plt.plot(rejected_df['test1'], rejected_df['test2'], 'x', label='Rejected')
        plt.legend(loc=3)

        # pf = PolynomialFeatures(degree=6)
        # pf.fit(data.iloc[:, :-1])
        def predictions(x1, x2, pf):
            return Helper2.calc_probability(
                pf.transform(
                    np.vstack((x1.ravel(), x2.ravel())).T
                ), theta
            )

        min_test1, max_test1 = data.test1.min(), data.test1.max()
        min_test2, max_test2 = data.test2.min(), data.test2.max()
        mesh_step = 0.1
        xs, ys = np.arange(min_test1 - .1, max_test1 + .1, mesh_step), np.arange(min_test2 - .1, max_test2 + .1, mesh_step)
        xv, yv = np.meshgrid(xs, ys)
        predictions_coords = predictions(xv, yv, pf)
        plt.contourf(xv, yv, predictions_coords.reshape(xv.shape), [0, .5, 1])

        plt.show()
