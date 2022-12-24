import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

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
    def build_training_3d(x, y):
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(x[:, 0], x[:, 1], y)

        plt.show()
