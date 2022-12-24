import numpy as np
import matplotlib.pyplot as plt

class ChartBuilder:
    @staticmethod
    def build_base_chart(passed_df, not_passed_df, theta, x):
        plt.plot(passed_df['ex1'], passed_df['ex2'], '*', label='Admitted')
        plt.plot(not_passed_df['ex1'], not_passed_df['ex2'], 'x', label='Not Admitted')
        plt.legend(loc=3)

        theta_f = list(theta.flat)

        min_x = (min(x[:, 1]) - 0).item()
        max_x = (max(x[:, 1]) + 0).item()
        plot_x = np.array([min_x, max_x])
        # Calculate the decision boundary line
        plot_y = (-1 / theta_f[2]) * (theta_f[1] * plot_x + theta_f[0])
        # h(x) = g(𝜃0 + 𝜃1x1 + 𝜃2x2) = g(−3 + x1 + x2).
        # Plot, and adjust axes for better viewing
        plt.plot(plot_x, plot_y)

        plt.show()
