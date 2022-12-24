import pandas as pd
from Helper import Helper
from ChartBuilder import ChartBuilder
import numpy as np
import matplotlib.pyplot as plt

# %matplotlib inline
data1 = pd.read_csv('ex1data1.txt', names=['population', 'profit'])

# the number of features.
n = len(data1.columns) - 1  # 1
x, y, theta = Helper.prepare_data(data1, n)
print('Целевая функция при нулевых тета:', Helper.compute_cost(x, y, theta))
# Initialize parameters for iterations and learning rate α.
# iterations = 1500
iterations = 10000
alpha = 0.01

theta, J_vals = Helper.gradient_descent(x, y, theta, alpha, iterations)
print('Рассчитанные тета:', theta[0, 0], theta[1, 0])
print('Целевая функция при рассчитанных тета:', Helper.compute_cost(x, y, theta))

ChartBuilder.build_prediction_chart(theta, data1)
ChartBuilder.build_j_vals_chart(J_vals)

# from mpl_toolkits.mplot3d import axes3d
#
# # Create meshgrid.
# xs = np.arange(-10, 10, 0.4)
# ys = np.arange(-2, 5, 0.14)
# xx, yy = np.meshgrid(xs, ys)
#
# # Initialize J values to a matrix of 0's.
# J_vals = np.zeros((xs.size, ys.size))
#
# # Fill out J values.
# for index, v in np.ndenumerate(J_vals):
#     J_vals[index] = Helper.compute_cost(x, y, [[xx[index]], [yy[index]]])
#
# # Create a set of subplots.
# fig = plt.figure(figsize=(16, 8))
# ax1 = fig.add_subplot(121, projection='3d')
# ax2 = fig.add_subplot(122)
#
# # Create surface plot.
# ax1.plot_surface(xx, yy, J_vals, alpha=0.5, cmap='jet')
# ax1.set_zlabel('Cost', fontsize=14)
# ax1.set_title('Surface plot of cost function')
#
# # Create contour plot.
# ax2.contour(xx, yy, J_vals, np.logspace(-2, 3, 20), cmap='jet')
# theta_f = list(theta.flat)
# ax2.plot(theta_f[0], theta_f[1], 'rx')
# ax2.set_title('Contour plot of cost function, showing minimum')
#
# # Create labels for both plots.
# for ax in fig.axes:
#     ax.set_xlabel(r'$\theta_0$ population', fontsize=14)
#     ax.set_ylabel(r'$\theta_1$ profit', fontsize=14)
# plt.show()
