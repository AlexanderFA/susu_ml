import pandas as pd
import numpy as np
from Helper import Helper
from ChartBuilder import ChartBuilder

data1 = pd.read_csv('ex1data2.txt', names=['size', 'bedrooms', 'price'])
# data1 = Helper.normalize(data1, 'price')

size_std, size_mean = data1['size'].std(), data1['size'].mean()
bedrooms_std, bedrooms_mean = data1['bedrooms'].std(), data1['bedrooms'].mean()
price_std, price_mean = data1['price'].std(), data1['price'].mean()
# data1['size'] = (data1['size'] - size_mean) / size_std
# data1['bedrooms'] = (data1['bedrooms'] - bedrooms_mean) / bedrooms_std
# data1['price'] = (data1['price'] - price_mean) / price_std
# x - mean = norm * std

# new_data = data1
# data1 = (new_data - new_data.mean(axis=0)) / new_data.std(axis=0)
# data1['price'] = new_data['price']

data1 = (data1 - data1.mean(axis=0)) / data1.std(axis=0)
# print(data1)
# quit()

n = len(data1.columns) - 1
# print(data1['size'])
# quit()
x, y, theta = Helper.prepare_data(data1, n)
print('Целевая функция при нулевых тета:', Helper.compute_cost(x, y, theta))
# Initialize parameters for iterations and learning rate α.
# iterations = 1500
iterations = 1000
alpha = 0.05

theta, J_vals = Helper.gradient_descent(x, y, theta, alpha, iterations)
print('Рассчитанные тета:', theta[0, 0], theta[1, 0], theta[2, 0])
print('Целевая функция при рассчитанных тета:', Helper.compute_cost(x, y, theta))

ChartBuilder.build_j_vals_chart(J_vals)

print('Среднее квадратичное и среднее математическое цены на квартиру:', [price_std, price_mean])
# real price = norm * std + mean
normalized_size = (3000 - size_mean) / size_std
normalized_bedrooms = (4 - bedrooms_mean) / bedrooms_std
# theta_list = np.array(theta)[:, 0]
th0, th1, th2 = list(theta.flat)
predicted_normalized_price = th0 + th1 * normalized_size + th2 * normalized_bedrooms
print(
    'Предполагаемая цена на дом площадью 3000 футов и числом комнат 4 =',
    round(predicted_normalized_price * price_std + price_mean, 2)
)
# print(data1.head())
# print(predicted_normalized_price)

