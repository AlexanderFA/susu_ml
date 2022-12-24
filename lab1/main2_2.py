import pandas as pd
from Helper import Helper
from ChartBuilder import ChartBuilder

data1 = pd.read_csv('ex1data2.txt', names=['size', 'bedrooms', 'price'])
# data1 = Helper.normalize(data1, 'price')

size_std, size_mean = data1['size'].std(), data1['size'].mean()
bedrooms_std, bedrooms_mean = data1['bedrooms'].std(), data1['bedrooms'].mean()
price_std, price_mean = data1['price'].std(), data1['price'].mean()
data1['size'] = (data1['size'] - size_mean) / size_std
data1['bedrooms'] = (data1['bedrooms'] - bedrooms_mean) / bedrooms_std
# print(data1)

n = len(data1.columns) - 1
x, y, theta = Helper.prepare_data(data1, n)
print('Целевая функция при нулевых тета:', Helper.compute_cost(x, y, theta))
iterations = 1000
alpha = 0.05

theta, J_vals = Helper.gradient_descent(x, y, theta, alpha, iterations)
print('Рассчитанные тета:', theta[0, 0], theta[1, 0], theta[2, 0])
print('Целевая функция при рассчитанных тета:', Helper.compute_cost(x, y, theta))

ChartBuilder.build_j_vals_chart(J_vals)

normalized_size = (3000 - size_mean) / size_std
normalized_bedrooms = (4 - bedrooms_mean) / bedrooms_std
th0, th1, th2 = list(theta.flat)
predicted_normalized_price = th0 + th1 * normalized_size + th2 * normalized_bedrooms
print(
    'Предполагаемая цена на дом площадью 3000 футов и числом комнат 4 =',
    round(predicted_normalized_price, 2)
)
