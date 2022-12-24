import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = np.genfromtxt('ex1data1.txt', delimiter=',')
population = np.array(data[:, 0]).reshape(-1, 1)
profit = np.array(data[:, 1])

model = LinearRegression()
model.fit(population, profit)

prediction1 = model.predict([[3.5]])
prediction2 = model.predict([[7]])
print('Предсказания: ', prediction1.item() * 10000, prediction2.item() * 10000)
population = np.append(population, 3.5).reshape(-1, 1)
profit = np.append(profit, prediction1)

f = model.predict(population)

plt.plot(population, profit, '.')
plt.plot(population, f)
plt.show()

# разница с нормал еквейшн и градиент дисент описана тут
# https://www.geeksforgeeks.org/difference-between-gradient-descent-and-normal-equation/