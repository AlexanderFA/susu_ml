import pandas as pd
from Helper2 import Helper2
from ChartBuilder2 import ChartBuilder2

data1 = pd.read_csv('ex2data2.txt', names=['test1', 'test2', 'is_accepted'])
accepted_df = data1[data1['is_accepted'] == 1]
rejected_df = data1[data1['is_accepted'] == 0]

x, y, x_poly, theta = Helper2.prepare_data(data1)
ChartBuilder2.build_training_data_chart(accepted_df, rejected_df)
ChartBuilder2.build_training_3d(x, y)

print(pd.DataFrame(x_poly).head(5))
lam = 1
print('Целевая функция при нулевых тета:', Helper2.compute_cost(theta, x_poly, y, lam))
print('Производная целевой функции при нулевых тета:', Helper2.compute_gradient(theta, x_poly, y, lam))
