import pandas as pd
import numpy as np
from Helper import Helper
from ChartBuilder import ChartBuilder
import scipy.optimize as opt

data1 = pd.read_csv('ex2data1.txt', names=['ex1', 'ex2', 'is_passed'])
passed_df = data1[data1['is_passed'] == 1]
not_passed_df = data1[data1['is_passed'] == 0]

x, y, theta = Helper.prepare_data(data1)

print('Целевая функция при нулевых тета:', Helper.compute_cost(theta, x, y))
print('Производная целевой функции при нулевых тета:', Helper.gradient_func(theta, x, y))

result = opt.fmin_tnc(
    func=Helper.compute_cost,
    x0=theta,
    fprime=Helper.gradient_func,
    args=(x, y),
    # approx_grad=True
)
theta_optimized = result[0]
print('Посчитанные тета: ', theta_optimized)
# Постройте границу решения.
ChartBuilder.build_base_chart(passed_df, not_passed_df, theta_optimized, x)

# pr2 = Helper.predict([1, 1, 1], [[1, 77, 55], [1, 7, 55]])
# print(pr2)

theta_min = np.matrix(result[0])
predictions = Helper.predict(theta_min, x)
combined_array = zip(predictions, y)
correct = [
    predicted_outcome == training_outcome
    for (predicted_outcome, training_outcome) in combined_array
]
coincides = sum(map(int, correct))
accuracy = coincides / (len(correct) / 100)
print('Точность классификатора = {0}%' . format(accuracy))

# Для студента, сдавшего экзамены на 45 и 85 баллов, оцените вероятность поступления.
ex1 = int(input("Введите оценку по 1 экзамену: "))  # введите 1ю оценку
ex2 = int(input("Введите оценку по 2 экзамену: "))  # и вторую
prediction1 = Helper.predict(theta_optimized, np.array([1, ex1, ex2]))

print('Поступит!' if prediction1[0][0] else 'Не поступит!')
