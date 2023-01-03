import pandas as pd
from Helper2 import Helper2
from ChartBuilder2 import ChartBuilder2
import scipy.optimize as opt


data1 = pd.read_csv('ex2data2.txt', names=['test1', 'test2', 'is_accepted'])
accepted_df = data1[data1['is_accepted'] == 1]
rejected_df = data1[data1['is_accepted'] == 0]

x_poly, y, theta, pf = Helper2.prepare_data(data1)
ChartBuilder2.build_training_data_chart(accepted_df, rejected_df)
ChartBuilder2.build_training_3d(data1)

# print(pd.DataFrame(x_poly).head(5))
print('Целевая функция при нулевых тета:', Helper2.compute_cost(theta, x_poly, y, 0))
print('Производная целевой функции при нулевых тета:', Helper2.compute_gradient(theta, x_poly, y, 0))

def find_optimized_theta(lam1):
    return opt.fmin_tnc(
        func=Helper2.compute_cost,
        x0=theta,
        fprime=Helper2.compute_gradient,
        args=(x_poly, y, lam1),
        # accuracy=.9  # еще один способ бороться с переобучениемб при нулевой лямбда  точность 98%
    )


theta_optimized = find_optimized_theta(0)[0]
ChartBuilder2.build_boundary_plot(data1, accepted_df, rejected_df, theta_optimized, x_poly, pf)

# При различных лямбда:
theta_optimized = find_optimized_theta(100)[0]
ChartBuilder2.build_boundary_plot(data1, accepted_df, rejected_df, theta_optimized, x_poly, pf)
theta_optimized = find_optimized_theta(1)[0]
print('Посчитанные тета при lambda = 1: ', theta_optimized)
ChartBuilder2.build_boundary_plot(data1, accepted_df, rejected_df, theta_optimized, x_poly, pf)

predictions = Helper2.predict(theta_optimized, x_poly)
combined_array = zip(predictions, y)
correct = [
    predicted_outcome == training_outcome
    for (predicted_outcome, training_outcome) in combined_array
]
coincides = sum(map(int, correct))
percent = (len(correct) / 100)
accuracy = coincides / percent
print('Точность классификатора = {0}%' . format(accuracy))  # 96 при лямбда = 1
