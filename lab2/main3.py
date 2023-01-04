import pandas as pd
import numpy as np
import scipy.optimize as opt

# скопируем основные функции из предыдущей задачи
def sigmoid(z):
    return 1 / (1 + np.exp(-1 * z))

def calc_probability(x, theta):
    x_theta = np.dot(x, theta)
    probability = sigmoid(x_theta)

    return probability

def compute_cost(theta, x, y, lam):
    m = len(x)
    probability = calc_probability(x, theta)
    one = y * (np.log(probability))
    two = ((1 - y) * (np.log(1 - probability)))
    slag = lam * np.sum(theta ** 2) / (2 * m)  # lam / 2 / len(x) * np.sum(np.square(theta))
    goal_func = -(1 / m) * (one + two).sum()

    return goal_func + slag

def compute_gradient(theta, x, y, lam):
    m = len(x)
    probability = calc_probability(x, theta)
    error = probability - y

    tmp = error[:, np.newaxis] * x  # ошибку умножаем для каождого признака (столбца)
    gradient = (1 / m) * np.sum(tmp, axis=0)  # считаем сумму по каждому столбцу (признаку)

    return gradient + (lam / m) * theta

def predict(theta, x):
    probability = calc_probability(x, theta)
    return [x >= 0.5 for x in probability]

def find_optimized_theta(XX, yy, lam1):
    zero_theta = np.zeros(len(XX.columns))  # нулевые тета

    return opt.fmin_tnc(
        func=compute_cost,
        x0=zero_theta,
        fprime=compute_gradient,
        args=(XX, yy, lam1),
        # disp=False
    )[0]

data1 = pd.read_csv('../ex2data3.txt', header=None)
data1.insert(0, 'Ones', 1)  # Добавим нулевую колонку с единичками
X = data1.iloc[:, :-1]  # все кроме последней (картинка 20*20 пикселей разернутая в строку из 400 знвчений (каждое значение - оттенок серого))
# Y = data1.iloc[:, -1]  # последний столбец (со значением цифры) вернет series
y = data1[400].values.astype(int)  # вернет массив

classifiers = np.zeros((10, len(X.columns)))

# создадим по бинарному классификатору для каждой цифры
for i in range(10):
    true_false_y = (y == i)  # для каждой цифры сформирует свой массив Y (true or false)
    one_zero_y = true_false_y.astype(int)  # сконвертирует его в 1 или 0
    theta_for_i = find_optimized_theta(X, one_zero_y, 0.5)
    classifiers[i, :] = theta_for_i

h = calc_probability(X, classifiers.T)  # получим вероятность принадлежности для каждой картинки из набора к той или иной цифре
h_argmax = np.argmax(h, axis=1)  # получим максимальную вероятность соответствия каждой картинки к той или иной цифре, то есть это и есть предсказания для каждой цифры зашифрованной в строке набора

print(h_argmax)
print(y)
print('Точность классификатора = {0}%' . format(np.mean(h_argmax == y) * 100))  # 95.1 при лямбда = 0.5
