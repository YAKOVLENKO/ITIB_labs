# coding: utf-8
# Применение однослойной нейронной сети с линейной функцией активации для прогнозирования временных рядов

import numpy as np
import matplotlib.pyplot as plt


def GetCompPlot(data_curr, data_true):
    plt.figure(figsize=(8, 6))
    plt.plot(data_curr[1], data_curr[0], 'b', lw=1.5, color='red')
    plt.plot(data_true[1], data_true[0], 'b', lw=1.5, color='green')
    plt.axvline(x=1, linestyle='--', color='black', lw=1.5)
    plt.axvspan(-2, 1, facecolor='g', alpha=0.1)
    plt.axvspan(1, 4, facecolor='r', alpha=0.1)
    plt.title('Исходная функция и ее прогноз')
    plt.legend(['Прогноз', 'Исходная функция'])
    plt.xlim(-1.2, 3.2)
    plt.grid()
    plt.show()


def GetErrorEraPlot():

    x_array = list()
    y_array = list()

    window = int(input('Введите размер окна: '))
    norm = float(input('Введите норму обучения: '))

    for i in np.arange(1000, 12000, 1000):
        weights = LearningProcess(window, [-1, 1], i, n=norm)[0]
        data = GetPlotByWeights(window, [1, 3], [-1, 1], weights)
        error = np.round(GetError(data[1][0], data[0][0]), 3)
        x_array.append(i)
        y_array.append(error)

    plt.figure(figsize=(8, 6))
    plt.plot(x_array, y_array, 'b', lw=1.5, color='red')
    plt.plot(x_array, y_array, 'o', lw=0.5, color='darkred', markerfacecolor='white')
    plt.fill_between(x_array, y_array, color='r', alpha=0.1)
    plt.xticks(np.arange(0, 12000, 1000))
    plt.title('График зависимости погрешности от количества эпох')
    plt.xlabel('Количество эпох')
    plt.ylabel('Погрешность')
    plt.grid()
    plt.xlim(0, 12000)
    plt.show()

def GetErrorNormPlot():

    x_array = list()
    y_array = list()

    window = int(input('Введите размер окна: '))
    eras_count = int(input('Введите количество эпох: '))

    for i in np.arange(0.1, 1.1, 0.1):
        weights = LearningProcess(window, [-1, 1], eras_count, n=i)[0]
        data = GetPlotByWeights(window, [1, 3], [-1, 1], weights)
        error = np.round(GetError(data[1][0], data[0][0]), 3)
        x_array.append(i)
        y_array.append(error)

    plt.figure(figsize=(8, 6))
    plt.plot(x_array, y_array, 'b', lw=1.5, color='blue')
    plt.plot(x_array, y_array, 'o', lw=0.5, color='darkblue', markerfacecolor='white')
    plt.fill_between(x_array, y_array, color='b', alpha=0.1)
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.title('График зависимости погрешности от нормы обучения')
    plt.xlabel('Норма обучения')
    plt.ylabel('Погрешность')
    plt.grid()
    plt.xlim(0, 1.1)
    plt.show()


def GetErrorWindowPlot():

    x_array = list()
    y_array = list()

    eras_count = int(input('Введите количество эпох: '))
    norm = float(input('Введите норму обучения: '))

    for i in np.arange(1, 20, 1):
        weights = LearningProcess(i, [-1, 1], eras_count, n=norm)[0]
        data = GetPlotByWeights(i, [1, 3], [-1, 1], weights)
        error = np.round(GetError(data[1][0], data[0][0]), 3)
        x_array.append(i)
        y_array.append(error)

    plt.figure(figsize=(8, 6))
    plt.plot(x_array, y_array, 'b', lw=1.5, color='green')
    plt.plot(x_array, y_array, 'o', lw=0.5, color='darkgreen', markerfacecolor='white')
    plt.fill_between(x_array, y_array, color='g', alpha=0.1)
    plt.xticks(np.arange(0, 20, 1))
    plt.title('График зависимости погрешности от величины окна')
    plt.xlabel('Величина окна')
    plt.ylabel('Погрешность')
    plt.grid()
    plt.xlim(0, 20)
    plt.show()


def GetVariantFunction(x):
    # return 0.5 * np.sin(0.5 * x) - 0.5
    # return x ** 2 * np.sin(x)
    return (np.cos(x)) ** 2 - 0.5

def GetMatrix(window, true_y_array):
    matrix = [true_y_array[i:i + window] for i in range(0, len(true_y_array) - window)]
    return np.array(matrix)


def GetCurrentY(true_window_y, weights_array):
    return sum([weight * y for weight, y in zip(weights_array, true_window_y)])


def GetSumError(true_values, current_values):
    return np.sqrt(np.sum((true_values - current_values) ** 2))


def LearningProcess(window_width,
                    x_interval,
                    eras,
                    n=1,
                    variant_function=GetVariantFunction):

    weights_array = np.zeros(window_width)
    x_array = np.linspace(*x_interval, 20)
    true_y_array = np.array([variant_function(index) for index in x_array])
    true_y_matrix = GetMatrix(window_width, true_y_array)

    for era in range(0, eras):
        current_y_array = list(true_y_matrix[0])

        for index, column in enumerate(true_y_matrix):
            current_y = np.dot(column, weights_array)
            current_y_array.append(current_y)
            error = true_y_array[window_width + index] - current_y
            delta = n * error * column
            weights_array += delta
    common_error = np.sqrt(np.sum((true_y_array - current_y_array) ** 2))
    return weights_array, common_error


def GetPlotByWeights(window_width,
                     x_interval_new,
                     x_interval_old,
                     weights,
                     variant_function=GetVariantFunction):

    x_old_array, step = np.linspace(*x_interval_old, 20, retstep=True)
    y_old_array = np.array([variant_function(index) for index in x_old_array])
    x_array = np.arange(*x_interval_new, step)
    true_y_array = np.array([variant_function(index) for index in x_array])
    current_y_array = list(true_y_array[0:window_width]).copy()

    for index in range(window_width, len(x_array)):
        current_y_array.append(GetCurrentY(current_y_array[len(current_y_array) - window_width:],
                                           weights))

    return (np.append(y_old_array, current_y_array), np.append(x_old_array, x_array)), \
           (np.append(y_old_array, true_y_array), np.append(x_old_array, x_array))


def GetError(true_y_array, current_y_array):
    return np.sqrt(np.sum((true_y_array - current_y_array) ** 2))


def GeneralProcess():

    window = int(input('Input window length: '))
    eras_count = int(input('Input era\'s count: '))

    print('Start computing.')
    weights = LearningProcess(window, [-1, 0.5], eras_count, n=0.9)[0]
    data = GetPlotByWeights(window, [0.5, 4.5], [-1, 0.5], weights)
    print('Finish computing.')

    print('Weights: ' + str(np.round(weights, 3)))
    error = np.round(GetError(data[1][0], data[0][0]), 3)
    print('Error: ' + str(error))
    GetCompPlot(data[0], data[1])
    return


def __main__():

    while 1:

        print('Start new process.')
        mode = input('Введите тип: general/era/norm/window: ')

        if mode == 'general':
            GeneralProcess()
        elif mode == 'era':
            GetErrorEraPlot()
        elif mode == 'norm':
            GetErrorNormPlot()
        elif mode == 'window':
            GetErrorWindowPlot()
        answer = input('\nStart new process? y/n ')
        if answer == 'n':
            return


__main__()

