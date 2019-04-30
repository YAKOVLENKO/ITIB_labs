import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def GetDerivative(out):
    return (1 - out ** 2) / 2


def GetActivation(net):
    out = (1 - np.exp(-net)) / (1 + np.exp(-net))
    return out, GetDerivative(out)


def FillData(data, filler):
    data['era'].append(filler[0])
    data['hide_weights1'].append(np.round(filler[1], 5))
    data['hide_weights2'].append(np.round(filler[2], 5))
    data['out_weights'].append(np.round(filler[3], 5))
    data['out'].append(np.round(filler[4], 5))
    data['error'].append(np.round(filler[5], 5))


def CreatePlot(data):

    plt.plot(data['era'], data['error'])
    plt.grid()
    plt.xlabel('Эпоха')
    plt.ylabel('Суммарная ошибкa')
    plt.title('График зависимости суммарной ошибки от эпохи')
    plt.show()


def LearningProcess(activation_func=GetActivation,  # функция активации
                    accuracy=10 ** -3,  # погрешность
                    n=1,  # норма обучения
                    input_arr=np.array([1, -3]),
                    true_value=0.1):  # входные данные

    E = np.inf
    hide_perc_weights_1 = np.zeros(2)#np.array([0.3, 0.4])  # np.zeros(2)
    hide_perc_weights_2 = np.zeros(2)#np.array([0.5, 0.6])  # np.zeros(2)
    y_perc_weights = np.zeros(3)#np.array([0.3, 0.4, 0.5])  # np.zeros(3)
    era = 0
    data = {'era': list(), 'hide_weights1': list(), 'hide_weights2': list(), 'out_weights': list(),
            'out': list(), 'error': list()}

    while E >= accuracy:
        # этап 1
        net_hide_perc_1 = np.dot(input_arr, hide_perc_weights_1)
        net_hide_perc_2 = np.dot(input_arr, hide_perc_weights_2)

        out_hide_perc_1, der_hide_perc_1 = activation_func(net_hide_perc_1)
        out_hide_perc_2, der_hide_perc_2 = activation_func(net_hide_perc_2)

        net_y_perc = np.dot(y_perc_weights, np.array([1, out_hide_perc_1, out_hide_perc_2]))
        out_y_perc, der_y_perc = activation_func(net_y_perc)

        E = np.sqrt((true_value - out_y_perc) ** 2)

        FillData(data, [era, hide_perc_weights_1, hide_perc_weights_2, y_perc_weights,
                        out_y_perc, E])

        # этап 2
        error_y_perc = der_y_perc * (true_value - out_y_perc)
        error_hide_perc1 = der_hide_perc_1 * error_y_perc * y_perc_weights[1]
        error_hide_perc2 = der_hide_perc_2 * error_y_perc * y_perc_weights[2]

        # этап 3
        delta_hide_perc1 = input_arr * error_hide_perc1 * n
        delta_hide_perc2 = input_arr * error_hide_perc2 * n
        delta_y_perc = n * np.array([1, out_hide_perc_1, out_hide_perc_2]) * error_y_perc

        hide_perc_weights_1 = hide_perc_weights_1 + delta_hide_perc1
        hide_perc_weights_2 = hide_perc_weights_2 + delta_hide_perc2
        y_perc_weights = y_perc_weights + delta_y_perc

        era += 1
    return data


def __main__():

    while 1:
        n = float(input('Введите норму обучения: '))
        accuracy = float(input('Введите погрешность: '))
        data = LearningProcess(n = n, accuracy = accuracy)
        print(pd.DataFrame(data).to_string())
        CreatePlot(data)

        new_proc = input('Начать заново? y/n ')
        if new_proc != 'y':
            return

__main__()
