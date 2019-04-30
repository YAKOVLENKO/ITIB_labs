# coding: utf-8
# Исследование нейронных сетей с радиальными базисными функциями (RBF) на примере моделирования булевых выражений

import numpy as np
from itertools import product, combinations
import collections
import pandas as pd
import matplotlib.pyplot as plt


# In[114]:

#
def GetTrueResult(x_array):

    result = np.logical_not(x_array[:,1])
    result = np.logical_or(result, x_array[:,2])
    result = np.logical_and(result, x_array[:,4])
    result = np.logical_and(result, np.logical_not(x_array[:,3]))
    result = np.logical_not(result)

    return result


# def  GetTrueResult(x_array):
#     # int(not (x[0] or x[1]) or x[2] or x[3])
#     result = np.logical_or(x_array[:, 1], x_array[:, 2])
#     result = np.logical_not(result)
#     result = np.logical_or(result, x_array[:, 3])
#     result = np.logical_or(result, x_array[:, 4])
#     return result

# In[168]:


# def GetTrueResult(x_array):
#     result = np.logical_and(x_array[:, 1], x_array[:, 2])
#     result = np.logical_not(result)
#     result = np.logical_and(result, x_array[:, 3])
#     result = np.logical_and(result, x_array[:, 4])
#
#     return result


def GetActivation_3(net):
    out = 1 / (1 + np.exp(-net))

    if out >= 0.5:
        return 1, GetDerivative(3, net)

    return 0, GetDerivative(3, net)


def GetDerivative(func_number, net=0):
    if func_number == 1:
        return 1

    if func_number == 3:
        return np.exp(-net) / (1 + np.exp(-net)) ** 2

    if func_number == 2:
        return 1 / (2 * (np.abs(net) + 1) ** 2)

    if func_number == 4:
        return 1 - (1 / (2 * np.cos(net ** 2)))


def GetActivation_1(net):
    if net >= 0:
        return 1, GetDerivative(1)
    return 0, GetDerivative(1)


def GetActivation_4(net):
    out = (np.tanh(net) + 1) / 2

    if out >= 0.5:
        return 1, GetDerivative(4, net)

    return 0, GetDerivative(4, net)


def GetActivation_2(net):
    out = ((net / (1 + np.abs(net))) + 1) / 2

    if out >= 0.5:
        return 1, GetDerivative(2, net)

    return 0, GetDerivative(2, net)


def GetMidLayerInfo(true_function):
    sets = np.array(list(product([0, 1], repeat=5))[16:])
    true_values = true_function(sets)
    count_true_values = collections.Counter(true_values)
    return sets[np.where(true_values == min(count_true_values, key=count_true_values.get))[0]]


def GetPhi(x_values, neuron_array):
    return np.exp(-np.sum((x_values - neuron_array) ** 2))


def GetNet(weights,
           x_values,
           center_neurons_arrays):
    net = 0

    for neuron_array, weight in zip(center_neurons_arrays, weights[1:]):
        phi = GetPhi(x_values, neuron_array)
        net += weight * phi
    net += weights[0]

    return net


def FillData(data, era, weights, out, error):
    data['Номер эпохи'].append(era)
    data['Вектор весов w'].append(np.round(weights, 3))
    data['Выходной вектор y'].append(out)
    data['Суммарная ошибка Е'].append(error)


def LearningProcess(array_for_learning,
                    activation_function,
                    true_function=GetTrueResult,
                    n=0.3, eralim = False):
    neuron_centers = GetMidLayerInfo(true_function)
    weights = np.zeros(len(neuron_centers) + 1)
    hamming_distance = GetHammingDistance(true_function, weights, activation_function)
    eras_count = 0
    data = {'Номер эпохи': list(), 'Вектор весов w': list(), 'Выходной вектор y': list(),
            'Суммарная ошибка Е': list()}

    while hamming_distance != 0:  # era
        FillData(data, eras_count, weights,
                 GetCurrentResult(weights, activation_function, array_for_learning), hamming_distance)
        eras_count += 1

        for set_i in array_for_learning:  # step
            net = GetNet(weights, set_i, neuron_centers)
            y, derivative = activation_function(net)
            error = true_function(np.array([set_i]))[0] - y
            phi_array = [1] + [GetPhi(set_i, neuro_i) for neuro_i in neuron_centers]
            delta = n * error * derivative * np.array(phi_array)
            weights += delta

        hamming_distance = GetHammingDistance(true_function, weights, activation_function)

        if eralim - 1 == 0:
            return np.round(weights, 3), data, False
        if eralim:
            eralim -= 1

    FillData(data, eras_count, weights,
             GetCurrentResult(weights, activation_function, array_for_learning), hamming_distance)
    return np.round(weights, 3), data, True


def GetHammingDistance(true_function,
                       weights,
                       activation_function):
    sets = np.array(list(product([0, 1], repeat=5))[16:])
    true_values = true_function(sets)
    curr_values = [WorkingProcess(set_i, weights, activation_function, true_function) for set_i in sets]
    curr_values = np.array(curr_values)
    return np.count_nonzero(curr_values != true_values)


def FindLessProcess(activation_func, lim, x_array):
    sample = list()
    sample_data = None

    for check_length in range(len(x_array), 0, -1):
        flag = False
        print('Проверка набора длины ' + str(check_length))
        for bin_s in list(combinations(x_array, check_length)):
            bin_s = np.array(bin_s)
            weights, data, flag = LearningProcess(bin_s, activation_func, eralim=lim)

            if flag:
                sample = bin_s
                sample_data = data
                break

        if not flag:
            return sample, sample_data

    return sample, sample_data


def GetCurrentResult(weights,
                     activation_function,
                     learning_array):
    result = list()
    for sample in learning_array:
        result.append(WorkingProcess(sample, weights, activation_function))
    return result


def WorkingProcess(x_values,
                   weights,
                   activation_function,
                   true_function=GetTrueResult):
    neuron_centers = GetMidLayerInfo(true_function)
    net = GetNet(weights, x_values, neuron_centers)
    return activation_function(net)[0]


def GetPlot(data):
    plt.figure(figsize=(8, 6))
    plt.plot(data['Номер эпохи'], data['Суммарная ошибка Е'], 'b', lw=1.5, color='red')
    plt.plot(data['Номер эпохи'], data['Суммарная ошибка Е'], 'o', lw=0.5, color='darkred', markerfacecolor='white')
    plt.fill_between(data['Номер эпохи'], data['Суммарная ошибка Е'], color='r', alpha=0.1)
    # plt.xticks(np.arange(0, 12000, 1000))
    plt.title('График суммарной ошибки НС по эпохам обучения')
    plt.xlabel('Номер эпохи')
    plt.ylabel('Суммарная ошибка Е')
    plt.grid()
    plt.show()


def __main__():
    while 1:
        activation_number = input('\nВыберите функцию активации 1/2/3/4: ')

        if activation_number == '1':
            function_activation = GetActivation_1
        elif activation_number == '3':
            function_activation = GetActivation_3
        elif activation_number == '2':
            function_activation = GetActivation_2
        elif activation_number == '4':
            function_activation = GetActivation_4
        else:
            print('Ошибка.')
            continue

        print('\nОбучение на полном наборе: ')
        data = LearningProcess(np.array(list(product([0, 1], repeat=5))[16:]),
                function_activation)[1]
        print(pd.DataFrame(data).to_string())
        GetPlot(data)

        print('\nПоиск минимального набора: ')
        sample, sample_data = FindLessProcess(function_activation,
                                              50, np.array(list(product([0, 1], repeat=5))[16:]))
        print('\nМинимальный набор:\n' + str(sample))
        print(pd.DataFrame(sample_data).to_string())
        GetPlot(sample_data)


__main__()
