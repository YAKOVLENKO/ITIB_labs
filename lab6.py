import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

def get_DataTheatres(dataname):
    data = pd.read_csv(dataname, sep=';')
    # data = data.fillna(0)
    data['Capacity'] = data['MainHallCapacity'] + data['AdditionalHallCapacity'].fillna(0)
    data = data.drop(columns=['MainHallCapacity', 'AdditionalHallCapacity'])
    data = data[pd.notnull(data['Capacity'])]
    return data[['CommonName','Capacity']]


def get_DataParking(dataname):
    data = pd.read_csv(dataname, sep=';')
    data = data.sample(200)
    data = data.fillna(0)
    return data


class Center:
    def __init__(self, center):
        self.weight = center

    def getResult(self, input_value):
        return np.abs(self.weight - input_value)

    def changeWeight(self, norm, input_value):
        self.weight += norm * (input_value - self.weight)


class Net:
    def __init__(self, norm, center_weights):

        self.norm = norm
        self.centers = [Center(weight) for weight in center_weights]

    def learningProcess(self, data):
        current_winners = np.nan
        previous_winners = np.nan
        while previous_winners != current_winners:  # current_winners != previous_winners:
            previous_winners = current_winners
            current_winners = list()
            # random.shuffle(data)
            for value in data:
                centers_results = list()
                for center in self.centers:
                    centers_results.append(center.getResult(value))
                winner_index = np.argmin(centers_results)
                winner = self.centers[winner_index]
                winner.changeWeight(self.norm, value)
                current_winners.append(winner_index)
        # return {'values': data, 'centers': current_winners}

    def workingProcess(self, data):
        winners = list()
        for value in data:
            centers_results = list()
            for center in self.centers:
                centers_results.append(center.getResult(value))
            winner_index = np.argmin(centers_results)
            winner = self.centers[winner_index]
            winners.append(winner_index)
        return winners


def getPlot(x_array, y_array, centers_weights):
    plt.figure(figsize=(8, 6))
    for center, center_weight in enumerate(centers_weights):
        plt.scatter(center, center_weight, c='blue')
    plt.plot(x_array, y_array, '.', alpha=0.2, c='red')
    plt.title('Разбиение на кластеры')
    plt.xlabel('Кластер')
    plt.ylabel('Количество мест')
    plt.grid()
    plt.show()


def __main__():
    while 1:
        if input('Начать новый процесс? ("n" если нет)') == 'n':
            return
        data_type = input('Выберите данные: театры/паркинг ')
        if data_type == 'театры':
            data = get_DataTheatres('data.csv')
            data_list = list(data['Capacity'])
        elif data_type == 'паркинг':
            data = get_DataParking('data_parking.csv')
            data = data.sample(200)
            data_list = list(data['CarCapacity'])
        else:
            print('Неправильные данные!')
            continue
        centers = input('Введите центры (через пробел): ')
        centers = centers.split(' ')
        centers = np.array(centers)
        centers = centers.astype(float)
        norm = float(input('Введите n: '))
        net = Net(norm, centers)
        net.learningProcess(data_list)
        result = net.workingProcess(data_list)
        data['Cluster'] = result
        data.sort_values(['Cluster', 'Capacity']).to_csv('result.csv', sep=';', encoding='cp1251')
        try:
            print(data.sort_values(['Cluster', 'Capacity']).to_string())
        except:
            print(data.sort_values(['Cluster', 'CarCapacity']).to_string())
        print('\n')

        for index, center in enumerate(net.centers):
            print(('Вес кластера ' + str(index) + ': ' + str(np.round(center.weight, 3))))

        getPlot(result, data_list, [center.weight for center in net.centers])

__main__()





