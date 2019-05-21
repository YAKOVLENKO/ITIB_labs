import numpy as np
import pandas as pd

class NeuroNet:
    def __init__(self, image_array):
        self.weights = self.get_weight_matrix(image_array.T)
        pd.DataFrame(self.weights).to_csv('weight_matrix.csv', sep=';', encoding='cp1251')

    def activate(self, curr_net, last_net):
        if curr_net > 0:
            return 1
        elif curr_net < 0:
            return -1
        else:
            return last_net

    def get_weight_matrix(self, images):
        matrix = np.array([np.sum(image * images, axis=1) for image in images])
        np.fill_diagonal(matrix, 0)
        return matrix

    def get_net(self, input_signal, row_index):
        return np.dot(input_signal, self.weights[row_index])

    # асинхронный режим
    def find_image(self, input_signal):
        signal = input_signal.copy()
        net_list = [signal]
        loop = 0
        while len(net_list) == 1 or flag == 1:
            flag = 0
            loop += 1
            current_net = list()
            for index, y in enumerate(signal):
                net = self.get_net(signal, index)
                current_net.append(net)
                new_y = self.activate(net, net_list[loop - 1][index])
                signal[index] = new_y
            net_list.append(current_net)
            if net_list[-1] != net_list[-2]:
                flag = 1
        return signal