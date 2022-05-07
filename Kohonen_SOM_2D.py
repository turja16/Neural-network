

import matplotlib.pyplot as plt

import numpy as np
import itertools


class SOM(object):

    def __init__(self, len, dim_feat):
        self.shape = (len, dim_feat)
        self.som = np.random.uniform(.9, 1.1, size=(len, dim_feat))

#np.zeros((h, w, dim_feat))

        # Training parameters
        self.L0 = 0.0
        self.lam = 0.0
        self.sigma0 = 0.0

        self.data = []

        self.hit_score = np.zeros((len))

    def train(self, data, L0, lam, sigma0, initializer=np.random.rand, frames=None):
        self.L0 = L0
        self.lam = lam
        self.sigma0 = sigma0

        self.som = initializer(*self.shape)

        self.data = data

        for t in itertools.count():
            if frames != None:
                frames.append(self.som.copy())

            if self.sigma(t) < 0.25:
                print("final t:", t)
                # print("quantization error:", self.quant_err())
                break
            if t% 50==0:
                plot_som_topology(som_square)
                test = 1

            i_data = np.random.choice(range(len(data)))

            bmu = self.find_bmu(data[i_data])
            self.hit_score[bmu] += 1

            self.update_som(bmu, data[i_data], t)

    def quant_err(self):
        bmu_dists = []
        for input_vector in self.data:
            bmu = self.find_bmu(input_vector)
            bmu_feat = self.som[bmu]
            bmu_dists.append(np.linalg.norm(input_vector - bmu_feat))
        return np.array(bmu_dists).mean()

    def find_bmu(self, input_vec):
        list_bmu = []
        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                dist = np.linalg.norm((input_vec - self.som[y, x]))
                list_bmu.append(((y, x), dist))
        list_bmu.sort(key=lambda x: x[1])
        return list_bmu[0][0]

    def update_som(self, bmu, input_vector, t):
        for y in range(self.shape[0]):
            for x in range(self.shape[1]):
                dist_to_bmu = np.linalg.norm((np.array(bmu) - np.array((y, x))))
                self.update_cell((y, x), dist_to_bmu, input_vector, t)

    def update_cell(self, cell, dist_to_bmu, input_vector, t):
        self.som[cell] += self.N(dist_to_bmu, t) * self.L(t) * (input_vector - self.som[cell])

    def update_bmu(self, bmu, input_vector, t):
        self.som[bmu] += self.L(t) * (input_vector - self.som[bmu])

    def L(self, t):
        return self.L0 * np.exp(-t / self.lam)

    def N(self, dist_to_bmu, t):
        curr_sigma = self.sigma(t)
        return np.exp(-(dist_to_bmu ** 2) / (2 * curr_sigma ** 2))

    def sigma(self, t):
        return self.sigma0 * np.exp(-t / self.lam)



    # Press the green button in the gutter to run the script.
def get_x_y(som):
    x = som.som[:,:,0].flatten()
    y = som.som[:,:,1].flatten()
    return x,y

def plot_2D_som(som):
    x,y = get_x_y(som)
    plt.scatter(x,y)
    plt.show()

def update_plot(i, frames, scat):
    scat.set_offsets(frames[i])
    return [scat]

def plot_som_topology(som):
    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 16,
            }
    k = 0
    plt.figure(figsize=(20,20))
    x,y = get_x_y(som)
    plt.scatter(x,y)
    for y in range(som.shape[0]):
        for x in range(som.shape[1]):
            x_f,y_f = som.som[y,x]
            plt.text(x_f, y_f, str(k), fontdict=font)
            k += 1
    plt.show()

def plot_input(data):
    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 16,
            }
    k = 0
    plt.figure(figsize=(20,20))
    for x in range(len(data)):
            x_f,y_f = data[x][0], data[x][1]
            plt.text(x_f, y_f, str(k), fontdict=font)
            k += 1
    plt.show()


if __name__ == '__main__':
    square_data = np.random.rand(1000, 2)
    plot_input(square_data)
    som_square = SOM(100, 2)
    frames_square = []
    plot_som_topology(som_square)
    som_square.train(square_data, L0=0.8, lam=1e2, sigma0=10, frames=frames_square)
    plot_som_topology(som_square)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/


