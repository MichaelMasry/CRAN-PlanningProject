import itertools
import numpy as np


def swap_bits(bit):
    if bit == 0:
        return 1
    else:
        return 0


def get_neighbor_combinations(k):
    old_shape = k.shape
    neighbor_combinations = np.empty(shape=(np.size(k),), dtype=object)
    for i in range(np.size(k)):
        temp = k.copy()
        temp = np.reshape(temp, (1, np.size(k)))
        temp[0, i] = swap_bits(temp[0, i])
        temp = np.reshape(temp, old_shape)
        neighbor_combinations[i] = temp
    return neighbor_combinations


a = np.array([[0, 1, 0], [1, 0, 0], [0, 1, 0]])
print(a)
r = get_neighbor_combinations(a)