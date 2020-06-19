import numpy as np
from _collections import deque


def shift_bit(array, direction):
    items = deque(array)
    items.rotate(direction)
    return items


def get_neighbor_combinations(k):
    size = (np.size(k, 1) - 1) * np.size(k, 0)
    neighbor_combinations = np.empty(shape=(size,), dtype=object)
    count = 0
    for i in range(np.size(k, 0)):
        temp = k.copy()
        for j in range(np.size(k, 1)-1):
            temp[i] = shift_bit(temp[i], 1)
            neighbor_combinations[count] = temp.copy()
            count += 1
    return neighbor_combinations


aaa = np.arange(600).reshape(100, 6)
a = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
r = get_neighbor_combinations(aaa)
