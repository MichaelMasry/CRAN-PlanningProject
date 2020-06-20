import numpy as np
from _collections import deque
import random
def swap_bits(bit):
    if bit == 0:
        return 1
    else:
        return 0

def shiftbit(array,direction):
    items=deque(array)
    items.rotate(direction)
    return items

def get_neighbor_combinations(k):
    old_shape = k.shape
    neighbor_combinations = np.empty(shape=(np.size(k),), dtype=object)
    count=0
    temp = k.copy()
    for i in range(np.size(k,0)):
        for j in range(np.size(k,1)-1):
            temp[i] = shiftbit(temp[i],1)
            print(temp[i])
            neighbor_combinations[count] = temp
            count += 1
    return neighbor_combinations


def mutate(parent):
    out = np.copy(parent)
    x = random.randint(0, np.size(out, 0)-1)
    ra = random.randint(0, np.size(out, 1)-1)
    z = np.zeros(np.size(out, 1))
    z[ra] = 1
    out[x] = z
    return out

a = np.arange(600).reshape((100, 6))
# print(a)
print(mutate(a))