import numpy as np
from _collections import deque

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


a = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
# print(a)
# print("###########################################################################################")
r = get_neighbor_combinations(a)
print("###########################################################################################")
print(r)
# print(r)
# print(np.size(a,0),np.size(a,0))
# for i in range(np.size(a,0)):
#     for j in range(np.size(a,0)-1):
#         a[i]=shiftbit(a[i],1)
#
# print(a)