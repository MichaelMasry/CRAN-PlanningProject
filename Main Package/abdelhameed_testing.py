import numpy as np


def array_2_gamma(arr, u, rrh):
    gamma = np.empty(shape=(u, rrh))
    for i in range(np.size(arr, 0)):
        temp = np.zeros((1, rrh))
        temp[0][arr[i]] = 1
        gamma[i, :] = temp[0]
    return gamma


def local_search(indices, factor, rrh):
    splitting_factor = indices.size / int(factor)
    split_arrays = np.split(indices, splitting_factor)
    for array in split_arrays:
        gamma = array_2_gamma(array, factor, rrh)

    return 0


print('Thanks')
x = np.array([2,1,0,0,2])
print(array_2_gamma(x, 5, 3))
