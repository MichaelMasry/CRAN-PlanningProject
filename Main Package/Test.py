import itertools
# import numpy as np
# a = np.array([1,2,3], [4,5,6])
# a = np.array([1,2,3], [4,5,6])

n = 20
aston = list(itertools.product([0, 1], repeat=n))
print(len(aston))
print(aston[15])
# K = 2
# N = 3
# x = [np.reshape(np.array(i), (K, N)) for i in itertools.product([0, 1], repeat=K * N)]
# print(x)
# x = np.array(['karim', 'michael', 'abdelhameed'])
# v = np.array([10, 15, 8])
# t = np.argsort(x)
# x.sort()
# print(x)
# print(t)
# print(v[t])
# def crossover(part1, part2, position):
#     part1 = np.reshape(part1, (1, 6))
#     part2 = np.reshape(part2, (1, 6))
#     child1 = np.concatenate((part1[0, 0:position], part2[0, position:]))
#     child2 = np.concatenate((part2[0, 0:position], part1[0, position:]))
#     return np.reshape(child1, (3,2)), np.reshape(child2, (3,2))
# a1 = np.array([5,6,7,11,12,13]).reshape((3,2))
# a2 = np.array([8,9,10,2,3,4]).reshape((3,2))
# c1, c2 = crossover(a1,a2,4)
# print(c1)
# print(c2)
