import itertools
import numpy as np


K = 2
N = 3
x = [np.reshape(np.array(i), (K, N)) for i in itertools.product([0, 1], repeat=K * N)]
print(len(x))

x = np.array(['karim', 'michael', 'abdelhameed'])
v = np.array([10, 15, 8])
t = np.argsort(x)
x.sort()
print(x)
print(t)
print(v[t])
