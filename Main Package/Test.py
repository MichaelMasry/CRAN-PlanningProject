import itertools
import numpy as np


K = 2
N = 3
x = [np.reshape(np.array(i), (K, N)) for i in itertools.product([0, 1], repeat =K * N)]
print(len(x))
