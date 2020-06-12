import itertools
import math as mth
import numpy as np


K = 2
N = 3
x = [np.reshape(np.array(i), (K, N)) for i in itertools.product([0, 1], repeat =K * N)]
print(len(x))


def rbs_calculate(distance):
    # Fitness Function
    # Power per Resource Block
    s = 5 - (10 * mth.log10(25)) - (20 * mth.log10(2350000)) - (10 * 3 * mth.log10(distance)) - (
                6 * (distance / 5) + 30) + 28
    # Rate per Resource Block
    n = (-174 + (10 * mth.log10(180000)))
    c = 180000 * mth.log2(1 + mth.pow(10, ((s / n) / 10)))
    # Number of RBs
    r = 2000000 / c
    return r

x = np.vectorize(rbs_calculate)([1,30,15])
print(x)
