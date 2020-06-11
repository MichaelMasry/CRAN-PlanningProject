# Project
import numpy as np
import math as mth


def crossover(part1, part2, position):
    child1 = np.concatenate((part1[0:position], part2[position:]))
    child2 = np.concatenate((part2[0:position], part1[position:]))
    return child1, child2


def mutate(parent):
    x = np.random.randint(0, np.size(parent), 1)
    out = np.copy(parent)
    if out[x] == 0:
        out[x] = 1
    else:
        out[x] = 0
    return out


def evaluate(chromosome):
    # Fitness Function
    print(chromosome)


# Power per Resource Block
S = 5 - 10*mth.log10(25) - 20*mth.log10(2350000000) - 10*3*mth.log10(''' Distance ''') + 28 - 6/5
# Rate per Resource Block
N = -174 + 10 * mth.log10(180000)
C = 180000 * mth.log2(1+mth.pow(10, (S/N/10)))
# Number of RBs
R = 2000000/C
# Create Population
print('Hello Guys')
print('200OK')
print ("keemolandz")