# Project
import numpy as np
import math as mth
import random


users = 100
dimension = 40
rrh_x = np.array([16, 15, 7, 27, 38, 9])
rrh_y = np.array([19, 38, 35, 21, 1, 0])
gamma=np.zeros((users,6))

def crossover(part1, part2, position):
    child1 = np.concatenate((part1[0:position], part2[position:]))
    child2 = np.concatenate((part2[0:position], part1[position:]))
    return child1, child2


def map_40_by_40():
    my_map = np.zeros((dimension, dimension))
    ran_x = np.random.randint(0, dimension, users)
    ran_y = np.random.randint(0, dimension, users)
    for i in range(users):
        my_map[ran_x[i], ran_y[i]] = 1
    return my_map, ran_x, ran_y


def distancebetweenpoints(ran_x, ran_y):
    for i in range(users):
        distancehelper=np.zeros((users,6))
        for j in range(6):
            x1,x2=ran_x[i],rrh_x[j]
            y1,y2=ran_y[i],rrh_y[j]
            dist = np.math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            distancehelper[i][j]=dist
        # whichrrh= distancehelper.index(min(distancehelper))
        # gamma[i,whichrrh]=1

    return distancehelper        # returns a matrix  of users with distance 2 each RRH


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
S = 5 - 10*mth.log10(25) - 20*mth.log10(2350000000) - 10*3*mth.log10(5) + 28 - 6/5
# Rate per Resource Block
N = -174 + 10 * mth.log10(180000)
C = 180000 * mth.log2(1+mth.pow(10, (S/N/10)))
# Number of RBs
R = 2000000/C
# Create Population


# random number RRHs variable holders vector
randomh = np.empty((6,6))
ran_x = np.random.randint(0, dimension, 6)
ran_y = np.random.randint(0, dimension, 6)




