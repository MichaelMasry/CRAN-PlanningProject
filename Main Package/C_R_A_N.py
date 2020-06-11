# Project
import numpy as np
import math as mth
import matplotlib.pyplot as plt


users = 100
dimension = 40
rrh_x = np.array([16, 15, 7, 27, 38, 9])
rrh_y = np.array([19, 38, 35, 21, 1, 0])


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


def distance_between_points(ran_x, ran_y):
    gamma = np.zeros((users, 6))
    for i in range(users):
        distance_helper = np.zeros(6)
        for j in range(6):
            x1,x2=ran_x[i], rrh_x[j]
            y1,y2=ran_y[i], rrh_y[j]
            dist = np.math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            distance_helper[j] = dist
        which_rrh = np.argmin(distance_helper)
        gamma[i, which_rrh] = 1
    return gamma


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
    # Power per Resource Block
    S = 5 - 10 * mth.log10(25) - 20 * mth.log10(2350000000) - 10 * 3 * mth.log10(5) + 28 - 6 / 5
    # Rate per Resource Block
    N = -174 + 10 * mth.log10(180000)
    C = 180000 * mth.log2(1 + mth.pow(10, (S / N / 10)))
    # Number of RBs
    R = 2000000 / C
    print(chromosome)


my_map, user_x, user_y = map_40_by_40()
gamma = distance_between_points(user_x, user_y)
print(gamma)
plt.plot(user_x, user_y, 'go')
plt.plot(rrh_x, rrh_y, 'rx')
plt.show()
