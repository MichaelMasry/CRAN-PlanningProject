# Project
import numpy as np
import math as mth
import matplotlib.pyplot as plt


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
    distance_helper = np.zeros((users, radio_r_h))
    for i in range(users):
        for j in range(radio_r_h):
            x1, x2 = ran_x[i], rrh_x[j]
            y1, y2 = ran_y[i], rrh_y[j]
            dist = np.math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            distance_helper[i][j] = dist
    return distance_helper  # returns a matrix  of users with distance 2 each RRH


def mutate(parent):
    x = np.random.randint(0, np.size(parent), 1)
    out = np.copy(parent)
    if out[x] == 0:
        out[x] = 1
    else:
        out[x] = 0
    return out


def rbs_calculate(distance):
    # Fitness Function
    # Power per Resource Block
    s = 5 - (10 * mth.log10(25)) + 30 + 30 - (20 * mth.log10(2350)) - (10 * 3 * mth.log10(distance)) - ((
                6 * (distance / 5)) + 30) + 28
    # Rate per Resource Block
    n = (-174 + (10 * mth.log10(180000)))
    c = 180000 * mth.log2(1 + mth.pow(10, ((s - n) / 10)))
    # Number of RBs
    r = 2000000 / c
    return r


def rbs_matrix(distances_matrix):
    final_matrix = np.zeros((distances_matrix.shape[0], distances_matrix.shape[1]))
    for user in range(distances_matrix.shape[0]):
        for rrh in range(distances_matrix.shape[1]):
            final_matrix[user][rrh] = rbs_calculate(distances_matrix[user][rrh])
    return final_matrix


users = 100
radio_r_h = 6
dimension = 40
rrh_x = np.array([16, 15, 7, 27, 38, 9])
rrh_y = np.array([19, 38, 35, 21, 1, 0])
my_map, user_x, user_y = map_40_by_40()
actual_distance = distance_between_points(user_x, user_y)
actual_distance = np.round(actual_distance, 2)
rbs_for_each_user = rbs_matrix(actual_distance)
print(actual_distance)
print(rbs_for_each_user)
plt.plot(user_x, user_y, 'gx')
plt.plot(rrh_x, rrh_y, 'ro')
plt.title('Users and RRHs Map')
plt.legend(('Users', 'RRHs'), loc=1)
plt.show()
