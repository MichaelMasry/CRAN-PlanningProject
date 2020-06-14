import numpy as np
import math as mth
import matplotlib.pyplot as plt


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


def distance_between_points(ran_x, ran_y):
    distance_helper = np.zeros((users, radio_r_h))
    for i in range(users):
        for j in range(radio_r_h):
            x1, x2 = ran_x[i], rrh_x[j]
            y1, y2 = ran_y[i], rrh_y[j]
            dist = np.math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            distance_helper[i][j] = dist
    return distance_helper  # returns a matrix  of users with distance 2 each RRH


def rbs_calculate(distance):
    if distance == 0:
        distance = 1
    # Power per Resource Block
    s = 5 - (10 * mth.log10(25)) + 30 + 30 - (20 * mth.log10(2350)) - (10 * 3 * mth.log10(distance)) - \
        ((6 * (distance / 5)) + 30) + 28
    # Rate per Resource Block
    n = (-174 + (10 * mth.log10(180000)))
    c = 180000 * mth.log2(1 + mth.pow(10, ((s - n) / 10)))
    # Number of RBs
    r = 2000000 / c
    return r


def total_rbs_check(connected_users, total_sys_capacity):
    total_rbs = connected_users.sum()
    if total_rbs <= total_sys_capacity:
        return True
    else:
        return False


# Data
users = 100
radio_r_h = 6
user_x = np.array([32, 16, 16, 17, 3, 28, 24, 37, 9, 37, 32, 5, 3, 33, 6, 23, 27, 5, 39, 29,
                   16, 11, 39, 39, 16, 13, 32, 4, 28, 11, 37, 15, 19, 19, 6, 14, 9, 19, 30, 26,
                   00, 38, 4, 0, 12, 31, 4, 17, 39, 35, 31, 11, 9, 36, 17, 35, 25, 9, 16, 10,
                   37, 19, 17, 3, 31, 15, 13, 3, 3, 30, 23, 18, 7, 21, 2, 34, 7, 15, 21, 36,
                   5, 23, 34, 20, 39, 3, 32, 20, 30, 0, 39, 19, 26, 31, 13, 24, 9, 25, 9, 0])
user_y = np.array([27, 22, 32, 10, 17, 9, 22, 2, 30, 12, 12, 24, 35, 16, 39, 16, 15, 16, 23, 23,
                   6, 8, 36, 25, 18, 16, 8, 14, 35, 19, 39, 24, 15, 31, 21, 8, 13, 18, 27, 3,
                   2, 10, 36, 30, 26, 17, 26, 19, 24, 19, 33, 28, 39, 21, 32, 23, 16, 31, 1, 14,
                   19, 17, 20, 15, 14, 2, 34, 25, 7, 24, 17, 38, 21, 14, 17, 1, 15, 1, 28, 4,
                   8, 12, 30, 27, 1, 4, 23, 31, 30, 21, 22, 6, 34, 36, 30, 16, 22, 13, 4, 18])
rrh_x = np.array([16, 15, 7, 27, 38, 9])
rrh_y = np.array([19, 38, 35, 21, 1, 0])
# Visualizing Data
plt.plot(user_x, user_y, 'gx')
plt.plot(rrh_x, rrh_y, 'ro')
plt.title('40x40 Users and RRHs Map')
plt.legend(('Users', 'RRHs'), loc=1)
plt.show()

# Initial Code
actual_distance = distance_between_points(user_x, user_y)
actual_distance = np.round(actual_distance, 2)
rbs_for_each_user = np.vectorize(rbs_calculate)(actual_distance)
rbs_for_each_user = np.round(rbs_for_each_user, 2)
print(rbs_for_each_user.sum())
print(total_rbs_check(rbs_for_each_user, 1000))

# Till Here We are Ready for both LOCAL SEARCH and GENETIC ALGORITHM
