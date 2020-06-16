import numpy as np
import math as mth
import matplotlib.pyplot as plt
import random
import sys
sys.path.append('...')
from Test.py import Table as plate


def int_random_generator_rows_gamma(num):
    return np.random.randint(0, 6, num)


def single_population_generator(num):
    tem = np.zeros([num, 6], int)
    for each in range(num):
        temp_random = int_random_generator_rows_gamma(num)
        tem[each, temp_random[each]] = 1
    return tem


def refill(pops, new_guy):
    tem = np.empty(shape=(np.size(pops)+1,), dtype=object)
    for each in range(np.size(pops)):
        tem[each] = pops[each]
    tem[-1] = new_guy
    return tem


# this function returns all the Gammas generated references in an array
def population_generator(u, num_population):
    tem = np.empty(shape=(num_population,), dtype=object)
    for each in range(num_population):
        tem[each] = single_population_generator(u)
    return tem


def crossover(part1, part2, position, u, rrh):
    part1 = np.reshape(part1, (1, u*rrh))
    part2 = np.reshape(part2, (1, u*rrh))
    child1 = np.concatenate((part1[0, 0:position], part2[0, position:]))
    child2 = np.concatenate((part2[0, 0:position], part1[0, position:]))
    return np.reshape(child1, (u, rrh)), np.reshape(child2, (u, rrh))


def mutate(parent, u, rrh):
    out = np.copy(parent)
    tem = np.reshape(out, (1, u*rrh))
    x = random.randint(0, u*rrh-1)
    if tem[0, x] == 0:
        tem[0, x] = 1
    else:
        tem[0, x] = 0
    return np.reshape(tem, (u, rrh))


def distance_between_points(ran_x, ran_y):
    distance_helper = np.zeros((users, remote_radio_h))
    for each in range(users):
        for j in range(remote_radio_h):
            x1, x2 = ran_x[each], rrh_x[j]
            y1, y2 = ran_y[each], rrh_y[j]
            distance_helper[each][j] = np.math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
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
    rate = 2000000 / c
    return rate


# Function checks that total RBs needed from the system check
def total_rbs_check(connected_users, total_sys_capacity):
    total_rbs = connected_users.sum()
    return total_rbs <= total_sys_capacity, total_rbs


# Function checks that One rrh for each user check
def constrain_one_rrh_user(gamma):
    tem = np.sum(gamma, axis=1)
    return np.all(tem == 1)


# Function checks that each rrh supply only q RBs
def check_2(res_block, q):
    tem = np.sum(res_block, axis=0)
    return np.all(tem <= q)


# Function checks that the inserted users RBs and Gamma satisfies all constraints
def evaluate_chromosome(gamma, rbs_matrix, q, total_sys_capacity):
    first_constraint = constrain_one_rrh_user(gamma)
    second_constraint = check_2(np.multiply(gamma, rbs_matrix), q)
    third_constraint, total_rbs = total_rbs_check(np.multiply(gamma, rbs_matrix), total_sys_capacity)
    if first_constraint & second_constraint & third_constraint:
        return total_rbs
    else:
        return total_rbs * 1000


def clean_and_sort(pops, rbs, q, remote):
    pos = np.argsort(rbs)
    rbs.sort()
    nation = pops[pos]
    e = np.where(rbs < q * remote + 1)
    nation = nation[e]
    _rbs = rbs[e]
    size = np.size(nation)
    return nation, _rbs, size


# Data
users = 100
remote_radio_h = 6
Q = 25
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
# plt.show()

# Initial Code
actual_distance = distance_between_points(user_x, user_y)
actual_distance = np.round(actual_distance, 2)
rbs_for_each_user = np.vectorize(rbs_calculate)(actual_distance)
rbs_for_each_user = np.round(rbs_for_each_user, 2)
# Till Here We are Ready for both LOCAL SEARCH and GENETIC ALGORITHM
# Genetic Algorithm
crossover_percentage = 80
pop_size = 1000
mutation_percentage = 20
elite = pop_size*0.4
# Creating Population
population = population_generator(users, pop_size)
best_rbs = np.zeros(pop_size)
# Evaluating and Sorting
for i in range(pop_size):
    best_rbs[i] = evaluate_chromosome(population[i], rbs_for_each_user, Q, Q*remote_radio_h)
population, best_rbs, pop_size = clean_and_sort(population, best_rbs, Q, remote_radio_h)

best_so_far = best_rbs[0]
counter = 0
ending = 6

while counter < ending:

    # Crossover
    if pop_size % 2 == 0:
        t = 0
    else:
        t = 1
    temp = pop_size
    for i in range((temp-t)//2):
        ra = random.randint(1, users*remote_radio_h)
        p1, p2 = crossover(population[i], population[-1-i], ra, users, remote_radio_h)
        population = refill(population, p1)
        population = refill(population, p1)
        best_rbs = np.append(best_rbs, evaluate_chromosome(p1, rbs_for_each_user, Q, Q*remote_radio_h))
        best_rbs = np.append(best_rbs, evaluate_chromosome(p1, rbs_for_each_user, Q, Q*remote_radio_h))
        pop_size += 2
        i += 1

    # Mutation
    temp = pop_size
    for i in range(temp):
        ra = random.randint(0, 100)
        if ra < mutation_percentage:
            child = mutate(population[i], users, remote_radio_h)
            population = refill(population, child)
            best_rbs = np.append(best_rbs, evaluate_chromosome(child, rbs_for_each_user, Q, Q*remote_radio_h))
            pop_size += 1
    # Evaluate
    population, best_rbs, pop_size = clean_and_sort(population, best_rbs, Q, remote_radio_h)
    print(pop_size)
    if best_so_far > best_rbs[0]:
        best_so_far = best_rbs[0]
        counter = 0
    else:
        counter += 1
        print(counter)
