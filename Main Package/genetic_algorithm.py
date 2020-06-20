import numpy as np
import math as mth
import random
from termcolor import colored
from _collections import deque
import matplotlib.pyplot as plt


def crossover(part1, part2, position, u, rrh):
    part1 = np.reshape(part1, (1, u * rrh))
    part2 = np.reshape(part2, (1, u * rrh))
    child1 = np.concatenate((part1[0, 0:position], part2[0, position:]))
    child2 = np.concatenate((part2[0, 0:position], part1[0, position:]))
    return np.reshape(child1, (u, rrh)), np.reshape(child2, (u, rrh))


# def mutate(parent, u, rrh):
#     out = np.copy(parent)
#     tem = np.reshape(out, (1, u * rrh))
#     x = random.randint(0, u * rrh - 1)
#     if tem[0, x] == 0:
#         tem[0, x] = 1
#     else:
#         tem[0, x] = 0
#     return np.reshape(tem, (u, rrh))
def mutate(parent):
    out = np.copy(parent)
    x = random.randint(0, np.size(out, 0)-1)
    ra = random.randint(0, np.size(out, 1)-1)
    z = np.zeros(np.size(out, 1))
    z[ra] = 1
    out[x] = z
    return out


def single_population_generator(num, rrh):
    tem = np.zeros([num, rrh], int)
    for each in range(num):
        temp_random = np.random.randint(0, rrh, num)
        tem[each, temp_random[each]] = 1
    return tem


def population_generator(u, num_population, rrh):
    tem = np.empty(shape=(num_population,), dtype=object)
    for each in range(num_population):
        tem[each] = single_population_generator(u, rrh)
    return tem


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


def clean_and_sort(pops, rbs, q, remote, eli):
    pos = np.argsort(rbs)
    rbs.sort()
    nation = pops[pos]
    e = np.where(rbs < q * remote + 1)
    nation = nation[e]
    _rbs = rbs[e]
    size = np.size(nation)
    if size > eli:
        nation = nation[0: int(eli)]
        _rbs = rbs[0: int(eli)]
        size = int(eli)
    return nation, _rbs, size


def refill(pops, new_guy):
    tem = np.empty(shape=(np.size(pops) + 1,), dtype=object)
    for each in range(np.size(pops)):
        tem[each] = pops[each]
    tem[-1] = new_guy
    return tem


def shift_bit(array, direction):
    items = deque(array)
    items.rotate(direction)
    return items


def get_neighbor_combinations(k):
    size = (np.size(k, 1) - 1) * np.size(k, 0)
    neighbor_combinations = np.empty(shape=(size,), dtype=object)
    count = 0
    for i in range(np.size(k, 0)):
        temp = k.copy()
        for j in range(np.size(k, 1)-1):
            temp[i] = shift_bit(temp[i], 1)
            neighbor_combinations[count] = temp.copy()
            count += 1
    return neighbor_combinations


def array_2_gamma(arr, u, rrh):
    gamma = np.empty(shape=(u, rrh))
    for i in range(np.size(arr, 0)):
        temp = np.zeros((1, rrh))
        temp[0][arr[i]] = 1
        gamma[i, :] = temp[0]
    return gamma


def distance_between_points(u, rrh, ran_x, ran_y, rrh_x, rrh_y):
    distance_helper = np.zeros((u, rrh))
    for each in range(u):
        for j in range(rrh):
            x1, x2 = ran_x[each], rrh_x[j]
            y1, y2 = ran_y[each], rrh_y[j]
            distance_helper[each][j] = np.math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance_helper  # returns a matrix  of users with distance 2 each RRH


def genetic_algorithm(number_of_users, user_x, user_y, remote_radio_h, rrh_x, rrh_y, Q, stopping_cond):
    pop_size = 1000
    mutation_percentage = 20
    elite = 0.4*pop_size
    print(colored("Starting GA Algorithm....", 'blue'))
    actual_distance = distance_between_points(number_of_users, remote_radio_h, user_x, user_y, rrh_x, rrh_y)
    actual_distance = np.round(actual_distance, 2)
    rbs_for_each_user = np.vectorize(rbs_calculate)(actual_distance)
    rbs_for_each_user = np.round(rbs_for_each_user, 2)
    print(colored("Generating Population....", 'blue'))
    population = population_generator(number_of_users, pop_size, remote_radio_h)
    best_rbs = np.zeros(pop_size)
    # Evaluating and Sorting
    for i in range(pop_size):
        # total system capacity = Q* remote_radio_h
        best_rbs[i] = evaluate_chromosome(population[i], rbs_for_each_user, Q, Q * remote_radio_h)
    population, best_rbs, pop_size = clean_and_sort(population, best_rbs, Q, remote_radio_h, elite)
    print(colored("Crossover and Mutation....", 'blue'))
    best_so_far = best_rbs[0]
    counter = 0
    ending = stopping_cond
    while counter < ending:

        # Crossover
        temp = pop_size
        for i in range(temp // 2):
            ra = random.randint(1, number_of_users * remote_radio_h)
            r1 = random.randint(1, pop_size - 1)
            r2 = random.randint(1, pop_size - 1)
            p1, p2 = crossover(population[r1], population[r2], ra, number_of_users, remote_radio_h)
            population = refill(population, p1)
            population = refill(population, p1)
            best_rbs = np.append(best_rbs, evaluate_chromosome(p1, rbs_for_each_user, Q, Q * remote_radio_h))
            best_rbs = np.append(best_rbs, evaluate_chromosome(p1, rbs_for_each_user, Q, Q * remote_radio_h))
            pop_size += 2

        # Mutation
        temp = pop_size
        for i in range(temp):
            ra = random.randint(0, 100)
            if ra < mutation_percentage:
                child = mutate(population[i])
                population = refill(population, child)
                best_rbs = np.append(best_rbs, evaluate_chromosome(child, rbs_for_each_user, Q, Q * remote_radio_h))
                pop_size += 1
        # Evaluate
        population, best_rbs, pop_size = clean_and_sort(population, best_rbs, Q, remote_radio_h, elite)
        if best_so_far > best_rbs[0]:
            best_so_far = best_rbs[0]
            counter = 0
        else:
            counter += 1
            print("Lap: " + str(counter))
            print("pop size: " + str(pop_size))
            print("min rbs found: " + str(np.round(best_so_far, 2)))
            print('-----------------------')
    print(colored("The Best Solution has Min Total RBs: ", 'green'))
    print(colored(np.round(best_so_far, 2), 'green'))
    return best_so_far, population[0]


def local_search(population, u, user_x, user_y, rrh, rrh_x, rrh_y, Q, num_population):
    actual_distance = distance_between_points(u, rrh, user_x, user_y, rrh_x, rrh_y)
    actual_distance = np.round(actual_distance, 2)
    rbs_for_each_user = np.vectorize(rbs_calculate)(actual_distance)
    rbs_for_each_user = np.round(rbs_for_each_user, 2)
    the_best_ever = 5000000
    the_best_gamma = None
    split_arrays = np.split(population, population.size / int(u))
    i = 0
    for array in split_arrays:
        gamma = array_2_gamma(array, u, rrh)
        rbs_assumed = evaluate_chromosome(gamma, rbs_for_each_user, Q, Q * rrh)
        if rbs_assumed < the_best_ever:
            the_best_ever = rbs_assumed
            the_best_gamma = gamma
        tenth = population.size//(10 * u)
        if i % tenth == 0:
            program_count = i // 10
            print(colored(str(program_count) + " % completed", 'red'))
        i += 1
    print(colored("100 % completed", 'green'))
    print(colored("The Best Solution has Min Total RBs: ", 'green'))
    print(colored(np.round(the_best_ever, 2), 'green'))
    return the_best_ever, the_best_gamma


# Test
_user_x = np.array([32, 16, 16, 17, 3, 28, 24, 37, 9, 37, 32, 5, 3, 33, 6, 23, 27, 5, 39, 29,
                   16, 11, 39, 39, 16, 13, 32, 4, 28, 11, 37, 15, 19, 19, 6, 14, 9, 19, 30, 26,
                   00, 38, 4, 0, 12, 31, 4, 17, 39, 35, 31, 11, 9, 36, 17, 35, 25, 9, 16, 10,
                   37, 19, 17, 3, 31, 15, 13, 3, 3, 30, 23, 18, 7, 21, 2, 34, 7, 15, 21, 36,
                   5, 23, 34, 20, 39, 3, 32, 20, 30, 0, 39, 19, 26, 31, 13, 24, 9, 25, 9, 0])
_user_y = np.array([27, 22, 32, 10, 17, 9, 22, 2, 30, 12, 12, 24, 35, 16, 39, 16, 15, 16, 23, 23,
                   6, 8, 36, 25, 18, 16, 8, 14, 35, 19, 39, 24, 15, 31, 21, 8, 13, 18, 27, 3,
                   2, 10, 36, 30, 26, 17, 26, 19, 24, 19, 33, 28, 39, 21, 32, 23, 16, 31, 1, 14,
                   19, 17, 20, 15, 14, 2, 34, 25, 7, 24, 17, 38, 21, 14, 17, 1, 15, 1, 28, 4,
                   8, 12, 30, 27, 1, 4, 23, 31, 30, 21, 22, 6, 34, 36, 30, 16, 22, 13, 4, 18])
_rrh_x = np.array([16, 15, 7, 27, 38, 9])
_rrh_y = np.array([19, 38, 35, 21, 1, 0])
ls_pop = 10
rrh = 6
users = 100  # Array 100,150,200,250,....
Q = 25
GA_stopping_cond = 8
x, gamma = genetic_algorithm(users, _user_x, _user_y, rrh, _rrh_x, _rrh_y, Q, GA_stopping_cond)
rand_arr = np.random.randint(0, rrh, 100000)
t, gamma = local_search(rand_arr, users, _user_x, _user_y, rrh, _rrh_x, _rrh_y, Q, ls_pop)
plt.plot(users, x, 'gx')
plt.plot(users, t, 'ro')
plt.title('40x40 Users and RRHs Map')
plt.legend(('Users', 'RRHs'), loc=1)
plt.show()
