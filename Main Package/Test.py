import numpy as np


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


a = population_generator(100, 500000, 6)
print(a.size)
