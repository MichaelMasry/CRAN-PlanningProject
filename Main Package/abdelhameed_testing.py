import numpy as np

random_indices = np.random.randint(0, 6, 2000)


def indices_to_gamma(indices):
    splitting_factor = indices.size / 100
    splitted_indices = np.split(indices, splitting_factor)
    final_gammas = []
    for array in splitted_indices:
        gamma = np.zeros((100, 6))
        i = 0
        for user in gamma:
            index_of_connected_node = array[i]
            user[index_of_connected_node] = 1
            i += 1
        final_gammas.append(gamma)
    return final_gammas


# test
print(indices_to_gamma(random_indices))
