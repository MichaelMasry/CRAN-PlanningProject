import itertools
import numpy as np





K = 2
N = 3
x = [np.reshape(np.array(i), (K, N)) for i in itertools.product([0, 1], repeat =K * N)]
print(len(x))


def total_rbs_check(connected_users, total_sys_capacity):
    total_rbs = connected_users.sum()
    if total_rbs <= total_sys_capacity:
        return True
    else:
        return False

