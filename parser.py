import numpy as np

def parse_instance(filename):
    with open(filename) as file:
        lines = file.readlines()

    n, m = lines[0].split()
    n = int(n)
    m = int(m)
    proc_times = np.empty((n, m))
    deadlines = np.empty(n)
    weights = np.empty(n)

    for i in range(n):
        proc_times_line = lines[i + 1].split()
        for j in range(m):
            proc_times[i, j] = float(proc_times_line[2 * j + 1])

    for i in range(n):
        deadline_weight = lines[i + n + 2].split()
        deadlines[i] = float(deadline_weight[1])
        weights[i] = float(deadline_weight[3])

    return proc_times, weights, deadlines
