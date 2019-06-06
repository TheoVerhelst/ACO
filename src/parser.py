import numpy as np

def parse_instance(filename):
    """Reads a file representing an instance of the permutation flow shop
    problem, and returns the corresponding matrix of processing times, the
    weights and the deadlines.
    """
    # Read the file, depending if it is a path string or a pathlib.Path
    if isinstance(filename, str):
        with open(filename) as file:
            lines = file.readlines()
    else: # Path from pathlib
        with filename.open() as file:
            lines = file.readlines()

    n, m = lines[0].split()
    n = int(n)
    m = int(m)
    proc_times = np.empty((n, m))
    deadlines = np.empty(n)
    weights = np.empty(n)

    # Read the processing times
    for i in range(n):
        proc_times_line = lines[i + 1].split()
        for j in range(m):
            proc_times[i, j] = float(proc_times_line[2 * j + 1])

    # Read the deadlines and weights.
    for i in range(n):
        deadline_weight = lines[i + n + 2].split()
        deadlines[i] = float(deadline_weight[1])
        weights[i] = float(deadline_weight[3])

    return proc_times, weights, deadlines
