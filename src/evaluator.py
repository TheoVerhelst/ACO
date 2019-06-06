import numpy as np

def evaluate_tardiness(solution, proc_times, weights, deadlines):
    """Evaluation function of the total weighted tardiness of a solution."""
    (n, m) = proc_times.shape

    assert weights.shape == (n,)
    assert deadlines.shape == (n,)

    comp_times = np.empty((n, m))

    # iterating up to len(solution) instead of n allows to evaluate partial solutions
    for i in range(len(solution)):
        for j in range(m):
            prev_job = comp_times[solution[i - 1], j] if i > 0 else 0
            prev_machine = comp_times[solution[i], j - 1] if j > 0 else 0
            comp_times[solution[i], j] = max(prev_job, prev_machine) + proc_times[solution[i], j]

    tardiness = np.zeros(n, dtype = float)
    for i in solution:
        tardiness[i] = max(comp_times[i, m - 1] - deadlines[i], 0)

    return np.dot(weights, tardiness)

def evaluate_tardiness_partial(solution, proc_times, weights, deadlines, pos, comp_times):
    """Evaluation function of the total weighted tardiness of a solution, with
    computation starting only at job at position pos in the solution, and using
    the pre-computed comp_times matrix. This allows to speed up the evaluation
    when we know that the beginning of the matrix is already valid.
    """

    (n, m) = proc_times.shape

    assert weights.shape == (n,)
    assert deadlines.shape == (n,)

    # iterating up to len(solution) instead of n allows to evaluate partial solutions
    # and iterating from pos allows to evaluate the end of the solution, when
    # we already know the beginning
    for i in range(pos, len(solution)):
        for j in range(m):
            prev_job = comp_times[solution[i - 1], j] if i > 0 else 0
            prev_machine = comp_times[solution[i], j - 1] if j > 0 else 0
            comp_times[solution[i], j] = max(prev_job, prev_machine) + proc_times[solution[i], j]

    tardiness = np.zeros(n, dtype = float)
    for i in solution:
        tardiness[i] = max(comp_times[i, m - 1] - deadlines[i], 0)

    return np.dot(weights, tardiness)
