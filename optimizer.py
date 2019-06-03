from functools import partial
from random import random
from math import exp
import numpy as np

class IG_RLS:
    """Iterated Greedy Random Local Search algorithm.

    Reference: Karabulut, Korhan. "A hybrid iterated greedy algorithm for total
    tardiness minimization in permutation flowshops." Computers & Industrial
    Engineering 98 (2016): 300-307.
    """

    def __init__(self, evaluator, proc_times, weights, deadlines, d = 4, T = 1, weighted_temperature = False):
        self.proc_times = proc_times
        self.weights = weights
        self.deadlines = deadlines
        self.evaluator = partial(evaluator, proc_times=proc_times, weights=weights, deadlines=deadlines)
        (self.n, self.m) = proc_times.shape
        self.d = d
        self.T = T
        self.weighted_temperature = weighted_temperature
        self.calculate_temperature()

    def optimize(self, n_iterations):
        current_sol, current_eval = self.local_search(self.NEH_edd())
        best_sol = current_sol
        best_eval = current_eval

        for i in range(n_iterations):
            new_sol, new_eval = self.destruction_construction(current_sol)
            new_sol, new_eval = self.local_search(new_sol)
            if new_eval < current_eval:
                current_sol = new_sol
                current_eval = new_eval
                if new_eval < best_eval:
                    best_sol = new_sol
                    best_eval = new_eval
            elif random() < exp((current_eval - new_eval) / self.temperature):
                current_sol = new_sol
                current_eval = new_eval

        return best_sol, best_eval

    def calculate_temperature(self):
        time_machines = [sum([self.proc_times[i, j] for i in range(self.n)]) for j in range(self.m)]
        time_jobs = [sum([self.proc_times[i, j] for j in range(self.m)]) for i in range(self.n)]
        makespan_lower_bound = max(max(time_machines), max(time_jobs))

        if self.weighted_temperature:
            self.temperature = (self.T / 10) * np.mean(np.dot(makespan_lower_bound - self.deadlines, self.weights))
        else:
            self.temperature = (self.T / 10) * np.mean(makespan_lower_bound - self.deadlines)

    def destruction_construction(self, current_sol):
        removed_idx = np.random.choice(current_sol.shape[0], self.d, replace = False)
        removed_jobs = current_sol[removed_idx]
        current_sol = np.delete(current_sol, removed_idx)
        current_eval = float("inf")

        for i in range(self.d):
            best_new_sol = None
            best_new_eval = float("inf")
            for j in range(len(current_sol)):
                new_sol = np.insert(current_sol, j, removed_jobs[i])
                new_eval = self.evaluator(new_sol)
                if new_eval < best_new_eval:
                    best_new_sol = new_sol
                    best_new_eval = new_eval

            current_sol = best_new_sol
            current_eval = best_new_eval

        return current_sol, current_eval

    def local_search(self, current_sol):
        k = 1
        current_eval = self.evaluator(current_sol)
        while k <= self.n:
            (pos_1, pos_2) = np.random.choice(current_sol.shape[0], 2, replace = False)
            if random() < 0.5:
                new_sol = self.numpy_move(current_sol, pos_1, pos_2)
            else:
                new_sol = self.numpy_swap(current_sol, pos_1, pos_2)
            new_eval = self.evaluator(new_sol) # We could optimize by evaluating only starting from min(pos_1, pos_2)

            if new_eval < current_eval:
                current_sol = new_sol
                current_eval = new_eval
                k = 1
            else:
                k += 1

        return current_sol, current_eval

    def numpy_move(self, array, from_pos, to_pos):
        val_from = array[from_pos]
        if from_pos > to_pos:
            res = np.delete(array, from_pos)
            res = np.insert(res, to_pos, val_from)
        else:
            res = np.insert(array, to_pos, val_from)
            res = np.delete(res, from_pos)
        return res

    def numpy_swap(self, array, from_pos, to_pos):
        res = np.copy(array)
        res[from_pos] = array[to_pos]
        res[to_pos] = array[from_pos]
        return res

    def NEH_edd(self):
        # Sort jobs by deadline
        jobs = list(range(self.n))
        jobs.sort(key = lambda i:self.deadlines[i])

        solution = np.empty(0, dtype = int)

        for i, job in enumerate(jobs):
            best_new_sol = None
            best_new_eval = float("inf")
            for pos in range(i + 1):
                new_sol = np.insert(solution, pos, job)
                new_eval = self.evaluator(new_sol)
                if new_eval < best_new_eval:
                    best_new_sol = new_sol
                    best_new_eval = new_eval
            solution = best_new_sol

        return solution


    # We will test this optimization as an experiment, and use the default
    # evaluator in the meantime
    """def construct_matrix(self, solution, pos):
        comp_times = np.empty((n, m))
        for i in range(self.n):
            for j in range(pos, self.m):
                prev_job = comp_times[solution[i - 1], j] if i > 0 else 0
                prev_machine = comp_times[solution[i], j - 1] if j > 0 else 0
                comp_times[solution[i], j] = max(prev_job, prev_machine) + proc_times[solution[i], j]
        return comp_times"""



class MMAS:
    """Max-Min Ant System for flow shop problem.

    Reference: Stützle, Thomas. "An ant approach to the flow shop problem."
    Proceedings of the 6th European Congress on Intelligent Techniques and Soft
    Computing. Vol. 3. 1998.
    """

    def __init__(self, evaluator, proc_times, weights, deadlines, p_0 = 0.9, pher_min = 0, pher_max = 1, pher_persistence = 0.75):
        self.proc_times = proc_times
        self.weights = weights
        self.deadlines = deadlines
        self.evaluator = partial(evaluator, proc_times=proc_times, weights=weights, deadlines=deadlines)
        (self.n, self.m) = proc_times.shape
        self.p_0 = p_0
        self.pher_min = pher_min
        self.pher_max = pher_max
        self.pher_persistence = pher_persistence

    def optimize(self, n_iterations, epsilon):
        pass
