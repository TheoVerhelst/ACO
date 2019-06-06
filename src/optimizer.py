from random import random
from math import exp
from time import time
import numpy as np

class Optimizer:
    """Abstract class for all optimization algorithms used in this project. It
    solely contains attributes common to all implementation (the job weights,
    deadlines and processing times), along with a few helper functions.
    """

    def __init__(self, evaluator, partial_evaluator, proc_times, weights, deadlines):
        """Constructor. The evaluator and partial_evaluator arguments are
        function objects pointing to the solution evaluation functions. This
        class is responsible for providing an evaluation function interface
        without the need to provide the proc_times, weights and deadlines
        arguments. This is done with the evaluator and partial_evaluator
        methods, defined below.
         """
        self.proc_times = proc_times
        self.weights = weights
        self.deadlines = deadlines
        self.evaluator_func = evaluator
        self.partial_evaluator_func = partial_evaluator
        (self.n, self.m) = proc_times.shape
        # Data used to plot convergence graphs
        self.convergence_data = {"time": [], "iteration": [], "evaluation": []}

    def evaluator(self, solution):
        """Function called by child classes when they need to evaluate a
        solution. It in turn calls the function object given in the constructor.
        """
        return self.evaluator_func(solution, self.proc_times, self.weights,
                self.deadlines)

    def partial_evaluator(self, solution, pos, comp_times):
        """Function called by child classes when they need to evaluate a
        solution with the partial speed-up trick. It in turn calls the function
        object given in the constructor. """
        return self.partial_evaluator_func(solution, self.proc_times,
                self.weights, self.deadlines, pos, comp_times)

    def log_convergence(self, time, iteration, evaluation):
        """At each iteration, child classes can call this function to log the
        current state of convergence, allowing external caller to plot the
        convergence by acessing self.convergence_data.
        """
        self.convergence_data["time"].append(time)
        self.convergence_data["iteration"].append(iteration)
        self.convergence_data["evaluation"].append(evaluation)

    def NEH_edd(self):
        """Initial solution heuristic used by child classes, based on
        Kim, Y.D.: A new branch and bound algorithm for minimizing mean
        tardiness in two-machine flowshops. Computers & Operations Research
        20(4), 391–401 (1993).
        """
        # Sort jobs by deadline
        jobs = list(range(self.n))
        jobs.sort(key = lambda i:self.deadlines[i])

        solution = np.empty(0, dtype = int)

        for i, job in enumerate(jobs):
            best_new_sol = None
            best_new_eval = float("inf")
            # Find the best position where to insert the job, among the already
            # scheduled jobs
            for pos in range(i + 1):
                new_sol = np.insert(solution, pos, job)
                new_eval = self.evaluator(new_sol)
                if new_eval < best_new_eval:
                    best_new_sol = new_sol
                    best_new_eval = new_eval
            solution = best_new_sol

        return solution

    def numpy_move(self, array, from_pos, to_pos):
        """Moves the element at from_pos in array, and place it before to_pos."""
        val_from = array[from_pos]
        if from_pos > to_pos:
            res = np.delete(array, from_pos)
            res = np.insert(res, to_pos, val_from)
        else:
            res = np.insert(array, to_pos, val_from)
            res = np.delete(res, from_pos)
        return res

    def numpy_swap(self, array, from_pos, to_pos):
        """Swaps elements at from_pos and to_pos in array."""
        res = np.copy(array)
        res[from_pos] = array[to_pos]
        res[to_pos] = array[from_pos]
        return res


class IG_RLS(Optimizer):
    """Iterated Greedy Random Local Search algorithm.

    Reference: Karabulut, Korhan. "A hybrid iterated greedy algorithm for total
    tardiness minimization in permutation flowshops." Computers & Industrial
    Engineering 98 (2016): 300-307.
    """

    def __init__(self, evaluator, partial_evaluator, proc_times, weights, deadlines, d, weighted_temperature = False):
        """Constructor. d is the number of jobs to destruct-construct at each
        iteration, and weighted_temperature is a flag if the weighted version
        of the temperature formula should be used.
        """
        super().__init__(evaluator, partial_evaluator, proc_times, weights, deadlines)
        self.d = d
        self.weighted_temperature = weighted_temperature
        self.calculate_temperature()

    def optimize(self, max_time):
        """Main optimization procedure. Continues until max_time seconds elapsed."""
        start_time = time()
        current_sol, current_eval = self.local_search(self.NEH_edd())
        best_sol = current_sol
        best_eval = current_eval
        i = 0

        while (time() - start_time) < max_time:
            new_sol, new_eval = self.destruction_construction(current_sol)
            new_sol, new_eval = self.local_search(new_sol)
            # If the new solution is better than the current one, keep it
            if new_eval < current_eval:
                current_sol = new_sol
                current_eval = new_eval
                if new_eval < best_eval:
                    best_sol = new_sol
                    best_eval = new_eval
            # Keep the new solution even if it not better, with a small probability
            elif random() < exp((current_eval - new_eval) / self.temperature):
                current_sol = new_sol
                current_eval = new_eval
            self.log_convergence(time() - start_time, i, current_eval)
            #print("Iteration", i, "done")
            i += 1

        print("Solution: ", best_sol)
        print("Evaluation:", best_eval)
        print("Iterations: ", i)
        print("Elapsed time:", time() - start_time)

        return best_sol, best_eval

    def calculate_temperature(self):
        """Computes the temperature according to the formula given in the
        reference paper. The lower bound on the makespan comes from
        Taillard, E.: Benchmarks for basic scheduling problems. european journal
        of opera- tional research 64(2), 278–285 (1993).
        """
        time_machines = [sum([self.proc_times[i, j] for i in range(self.n)]) for j in range(self.m)]
        time_jobs = [sum([self.proc_times[i, j] for j in range(self.m)]) for i in range(self.n)]
        makespan_lower_bound = max(max(time_machines), max(time_jobs))

        if self.weighted_temperature:
            self.temperature = (1 / 10) * np.mean(np.dot(makespan_lower_bound - self.deadlines, self.weights))
        else:
            self.temperature = (1 / 10) * np.mean(makespan_lower_bound - self.deadlines)

    def destruction_construction(self, current_sol):
        """Removes randomly self.d jobs from the current solution, and places
        them back iteratively at the best location.
        """
        removed_idx = np.random.choice(current_sol.shape[0], self.d, replace = False)
        removed_jobs = current_sol[removed_idx]
        current_sol = np.delete(current_sol, removed_idx)
        current_eval = float("inf")

        # Iterate on all removed jobs to place back
        for i in range(self.d):
            best_new_sol = None
            best_new_eval = float("inf")
            # Initialize the completion time matrix, for optimized partial evaluations
            comp_times = np.empty((self.n, self.m))
            self.partial_evaluator(current_sol, pos = 0, comp_times = comp_times)

            # Find the best place to insert the removed job
            for j in range(len(current_sol)):
                new_sol = np.insert(current_sol, j, removed_jobs[i])
                new_eval = self.partial_evaluator(new_sol, pos = j, comp_times = comp_times)
                if new_eval < best_new_eval:
                    best_new_sol = new_sol
                    best_new_eval = new_eval

            current_sol = best_new_sol
            current_eval = best_new_eval

        return current_sol, current_eval

    def local_search(self, current_sol):
        """Local search on the current solution: with probability 0.5, swap to
        random jobs. Otherwise, move a random job at a random position. Repeat
        until no improvement is made for self.n iterations.
        """
        k = 1
        current_eval = float("inf")
        comp_times = np.empty((self.n, self.m))
        self.partial_evaluator(current_sol, pos = 0, comp_times = comp_times)
        while k <= self.n:
            (pos_1, pos_2) = np.random.choice(current_sol.shape[0], 2, replace = False)
            if random() < 0.5:
                new_sol = self.numpy_move(current_sol, pos_1, pos_2)
            else:
                new_sol = self.numpy_swap(current_sol, pos_1, pos_2)
            new_eval = self.partial_evaluator(new_sol, pos = min(pos_1, pos_2), comp_times = comp_times)

            if new_eval < current_eval:
                current_sol = new_sol
                current_eval = new_eval
                k = 1
            else:
                k += 1

        return current_sol, current_eval
