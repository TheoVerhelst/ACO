from time import time
from random import random
import numpy as np
from optimizer import Optimizer

class AntSystem(Optimizer):
    """Generic Ant System for the Flow Shop Problem."""

    def __init__(self, evaluator, proc_times, weights, deadlines, n_ants):
        super().__init__(evaluator, proc_times, weights, deadlines)
        self.n_ants = n_ants

    def optimize(self, max_time):
        start_time = time()
        trail = self.initialize_trail()
        print("Trail initialized")
        best_sol = None
        best_eval = float("inf")
        i = 0

        while (time() - start_time) < max_time:
            new_sols = []
            new_evals = []
            for k in range(self.n_ants):
                new_sol = self.construct_from_trail(trail)
                new_sol, new_eval = self.local_search(new_sol)
                if new_eval < best_eval:
                    best_sol = new_sol
                    best_eval = new_eval
                new_sols.append(new_sol)
                new_evals.append(new_eval)
            trail = self.update_trail(trail, new_sols, new_evals)
            #print("Iteration", i, "done")
            i += 1

        print("Solution: ", best_sol)
        print("Evaluation:", best_eval)
        print("Iterations: ", i)
        print("Elapsed time:", time() - start_time)

        return best_sol, best_eval


class MaxMinAS(AntSystem):
    """Max-Min Ant System for the Flow Shop Problem.

    Reference: Stützle, Thomas. "An ant approach to the flow shop problem."
    Proceedings of the 6th European Congress on Intelligent Techniques and Soft
    Computing. Vol. 3. 1998.
    """

    def __init__(self, evaluator, proc_times, weights, deadlines, n_ants, p_0, trail_min_max_ratio, stagnation_threshold, trail_persistence):
        super().__init__(evaluator, proc_times, weights, deadlines, n_ants)
        self.p_0 = p_0
        self.trail_min_max_ratio = trail_min_max_ratio
        self.trail_persistence = trail_persistence
        self.best_eval = float("inf")
        self.best_eval_counter = 0
        self.stagnation_threshold = stagnation_threshold

    def local_search(self, current_sol):
        # no local search
        return current_sol, self.evaluator(current_sol)
        best_sol = None
        comp_times = np.empty((self.n, self.m))
        # Fill the initial evaluation matrix
        best_eval = self.partial_evaluator(current_sol, pos = 0, comp_times = comp_times)

        for pos_1 in range(self.n):
            for pos_2 in range(self.n):
                if pos_1 != pos_2:
                    new_sol = self.numpy_move(current_sol, pos_1, pos_2)
                    new_eval = self.partial_evaluator(new_sol, pos = min(pos_1, pos_2), comp_times = comp_times)

                    if new_eval < best_eval:
                        best_sol = new_sol
                        best_eval = new_eval

        return best_sol, best_eval


    def initialize_trail(self):
        current_sol, current_eval = self.local_search(self.NEH_edd())
        trail = np.zeros((self.n, self.n))
        return self.update_trail(trail, [current_sol], [current_eval])

    def update_trail(self, trail, solutions, evaluations):
        best_idx = np.argmax(evaluations)
        best_sol = solutions[best_idx]
        best_eval = evaluations[best_idx]

        trail = self.trail_persistence * trail
        trail[best_sol, list(range(self.n))] += 1 / best_eval
        trail_max = 1 / ((1 - self.trail_persistence) * min(self.best_eval, best_eval))
        trail_min = trail_max / self.trail_min_max_ratio
        trail = np.maximum(trail_min, np.minimum(trail_max, trail))

        if best_eval < self.best_eval:
            self.best_eval = best_eval
            self.best_eval_counter = 0
        else:
            self.best_eval_counter += 1
            if self.best_eval_counter > self.stagnation_threshold:
                self.best_eval_counter = 0
                trail = self.initialize_trail()


        return trail

    def construct_from_trail(self, trail):
        solution = np.empty(self.n, dtype = int)
        not_scheduled = list(range(self.n))

        for pos in range(self.n):
            if random() < self.p_0:
                chosen = np.argmax(trail[not_scheduled, pos])
            else:
                proba_val = trail[not_scheduled, pos]
                chosen = np.random.multinomial(1, proba_val / np.sum(proba_val))
                chosen = np.nonzero(chosen)[0][0]

            solution[pos] = not_scheduled[chosen]
            del not_scheduled[chosen]

        assert(len(not_scheduled) == 0)

        return solution


class RankBasedAS(AntSystem):
    """Max-Min Ant System for the Flow Shop Problem.

    Reference: Stützle, Thomas. "An ant approach to the flow shop problem."
    Proceedings of the 6th European Congress on Intelligent Techniques and Soft
    Computing. Vol. 3. 1998.
    """

    def __init__(self, evaluator, proc_times, weights, deadlines, n_ants, number_top, trail_persistence):
        super().__init__(evaluator, proc_times, weights, deadlines, n_ants)
        self.number_top = number_top
        self.trail_persistence = trail_persistence

    def local_search(self, current_sol):
        # no local search
        return current_sol, self.evaluator(current_sol)

    def initialize_trail(self):
        current_sol, current_eval = self.local_search(self.NEH_edd())
        trail = np.zeros((self.n, self.n))
        return self.update_trail(trail, [current_sol], [current_eval])

    def update_trail(self, trail, solutions, evaluations):
        ranking = list(range(len(evaluations)))
        ranking.sort(key = lambda i:evaluations[i])
        ranking = ranking[:self.number_top]
        solutions = [solutions[r] for r in ranking]
        evaluations = [evaluations[r] for r in ranking]
        weights = [len(ranking) - i for i in range(len(ranking))]

        trail = self.trail_persistence * trail
        for solution, evaluation, weight in zip(solutions, evaluations, weights):
            trail[solution, list(range(self.n))] += weight / evaluation

        return trail

    def construct_from_trail(self, trail):
        solution = np.empty(self.n, dtype = int)
        not_scheduled = list(range(self.n))

        for pos in range(self.n):
            proba_val = trail[not_scheduled, pos]
            chosen = np.random.multinomial(1, proba_val / np.sum(proba_val))
            chosen = np.nonzero(chosen)[0][0]
            solution[pos] = not_scheduled[chosen]
            del not_scheduled[chosen]

        assert(len(not_scheduled) == 0)

        return solution
