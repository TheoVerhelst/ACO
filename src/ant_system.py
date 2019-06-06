from time import time
from random import random, randrange
import numpy as np
from optimizer import Optimizer

class AntSystem(Optimizer):
    """Generic Ant System for the Flow Shop Problem. It is the base class
    for Ant System implementation. The main optimization procedure is
    implemented, and child classes have to implement initialize_trail,
    construct_from_trail and local_search. If no local search is used, simply
    return the solution unmodified in local_search.
    """

    def __init__(self, evaluator, partial_evaluator, proc_times, weights, deadlines, n_ants):
        super().__init__(evaluator, partial_evaluator, proc_times, weights, deadlines)
        self.n_ants = n_ants

    def optimize(self, max_time):
        """Main optimization procedure, common to all Ant System algorithms.
        Continues until max_time seconds elapsed.
        """
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
            self.log_convergence(time() - start_time, i, max(new_evals)) # Log best iteration evaluation
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

    def __init__(self, evaluator, partial_evaluator, proc_times, weights, deadlines, n_ants, p_0, trail_min_max_ratio, stagnation_threshold, trail_persistence):
        super().__init__(evaluator, partial_evaluator, proc_times, weights, deadlines, n_ants)
        self.p_0 = p_0
        self.trail_min_max_ratio = trail_min_max_ratio
        self.trail_persistence = trail_persistence
        self.best_eval = float("inf")
        self.best_eval_counter = 0
        self.stagnation_threshold = stagnation_threshold

    def local_search(self, current_sol):
        # no local search, just return the solution and its evaluation
        return current_sol, self.evaluator(current_sol)

    def initialize_trail(self):
        """Creates the trail at algorithm initialization."""
        current_sol, current_eval = self.local_search(self.NEH_edd())
        trail = np.zeros((self.n, self.n))
        return self.update_trail(trail, [current_sol], [current_eval])

    def update_trail(self, trail, solutions, evaluations):
        """Updates the pheromone trail from a list of ant solutions and
        their evaluations."""
        # Take the best solution for updating the trail
        best_idx = np.argmax(evaluations)
        best_sol = solutions[best_idx]
        best_eval = evaluations[best_idx]

        # Apply the evaporation
        trail = self.trail_persistence * trail
        # Add 1/best_eval to all trail cells i, j if job i is scheduled at j
        # in the best solution
        trail[best_sol, list(range(self.n))] += 1 / best_eval
        # Apply max-min clipping
        trail_max = 1 / ((1 - self.trail_persistence) * min(self.best_eval, best_eval))
        trail_min = trail_max / self.trail_min_max_ratio
        trail = np.maximum(trail_min, np.minimum(trail_max, trail))

        # If there is no solution improvement for self.stagnation_threshold
        # iteration, reset the trail
        if best_eval < self.best_eval:
            self.best_eval = best_eval
            self.best_eval_counter = 0
        else:
            self.best_eval_counter += 1
            if self.best_eval_counter > self.stagnation_threshold:
                self.best_eval = best_eval
                self.best_eval_counter = 0
                trail = self.initialize_trail()


        return trail

    def construct_from_trail(self, trail):
        """Creates a solution according to the current pheromone trail."""
        solution = np.empty(self.n, dtype = int)
        not_scheduled = list(range(self.n))

        for pos in range(self.n):
            # with probability p_0, use the job with highest pheromone
            if random() < self.p_0:
                chosen = np.argmax(trail[not_scheduled, pos])
            else:
                # Chose the job according to a discrete probability distribution
                proba_val = trail[not_scheduled, pos]
                chosen = np.random.multinomial(1, proba_val / np.sum(proba_val))
                chosen = np.nonzero(chosen)[0][0]

            solution[pos] = not_scheduled[chosen]
            del not_scheduled[chosen]

        assert(len(not_scheduled) == 0)

        return solution


class RankBasedAS(AntSystem):
    """Rank-based Ant System for the Flow Shop Problem. Inspired from the
    course material, with an optional local search.
    """

    def __init__(self, evaluator, partial_evaluator, proc_times, weights, deadlines, n_ants, number_top, trail_persistence, use_local_search = False):
        super().__init__(evaluator, partial_evaluator, proc_times, weights, deadlines, n_ants)
        self.number_top = number_top
        self.trail_persistence = trail_persistence
        self.use_local_search = use_local_search

    def local_search(self, current_sol):
        """Local search on the current solution, if self.local_search is set.
        The algorithm is the same as in Iterated Greedy Local Random Search:
        with probability 0.5, swap to random jobs. Otherwise, move a random
        job at a random position. Repeat until no improvement is made for
        self.n iterations.
        """
        if self.use_local_search:
            k = 0
            comp_times = np.empty((self.n, self.m))
            current_eval = self.partial_evaluator(current_sol, pos = 0, comp_times = comp_times)
            while k < self.n:
                (pos_1, pos_2) = np.random.choice(current_sol.shape[0], 2, replace = False)
                if random() < 0.5:
                    new_sol = self.numpy_move(current_sol, pos_1, pos_2)
                else:
                    new_sol = self.numpy_swap(current_sol, pos_1, pos_2)
                new_eval = self.partial_evaluator(new_sol, pos = min(pos_1, pos_2), comp_times = comp_times)

                if new_eval < current_eval:
                    current_sol = new_sol
                    current_eval = new_eval
                    k = 0
                else:
                    k += 1

            return current_sol, current_eval
        else:
            return current_sol, self.evaluator(current_sol)

    def initialize_trail(self):
        """Trail initialization, using the NEH-edd heuristic as the first
        solution.
        """
        current_sol, current_eval = self.local_search(self.NEH_edd())
        trail = np.zeros((self.n, self.n))
        return self.update_trail(trail, [current_sol], [current_eval])

    def update_trail(self, trail, solutions, evaluations):
        """Update the trail according to the rank-based approach: only the
        best self.number_top solution are retained, and their contribution to the
        trail update is proportional to their rank.
        """
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
        """Constructs a solution from the current pheromone trail. The discrete
        probability distribution is always used to choose the jobs to place. If
        the trail is zero for all remaining jobs, a uniform probability
        distribution is used.
        """
        solution = np.empty(self.n, dtype = int)
        not_scheduled = list(range(self.n))

        for pos in range(self.n):
            proba_val = trail[not_scheduled, pos]
            sum_trail = np.sum(proba_val)
            # If the trail is zero for all not yet scheduled jobs
            if sum_trail == 0:
                chosen = randrange(len(not_scheduled))
            else:
                # Choose according to a discrete probability density
                chosen = np.random.multinomial(1, proba_val / sum_trail)
                chosen = np.nonzero(chosen)[0][0]
            solution[pos] = not_scheduled[chosen]
            del not_scheduled[chosen]

        assert(len(not_scheduled) == 0)

        return solution
