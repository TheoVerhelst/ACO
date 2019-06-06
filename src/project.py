from sys import argv
from parser import parse_instance
from evaluator import evaluate_tardiness
from random import shuffle, seed
import numpy as np
from optimizer import IG_RLS
from ant_system import MaxMinAS, RankBasedAS

if len(argv) != 2:
    print("error: need an instance file")
    print("usage:", argv[0], "FILE")
    exit(1)

proc_times, weights, deadlines = parse_instance(argv[1])

seed(42)
np.random.seed(42)

#optimizer = IG_RLS(evaluate_tardiness, proc_times, weights, deadlines, d = 4, T = 1, weighted_temperature = False)
#optimizer = MaxMinAS(evaluate_tardiness, proc_times, weights, deadlines, n_ants = 10, p_0 = 0.9, trail_min_max_ratio = 5, stagnation_threshold = 5, trail_persistence = 0.75)
optimizer = RankBasedAS(evaluate_tardiness, proc_times, weights, deadlines, n_ants = 10, number_top = 5, trail_persistence = 0.75)
solution, evaluation = optimizer.optimize(max_time = 30)
