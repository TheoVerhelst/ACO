from sys import argv
from parser import parse_instance
from evaluator import evaluate_tardiness
from random import shuffle
from optimizer import IG_RLS

if len(argv) != 2:
    print("error: need an instance file")
    print("usage:", argv[0], "FILE")
    exit(1)

proc_times, weights, deadlines = parse_instance(argv[1])

optimizer = IG_RLS(evaluate_tardiness, proc_times, weights, deadlines,
    d = 4, T = 1, weighted_temperature = False)
solution, evaluation = optimizer.optimize(n_iterations = 100)

print("solution :", solution)
print("tardiness:", evaluation)
