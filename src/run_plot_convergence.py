"""This scripts runs all three algorithms on a problem instance, and plots the
convergence of the solutions of the three algorithms. The resulting plot is
saved in an image.
"""

from sys import argv
from parser import parse_instance
from evaluator import evaluate_tardiness, evaluate_tardiness_partial
from random import seed
import numpy as np
from optimizer import IG_RLS
from ant_system import MaxMinAS, RankBasedAS
import matplotlib.pyplot as plt
import pickle

if len(argv) != 3:
    print("error: need a path to instance file and to output file")
    print("usage: python3", argv[0], "INST_FILE OUT_FILE")
    exit(1)

inst_file = argv[1]
out_file = argv[2]

max_time = 30

parameter_space = {
    "IG_RLS": {
        "optimizer": IG_RLS,
        "parameters": {"d": 2, "weighted_temperature": False}
    },
    "MaxMinAS": {
        "optimizer": MaxMinAS,
        "parameters": {"n_ants": 10,  "p_0": 0.9, "trail_min_max_ratio": 5, "stagnation_threshold": 50, "trail_persistence": 0.75}
    },
    "RankBasedAS": {
        "optimizer": RankBasedAS,
        "parameters": {"n_ants":  10, "number_top": 3, "use_local_search": True, "trail_persistence": 0.75}
    }
}

results = {}

proc_times, weights, deadlines = parse_instance(inst_file)

for algo in parameter_space:
    print("***************")
    print("Algorithm", algo)
    print("***************")
    optimizer_class = parameter_space[algo]["optimizer"]
    parameters = parameter_space[algo]["parameters"]

    seed(42)
    np.random.seed(42)
    optimizer = optimizer_class(evaluate_tardiness, evaluate_tardiness_partial, proc_times, weights, deadlines, **parameters)
    solution, evaluation = optimizer.optimize(max_time = max_time)
    results[algo] = optimizer.convergence_data

fig, ax = plt.subplots(figsize=(5,4))

for algo in sorted(results.keys()):
    plt.plot(results[algo]["time"], results[algo]["evaluation"])

plt.legend(sorted(results.keys()))
ax.set_ylabel("Weighted tardiness (best of iteration)")
ax.set_xlabel("Time (s)")
fig.tight_layout()
fig.savefig(out_file)
