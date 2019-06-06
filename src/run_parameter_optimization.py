"""This script evaluates the performance of each three algorithms on a set of
parameter assignments, and save the results in three CSV files.
"""

from sys import argv
import csv
from pathlib import Path
from os.path import join
from parser import parse_instance
from evaluator import evaluate_tardiness, evaluate_tardiness_partial
from random import shuffle, seed
import numpy as np
from optimizer import IG_RLS
from ant_system import MaxMinAS, RankBasedAS

def param_to_string(param_dict):
    """Converts a dictionary to a string, where keys and values are separated
    by "_" characters. The keys are sorted to keep a deterministic order.
    """
    return "_".join(key + "_" + str(val) for (key, val) in sorted(param_dict.items()))

if len(argv) != 3:
    print("error: need a path to instance folder and to output folder")
    print("usage: python3", argv[0], "INST_PATH OUT_PATH")
    exit(1)

inst_path = argv[1]
out_path = argv[2]

max_time = 30
"""
parameter_space = {
    "IG_RLS": {
        "optimizer": IG_RLS,
        "parameters": [
            {"d": 2, "weighted_temperature": False},
            {"d": 3, "weighted_temperature": False},
            {"d": 4, "weighted_temperature": False},
            {"d": 2, "weighted_temperature": True},
            {"d": 3, "weighted_temperature": True},
            {"d": 4, "weighted_temperature": True}
        ]
    },
    "MaxMinAS": {
        "optimizer": MaxMinAS,
        "parameters": [
            {"n_ants": 3,  "p_0": 0.9, "trail_min_max_ratio": 5, "stagnation_threshold": 5, "trail_persistence": 0.75},
            {"n_ants": 3,  "p_0": 0.5, "trail_min_max_ratio": 5, "stagnation_threshold": 5, "trail_persistence": 0.5},
            {"n_ants": 3,  "p_0": 0.2, "trail_min_max_ratio": 5, "stagnation_threshold": 5, "trail_persistence": 0.25},
            {"n_ants": 10, "p_0": 0.9, "trail_min_max_ratio": 5, "stagnation_threshold": 5, "trail_persistence": 0.75},
            {"n_ants": 10, "p_0": 0.5, "trail_min_max_ratio": 5, "stagnation_threshold": 5, "trail_persistence": 0.5},
            {"n_ants": 10, "p_0": 0.2, "trail_min_max_ratio": 5, "stagnation_threshold": 5, "trail_persistence": 0.25},
        ]
    },
    "RankBasedAS": {
        "optimizer": RankBasedAS,
        "parameters": [
            {"n_ants":  3, "number_top": 1, "use_local_search": True, "trail_persistence": 0.75},
            {"n_ants":  3, "number_top": 2, "use_local_search": True, "trail_persistence": 0.5},
            {"n_ants":  3, "number_top": 3, "use_local_search": True, "trail_persistence": 0.25},
            {"n_ants": 10, "number_top": 2, "use_local_search": True, "trail_persistence": 0.75},
            {"n_ants": 10, "number_top": 4, "use_local_search": True, "trail_persistence": 0.5},
            {"n_ants": 10, "number_top": 6, "use_local_search": True, "trail_persistence": 0.25},
        ]
    }
}"""

parameter_space = {
    "RankBasedAS-LS": {
        "optimizer": RankBasedAS,
        "parameters": [
            {"n_ants":  3, "number_top": 1, "use_local_search": True, "trail_persistence": 0.75},
            {"n_ants":  3, "number_top": 2, "use_local_search": True, "trail_persistence": 0.5},
            {"n_ants":  3, "number_top": 3, "use_local_search": True, "trail_persistence": 0.25},
            {"n_ants": 10, "number_top": 2, "use_local_search": True, "trail_persistence": 0.75},
            {"n_ants": 10, "number_top": 4, "use_local_search": True, "trail_persistence": 0.5},
            {"n_ants": 10, "number_top": 6, "use_local_search": True, "trail_persistence": 0.25},
        ]
    }
}

all_paths = list(Path(inst_path).rglob("*.txt"))
fieldnames = ["params"] + [inst.name for inst in all_paths]

for algo in parameter_space:
    print("***************")
    print("Algorithm", algo)
    print("***************")
    optimizer_class = parameter_space[algo]["optimizer"]
    parameters = parameter_space[algo]["parameters"]

    results = []
    for parameter_dict in parameters:
        row = {"params": param_to_string(parameter_dict)}
        print("***************")
        print("Parameters", param_to_string(parameter_dict))
        print("***************")
        seed(42)
        np.random.seed(42)
        for instance_path in all_paths:
            proc_times, weights, deadlines = parse_instance(instance_path)
            optimizer = optimizer_class(evaluate_tardiness, evaluate_tardiness_partial, proc_times, weights, deadlines, **parameter_dict)
            solution, evaluation = optimizer.optimize(max_time = max_time)
            row[instance_path.name] = evaluation
        results.append(row)

        with open(join(out_path, algo + "-opti-param-results.csv"), "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
            writer.writeheader()
            writer.writerows(results)
