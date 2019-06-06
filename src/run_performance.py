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

if len(argv) != 3:
    print("error: need a path to instance folder and to output folder")
    print("usage: python3", argv[0], "INST_PATH OUT_PATH")
    exit(1)

inst_path = argv[1]
out_path = argv[2]

max_time = 30
repetitions = 10

parameter_space = {
    "IG_RLS": {
        "optimizer": IG_RLS,
        "parameters": {"d": 2, "T": 1, "weighted_temperature": False}
    },
    "MaxMinAS": {
        "optimizer": MaxMinAS,
        "parameters": {"n_ants": 10,  "p_0": 0.9, "trail_min_max_ratio": 5, "stagnation_threshold": 5, "trail_persistence": 0.75}
    },
    "RankBasedAS": {
        "optimizer": RankBasedAS,
        "parameters": {"n_ants":  10, "number_top": 3, "trail_persistence": 0.75}
    }
}

all_paths = list(Path(inst_path).rglob("*.txt"))
fieldnames = ["instance"] + [str(i) for i in range(repetitions)]

for algo in parameter_space:
    print("***************")
    print("Algorithm", algo)
    print("***************")
    optimizer_class = parameter_space[algo]["optimizer"]
    parameters = parameter_space[algo]["parameters"]

    with open(join(out_path, algo + "-results.csv"), "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = fieldnames)
        writer.writeheader()

        for instance_path in all_paths:
            proc_times, weights, deadlines = parse_instance(instance_path)
            row = {"instance": instance_path.name}
            print("***************")
            print("Instance", instance_path.name)
            print("***************")
            seed(42)
            np.random.seed(42)
            for i in range(repetitions):
                optimizer = optimizer_class(evaluate_tardiness, evaluate_tardiness_partial, proc_times, weights, deadlines, **parameters)
                solution, evaluation = optimizer.optimize(max_time = max_time)
                row[str(i)] = evaluation
            writer.writerow(row)
