from pathlib import Path
from os.path import join
from sys import argv
import pandas as pd
from math import factorial as fact
from scipy.stats import mannwhitneyu

if len(argv) != 3:
    print("error: need a path to results folder and to output file")
    print("usage: python3", argv[0], "RES_PATH OUT_PATH")
    exit(1)

res_path = argv[1]
out_file = argv[2]

all_paths = list(Path(res_path).rglob("*-results.csv"))
all_data = pd.DataFrame()

# Read all result files
for path in all_paths:
    data = pd.read_csv(path)
    data = pd.melt(data, id_vars = "instance")
    data["algo"] = path.name[:-len("-results.csv")]
    del data["variable"]
    all_data = all_data.append(data)

algorithms = list(all_data["algo"].unique())
instances = list(all_data["instance"].unique())

# Create an empty data frame for the statistical test results
results = pd.DataFrame([], columns = ["algo1", "algo2", "instance", "pval", "statistic"])

# For each pair of different algorithms
for i in range(len(algorithms)):
    for j in range(i + 1, len(algorithms)):
        # Conduct a Wilcoxon rank-sum test for each instance
        for instance in instances:
            algo_1 = algorithms[i]
            algo_2 = algorithms[j]
            algo_1_values = list(all_data[(all_data["algo"] == algo_1) & (all_data["instance"] == instance)]["value"])
            algo_2_values = list(all_data[(all_data["algo"] == algo_2) & (all_data["instance"] == instance)]["value"])
            statistic, pval = mannwhitneyu(algo_1_values, algo_2_values, use_continuity = False, alternative = "two-sided")
            results = results.append([{
                "algo1": algo_1,
                "algo2": algo_2,
                "instance": instance,
                # Bonferroni correction, the test between the different algorithms
                # are not independent
                "pval": pval * (fact(len(algorithms)) // (2 * fact(len(algorithms) - 2))),
                "statistic": statistic
            }])

results.to_csv(out_file, index = False)
