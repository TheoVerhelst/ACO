"""This script plots the results of the run of all algorithms on all instances.
It is intended to process the results of run_performance.py. It reads all
results in a folder of result CSV files, and outputs two plot images: one with
box plots with small problem instances, and another box plot with large problem
instances.
"""

from pathlib import Path
from os.path import join
from sys import argv
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

if len(argv) != 4:
    print("error: need a path to results folder and to output folder")
    print("usage: python3", argv[0], "RES_PATH SMALL_OUT_PATH LARGE_OUT_PATH")
    exit(1)

res_path = argv[1]
out_path = {"small": argv[2], "large": argv[3]}

all_paths = list(Path(res_path).rglob("*-results.csv"))
all_data = pd.DataFrame()

for path in all_paths:
    data = pd.read_csv(path)
    data = pd.melt(data, id_vars = "instance")
    data["algo"] = path.name[:-len("-results.csv")]
    del data["variable"]
    all_data = all_data.append(data)

for instances_to_show in ("small", "large"):
    instances = {
        "small": [
            "DD_Ta051.txt",
            "DD_Ta052.txt",
            "DD_Ta053.txt",
            "DD_Ta054.txt",
            "DD_Ta055.txt",
            "DD_Ta056.txt",
            "DD_Ta057.txt",
            "DD_Ta058.txt",
            "DD_Ta059.txt",
            "DD_Ta060.txt"
        ],
        "large": [
            "DD_Ta081.txt",
            "DD_Ta082.txt",
            "DD_Ta083.txt",
            "DD_Ta084.txt",
            "DD_Ta085.txt",
            "DD_Ta086.txt",
            "DD_Ta087.txt",
            "DD_Ta088.txt",
            "DD_Ta089.txt",
            "DD_Ta090.txt"]
    }[instances_to_show]

    data_to_plot = all_data[all_data["instance"].isin(instances)]
    fig, ax = plt.subplots(figsize=(5,4))
    sns.stripplot(x="instance", y="value", hue = "algo", data=data_to_plot, jitter=True, dodge = True)
    plt.legend(title="")
    plt.xticks(rotation=45, horizontalalignment = "right")
    ax.set_ylabel("Weighted tardiness")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(out_path[instances_to_show])
