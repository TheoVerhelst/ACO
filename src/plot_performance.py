from pathlib import Path
from os.path import join
from sys import argv
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

if len(argv) != 3:
    print("error: need a path to results folder and to output folder")
    print("usage: python3", argv[0], "RES_PATH OUT_PATH")
    exit(1)

res_path = argv[1]
out_path = argv[2]
show_small_instance = False

if show_small_instance:
    to_show = [
        "DD_Ta051.txt",
        "DD_Ta052.txt",
        "DD_Ta053.txt",
        "DD_Ta054.txt",
        "DD_Ta055.txt",
        "DD_Ta056.txt",
        "DD_Ta057.txt",
        "DD_Ta058.txt",
        "DD_Ta059.txt",
        "DD_Ta060.txt"]
else:
    to_show = [
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

all_paths = list(Path(res_path).rglob("*.csv"))
all_data = pd.DataFrame()

for path in all_paths:
    data = pd.read_csv(path)
    data = pd.melt(data, id_vars = "instance")
    data["algo"] = path.name[:-len("-results.csv")]
    del data["variable"]
    all_data = all_data.append(data)

all_data = all_data[all_data["instance"].isin(to_show)]

fig, ax = plt.subplots(figsize=(5,4))

#sns.boxplot(x="instance", y="value", data=all_data, hue = "algo")
sns.stripplot(x="instance", y="value", data=all_data, hue = "algo", jitter=True, dodge = True)

plt.legend(title="")
plt.xticks(rotation=45, horizontalalignment = "right")
ax.set_ylabel("Weighted tardiness")
ax.set_xlabel("")
fig.tight_layout()
fig.savefig(join(out_path, ("small-instances" if show_small_instance else "large-instances") + "-results.eps"))
