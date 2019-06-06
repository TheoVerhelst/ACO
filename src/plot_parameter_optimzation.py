from pathlib import Path
from sys import argv
import seaborn as sns
import pandas as pd
from os.path import join, basename
import matplotlib.pyplot as plt

if len(argv) != 3:
    print("error: need a path to results file and to output folder")
    print("usage: python3", argv[0], "RES_FILE OUT_PATH")
    exit(1)

res_file = argv[1]
out_path = argv[2]

data = pd.read_csv(res_file)
data = pd.melt(data, id_vars = "params")
del data["variable"]

fig, ax = plt.subplots(figsize=(5,4))

sns.boxplot(y="params", x="value", data=data, color = "white", orient = "h")
sns.stripplot(y="params", x="value", data=data, jitter=True, color = "black", orient = "h")

ticks_labels = {
    "RankBasedAS": ["$K=3,\\,\omega=1,\\rho=0.75$", "$K=3,\\,\omega=2,\\rho=0.5$", "$K=3,\\,\omega=3,\\rho=0.25$",
                    "$K=10,\\,\omega=2,\\rho=0.75$", "$K=10,\\,\omega=4,\\rho=0.5$", "$K=10,\\,\omega=6,\\rho=0.25$"],
    "IG_RLS": ["$d=2$, not weighted", "$d=3$, not weighted", "$d=4$, not weighted",
               "$d=2$, weighted", "$d=3$, weighted", "$d=4$, weighted"],
    "MaxMinAS": ["$K=3,\\,p_0=0.9,\\rho=0.75$", "$K=3,\\,p_0=0.5,\\rho=0.5$", "$K=3,\\,p_0=0.2,\\rho=0.25$",
                 "$K=10,\\,p_0=0.9,\\rho=0.75$", "$K=10,\\,p_0=0.5,\\rho=0.5$", "$K=10,\\,p_0=0.2,\\rho=0.25$"]
}

algo_name = basename(res_file)[:-len("-results.csv")]

ax.set_yticklabels(ticks_labels[algo_name])
plt.xticks(rotation=45)
ax.set_ylabel("")
ax.set_xlabel("Weighted tardiness")
fig.tight_layout()
fig.savefig(join(out_path, algo_name + "-params.eps"))
plt.show()
