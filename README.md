# Project on Ant Colony Optimization applied to the Permutation Flow Shop Problem with Weighted Tardiness

## Organization

This project is divided into folders:

* `instances/` contains the problem instances (i.e. numerical data)
* `src/` contains the Python code
* `results/` contain the numerical results in CSV files
* `plots/` contains the EPS plots of the results used in the report
* `report/` contains the LaTeX report


The code is divided into two different types of source files: the algorithm code
and the scripts.

The script files are all those that begins with `run_` or `plot_`, and are
responsible for running the experiments and exporting the results. They all take
two command-line arguments, the first being the input file or folder, the second
being the output file or folder. The `run_` files take as input the path to the
instance file or folder, and outputs a CSV file with numerical results. The
`plot_` files take as input the path to the CSV result files (or folder thereof)
and output an EPS plot.

The algorithm code files are all the others, and implement the various
algorithms we use in our experiments.

## Requirements

This project is fully written in Python 3. For installing the required
libraries, run

```
pip install -r requirements.txt
```

## Running the project

To run the parameter optimization experiment, run in the command line

```
python3 src/run_parameter_optimization.py instances/ results/
```
To plot the results of the parameter optimization experiment, run

```
python3 src/plot_parameter_optimization.py results/<algo name>-results-opti-param.csv plots/<algo name>-params.eps
```

To run all algorithm 10 times on each instance, run

```
python3 src/run_performance.py instances/ results/
```

To plot the results of the runs, run

```
python3 src/plot_performance.py results/ plots/small-instances-results.eps plots/large-instances-results.eps
```

To perform the Wilcoxon rank-sum test to compare different algorithms, run

```
python3 src/run_wilcoxon.py results/ results/all-wcn.csv
```

To run algorithms and plot their convergence, run

```
python3 src/run_plot_convergence.py instances/<instance name>.txt plots/<instance name>-convergence.eps
```

In order to keep the code simple, we do not provide a simple script to run the
experiment where we the local search is evaluated. Instead, we use the previous
scripts by changing the arguments and some of the code. The local search is
activated by giving the argument `use_local_search = True` to the constructor
of the RankBasedAS class.
