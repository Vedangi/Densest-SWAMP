# Densest-SWAMP
Code for The Densest SWAMP Problem. This repo accompanies the paper 
> The Densest SWAMP problem: subhypergraphs with arbitrary monotonic partial edge rewards


## Data source
The real world datasets are publicly available at [Hypergraph-Data](https://www.cs.cornell.edu/~arb/data/).
The raw data has been preprocessed to remove self-loops and dangling nodes using the methods in [Data preprossing](https://github.com/luotuoqingshan/local-DSHG).
A copy of the code along with the license is included in the 'include' folder. The 'include' folder also contains files for some utility functions and a maxflow solver that have been used in our code.

The processed data is then stored in datafolder/large-datasets and datafolder/small_benchmarks.

## src files
The 'src' folder contains all the files for objective, methods and other helper-function definitions, which have been used in the experiments.
We also include some demo files for running the ILP-based Exact method, the flow-based methods and greedy methods.

The run_expts.jl file contains code to run each of these methods for each objective over a given hypergraph.
The result is saved in the results folder. We add some sample results for a small Trivago dataset.

