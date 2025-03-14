# Densest-SWAMP
This repository contains code for The Densest SWAMP Problem and is associated with the paper: 
> The Densest SWAMP problem: subhypergraphs with arbitrary monotonic partial edge rewards


## Data source
Real-world datasets are publicly available at [Hypergraph-Data](https://www.cs.cornell.edu/~arb/data/).
The raw data was preprocessed to remove self-loops and dangling nodes following the methods outlined in [Data preprocessing](https://github.com/luotuoqingshan/local-DSHG).
A copy of the code along with its license is provided in the 'include' folder. This folder also contains various utility functions and a maxflow solver used in our implementation.
The processed data is organized  into two directories 
 - datafolder/large-datasets
 - datafolder/small_benchmarks.

## Source Files
The 'src' folder contains all the source code for objective functions, methods, and supporting helper functions.
Demo files are provided for running the ILP-based methods, the flow-based projection methods, and various greedy methods.

The run_expts.jl file contains code to execute each of these methods for each objective over a given hypergraph, and the results are saved in the results folder. We add some A set of sample results for small Trivago dataset is included.

