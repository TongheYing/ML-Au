# Machine learning potential aided structure search for low-lying candidates of Au clusters
This program implements the automated Au clusters structure search using machine learning potential.

## Prerequisites
This program requires:
- [SchNet](https://github.com/atomistic-machine-learning/schnetpack)

If you are new to Python, the easiest way of installing the prerequisites is via [conda](https://conda.io/docs/index.html).

## Usage

### Training Dataset
The training dataset used to get the machine learning potential is in the directory `training_dataset`

### Configuration File
The configuration file of PSO is `local/pso/pso_init`

`NPAR`: the number of PSO particles

`GER`: the generation number

`Radius`: Au 1.44

`EDIFF`: the convergence accuracy of energy

`ELIR`: the elimination proportion of every generation

`PBC`: 0 0 0

`CUO`: the min and max distance between atoms

the main program entry is `run_iter.py`

## Authors
This program was primarily written by Tonghe Ying.

Email: yingth@mail.ustc.edu.cn
