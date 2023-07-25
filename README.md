# Counterexample explanation in MDPs using Storm - User Manual
This guide provides an overview on how to get started with countexex.
Further information on how to customize and extend countexex can be found in the [Developer manual](doc/develop.md)

## Capabilities - What is countexex?
Counterexample Explanation (Countexex) is a tool that allows comprehensive representations of strategies using decision trees.
Assume we are given a probabilistic and non-deterministic sytem, modeled as Markov Decision Process (MDP), and a property to be checked via a model checker such as [Storm](https://www.stormchecker.org/index.html). In order to resolve the non-determinism we need strategies that indicate which action(s) to take in each state. The strategy might act as a counterexample, providing information on how to reach an error state, or could be a synthesized strategy. Due to the nature of the system (probabilistic AND non-deterministic) those strategies easily grow very large and incomprehensible. Countexex implements the approach proposed in [Counterexample Explanation by Learning Small Strategies in Markov Decision Processes](https://link.springer.com/chapter/10.1007/978-3-319-21690-4_10) to reduce the size of such strategies and provide a understandable representation using decision tree learning. The approach is threefold:

1. Compute liberal, &epsilon;-optimal strategy:
2. Compute importance of states
3. Learn a decision tree representation of the strategy 
    

## Getting started
### System requirements 
As countexex is based on the model checker storm, the support is limited to the following operating systems: 
* macOS on either x86- or ARM-based CPUs
* Debian 11 and higher
* Ubuntu 20.04 and higher
* Arch Linux 

For updates see: [Storm](https://www.stormchecker.org/documentation/obtain-storm/build.html)
We have tested the installation on Ubuntu Linux and ....

## Building
### Dependencies
- [Storm dependencies](https://www.stormchecker.org/documentation/obtain-storm/dependencies.html#general-dependencies)
- As we had to change the mlpack library, currently it needs to be built from source. See: [mlpack](https://github.com/mlpack/mlpack)
- [mlpack dependencies](https://github.com/mlpack/mlpack#2-dependencies)
- mklearn (on fedora mklearn-devel)


### Cloning and Compiling
```bash
$ # Use --recursive to clone the submodules; ssh required
$ git clone git@github.com:flyingkukri/countexex.git --recursive
$ mkdir build
$ cd build
$ cmake ..
$ make
```
### Common installation issues

## Running
The executable countexex is located in /countexex/build/app/
Run the following command:
```bash
$ countexex model.nm property.pctl
```
### Model requirements
- Supply a model at the end of which the label goal is given to the states F, e.g.
    label "goal" = s=1|s=2;
- All final states are sink states


### Command Line Interface
### Configurable options
## Reading the output
The decision tree is stored in the ... folder as a DOT file and can be converted to a pdf via the command ```bash
$ dot -Tpdf default.dot -o default.pdf
```
or visualized e.g. by [Graphviz Online](https://dreampuf.github.io/GraphvizOnline/)
## Development
Click [here](doc/develop.md) for information regarding our code structure
