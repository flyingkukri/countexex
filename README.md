# Counterexample explanation in MDPs using Storm - User Manual
This guide provides an overview of the tool countexex.
Further information on how to customize and extend countexex can be found in the [Developer manual](doc/develop.md).

## Capabilities - What is countexex?
Counterexample Explanation (Countexex) is a tool that allows comprehensible representations of strategies using decision trees.
Given a probabilistic and non-deterministic system, modeled as Markov Decision Process (MDP), and a property to be checked via a model checker, we need strategies in order to resolve the non-determinism, indicating which action to take in each state. The strategy might act as a counterexample, providing information on how to reach error states, or could be a synthesized strategy. Due to the nature of the system (probabilistic AND non-deterministic) those strategies easily grow very large and incomprehensible. Countexex implements the approach presented in [this paper](https://link.springer.com/chapter/10.1007/978-3-319-21690-4_10) to reduce the size of strategies and provide a succinct representation using decision tree learning. 

The approach is threefold:

1. **Compute liberal, &epsilon;-optimal strategy:**  
    We allow multiple, equally good actions to be chosen for each state to give the learning algorithm more freedom. Additionally, we apply a maximal end component (MEC) decomposition on the MDP. If a state is an exit, we only select maximal-external state-action pairs for that state. More information on the definitions can be found [here](https://link.springer.com/chapter/10.1007/978-3-319-21690-4_10).  

2. **Compute importance of states:**  
    We simulate 10,000 runs on the MDP under the strategy and count how often each state was reached. This importance value determines how frequently the state-action pairs will occur in the training data. Therefore, states that do not lead to a target state get an importance value of zero, thereby reducing the amount of relevant states.

3. **Learn a decision tree representation of the strategy:**   
    In the last step, we apply a decision tree learning algorithm on the modified strategy. Via tuning parameters, we are able to influence the size of the resulting tree further, in order to obtain a succinct representation.
    

## Getting started
### System requirements 
As countexex is based on the model checker storm, the support is limited to the following operating systems: 
* macOS on either x86- or ARM-based CPUs
* Debian 11 and higher
* Ubuntu 20.04 and higher
* Arch Linux 

For updates see: [Storm](https://www.stormchecker.org/documentation/obtain-storm/build.html).

We have tested the installation on Linux Ubuntu and Fedora.

## Building
### Dependencies
- [Storm dependencies](https://www.stormchecker.org/documentation/obtain-storm/dependencies.html#general-dependencies)
- [mlpack dependencies](https://github.com/mlpack/mlpack#2-dependencies)
- Install the [Gurobi solver](https://www.gurobi.com/) if you want to use MILP. For Linux systems, you can follow this [guide](https://ca.cs.uni-bonn.de/doku.php?id=tutorial:gurobi-install)
- example for ubuntu
```bash
$ sudo apt-get install build-essential git cmake libboost-all-dev libcln-dev libgmp-dev libginac-dev automake libglpk-dev libhwloc-dev libz3-dev libxerces-c-dev libeigen3-dev libarmadillo-dev libensmallen-dev libcereal-dev
```

### Cloning and Compiling
In the following, use *--recursive* in order to clone the submodules. Additionally, the project has to be cloned via SSH-URL. 

```bash
git clone git@github.com:flyingkukri/countexex.git --recursive
mkdir build
cd build
cmake ..
make
```

Due to the size of the project the compilation time is long. For a speedup use 
```bash
make -j${NUMBER_OF_CORES}
```
if you have multiple cores and at least *8GB* of memory.  

In order to be able to debug the system, set the option *STORM_DEVELOPER* to *ON* in *countexex/storm/CMakeLists.txt*.
If you want to use the Gurobi solver that is not shipped with Storm, you need to additionally enable *STORM_USE_GUROBI* and set the path in *GUROBI_ROOT*. Thereafter, execute make.

## Running

### Command Line Interface
Check if the installation was successful: The executable countexex is located in the folder *PathToCountexex/build/app/*, where PathToCountexex refers to the root directory of the project. Execute the following command:
```bash
./countexex -h
```

Now you should be able to see the following help page:
```
Usage:

General:
  -h [ --help ]                         Print help message and exit
  -v [ --verbose ]                      Print additional output during the 
                                        program execution.

Check task:
  -m [ --model ] arg                    Required argument: Path to model file. 
                                        Model has to be in PRISM format: e.g., 
                                        model.nm
  -p [ --propertyMax ] arg              Required argument: Specify whether you 
                                        want to check Pmax or Pmin. Set the 
                                        argument to max or min accordingly.

Configuration arguments:
  -c [ --config ] arg                   Path to a config file where the 
                                        following parameters can be specified 
                                        in alternative to specifying them via 
                                        the command line.
  -g [ --minimumGainSplit ] arg (=9.9999999999999995e-08)
                                        Set the minimumGainSplit parameter for 
                                        the decision tree learning.
  -l [ --minimumLeafSize ] arg (=5)     Set the minimumLeafSize parameter for 
                                        the decision tree learning.
  -d [ --maximumDepth ] arg (=10)       Set the maximumDepth parameter for the 
                                        decision tree learning.
  -i [ --importanceDelta ] arg (=0.001) Set the delta parameter for the 
                                        importance calculation.
  -s [ --safetyPrec ] arg (=16)         Set the precision for the safety 
                                        property bound.
  -o [ --optimizer ] arg (=smt)         Choose the method for computing the 
                                        permissive strategy: smt or milp. Note 
                                        that for MILP, you need to have Gurobi 
                                        installed.
```

### Input format

#### Model requirements - supported tools
Countexex currently only supports file formats generated by PRISM. For more information on how to add support for new file formats, see [Developer manual](doc/develop.md).  
Additionally, it is expected that all final states of the model are sink states and no variable is called "action".

#### Property requirements
Currently only reachability objectives are supported. As for the permissive strategy computation only eventually formulas are allowed, we restrict the properties to the following form: 

```
Pmax=? [ F "goal" ]
Pmin=? [ F "goal" ]
```

Therefore, the user only has to specify via *propertyMax* whether Pmax or Pmin should be checked.
Additionally, the set of states for which we want to check the reachability has to be labeled as 'goal'. This can be achieved by adding a label, as shown below, at the bottom of the model file:

```
# model.nm 
...
label "goal" = s=1|s=2;
```

### Example executions
As the configurable options have default values, they don't need to be specified.
In case you want to change them, you have two options: 

1. Options specified via command line:

```bash
./countexex --model "PathToCountexex/examples/cycle.nm" --propertyMax max -l 1 -d 100 -i 0.01
```

2. Options specified via config file: 
Create a config file and specify the desired parameters:

```
# config.txt:
minimumGainSplit = 1e-10
minimumLeafSize = 20
importanceDelta = 0.01
```

Then run: 
```bash
./countexex --model "PathToCountexex/examples/cycle.nm" --propertyMax max -c "PathToConfig/config.txt"
```

### Configurable options

For more information on the decision tree learning parameters see: [mlpack DecisionTree](https://mlpack.org/doc/mlpack-3.3.1/doxygen/classmlpack_1_1tree_1_1DecisionTree.html).


**ImportanceDelta**: we compute for each state an importance value *Imp_s*, that indicates how often this state is repeated in the training data. However, if *Imp_s* is below a certain threshold, *importanceDelta*, we will simply discard this state. Depending on your model structure and set of target states, you might want to change this parameter defaulting to 0.001. 

**safetyPrec**: the permissive strategy computation expects safety properties of the following form:  
for propertyMax = max:  P >= Pmax [F s]  
We therefore need to convert Pmax to a string with a fixed number of decimal places, which is specified by *safetyPrec*. Depending on the expected value of Pmax you might want to change safetyPrec, e.g., to a higher value for very small values of Pmax.  

**optimizer**: This option allows selecting the computation method for the permissive strategy. In certain cases, when using the SMT method with some models, an error indicating an empty expression list can occur. In such situations, we recommend switching to the MILP method. Note that this switch requires the Gurobi solver to be installed on your system.  


## Reading the output
The decision tree is stored as a DOT file named *graph.dot* within the "build/app/" folder and can be converted to a pdf via the command 
```bash
dot -Tpdf graph.dot -o graph.pdf
```
or visualized e.g., by [Graphviz Online](https://dreampuf.github.io/GraphvizOnline/)

## Examples
Click [here](examples/examples.md) to see a few examples.

## Development
Click [here](doc/develop.md) for information regarding our code structure.
