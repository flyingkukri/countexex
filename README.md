# Counterexample explanation in MDPs using Storm - User Manual
This guide provides an overview of the tool countexex.
Further information on how to customize and extend countexex can be found in the [Developer manual](doc/develop.md).

## Capabilities - What is countexex?
Counterexample Explanation (Countexex) is a tool that allows comprehensible representations of strategies using decision trees.
Given a probabilistic and non-deterministic system, modeled as Markov Decision Process (MDP), and a property to be checked via a model checker, we need strategies in order to resolve the non-determinism, indicating which action to take in each state. The strategy might act as a counterexample, providing information on how to reach an error state, or could be a synthesized strategy. Due to the nature of the system (probabilistic AND non-deterministic) those strategies easily grow very large and incomprehensible. Countexex implements the approach presented in [this paper](https://link.springer.com/chapter/10.1007/978-3-319-21690-4_10) to reduce the size of strategies and provide a succinct representation using decision tree learning. 

The approach is threefold:

1. **Compute liberal, &epsilon;-optimal strategy:**  
    We allow multiple actions to be chosen for each state to give the learning algorithm more freedom.

2. **Compute importance of states:**  
    We simulate 10,000 runs on the MDP under the strategy and count how often each state was reached. This importance value determines how frequently the state will occur in the training data. Therefore, states that do not lead to a target state get an importance value of zero, thereby reducing the amount of relevant states.

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

We have tested the installation on Ubuntu Linux and ....

Due to the size of the system ... certain requirements for running? 
## Building
### Dependencies
- [Storm dependencies](https://www.stormchecker.org/documentation/obtain-storm/dependencies.html#general-dependencies)
- As we had to change the mlpack library, currently it needs to be built from source. See: [mlpack](https://github.com/mlpack/mlpack)
- [mlpack dependencies](https://github.com/mlpack/mlpack#2-dependencies)
- mklearn (on fedora mklearn-devel)


### Cloning and Compiling
```bash
# Use --recursive to clone the submodules; ssh required
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

if you have multiple cores and at least **8GB** of memory.

## Running

### Command Line Interface
Check if the installation was successful: The executable countexex is located in the folder **{PathToCountexex}/countexex/build/app/**. Execute the following command:
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
  -p [ --propertyMax ] arg              Required argument: Specify wether you 
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
```

### Input format

#### Model requirements - supported tools
Countexex currently only supports file formats generated by PRISM. For more information on how to add support for new file formats, see [Developer manual](doc/develop.md).  
Additionally, is is expected that all final states of the model are sink states and no variable is called "action".

#### Property requirements
Currently only reachability objectives are supported. As for the permissive strategy computation only eventually formulas are allowed, we restrict the properties to the following form: 

```
Pmax=? [ F "goal" ]
Pmin=? [ F "goal" ]
```

Therefore, the user only has to specify via *propertyMax* wether Pmax or Pmin should be checked.
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
./countexex --model "{PathToModelFile}/cicle.nm" --property "{PathToPropertyFile}/default.props" -l 1 -d 100 -i 0.01
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
./countexex --model "{PathToModelFile}/cicle.nm" --property "{PathToPropertyFile}/default.props" -c "{PathToConfigFile}/config.txt"
```

### Configurable options

For more information on the decision tree learning parameters see: [mlpack DecisionTree](https://mlpack.org/doc/mlpack-3.3.1/doxygen/classmlpack_1_1tree_1_1DecisionTree.html).


**ImportanceDelta**: we compute for each state an importance value **Imp_s**, that indicates how often this state is repeated in the training data. However, if **Imp_s** is below a certain threshold: **importanceDelta**, we will simply discard this state. Depending on your model structure and set of target states, you might want to change this parameter defaulting to 0.001. 

**safetyPrec**: the permissive strategy computation expects safety properties of the following form:  
for propertyMax = max:  P >= Pmax [F s]  
We therefore need to convert Pmax to a string with a fixed number of decimal places, which is specified by **safetyPrec**. Depending on the expected value of Pmax you might want to change safetyPrec, e.g., to a higher value for very small values of Pmax.  

## Reading the output
The decision tree is located in the folder **{PathToCountexex}/countexex/build/app/graph.dot** as a **DOT file** and can be converted to a pdf via the command 
```bash
dot -Tpdf graph.dot -o graph.pdf
```
or visualized e.g., by [Graphviz Online](https://dreampuf.github.io/GraphvizOnline/)

## Examples
Click [here](examples/examples.md) to see a few examples.

## Development
Click [here](doc/develop.md) for information regarding our code structure.