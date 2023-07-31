# Developer Guide
This guide contains necessary information to extend or customize countexex to suit your specific needs.

## Overview
In this section we provide an overview of the software architecture of the tool.
The following is a schematic overview of the folder structure of our project.
```
├── app
│   ├── main.cpp
├── CMakeLists.txt
├── doc
├── examples
├── LICENSE
├── README.md
├── src
│   ├── dtree_visualization
│   ├── importance_calculation
│   ├── model_builder
│   ├── train_data_generation
├── tests

```

The folder app contains the main.cpp file.
In the main.cpp file, the function pipeline is the main process. 
Each folder in src contains a header and a cpp file.
This contain library function that get used by the pipeline.

### Important Data Structures
#### value_map
```cpp
std::map<std::string, std::variant<std::vector<int>, std::vector<bool>>> value_map
```
This data structure is our representation of the state-action pairs. 
it is a map from a string that is either 
1. the name of a variable (or dimension in mlpack)
2. "action"
3. "imps"

to a vector of values. 
The cartesian product of the n-th entry of the vectors for every key and "action" constitute a state-action pair.
We further add the vector imps which is the id of the state, so that we can later repeat this state-action pair as often as needed.
#### data
```cpp
arma::mat data
```
This contains the state-action pairs in matrix format. 
We will have repeated each state-action pair as often as indicated by the importance vector.
As arma is column-major, mlpack treats each column as a data point and each row as a dimension.

### main.cpp
After setting up the model of type storm::models::sparse::Mdp<double>, we calculate the maximum possible probability of reaching our goal states, because for calculating a permissive strategy, for need a formula of the form P >= p [F s].
After this, we calculate the importance using calculateImps from importance_calculation/impCalc.cpp.
Then the pipeline calls createStateActionPairs from genTrainData.cpp to convert the state-actions pairs of the mdp into the value_map.
Finally the function createTrainingData in genTrainData.cpp converts this value map into a matrix and label pair that we can use as input for the decision tree.

### genTrainData.cpp
In this section we will give an overview of the 
#### 

```mermaid
  graph TD;
      A-->B;
      A-->C;
      B-->D;
      C-->D;
```

## CMake structure

## Mlpack
- Mlpack is column major in arma; Thus each column represents a data point and each row represents a dimension
## Setup
In order to be able to debug the system set the option *STORM_DEVELOPER* to *ON* in **/countexex/storm/CMakeLists.txt**
## Extending countexex
### Supporting New Objectives
Currently only reachability objectives are supported ...