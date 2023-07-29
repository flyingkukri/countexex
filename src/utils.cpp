#include "utils.h"
#include <armadillo>
#include <iostream>
#include <cmath>

int printTreeToDotHelp(mlpack::DecisionTree<mlpack::GiniGain, mlpack::BestBinaryNumericSplit, mlpack::AllCategoricalSplit, mlpack::AllDimensionSelect, false>& dt, std::ofstream& output, size_t nodeIndex, std::map<int,std::string>& featureMap, std::map<int, std::string> datasetInfo, int numOfActId) {
    // Print this node.
    output << "node" << nodeIndex << " [label=\"";
    // This is a leaf node.
    if (dt.NumChildren() == 0){
        // In this case classProbabilities will hold the information for the
        // probabilites of each class.
        auto probs = dt.ClassProbabilities();
        int maxClass = probs.index_max();
        output << (maxClass ? "good" : "bad") << "\\n";
    } else { // This is a splitting node.
        // If the node isn't a leaf, getClassProbabilities() returns the splitinfo
        auto featureNumber = dt.SplitDimension();        
        auto it = featureMap.find(featureNumber);
        if( it != featureMap.end()){
            // Get the feature variable 
            output << it->second;
            if(featureNumber<numOfActId){ // categorical feature
                // Get action represented by featureNumber due to one-hot-encoding
                auto act = datasetInfo[featureNumber];
                output << " = [" << act << "]";
            }else{ // numeric feature
                // TODO: how do we know the operator: <=, <, >=?
                output << " <=" << dt.ClassProbabilities();
            }
        }
    }

    output << "\"];\n";
    // Recurse to children.
    int highestIndex = nodeIndex;
    for (size_t i = 0; i < dt.NumChildren(); ++i)
    {
        output << "node" << nodeIndex << " -> node" << (highestIndex + 1) << ";\n";
        highestIndex = printTreeToDotHelp(dt.Child(i), output, highestIndex + 1, featureMap, datasetInfo, numOfActId);
    }
    return highestIndex;
}

void printTreeToDot(mlpack::DecisionTree<mlpack::GiniGain, mlpack::BestBinaryNumericSplit, mlpack::AllCategoricalSplit, mlpack::AllDimensionSelect, false>& dt, std::ofstream& output, std::map<int,std::string>& featureMap, std::map<int, std::string> datasetInfo, int numOfActId) {
    output << "digraph G {\n";
    printTreeToDotHelp(dt, output, 0, featureMap, datasetInfo, numOfActId);
    output << "}\n";
}
