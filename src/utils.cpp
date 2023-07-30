#include "utils.h"
#include <armadillo>
#include <iostream>
#include <cmath>

int printTreeToDotHelp(mlpack::DecisionTree<>& dt, std::ofstream& output, size_t nodeIndex, const MdpInfo& mdpInfo) {
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
        auto it = mdpInfo.featureMap.find(featureNumber);
        if( it != mdpInfo.featureMap.end()){
            // Get the feature variable 
            output << it->second;
            if(featureNumber<mdpInfo.numOfActId){ // categorical feature
                // Get action represented by featureNumber due to one-hot-encoding
                auto it = mdpInfo.identifierActionMap.find(featureNumber);
                std::string act = "";
                if(it!=mdpInfo.identifierActionMap.end()){
                    act = it->second;
                }
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
        highestIndex = printTreeToDotHelp(dt.Child(i), output, highestIndex + 1, mdpInfo);
    }
    return highestIndex;
}

void printTreeToDot(mlpack::DecisionTree<>& dt, std::ofstream& output, const MdpInfo& mdpInfo) {
    output << "digraph G {\n";
    printTreeToDotHelp(dt, output, 0, mdpInfo);
    output << "}\n";
}
