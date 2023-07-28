#include "utils.h"
#include <armadillo>

int printTreeToDotHelp(mlpack::DecisionTree<mlpack::GiniGain, mlpack::BestBinaryNumericSplit, mlpack::AllCategoricalSplit, mlpack::AllDimensionSelect, false>& dt, std::ofstream& output, size_t nodeIndex, std::map<int,std::string>& featureMap, mlpack::data::DatasetInfo& datasetInfo) {
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
            output << it->second;
            if(featureNumber==0){ // categorical feature
                auto tmp = dt.ClassProbabilities().at(0);
                for(auto i : dt.ClassProbabilities()){
                    std::cout << "Print value of ClassProbs: " << std::endl;
                    std::cout << i << std::endl;
                }
                // output << " = [" << dt.ClassProbabilities() << "]";
                output << " = [" << datasetInfo.UnmapString(static_cast<int>(tmp),featureNumber) << "]";   
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
        highestIndex = printTreeToDotHelp(dt.Child(i), output, highestIndex + 1, featureMap, datasetInfo);
    }
    return highestIndex;
}

void printTreeToDot(mlpack::DecisionTree<mlpack::GiniGain, mlpack::BestBinaryNumericSplit, mlpack::AllCategoricalSplit, mlpack::AllDimensionSelect, false>& dt, std::ofstream& output, std::map<int,std::string>& featureMap, mlpack::data::DatasetInfo& datasetInfo) {
    output << "digraph G {\n";
    printTreeToDotHelp(dt, output, 0, featureMap, datasetInfo);
    output << "}\n";
}
