#include "utils.h"
#include <armadillo>

int printTreeToDotHelp(mlpack::DecisionTree<mlpack::GiniGain, mlpack::BestBinaryNumericSplit, mlpack::AllCategoricalSplit, mlpack::AllDimensionSelect, false>& dt, std::ofstream& output, size_t nodeIndex) {
    // Print this node.
    output << "node" << nodeIndex << " [label=\"";
    // This is a leaf node.
    if (dt.NumChildren() == 0){

        // In this case classProbabilities will hold the information for the
        // probabilites if each class.
        auto probs = dt.ClassProbabilities();
        int maxClass = probs.index_max();
        output << "Leaf\nLabel: " << maxClass << "\\n";

    } else {
        // This is a splitting node.
        output << "Split\nFeature: " << dt.SplitDimension() << "\\n";
        // If the node isn't a leaf, getClassProbabilities() returns the splitinfo
        output << "Threshold: " << dt.ClassProbabilities() << "\\n";
    }
    output << "\"];\n";
    // Recurse to children.
    int highestIndex = nodeIndex;
    for (size_t i = 0; i < dt.NumChildren(); ++i)
    {
        output << "node" << nodeIndex << " -> node" << (highestIndex + 1) << ";\n";
        highestIndex = printTreeToDotHelp(dt.Child(i), output, highestIndex + 1);
    }
    return highestIndex;
}

void printTreeToDot(mlpack::DecisionTree<mlpack::GiniGain, mlpack::BestBinaryNumericSplit, mlpack::AllCategoricalSplit, mlpack::AllDimensionSelect, false>& dt, std::ofstream& output) {
    output << "digraph G {\n";
    printTreeToDotHelp(dt, output, 0);
    output << "}\n";
}
