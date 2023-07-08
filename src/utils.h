#include <iostream>
#include <mlpack/methods/decision_tree.hpp>

void printTreeToDot(mlpack::DecisionTree<>& dt, std::ostream& output);