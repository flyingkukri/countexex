#pragma once
#include <iostream>
#undef As
#include <mlpack.hpp>
#include <mlpack/core.hpp>
#include <mlpack/core/data/load.hpp>
#include <mlpack/core/data/save.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>
int printTreeToDotHelp(mlpack::DecisionTree<mlpack::GiniGain, mlpack::BestBinaryNumericSplit, mlpack::AllCategoricalSplit, mlpack::AllDimensionSelect, false>& dt, std::ofstream& output, size_t nodeIndex, std::map<int,std::string>& featureMap, std::map<int, std::string> datasetInfo, int numOfActId);
void printTreeToDot(mlpack::DecisionTree<mlpack::GiniGain, mlpack::BestBinaryNumericSplit, mlpack::AllCategoricalSplit, mlpack::AllDimensionSelect, false>& dt, std::ofstream& output, std::map<int,std::string>& featureMap, std::map<int, std::string> datasetInfo, int numOfActId);
    