#pragma once
#include <iostream>
#undef As
#include <mlpack.hpp>
#include <mlpack/core.hpp>
#include <mlpack/core/data/load.hpp>
#include <mlpack/core/data/save.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include <string>
#include <map>
#include <vector>
#include <memory>
#include <variant>
#include "../train_data_generation/genTrainData.h"

/*!
*  Helper to print the tree
* @param dt: decision tree to be printed
* @param output: output stream to write to
* @param nodeIndex: helper variable for giving each node a unique index
* @param mdpInfo: information about the MDP
* @return: The highest index of the subtree starting at this node
*/
int printTreeToDotHelp(mlpack::DecisionTree<>& dt, std::ofstream& output, size_t nodeIndex, const MdpInfo& mdpInfo);

/*!
* Print the tree to a dot file
* @param dt: decision tree to be printed
* @param output: output stream to write to
* @param mdpInfo: information about the MDP
*/
void printTreeToDot(mlpack::DecisionTree<>& dt, std::ofstream& output, const MdpInfo& mdpInfo);
    