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
int printTreeToDotHelp(mlpack::DecisionTree<>& dt, std::ofstream& output, size_t nodeIndex, const MdpInfo& mdpInfo);
void printTreeToDot(mlpack::DecisionTree<>& dt, std::ofstream& output, const MdpInfo& mdpInfo);
    