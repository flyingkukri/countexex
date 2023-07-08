#include "genTrainData.h"
#include <random>
#include <storm-parsers/parser/PrismParser.h>
#include <storm/storage/prism/Program.h>
#include <storm/utility/initialize.h>
#include "storm/environment/solver/MinMaxSolverEnvironment.h"
#include <array>
#include <storm/environment/Environment.h>
#include <storm/models/sparse/StandardRewardModel.h>
#include <bits/stdc++.h>
#include <iostream>

arma::mat createMatrixFromValueMap(std::map<std::string, std::variant<std::vector<int>, std::vector<bool>, std::vector<storm::RationalNumber>>>& value_map){
    arma::mat armaData;
    for (const auto& pair : value_map) {
        // Get the vector corresponding to the key
        const std::variant<std::vector<int>, std::vector<bool>, std::vector<storm::RationalNumber>>& valueVector = pair.second;

        // Create an arma::rowvec from the vector
        arma::rowvec rowVec;
        if (const auto intVector = std::get_if<std::vector<int>>(&valueVector)) {
            rowVec = arma::conv_to<arma::rowvec>::from(*intVector);
        } else if (const auto boolVector = std::get_if<std::vector<bool>>(&valueVector)) {
            std::vector<int> boolToIntVector;
            boolToIntVector.reserve(boolVector->size());
            for (bool value : *boolVector) {
                boolToIntVector.push_back(value ? 1 : 0);
            }
            rowVec = arma::conv_to<arma::rowvec>::from(boolToIntVector);
        } else if (const auto ratVector = std::get_if<std::vector<storm::RationalNumber>>(&valueVector)) {
            // TODO: convert storm::RationalNumber to double
            //            arma::rowvec rowVec(ratVector->size());
            //            for (std::size_t i = 0; i < ratVector->size(); ++i) {
            //                rowVec(i) = storm::utility::convertNumber<double>(ratVector[i]);
            //            }
        }
        // Append the row vector to the matrix
        armaData = arma::join_vert(armaData, rowVec);
    }
    return armaData;
}

arma::Row<size_t> createDataLabels(arma::mat& allPairs, arma::mat& strategyPairs){
    // column-major in arma: thus each column represents a data point
    size_t numColumns = allPairs.n_cols;
    arma::Row<size_t> labels(numColumns, arma::fill::zeros);
    for (size_t i = 0; i < numColumns; ++i) {
        // Get the i-th column of allPairs
        auto found = 0;
        arma::vec column = allPairs.col(i);
        for (size_t j = 0; j < strategyPairs.n_cols; ++j) {
            arma::vec col2Cmp = strategyPairs.col(j);
            if (arma::all(col2Cmp == column)) {
                found = 1;
                labels(i) = 1;
                break;
            }
        }
        if(!found) {
            labels(i) = 0;
        }
    }
    return labels;
}

std::pair<arma::mat, arma::Row<size_t>> createTrainingData(std::map<std::string, std::variant<std::vector<int>, std::vector<bool>, std::vector<storm::RationalNumber>>>& value_map, std::map<std::string, std::variant<std::vector<int>, std::vector<bool>, std::vector<storm::RationalNumber>>>& value_map_submdp){
    arma::mat all_pairs = createMatrixFromValueMap(value_map);
    auto strategy_pairs = createMatrixFromValueMap(value_map_submdp);
    arma::Row<size_t> labels = createDataLabels(all_pairs, strategy_pairs);
    return std::make_pair(all_pairs, labels);
}