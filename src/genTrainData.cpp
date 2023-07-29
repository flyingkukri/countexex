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

void createMatrixHelper(arma::mat& armaData, arma::rowvec& rowVec, std::variant<std::vector<int>, std::vector<bool>>& valueVector){
    // Create an arma::rowvec from the vector
    if (const auto intVector = std::get_if<std::vector<int>>(&valueVector)) {
        rowVec = arma::conv_to<arma::rowvec>::from(*intVector);
    } else if (const auto boolVector = std::get_if<std::vector<bool>>(&valueVector)) {
        std::vector<int> boolToIntVector;
        boolToIntVector.reserve(boolVector->size());
        for (bool value : *boolVector) {
            boolToIntVector.push_back(value ? 1 : 0);
        }
        rowVec = arma::conv_to<arma::rowvec>::from(boolToIntVector);
    }
    armaData = arma::join_vert(armaData, rowVec);
}


void categoricalFeatureOneHotEncoding(arma::mat& armaData, uint_fast64_t numOfActId, std::map<int,std::string>& featureMap, std::variant<std::vector<int>, std::vector<bool>>& valueVector){
    // We know that the variant holds an int vector in this case as it contains the action identifiers
    if (const auto intVector = std::get_if<std::vector<int>>(&valueVector)) {
        auto ncols = (*intVector).size();
        // Create a feature row in the matrix for every actionIdentifier: e.g. if there are 10 different actions we add 10 feature rows to the matrix
        for(int i=0; i<numOfActId; ++i){
            arma::Row<double> rowVec = arma::zeros<arma::Row<double>>(ncols);
            armaData = arma::join_vert(armaData, rowVec);
            // indicate for each row i that it represents an action feature
            featureMap.insert(std::make_pair(i,"action"));
        }
        // std::cout << "ArmaData before:\n " << armaData << std::endl;
        // intVector: each entry i corresponds to the actionIdentifier of the i-th data point
        // we thus set the entry of the row that corresponds to that actionIdentifier to 1 for the i-th data point
        for(int i=0;i<ncols;++i){
            // as we store the stateIndex in the first row temporarily we add +1 to access a row i logically 
            armaData.at((*intVector).at(i)+1,i)=1;
            // auto tmp = (*intVector).at(i);
            // std::cout << "ArmaData after round:\n " << i << " \n" << armaData << std::endl;
        }
    }
}

std::pair<arma::mat, std::map<int,std::string>> createMatrixFromValueMap(std::map<std::string, std::variant<std::vector<int>, std::vector<bool>>>& value_map, uint_fast64_t numOfActId){
    arma::mat armaData;
    arma::rowvec rowVec;
    std::string imps = "imps";
    std::string act = "action";

    // Every row of the resulting matrix corresponds to a feature
    // Create a mapping between variable names and row numbers(==feature number)
    std::map<int,std::string> featureMap;

    // make sure imps is the first row in the matrix
    auto it = value_map.find(imps);
    std::variant<std::vector<int>, std::vector<bool>>& valueVector = it->second;
    if(it!=value_map.end()){
        createMatrixHelper(armaData,rowVec, valueVector);    
    } // no entry in featureMap for imps as this row will be removed from the matrix for training

    // one-hot encoding for the categorical action features
    it = value_map.find(act);
    valueVector = it->second;
    if(it!=value_map.end()){
        categoricalFeatureOneHotEncoding(armaData, numOfActId, featureMap, valueVector);
    }
    
    auto featureIndex = numOfActId;
    // loop over all other key-value pairs
    for (const auto& pair : value_map) {
        // Get the vector corresponding to the key
        if(pair.first!="imps" && pair.first != "action"){
            valueVector = pair.second;
            createMatrixHelper(armaData,rowVec, valueVector);
            featureMap.insert(std::make_pair(featureIndex,pair.first));
            featureIndex +=1;
        }
    }
    return std::make_pair(armaData,featureMap);
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

std::pair<arma::mat, arma::Row<size_t>> repeatDataLabels(arma::mat data, arma::Row<size_t> labels, std::vector<int> importance){
    arma::mat data_new(data.n_rows, 0);
    arma::Row<size_t> labels_new;
    // TODO foreach loop
    for(int c = 0; c < data.n_cols; c++ ) {
        // Our stateIndex is the first row.
        int stateIndex = data.at(0, c); 
        // Repeat the s-a pair as often as the importance of the state
        arma::mat addToMat = arma::repmat(data.col(c), 1, importance[stateIndex]);
        arma::Row<size_t> addToVec = arma::repmat(labels.col(c), 1, importance[stateIndex]); 
        data_new = arma::join_horiz(data_new, addToMat);
        labels_new = arma::join_horiz(labels_new, addToVec);
    }

    // exclude the first row containing only stateIndex information
    arma::mat trainData = data_new.submat(1, 0, data_new.n_rows-1, data_new.n_cols-1);
    return std::make_pair(trainData, labels_new);
}

std::pair<std::pair<arma::mat, arma::Row<size_t>>,std::map<int,std::string>> createTrainingData(std::map<std::string, std::variant<std::vector<int>, std::vector<bool>>>& value_map, std::map<std::string, std::variant<std::vector<int>, std::vector<bool>>>& value_map_submdp, std::vector<int> imps, uint_fast64_t numOfActId){
    auto resAllValues = createMatrixFromValueMap(value_map,numOfActId);
    arma::mat allPairsMat = resAllValues.first;
    auto featureMap = resAllValues.second;
    auto resStrategyValues = createMatrixFromValueMap(value_map_submdp,numOfActId);
    arma::mat strategyPairsMat = resStrategyValues.first;
    arma::Row<size_t> labels = createDataLabels(allPairsMat, strategyPairsMat);
    return std::make_pair(repeatDataLabels(allPairsMat, labels, imps),featureMap);
}