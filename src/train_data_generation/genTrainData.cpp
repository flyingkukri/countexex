#include "genTrainData.h"
#include <random>
#include <storm/utility/initialize.h>
#include <array>
#include <bits/stdc++.h>
#include <armadillo>

void createMatrixHelper(arma::fmat &armaData, arma::frowvec &rowVec, ValueVector &valueVector)
{
    // Create an arma::frowvec from the vector
    if (const auto intVector = std::get_if<std::vector<int>>(&valueVector))
    {
        rowVec = arma::conv_to<arma::frowvec>::from(*intVector);
    }
    else if (const auto boolVector = std::get_if<std::vector<bool>>(&valueVector))
    {
        std::vector<int> boolToIntVector;
        boolToIntVector.reserve(boolVector->size());
        for (bool value : *boolVector)
        {
            boolToIntVector.push_back(value ? 1 : 0);
        }
        rowVec = arma::conv_to<arma::frowvec>::from(boolToIntVector);
    }
    armaData = arma::join_vert(armaData, rowVec);
}

void categoricalFeatureOneHotEncoding(arma::fmat &armaData, MdpInfo &mdpInfo, ValueVector &valueVector)
{
    // We know that the variant holds an int vector in this case as it contains the action identifiers
    if (const auto intVector = std::get_if<std::vector<int>>(&valueVector))
    {
        auto nCols = (*intVector).size();
        // Create a feature row in the matrix for every actionIdentifier: e.g. if there are 10 different actions we add 10 feature rows to the matrix
        for (int i = 0; i < mdpInfo.numOfActId; ++i)
        {
            arma::Row<float> rowVec = arma::zeros<arma::Row<float>>(nCols);
            armaData = arma::join_vert(armaData, rowVec);
            // Indicate for each row i that it represents an action feature
            mdpInfo.featureMap.insert(std::make_pair(i, "action"));
        }


        // intVector: each entry i corresponds to the actionIdentifier of the i-th data point
        // we thus set the entry of the row that corresponds to that actionIdentifier to 1 for the i-th data point
        for (int i = 0; i < nCols; ++i)
        {
            // As we store the stateIndex in the first row temporarily we add +1 to access a row i logically
            armaData.at((*intVector).at(i) + 1, i) = 1;

        }

    } else {
        throw std::invalid_argument("valueVector does not contain an int vector");
    }
}

arma::fmat createMatrixFromValueMap(ValueMap &valueMap, MdpInfo &mdpInfo)
{
    arma::fmat armaData;
    arma::frowvec rowVec;
    std::string imps = "imps";
    std::string act = "action";

    // Every row of the resulting matrix corresponds to a feature
    // Create a mapping between variable names and row numbers(==feature number)

    // Make sure imps is the first row in the matrix
    auto it = valueMap.find(imps);
    if (it != valueMap.end())
    {
        std::variant<std::vector<int>, std::vector<bool>> valueVector = it->second;
        createMatrixHelper(armaData, rowVec, valueVector);
        // No entry in featureMap for imps as this row will be removed from the matrix for training
    } else {
        throw std::invalid_argument("value_map does not have any state id's (a key \"imps\")");
    } 

    // One-hot encoding for the categorical action features
    it = valueMap.find(act);
    if (it != valueMap.end())
    {
        std::variant<std::vector<int>, std::vector<bool>> valueVector = it->second;
        categoricalFeatureOneHotEncoding(armaData, mdpInfo, valueVector);
    }

    auto featureIndex = mdpInfo.numOfActId;
    // Loop over all other key-value pairs
    for (const auto &pair : valueMap)
    {
        // Get the vector corresponding to the key
        if (pair.first != "imps" && pair.first != "action")
        {
            std::variant<std::vector<int>, std::vector<bool>> valueVector = pair.second;
            createMatrixHelper(armaData, rowVec, valueVector);
            mdpInfo.featureMap.insert(std::make_pair(featureIndex, pair.first));
            featureIndex += 1;
        }
    }
    return armaData;
}

arma::Row<size_t> createDataLabels(arma::fmat &allPairs, arma::fmat &strategyPairs)
{
    if (allPairs.n_rows != strategyPairs.n_rows) {
        throw std::invalid_argument("allPairs.n_rows != strategyPairs.n_rows");
    }
    // Column-major in arma: thus each column represents a data point
    size_t numColumns = allPairs.n_cols;
    arma::Row<size_t> labels(numColumns, arma::fill::zeros);
    for (size_t i = 0; i < numColumns; ++i)
    {
        // Get the i-th column of allPairs
        auto found = 0;
        arma::fvec column = allPairs.col(i);
        for (size_t j = 0; j < strategyPairs.n_cols; ++j)
        {
            arma::fvec col2Cmp = strategyPairs.col(j);
            if (arma::all(col2Cmp == column))
            {
                found = 1;
                labels(i) = 1;
                break;
            }
        }
        if (!found)
        {
            labels(i) = 0;
        }
    }
    return labels;
}

TrainData repeatDataLabels(arma::fmat data, arma::Row<size_t> labels, const MdpInfo &mdpInfo) 
{
    if (data.n_cols != labels.n_cols) {
        throw std::invalid_argument("data.n_cols != labels.n_cols");
    }
    arma::fmat dataNew(data.n_rows, 0);
    arma::Row<size_t> labelsNew;
    for (int c = 0; c < data.n_cols; ++c)
    {
        // Our stateIndex is the first row.
        int stateIndex = data.at(0, c);
        // Repeat the s-a pair as often as the importance of the state
        arma::fmat addToMat = arma::repmat(data.col(c), 1, mdpInfo.imps[stateIndex]);
        arma::Row<size_t> addToVec = arma::repmat(labels.col(c), 1, mdpInfo.imps[stateIndex]);
        dataNew = arma::join_horiz(dataNew, addToMat);
        labelsNew = arma::join_horiz(labelsNew, addToVec);
    }

    // Exclude the first row containing only stateIndex information
    arma::fmat trainData = dataNew.submat(1, 0, dataNew.n_rows - 1, dataNew.n_cols - 1);
    return std::make_pair(trainData, labelsNew);
}

TrainData createTrainingData(ValueMap &valueMap, ValueMap &valueMapSubMdp, MdpInfo &mdpInfo)
{
    arma::fmat allPairsMat = createMatrixFromValueMap(valueMap, mdpInfo);
    arma::fmat strategyPairsMat = createMatrixFromValueMap(valueMapSubMdp, mdpInfo);
    arma::Row<size_t> labels = createDataLabels(allPairsMat, strategyPairsMat);
    return repeatDataLabels(allPairsMat, labels, mdpInfo);
}