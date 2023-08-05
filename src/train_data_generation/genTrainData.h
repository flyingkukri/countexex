#pragma once
#include <random>
#include <storm/utility/initialize.h>
#include <storm/api/storm.h>
#include <storm-parsers/api/storm-parsers.h>
#include <storm-parsers/parser/PrismParser.h>
#include <storm/storage/prism/Program.h>
#include <storm/storage/sparse/PrismChoiceOrigins.h>
#include <storm/modelchecker/results/CheckResult.h>
#include <storm/modelchecker/results/ExplicitQuantitativeCheckResult.h>
#include <storm/utility/initialize.h>
#include "storm/environment/solver/MinMaxSolverEnvironment.h"
#include <array>
#include <storm-parsers/parser/FormulaParser.h>
#include <storm-permissive/analysis/PermissiveSchedulers.h>
#include <storm/builder/ExplicitModelBuilder.h>
#include <storm/environment/Environment.h>
#include <storm/models/sparse/Mdp.h>
#include <storm/modelchecker/prctl/SparseMdpPrctlModelChecker.h>
#include <storm/models/sparse/StandardRewardModel.h>
#include <storm-permissive/analysis/PermissiveSchedulerPenalty.h>
#include <storm-permissive/analysis/PermissiveSchedulers.h>
#include <bits/stdc++.h>
#include <iostream>
#undef As
#include <mlpack.hpp>
#include <mlpack/core.hpp>
#include <armadillo>
#include <string>
#include <map>
#include <vector>
#include <memory>
#include <variant>

/*! 
 * This struct contains information about the mdp
 */
typedef struct {
        /* This maps the feature id to its name*/
        std::map<int,std::string> featureMap; 
        /* This maps the action id to its name*/
        std::map<int, std::string> identifierActionMap;
        /* A vector containing the importance for each state */
        std::vector<int> imps;
        /* The number of action ids*/
        int numOfActId;
} MdpInfo;

/*!
 * This data structure is our representation of the state-action pairs. 
 * it is a map from a string that is either 
 * 1. the name of a variable (or dimension in mlpack)
 * 2. "action"
 * 3. "imps"
 * 
 * to a vector of values. 
 * The cartesian product of the n-th entry of the vectors for every key and "action" constitute a state-action pair.
 * We further add the vector imps which is the id of the state, so that we can later repeat this state-action pair as often as needed.
*/
typedef std::map<std::string, std::variant<std::vector<int>, std::vector<bool>>> ValueMap;

/*!
 * Print the state-action pairs of the mdp
 * @param mdp: the mdp for which we want to print the state-action pairs
*/
template <typename MdpType>
void printStateActPairs(std::shared_ptr<MdpType>& mdp){
    auto val = mdp->getStateValuations();
    std::cout << "Print (s-a)-pairs: " << std::endl;
    for(int i=0; i<mdp->getNumberOfStates(); ++i){
        auto choices = mdp->getNumberOfChoices(i);
        for(int k=0; k < choices; ++k){
            auto it = val.at(i);
            auto start = it.begin();
            while(start != it.end()){
                // TODO: extend to other datatypes: bool, double ... aber nur für uns also nicht unbedingt nötig
                std::cout << start.getVariable().getName() << ": " << start.getIntegerValue() << " ";
                start.operator++();
            }
            // This info should be present in the dt in the end and is represented via mdp->getChoiceOrigins()->getIdentifier(mdp->getTransitionMatrix().getRowGroupIndices()[i]+k);
            auto choiceInfo = mdp->getChoiceOrigins()->getChoiceInfo(mdp->getTransitionMatrix().getRowGroupIndices()[i] + k);
            std::cout << choiceInfo << " ";
            std::cout << "\n" << std::endl;
        }
    }
}

/*!
 * Create a ValueMap from the mdp
 * @param mdp: the mdp for which we want extract the state-action pairs
 * @param mdpInfo: the struct containing information about the mdp (we will fill the actionIdentifierMap)
*/
template <typename MdpType>
ValueMap createStateActPairs(std::shared_ptr<MdpType>& mdp, MdpInfo& mdpInfo){
    auto val = mdp->getStateValuations();
    ValueMap valueMap;

    for(int i=0; i<mdp->getNumberOfStates(); ++i){
        auto choices = mdp->getNumberOfChoices(i);
        for(int k=0; k < choices; ++k){
            auto stateValuation = val.at(i);
            auto varIt = stateValuation.begin();
            while(varIt != stateValuation.end()){
                auto key = varIt.getVariable().getName();
                auto it = valueMap.find(key);
                if(varIt.getVariable().hasBooleanType()){
                    auto varValue = varIt.getBooleanValue();
                    if( it == valueMap.end()){
                        valueMap.insert(std::make_pair(key, std::vector<bool>{varValue}));
                    }else{
                        auto& vector = std::get<std::vector<bool>>(it->second);
                        vector.push_back(varValue);
                    }
                }else if(varIt.getVariable().hasIntegerType()){
                    auto varValue = varIt.getIntegerValue();
                    if(it == valueMap.end()){
                        std::vector<int> int_vector;
                        int_vector.push_back(varValue);
                        valueMap.insert(std::make_pair(key, int_vector));
                    }else{
                        auto& vector = std::get<std::vector<int>>(it->second);
                        vector.push_back(varValue);                    
                    }
                }
                varIt.operator++();
            }
            auto key = "action";
            auto actionIdentifier = mdp->getChoiceOrigins()->getIdentifier(mdp->getTransitionMatrix().getRowGroupIndices()[i] + k);
            // insert key to valueMap
            auto it = valueMap.find(key);
            if(it == valueMap.end()){
                std::vector<int> int_vector;
                int_vector.push_back(actionIdentifier);
                valueMap.insert(std::make_pair(key, int_vector));
            }else{
                auto& vector = std::get<std::vector<int>>(it->second);
                vector.push_back(actionIdentifier);
            }
            // insert actionIdentifer together with actionInfo to identifierActionMap
            auto itIdent = mdpInfo.identifierActionMap.find(actionIdentifier);
            if(itIdent == mdpInfo.identifierActionMap.end()){
                mdpInfo.identifierActionMap.insert(std::make_pair(actionIdentifier, mdp->getChoiceOrigins()->getChoiceInfo(mdp->getTransitionMatrix().getRowGroupIndices()[i] + k)));
            }
            // Create additional key-vector pair imps: to indicate for each s-a pair to which state id it belongs 
            key = "imps";
            it = valueMap.find(key);
            if( it == valueMap.end()){
                valueMap.insert(std::make_pair(key, std::vector<int>{i}));
            }else{
                auto& vector = std::get<std::vector<int>>(it->second);
                vector.push_back(i);
            }
        }
    }
    
    return valueMap;
}

/*!
 * we add a new row to the bottom of the matrix
 * @param armaData: the arma::fmat to which we will add a rowVec
 * @param rowVec: a dummy vector that will contain the valueVector but converted to double
 * @param valueVector: the vector that contains the values that we want to add to the matrix
 * 
*/
void createMatrixHelper(arma::fmat& armaData, arma::frowvec& rowVec, std::variant<std::vector<int>, std::vector<bool>>& valueVector);


/*!
 * We add an n x n identity matrix to the bottom of the matrix, where n is mdpInfo.numOfActId
 * @param armaData: the arma::fmat to which we will add an identity matrix  
 * @param mdpInfo: the MdpInfo object that contains the featureMap; the dimension of each row that we add will be called "action"
 * @param valueVector: the vector that contains the action identifiers
*/
void categoricalFeatureOneHotEncoding(arma::fmat& armaData, MdpInfo& mdpInfo, std::variant<std::vector<int>, std::vector<bool>>& valueVector);

/*!
 * We create a matrix from the valueMap. We will use mdpInfo.imps to repeat data points.
 * Further we will fill out mdpInfo.featureMap an mdpInfo.actionIdentifierMap
 * Warning: This function will modify the value map!
 * @param valueMap: map containing the variable names as keys and the corresponding vectors as values
 * @param mdpInfo: struct containing information about the MDP; we will add the feature names to the featureMap
 * @return: matrix containing the data points of the MDP (see data in develop.md for details)
 */
arma::fmat createMatrixFromValueMap(std::map<std::string, std::variant<std::vector<int>, std::vector<bool>>>& valueMap, MdpInfo& mdpInfo);

/*!
 * Create Labels for the allPairs matrix. 1 if the pair is in the strategy, 0 otherwise
 * @param allPairs: matrix containing the s-a pairs of the MDP
 * @param strategyPairs: matrix containing the s-a pairs of the strategy
 * @return: vector containing the labels for the s-a pairs (1 if the pair is in the strategy, 0 otherwise)
 */
arma::Row<size_t> createDataLabels(arma::fmat &allPairs, arma::fmat &strategyPairs);

/*! 
 * Repeat the data points according to the importance of the state (and remove the importance row)
 * @param data: matrix containing the s-a pairs
 * @param labels: vector containing the labels for the s-a pairs
 * @param mdpInfo: struct containing struct.importance for the importance of each state
 * @return: pair of matrices: first matrix contains the s-a pairs repeated as often as the importance of the state.
 *          the row vector contains the labels for the s-a pairs also repeated as often as the importance of the state.
 *
 */
std::pair<arma::fmat, arma::Row<size_t>> repeatDataLabels(arma::fmat data, arma::Row<size_t> labels, const MdpInfo& mdpInfo);

/*!
 * The main function to create the training data from the value map
 * Warning: This function will change the valueMap!
 * @param valueMap The value map containing all the state-action pairs
 * @param valueMapSubMdp The value map containing the state-action pairs of the strategy
 * @param mdpInfo The MdpInfo object containing the information about the MDP
 * @return A pair of the training data and the labels.
 */
std::pair<arma::fmat, arma::Row<size_t>> createTrainingData(ValueMap& valueMap, ValueMap& value_map_submdp, MdpInfo& mdpInfo);
