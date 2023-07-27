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

template <typename MdpType>
std::map<std::string, std::variant<std::vector<int>, std::vector<bool>, std::vector<storm::RationalNumber>>> createStateActPairs(std::shared_ptr<MdpType>& mdp){
    auto val = mdp->getStateValuations();
    std::map<std::string, std::variant<std::vector<int>, std::vector<bool>, std::vector<storm::RationalNumber>>> value_map;

    for(int i=0; i<mdp->getNumberOfStates(); ++i){
        auto choices = mdp->getNumberOfChoices(i);
        for(int k=0; k < choices; ++k){
            auto stateValuation = val.at(i);
            auto varIt = stateValuation.begin();
            while(varIt != stateValuation.end()){
                auto key = varIt.getVariable().getName();
                auto it = value_map.find(key);
                if(varIt.getVariable().hasBooleanType()){
                    auto varValue = varIt.getBooleanValue();
                    if( it == value_map.end()){
                        value_map.insert(std::make_pair(key, std::vector<bool>{varValue}));
                    }else{
                        auto& vector = std::get<std::vector<bool>>(it->second);
                        vector.push_back(varValue);
                    }
                }else if(varIt.getVariable().hasIntegerType()){
                    auto varValue = varIt.getIntegerValue();
                    if(it == value_map.end()){
                        std::vector<int> int_vector;
                        int_vector.push_back(varValue);
                        value_map.insert(std::make_pair(key, int_vector));
                    }else{
                        auto& vector = std::get<std::vector<int>>(it->second);
                        vector.push_back(varValue);                    }
                }else if(varIt.getVariable().hasRationalType()){
                    auto varValue = varIt.getRationalValue();
                    if(it == value_map.end()){
                        std::vector<storm::RationalNumber> rat_vector;
                        rat_vector.push_back(varValue);
                        value_map.insert(std::make_pair(key, rat_vector));
                    }else{
                        auto& vector = std::get<std::vector<storm::RationalNumber>>(it->second);
                        vector.push_back(varValue);
                    }
                }
                varIt.operator++();
            }
            // TODO: was, wenn eine Variable aus dem PRISM code "action" heißt?
            auto key = "action";
            auto actionIdentifier = mdp->getChoiceOrigins()->getIdentifier(mdp->getTransitionMatrix().getRowGroupIndices()[i] + k);
            auto it = value_map.find(key);
            if(it == value_map.end()){
                std::vector<int> int_vector;
                int_vector.push_back(actionIdentifier);
                value_map.insert(std::make_pair(key, int_vector));
            }else{
                auto& vector = std::get<std::vector<int>>(it->second);
                vector.push_back(actionIdentifier);
            }
            // Create additional key-vector pair imps: to indicate for each s-a pair to which state id it belongs 
            key = "imps";
            it = value_map.find(key);
            if( it == value_map.end()){
                value_map.insert(std::make_pair(key, std::vector<int>{i}));
            }else{
                auto& vector = std::get<std::vector<int>>(it->second);
                vector.push_back(i);
            }
        }
    }
    return value_map;
}

void createMatrixHelper(arma::mat& armaData, arma::rowvec& rowVec, std::variant<std::vector<int>, std::vector<bool>, std::vector<storm::RationalNumber>>& valueVector);

arma::mat createMatrixFromValueMap(
        std::map<std::string, std::variant<std::vector<int>, std::vector<bool>, std::vector<storm::RationalNumber>>> &value_map);

arma::Row<size_t> createDataLabels(arma::mat &allPairs, arma::mat &strategyPairs);

std::pair<arma::mat, arma::Row<size_t>> createTrainingData(std::map<std::string, std::variant<std::vector<int>, std::vector<bool>, std::vector<storm::RationalNumber>>>& valueMap, std::map<std::string, std::variant<std::vector<int>, std::vector<bool>, std::vector<storm::RationalNumber>>>& valueMapSubmdp, std::vector<int> importance);

std::pair<arma::mat, arma::Row<size_t>> repeatDataLabels(arma::mat data, arma::Row<size_t> labels, std::vector<int> importance);
