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
std::map<std::string, std::variant<std::vector<int>, std::vector<bool>, std::vector<storm::RationalNumber>>> createStateActPairs(std::shared_ptr<MdpType>& mdp, std::vector<int> imps){
    auto val = mdp->getStateValuations();
    std::map<std::string, std::variant<std::vector<int>, std::vector<bool>, std::vector<storm::RationalNumber>>> value_map;

    for(int i=0; i<mdp->getNumberOfStates(); ++i){
        auto a_count = mdp->getNumberOfChoices(i);
        for(int k=0;k<a_count;++k){
            auto l = val.at(i);
            auto start = l.begin();
            while(start!=l.end()){
                auto key = start.getVariable().getName();
                auto it = value_map.find(key);
                // add the id of the current element
                if( it == value_map.end()){
                    value_map.insert(std::make_pair(key, std::vector<int>{i}));
                }else{
                    auto& vector = std::get<std::vector<int>>(it->second);
                    vector.push_back(i);
                }
                if(start.getVariable().hasBooleanType()){
                    auto e = start.getBooleanValue();
                    if( it == value_map.end()){
                        value_map.insert(std::make_pair(key, std::vector<bool>{e}));
                    }else{
                        auto& vector = std::get<std::vector<bool>>(it->second);
                        vector.push_back(e);
                    }
                }else if(start.getVariable().hasIntegerType()){
                    auto e = start.getIntegerValue();
                    if(it == value_map.end()){
                        std::vector<int> int_vector;
                        int_vector.push_back(e);
                        value_map.insert(std::make_pair(key, int_vector));
                    }else{
                        auto& vector = std::get<std::vector<int>>(it->second);
                        vector.push_back(e);                    }
                }else if(start.getVariable().hasRationalType()){
                    auto e = start.getRationalValue();
                    if(it == value_map.end()){
                        std::vector<storm::RationalNumber> rat_vector;
                        rat_vector.push_back(e);
                        value_map.insert(std::make_pair(key, rat_vector));
                    }else{
                        auto& vector = std::get<std::vector<storm::RationalNumber>>(it->second);
                        vector.push_back(e);
                    }
                }
                start.operator++();
            }
            // TODO: was, wenn eine Variable aus dem PRISM code "action" heißt?
            auto key = "action";
            auto elem = mdp->getChoiceOrigins()->getIdentifier(mdp->getTransitionMatrix().getRowGroupIndices()[i]+k);
            auto it = value_map.find(key);
            if(it == value_map.end()){
                std::vector<int> int_vector;
                int_vector.push_back(elem);
                value_map.insert(std::make_pair(key, int_vector));
            }else{
                auto& vector = std::get<std::vector<int>>(it->second);
                vector.push_back(elem);
            }
        
        }
    }
    return value_map;
}

arma::mat createMatrixFromValueMap(
        std::map<std::string, std::variant<std::vector<int>, std::vector<bool>, std::vector<storm::RationalNumber>>> &value_map);

arma::Row<size_t> createDataLabels(arma::mat &allPairs, arma::mat &strategyPairs);

std::pair<arma::mat, arma::Row<size_t>> createTrainingData(std::map<std::string, std::variant<std::vector<int>, std::vector<bool>, std::vector<storm::RationalNumber>>>& valueMap, std::map<std::string, std::variant<std::vector<int>, std::vector<bool>, std::vector<storm::RationalNumber>>>& valueMapSubmdp, std::vector<int> importance);

std::pair<arma::mat, arma::Row<size_t>> repeatDataLabels(arma::mat data, arma::Row<size_t> labels, std::vector<int> importance);
