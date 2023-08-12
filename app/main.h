#pragma once

#ifndef STORM_PROJECT_STARTER_MAIN_H
#define STORM_PROJECT_STARTER_MAIN_H

#endif //STORM_PROJECT_STARTER_MAIN_H
#include <cstddef>
#include <string>
#include <storm/models/sparse/Mdp.h>
#include "genTrainData.h"


typedef storm::models::sparse::Mdp<double> Mdp;

/*! 
*  This struct contains configuration information for the importance and safety property calculation
*/
typedef struct config{
        /* We will not consider importance values being smaller than delta*c */
        double delta;
        /* c represents the number of simulation runs*/
        int c;
        /* l represents the maximal number of simulation steps we perform in each simulation run*/
        int l;
        /* Precision for the safety property bound*/
        int prec;
} config;

/*! 
*  This struct contains information about the decision tree tuning parameterss
*/
struct DtConfig{
        double minimumGainSplit; 
        size_t minimumLeafSize;
        size_t maximumDepth;
};

/*! 
*  This struct contains the command line options
*/
struct Options{
        /* Path to the model file.*/
        std::string pathToModel;
        /* Specifies whether Pmax or Pmin will be checked. If set to true, Pmax will be checked.*/
        bool propertyMax;
        /* If set to true, prints additional output during the program execution*/
        bool verbose;
        /* Decides on permissive strategy computation: if set to true SMT is used, else MILP*/
        bool optimizerSMT;
        /* Struct containing configuration information for the importance and safety property calculation*/
        config conf;
        /* Struct containing information about the decision tree tuning parameters*/
        DtConfig dtConfig;
};

/*!
*  Helper to initialize the CL options
* @param argc: arguments from main function
* @param argv: arguments from main function
* @param clOptions: struct containing the command line options
* @return: 1 or 0 indicating whether the setup was successful or not
*/
int initializeOptions(int argc, char *argv[], Options& clOptions);

/*!
*  Helper to compute the importance values and create the value maps
* @param permissiveScheduler: permissive scheduler, either computed via SMT or MILP
* @param clOptions: struct containing the command line options
* @param label: target states label
* @param safetyMdp: MDP, from which the valueMap is constructed
* @return: a pair containing the valueMap of the safety mdp and the submdp, which results from applying the permissive scheduler 
*/
std::pair<ValueMap, ValueMap> createValueMapHelper(boost::optional<storm::ps::SubMDPPermissiveScheduler<>>& permissiveScheduler, Options& clOptions, MdpInfo& mdpInfo, std::string& label, std::shared_ptr<storm::models::sparse::Mdp<double>>& safetyMdp);

/*!
 * Pipeline, performing the following steps:
 * 1. Set up: Build MDP, parse formula, create check task
 * 2. Check task 1: Pmax=? [ F “goal” ]
 * 3. Create check task 2: P≥Pmax [ F “goal” ]
 * 4. Extract permissive strategy σ for check task 2
 * 5. Calculate importance: Simulate c runs on MDP under strategy σ
 * 6. Create training data and train decision tree 
 * 7. Visualize the strategy as decision tree stored in DOT file 
 * @param clOptions: struct containing the command line options
 */
void pipeline(Options& clOptions);