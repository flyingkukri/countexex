#pragma once

#ifndef STORM_PROJECT_STARTER_MAIN_H
#define STORM_PROJECT_STARTER_MAIN_H

#endif //STORM_PROJECT_STARTER_MAIN_H
#include <cstddef>
#include <string>
#include <storm/models/sparse/Mdp.h>


typedef storm::models::sparse::Mdp<double> Mdp;

typedef struct config{
        double delta;
        int C;
        int l;
        int prec;
} config;

struct DtConfig{
        const double minimumGainSplit; 
        const size_t minimumLeafSize;
        const size_t maximumDepth;
};

/*!
 * Pipeline, performing the following steps:
 * 1. Set up: Build MDP, parse formula, create check task
 * 2. Check task 1: Pmax=? [ F “goal” ]
 * 3. Create check task 2: P≥Pmax [ F “goal” ]
 * 4. Extract permissive strategy σ for check task 2
 * 5. Calculate importance: Simulate c runs on MDP under strategy σ
 * 6. Create training data and train decision tree 
 * 7. Visualize the strategy as decision tree stored in DOT file 
 * @param path_to_model: Path to the model file
 * @param max: Specifies wether Pmax or Pmin will be checked. If set to true, Pmax will be checked.
 * @param conf: Struct containing configuration information for the importance and safety property calculation
 * @param dtConfig: Struct containing information about the decision tree tuning parameters
 * @param verbose: If set to true, prints additional output during the program execution
 */
void pipeline(std::string const& path_to_model, bool max, config  const& conf, DtConfig& dtConfig, bool verbose);