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
 * Produces the dt_pipeline with input:
 * @param path_to_model, property_string
 * It performs the following steps:
 * 1. Compute e-optimal, liberal strategy
 * 2. Simulate c runs on MDP under the strategy
 * 3. Visualize the strategy as decision tree using DT learning
 * @return DOT file of the dt strategy representation
 */
bool pipeline(std::string const& path_to_model, bool max, config  const& conf, DtConfig& dtConfig, bool verbose);