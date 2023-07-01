#pragma once

#ifndef STORM_PROJECT_STARTER_MAIN_H
#define STORM_PROJECT_STARTER_MAIN_H

#endif //STORM_PROJECT_STARTER_MAIN_H


typedef storm::models::sparse::Mdp<double> Mdp;


/*!
 * Build the model and the formulas
 */
std::pair<std::shared_ptr<storm::models::sparse::Mdp<double>>, std::vector<std::shared_ptr<storm::logic::Formula const>>> buildModelFormulas(
        std::string const& pathToPrismFile, std::string const& formulasAsString, std::string const& constantDefinitionString = "");

/*!
 * Set up tasks for the model checking: here, produce a scheduler
 */
std::vector<storm::modelchecker::CheckTask<storm::logic::Formula, double>> getTasks(
        std::vector<std::shared_ptr<storm::logic::Formula const>> const& formulas);

/*!
 * Visualize a model: state & choice labels, transition matrix ...
 */
template <typename MdpType>
void model_vis(std::shared_ptr<MdpType>& model);

template <typename MdpType>
void print_state_act_pairs(std::shared_ptr<MdpType>& mdp);

template <typename MdpType>
std::map<std::string, std::variant<std::vector<int>, std::vector<bool>, std::vector<storm::RationalNumber>>> create_state_act_pairs(std::shared_ptr<MdpType>& mdp);

/*!
 * Produces the dt_pipeline with input:
 * @param path_to_model, property_string
 * It performs the following steps:
 * 1. Compute e-optimal, liberal strategy
 * 2. Simulate c runs on MDP under the strategy
 * 3. Visualize the strategy as decision tree using DT learning
 * @return DOT file of the dt strategy representation
 */
bool pipeline(std::string const& path_to_model, std::string const& property_string);