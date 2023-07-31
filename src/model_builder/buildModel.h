#pragma once
#include <storm/api/storm.h>
#include <storm-parsers/api/storm-parsers.h>

#include <storm/storage/prism/Program.h>
#include <storm/storage/sparse/PrismChoiceOrigins.h>

#include "storm/environment/solver/MinMaxSolverEnvironment.h"
#include <array>
#include <storm-permissive/analysis/PermissiveSchedulers.h>
#include <storm/builder/ExplicitModelBuilder.h>
#include <storm/environment/Environment.h>
#include <storm/models/sparse/Mdp.h>
#include <storm/modelchecker/prctl/SparseMdpPrctlModelChecker.h>
#include <string>
#include <map>
#include <vector>
#include <memory>


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
void modelVis(std::shared_ptr<MdpType>& model){
    model->printModelInformationToStream(std::cout);
    auto trans_M = model->getTransitionMatrix();
    trans_M.printAsMatlabMatrix(std::cout);
}

/*!
 * Set up the environment for the model checking
 */
storm::Environment setUpEnv();


std::string generateSafetyProperty(std::string const& formulasString, double initStateCheckResult);

std::pair<std::shared_ptr<storm::models::sparse::Mdp<double>>, storm::logic::ProbabilityOperatorFormula> buildModelForSafetyProperty(std::string const& pathToModel, std::string const& safetyProp);