
#include <random>
#include <storm/api/storm.h>
#include <storm-parsers/api/storm-parsers.h>
#include <storm-parsers/parser/PrismParser.h>
#include <storm/storage/prism/Program.h>
#include <storm/storage/sparse/PrismChoiceOrigins.h>
#include <storm-parsers/parser/FormulaParser.h>
#include "storm/environment/solver/MinMaxSolverEnvironment.h"
#include <array>
#include <storm-permissive/analysis/PermissiveSchedulers.h>
#include <storm/builder/ExplicitModelBuilder.h>
#include <storm/environment/Environment.h>
#include <storm/models/sparse/Mdp.h>
#include <storm/modelchecker/prctl/SparseMdpPrctlModelChecker.h>
#include <bits/stdc++.h>
//#include "main.h"
#include "buildModel.h"
#include <iostream>



std::pair<std::shared_ptr<storm::models::sparse::Mdp<double>>, std::vector<std::shared_ptr<storm::logic::Formula const>>> buildModelFormulas(
        std::string const& pathToPrismFile, std::string const& formulasAsString, std::string const& constantDefinitionString) {
    std::pair<std::shared_ptr<storm::models::sparse::Mdp<double>>, std::vector<std::shared_ptr<storm::logic::Formula const>>> result;
    storm::prism::Program program = storm::api::parseProgram(pathToPrismFile);
    program = storm::utility::prism::preprocess(program, constantDefinitionString);

    // Parse formulas
    result.second = storm::api::extractFormulasFromProperties(storm::api::parsePropertiesForPrismProgram(formulasAsString, program));

    // Options for model builder: gain information about states and actions
    storm::generator::NextStateGeneratorOptions options(result.second);
    options.setBuildChoiceLabels(true);
    options.setBuildStateValuations(true);
    options.setBuildObservationValuations(true);
    options.setBuildAllLabels(true);
    options.setBuildChoiceOrigins(true);

    // Build the model
    result.first = storm::builder::ExplicitModelBuilder<double>(program, options).build()->as<storm::models::sparse::Mdp<double>>();

    return result;
}

std::vector<storm::modelchecker::CheckTask<storm::logic::Formula, double>> getTasks(
        std::vector<std::shared_ptr<storm::logic::Formula const>> const& formulas) {
    std::vector<storm::modelchecker::CheckTask<storm::logic::Formula, double>> result;
    for (auto const& f : formulas) {
        result.emplace_back(*f);
        // Enable scheduler production for check tasks
        result.back().setProduceSchedulers(true);
    }
    return result;
}

template <typename MdpType>
void modelVis(std::shared_ptr<MdpType>& model){
    model->printModelInformationToStream(std::cout);
    auto trans_M = model->getTransitionMatrix();
    trans_M.printAsMatlabMatrix(std::cout);
}

storm::Environment setUpEnv(){
    // Set up environment: solver method and precision
    storm::Environment env;
    env.solver().minMax().setMethod(storm::solver::MinMaxMethod::ValueIteration);
    env.solver().minMax().setPrecision(storm::utility::convertNumber<storm::RationalNumber>(1e-8));
    return env;
}

std::string generateSafetyProperty(std::string const& formulasString, double initStateCheckResult){
    std::string reach_obj_substr = formulasString.substr(formulasString.find('['));

    // TODO: how to setprecision?
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(16) << initStateCheckResult;
    // Create safetyProperty from initStateCheckResult and reachability formula to generate only strategies that are as good as the e-optimal one
    std::string safetyProp = "P>=" + oss.str() + " " + reach_obj_substr;
    std::cout << safetyProp << std::endl;
    return safetyProp;
}

std::pair<std::shared_ptr<storm::models::sparse::Mdp<double>>, storm::logic::ProbabilityOperatorFormula> buildModelForSafetyProperty(std::string const& pathToModel, std::string& safetyProp){
    // Build new model with safety property
    auto program = storm::parser::PrismParser::parse(
            pathToModel);
    storm::parser::FormulaParser formulaParser(program);
    auto formulas = formulaParser.parseFromString(safetyProp);
    auto const& formula = formulas[0].getRawFormula()->asProbabilityOperatorFormula();
    storm::generator::NextStateGeneratorOptions options(formula);
    options.setBuildChoiceLabels(true);
    options.setBuildStateValuations(true);
    options.setBuildObservationValuations(true);
    options.setBuildAllLabels(true);
    options.setBuildChoiceOrigins(true);
    std::shared_ptr<storm::models::sparse::Mdp<double>> mdp =
            storm::builder::ExplicitModelBuilder<double>(program, options).build()->as<storm::models::sparse::Mdp<double>>();
    return std::make_pair(mdp,formula);
}