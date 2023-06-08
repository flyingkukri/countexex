#include <storm/api/storm.h>
#include <storm-parsers/api/storm-parsers.h>
#include <storm-parsers/parser/PrismParser.h>
#include <storm/storage/prism/Program.h>
#include <storm/storage/jani/Property.h>
#include <storm/modelchecker/results/CheckResult.h>
#include <storm/modelchecker/results/ExplicitQuantitativeCheckResult.h>
#include <storm/utility/initialize.h>
#include "storm/environment/solver/MinMaxSolverEnvironment.h"

typedef storm::models::sparse::Mdp<double> Mdp;

std::pair<std::shared_ptr<storm::models::sparse::Mdp<double>>, std::vector<std::shared_ptr<storm::logic::Formula const>>> buildModelFormulas(
        std::string const& pathToPrismFile, std::string const& formulasAsString, std::string const& constantDefinitionString = "") {
    std::pair<std::shared_ptr<storm::models::sparse::Mdp<double>>, std::vector<std::shared_ptr<storm::logic::Formula const>>> result;
    storm::prism::Program program = storm::api::parseProgram(pathToPrismFile);
    program = storm::utility::prism::preprocess(program, constantDefinitionString);
    result.second = storm::api::extractFormulasFromProperties(storm::api::parsePropertiesForPrismProgram(formulasAsString, program));
    result.first = storm::api::buildSparseModel<double>(program, result.second)->template as<storm::models::sparse::Mdp<double>>();
    return result;
}

std::vector<storm::modelchecker::CheckTask<storm::logic::Formula, double>> getTasks(
        std::vector<std::shared_ptr<storm::logic::Formula const>> const& formulas) {
    std::vector<storm::modelchecker::CheckTask<storm::logic::Formula, double>> result;
    for (auto const& f : formulas) {
        result.emplace_back(*f);
        result.back().setProduceSchedulers(true);
    }
    return result;
}

bool check(std::string const& path_to_model, std::string const& property_string) {

    // 1. Generate liberal, e-optimal strategy s

    // Generate e-optimal scheduler

    // Assumes that the model is in the prism program language format and parses the program.
    auto program = storm::parser::PrismParser::parse(path_to_model);
    // Code snippet assumes a Mdp
    assert(program.getModelType() == storm::prism::Program::ModelType::MDP);
    // Then parse the properties, passing the program to give context to some potential variables.
    auto properties = storm::api::parsePropertiesForPrismProgram(property_string, program);
    // Translate properties into the more low-level formulae.
    auto formulae = storm::api::extractFormulasFromProperties(properties);

    auto model = storm::api::buildSparseModel<double>(program, formulae)->template as<Mdp>();

    auto modelFormulas = buildModelFormulas(path_to_model, property_string);
    auto tasks = getTasks(modelFormulas.second);
    storm::Environment env;
    env.solver().minMax().setMethod(storm::solver::MinMaxMethod::ValueIteration);
    env.solver().minMax().setPrecision(storm::utility::convertNumber<storm::RationalNumber>(1e-8));

    storm::modelchecker::SparseMdpPrctlModelChecker<storm::models::sparse::Mdp<double>> checker(*model);

    auto result = checker.check(env, tasks[0]);
    assert(result->isExplicitQuantitativeCheckResult());

    // Use that we know that the model checker produces an explicit quantitative result
    auto quantRes = result->asExplicitQuantitativeCheckResult<double>();
    storm::storage::Scheduler<double> const& scheduler = result->asExplicitQuantitativeCheckResult<double>().getScheduler();
    // scheduler.printToStream(std::cout,model);

    // Create liberal strategy

    // 2. Simulate c runs under scheduler to approximate importance

    // 3. Create training data
    // Repeat the samples importance times
    // Label the data: Include state-action pairs from the scheduler as positive examples
    // and others as negative examples

    // 4. DT learning

    return quantRes[*model->getInitialStates().begin()] < 0.01;
}


int main (int argc, char *argv[]) {
    if (argc != 3) {
        std::cout << "Needs exactly 2 arguments: model file and property" << std::endl;
        return 1;
    }

    // Init loggers
    storm::utility::setUp();
    // Set some settings objects.
    storm::settings::initializeAll("storm-starter-project", "storm-starter-project");

    // Call function
    auto result = check(argv[1], argv[2]);
    // And print result
    std::cout << "Result < 0.01 ? " << (result ? "yes" : "no") << std::endl;
}