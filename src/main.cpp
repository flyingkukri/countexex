#include <cstdint>
#include <random>
#include <storm/api/storm.h>
#include <storm-parsers/api/storm-parsers.h>
#include <storm-parsers/parser/PrismParser.h>
#include <storm/storage/prism/Program.h>
#include <storm/storage/jani/Property.h>
#include <storm/modelchecker/results/CheckResult.h>
#include <storm/modelchecker/results/ExplicitQuantitativeCheckResult.h>
#include <storm/utility/initialize.h>
#include "storm/environment/solver/MinMaxSolverEnvironment.h"
#include <storm/simulator/DiscreteTimeSparseModelSimulator.h>
#include <storm/simulator/PrismProgramSimulator.h>
#include <storm/builder/BuilderOptions.h>
#include <storm/storage/Scheduler.h>
#include "storm/models/sparse/Model.h"

#undef As
// Define these to print extra informational output and warnings.
#include <mlpack/methods/decision_tree/decision_tree.hpp>
#define MLPACK_PRINT_INFO
#define MLPACK_PRINT_WARN
#include <mlpack.hpp>

using namespace arma;
using namespace mlpack;
using namespace mlpack::tree;
using namespace std;


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


void simulateRun(storm::simulator::DiscreteTimeSparseModelSimulator<double> simulator, storm::storage::Scheduler<double> scheduler, int *imps) {
    //create a random number generator for the liberal strategy
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(0.0, 1.0);

    while(true){ // break if we reach F or we don't have any choices left in our current state
         auto state = simulator.getCurrentState();
         auto choices = scheduler.getChoice(state);
         uint_fast64_t next;
         if (choices.isDeterministic()) {
            next = choices.getDeterministicChoice();
         } else {
            auto dist = choices.getChoiceAsDistribution();
            dist.normalize(); // normalize so that we can sample a number between 1 and 0
            int quantile = dis(gen);
            next = dist.sampleFromDistribution(quantile); // This returns a statevalue...
         }
        simulator.step(next);
    }
}

void testType(storm::models::sparse::Model<double> const& model) {
    return;
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

//    storm::generator::NextStateGeneratorOptions options = new storm::builder::BuilderOptions();
//    storm::simulator::DiscreteTimePrismProgramSimulator<double> simulator(program, options);


     // Our termination condition is that the property holds (we reach the target)
     // TODO when should we terminate if it doesn't hold? If we reach a sink state?
     // maybe we can log which actions we have already chosen and never take the same one twice.
     
    const int C = 10000;
    int states = model->getNumberOfStates();
    int *imps = (int*) malloc(states * sizeof(int));

    auto model2 = model.get();
    storm::simulator::DiscreteTimeSparseModelSimulator<double> simulator(*model);

    for (int i =0; i++; i < C){
       simulateRun(simulator, scheduler, imps);
    }
    
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