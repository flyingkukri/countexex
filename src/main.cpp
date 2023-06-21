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
#include <storm/modelchecker/prctl/SparseMdpPrctlModelChecker.h>
#include <storm/models/sparse/StandardRewardModel.h>
#include <bits/stdc++.h>
#include "main.h"
#include <iostream>
#include <iomanip>
#include <sstream>

//#include <mlpack.hpp>
//#include <mlpack/core.hpp>
//#include <mlpack/core/data/load.hpp>
//#include <mlpack/core/data/save.hpp>
//#include <mlpack/methods/decision_tree/decision_tree.hpp>
//#include <armadillo>


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
    options.setBuildChoiceOrigins();

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

void model_vis(std::shared_ptr<storm::models::sparse::Mdp<double>>& model){
    model->printModelInformationToStream(std::cout);
    auto trans_M = model->getTransitionMatrix();
    trans_M.printAsMatlabMatrix(std::cout);
}


bool pipeline(std::string const& path_to_model, std::string const& property_string = "") {

    // Setup:

    // TODO: currently works only for path_to_model = examples/die_c1.nm
    //  Properties should be in property_string in the future: lines below just for quick testing

    //    std::string formulasString = "Pmax=? [ F (l=4 & ip=1) ]; P>=x [ F \"goal1\" ]";
    //    std::string formulasString = "Pmax=? [ F \"goal1\"]; P<=0.5 [ F \"goal1\" ]";
    //    std::string formulasString = "Pmax=? [ F \"one\"]; P>=0.166665 [ F \"one\"];\n";
    std::string formulasString = "Pmax=? [ F \"one\"];";

    // Set up environment: solver method and precision
    storm::Environment env;
    env.solver().minMax().setMethod(storm::solver::MinMaxMethod::ValueIteration);
    env.solver().minMax().setPrecision(storm::utility::convertNumber<storm::RationalNumber>(1e-8));

    // Build model and check tasks
    auto modelFormulas = buildModelFormulas(path_to_model, formulasString);
    auto mdp = std::move(modelFormulas.first);
    auto tasks = getTasks(modelFormulas.second);


    // Check task and produce e-optimal strategy
    storm::modelchecker::SparseMdpPrctlModelChecker<storm::models::sparse::Mdp<double>> checker0(*mdp);
    std::unique_ptr<storm::modelchecker::CheckResult> result0 = checker0.check(env, tasks[0]);
    auto quantitativeResult0 = result0->asExplicitQuantitativeCheckResult<double>().getValueVector();

    // Get check result for Pmax property
    auto e_opt_res = result0->asExplicitQuantitativeCheckResult<double>()[*mdp->getInitialStates().begin()];
    std::cout << "Check result from Pmax=? [ F psi]: " << e_opt_res << std::endl;
    std::cout << "Values under e-optimal strategy: " << std::endl;
    for(auto i: quantitativeResult0){
        std::cout <<  i << std::endl;
    }
    storm::storage::Scheduler<double> const& scheduler = result0->asExplicitQuantitativeCheckResult<double>().getScheduler();
    scheduler.printToStream(std::cout, mdp);

    // Generate safety property for permissive scheduler from e_opt_res:
    // Extract reachability formula from formula string
    std::string reach_obj_substr = formulasString.substr(formulasString.find('['));

    // TODO: how to setprecision?
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(16) << e_opt_res;
    // Create safetyProperty from e_opt_res and reachability formula to generate only strategies that are as good as the e-optimal one
    std::string safetyProp = "P>=" + oss.str() + " " + reach_obj_substr;
    std::cout << safetyProp << std::endl;
    auto program = storm::parser::PrismParser::parse(
            path_to_model);
    storm::parser::FormulaParser formulaParser(program);
    auto formulas = formulaParser.parseFromString(safetyProp);
    auto const& formula = formulas[0].getRawFormula()->asProbabilityOperatorFormula();

    // Produce permissive scheduler & check task
    // TODO: Check: Is this valid or do we need to build a new model mdp with respect to the new formula?
    boost::optional<storm::ps::SubMDPPermissiveScheduler<>> permissive_scheduler = storm::ps::computePermissiveSchedulerViaSMT<>(*mdp, formula);
    std::cout << "Is the permissive scheduler initialized? " << (permissive_scheduler.is_initialized()) << std::endl;


    //TODO:
    // permissive_scheduler: error for robot example why?
    // permissive_scheduler: infinite run for zeroconfig example?
    // How to visualize the scheduler?
    // What happens when applying scheduler (next line)?
    // Produces submdp that represents all actions that are enabled under strategy? == deterministic permissive scheduler?
    // Then we would simply need to simulate runs on this MDP by choosing actions at each state uniformly at random?

    auto submdp = permissive_scheduler->apply();

    // Visualize submdp vs mdp
    std::cout << "Information about submdp under permissive strategy: " << std::endl;
    submdp.printModelInformationToStream(std::cout);
    auto trans_M = submdp.getTransitionMatrix();
    trans_M.printAsMatlabMatrix(std::cout);
    std::cout << "Information about mdp: " << std::endl;
    model_vis(mdp);

    storm::modelchecker::SparseMdpPrctlModelChecker<storm::models::sparse::Mdp<double>> checker1(submdp);
    std::unique_ptr<storm::modelchecker::CheckResult> result1 = checker1.check(env,tasks[0]);
    auto quantitativeResult = result1->asExplicitQuantitativeCheckResult<double>();
    std::cout << "Check max result under permissive strategy: " << (quantitativeResult[0]) <<std::endl;


    // TODO 2. Simulate c runs under scheduler to approximate importance

    // TODO 3. Create training data
    // TODO Repeat the samples importance times
    // TODO Label the data:
    //  Create list of all state-action pairs
    //  Label state-action pairs from the scheduler as positive examples and others as negative examples

    //    for(int i=0;i<choiceOrig->getNumberOfChoices();++i){
    //        std::cout << "choice info: " << choiceOrig->getChoiceInfo(i) << std::endl;
    //    }

    // TODO 4. DT learning:
    //  Implement Test visualization for storm::storage::Scheduler<double> const& scheduler
    //  How to preprocess string data
    //  How to get dt structure that we want: nodes with state action info, comparison=<>..., leaf labels

//    arma::Row<size_t> labels{1,1,1,1,1,1,1,0,0,0};
//    arma::mat data = {{0,0},{1,1},{2,0},{3,0},{4,0},{5,0},{6,0},{7,1},{8,1},{9,0}};
//    mlpack::DecisionTree<> dt(data,labels,2);
//    data::Save("dt.xml", "dt_model", dt);


    return true;
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
    pipeline(argv[1], argv[2]);
}