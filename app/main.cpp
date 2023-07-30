#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <random>
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
#include "main.h"
#include "genTrainData.h"
#include "buildModel.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>
#undef As
#include <mlpack.hpp>
#include <mlpack/core.hpp>
#include <mlpack/core/data/load.hpp>
#include <mlpack/core/data/save.hpp>
#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include <armadillo>
#include "impCalc.h"
#include "utils.h"

bool pipeline(std::string const& pathToModel, config  const& conf, std::string const& propertyString = "") {

    // TODO: currently works only for pathToModel = examples/die_c1.nm
    //  Properties should be in propertyString in the future: lines below just for quick testing

    std::string label = "goal";
    std::string formulasString = "Pmax=? [ F \"" + label + " \"];";

    // Setup: Build model, environment and check tasks
    auto env = setUpEnv();
    auto modelFormulas = buildModelFormulas(pathToModel, formulasString);
    auto mdp = std::move(modelFormulas.first);
    auto tasks = getTasks(modelFormulas.second);

    // Check task and produce e-optimal strategy
    storm::modelchecker::SparseMdpPrctlModelChecker<storm::models::sparse::Mdp<double>> checkerOriginalTask(*mdp);
    std::unique_ptr<storm::modelchecker::CheckResult> checkResult = checkerOriginalTask.check(env, tasks[0]);
    auto stateValueVector = checkResult->asExplicitQuantitativeCheckResult<double>().getValueVector();

    // Display deterministic scheduler
    storm::storage::Scheduler<double> const& scheduler = checkResult->asExplicitQuantitativeCheckResult<double>().getScheduler();
    scheduler.printToStream(std::cout, mdp);

    // Generate safety property for permissive scheduler from initStateCheckResult:
    auto initStateCheckResult = checkResult->asExplicitQuantitativeCheckResult<double>()[*mdp->getInitialStates().begin()];
    std::string safetyProp = generateSafetyProperty(formulasString, initStateCheckResult);
    // std::cout << "Check result from Pmax=? [ F psi]: " << initStateCheckResult << std::endl;

    // Generate safety property model and formula
    auto modelSafetyProp = buildModelForSafetyProperty(pathToModel, safetyProp);
    auto mdp2 = std::move(modelSafetyProp.first);
    auto formula = modelSafetyProp.second;

    // Produce permissive scheduler & check task
    boost::optional<storm::ps::SubMDPPermissiveScheduler<>> permissive_scheduler = storm::ps::computePermissiveSchedulerViaSMT<>(*mdp2, formula);
    std::cout << "Is the permissive scheduler initialized? " << (permissive_scheduler.is_initialized()) << std::endl;

    //TODO:
    // permissive_scheduler: error for robot example why?
    // permissive_scheduler: infinite run for zeroconfig example?

    auto submdp = permissive_scheduler->apply();
    auto submdp_ptr = std::make_shared<decltype(submdp)>(submdp);

    storm::modelchecker::SparseMdpPrctlModelChecker<storm::models::sparse::Mdp<double>> checker1(submdp);
    std::unique_ptr<storm::modelchecker::CheckResult> result1 = checker1.check(env,tasks[0]);
    auto quantitativeResult = result1->asExplicitQuantitativeCheckResult<double>();
    std::cout << "Check max result under permissive strategy: " << (quantitativeResult[0]) <<std::endl;

    // Simulate C runs under scheduler to approximate importance of states
    int l, C, delta;
    l = conf.l;
    C = conf.C;
    delta = C*conf.delta;

    MdpInfo mdpInfo;
    mdpInfo.imps = calculateImps(submdp, l, C, delta, label);
    std::cout << "imps: " << std::endl;
    for(int imp: mdpInfo.imps) {
        std::cout << imp << std::endl;
    }

    std::vector<int> myVector(13); // Create a vector with 13 elements
    // Fill the vector with values from 0 to 12
    for (int i = 0; i < 13; ++i) {
        myVector[i] = 1;
    }
    // myVector[4]=3;
    // myVector[10]=4;
    // myVector[6]=1;
    mdpInfo.imps=myVector;
    
    // Create training data: Repeat the samples importance times
    auto value_map = createStateActPairs<storm::models::sparse::Mdp<double>>(mdp, mdpInfo);
    mdpInfo.numOfActId = mdp->getChoiceOrigins()->getNumberOfIdentifiers();
    auto value_map_submdp = createStateActPairs<storm::models::sparse::Mdp<double, storm::models::sparse::StandardRewardModel<double>>>(submdp_ptr, mdpInfo);
    // printStateActPairs<storm::models::sparse::Mdp<double>>(mdp);
    // printStateActPairs<storm::models::sparse::Mdp<double>>(submdp_ptr);
    std::cout << "created value map" << std::endl;
    // arma::mat all_pairs = createMatrixFromValueMap(value_map);
    // auto strategy_pairs = createMatrixFromValueMap(value_map_submdp);
    // std::cout << "created matrices" << std::endl;
    // arma::cout << "All state-action pairs: \n" << all_pairs << arma::endl;
    // arma::cout << "State-action pairs of the strategy: \n" << strategy_pairs << arma::endl;

    auto result = createTrainingData(value_map, value_map_submdp, mdpInfo);
    std::cout << "Created training data" << std::endl;

    auto all_pairs = result.first;
    auto labels = result.second;
    std::cout << "Training data:\n " << all_pairs << std::endl;
    std::cout << "Labels:\n " << labels << std::endl;

    // arma::Row<size_t> myVector2(14);
    // // std::vector<int> myVector2(14);
    // for (int i = 0; i < 14; ++i) {
    //     myVector2(i) = 1;
    // }

    // myVector2(0)=0;
    // // myVector2(2)=0;
    // // myVector2(4)=0;
    // // myVector2(6)=0;
    // // myVector2(8)=0;
    // myVector2(9)=0;
    // myVector2(10)=0;
    // myVector2(11)=0;
    // myVector2(12)=0;
    // myVector2(13)=0;
    // labels=myVector2;


    // DT learning:

    mlpack::DecisionTree<> dt(all_pairs, labels,2, 1, 1e-7, 3);
    arma::Row<size_t> testPredictions;
    dt.Classify(all_pairs, testPredictions);
    for (size_t pred: testPredictions) {
        std::cout << pred << std::endl;
    }

    // Visualize the tree
    std::ofstream file;
    file.open ("graph.dot");
    printTreeToDot(dt, file, mdpInfo);
    file.close();

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
    storm::settings::initializeAll("countexex", "countexex");

    config conf;
    conf.C = 10000;
    conf.l = 10000;
    conf.delta = 0.001;

    // Call function
    pipeline(argv[1], conf, argv[2]);
}