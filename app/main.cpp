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
#include "dtreeToDot.h"
#include <boost/program_options.hpp>
#include <boost/program_options/cmdline.hpp>

namespace po = boost::program_options;

int initializeOptions(int argc, char *argv[], std::string& model, bool& max, bool& verbose, config& conf, DtConfig& dtConfig){
    
    conf.c = 10000;
    conf.l = 10000;

    // Declare the supported CL options.
    po::options_description generic("General");
    generic.add_options()
    ("help,h", "Print help message and exit")
    ("verbose,v", "Print additional output during the program execution.");
    
    po::options_description input("Check task");
    input.add_options()
    ("model,m", po::value<std::string>(), "Required argument: Path to model file. Model has to be in PRISM format: e.g., model.nm")
    ("propertyMax,p", po::value<std::string>(), "Required argument: Specify whether you want to check Pmax or Pmin. Set the argument to max or min accordingly.");

    po::options_description configuration("Configuration arguments");
    configuration.add_options()
    ("config,c", po::value<std::string>(), "Path to a config file where the following parameters can be specified in alternative to specifying them via the command line.")
    ("minimumGainSplit,g", po::value<double>()->default_value(1e-7), "Set the minimumGainSplit parameter for the decision tree learning.")
    ("minimumLeafSize,l", po::value<size_t>()->default_value(5), "Set the minimumLeafSize parameter for the decision tree learning.")
    ("maximumDepth,d", po::value<size_t>()->default_value(10), "Set the maximumDepth parameter for the decision tree learning.")
    ("importanceDelta,i", po::value<double>()->default_value(0.001), "Set the delta parameter for the importance calculation.")
    ("safetyPrec,s", po::value<int>()->default_value(16), "Set the precision for the safety property bound.");

    po::options_description cmdline_options("Usage");
    cmdline_options.add(generic).add(input).add(configuration);
    
    po::options_description config_file_options;
    config_file_options.add(configuration);

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv,cmdline_options), vm);
    if(vm.count("config")){
        std::string configFile = vm["config"].as<std::string>();
        std::ifstream configFileStream(configFile);
        if (configFileStream) {
            po::store(po::parse_config_file(configFileStream, config_file_options), vm);
            configFileStream.close();
        }
    }
    po::notify(vm);    

    if (vm.count("help")) {
        std::cout << cmdline_options << "\n";
        return 1;
    }

    if (!vm.count("model") || !vm.count("propertyMax")) {
        std::cerr << "Error: Model file is required!" << std::endl;
        return 1;
    }

    model = vm["model"].as<std::string>(); 

    if (vm.count("verbose")) {
        verbose = true;
    }

    if(vm.count("propertyMax")){
        if(vm["propertyMax"].as<std::string>()=="max"){
            if(verbose){
                std::cout << "Property: Pmax=? [ F \"goal\" ]" << std::endl;
            }
            max = true;
        } else if(vm["propertyMax"].as<std::string>()=="min"){
            if(verbose){
                std::cout << "Property: Pmin=? [ F \"goal\" ]" << std::endl;
            }
            max = false;
        } else {
            std::cerr << "Error: propertyMax can take either one of the following values: max, min. For more information, type -h" << std::endl;
            return 1;
        }
    }

    if(verbose){
        if (vm.count("minimumGainSplit")) {
            std::cout << "minimumGainSplit: " << vm["minimumGainSplit"].as<double>() << std::endl; 
        } 
        
        if (vm.count("minimumLeafSize")) {
            std::cout << "minimumLeafSize: " << vm["minimumLeafSize"].as<size_t>() << std::endl;
        } 
            
        if (vm.count("maximumDepth")) {
            std::cout << "maximumDepth: " << vm["maximumDepth"].as<size_t>() << std::endl;
        } 

        if (vm.count("importanceDelta")) {
            std::cout << "importanceDelta: " << vm["importanceDelta"].as<double>() << std::endl;
        }

        if (vm.count("safetyPrec")) {
            std::cout << "safetyPrec: " << vm["safetyPrec"].as<int>() << std::endl;
        }
    }

    conf.delta = vm["importanceDelta"].as<double>();
    conf.prec = vm["safetyPrec"].as<int>();
    dtConfig = {vm["minimumGainSplit"].as<double>(), vm["minimumLeafSize"].as<size_t>(), vm["maximumDepth"].as<size_t>()};
    return 1;
}

void pipeline(std::string const& pathToModel, bool propertyMax, config  const& conf, DtConfig& dtConfig, bool verbose) {

    std::string label = "goal";
    std::string formulaString = (propertyMax ? std::string("Pmax=? ") : std::string("Pmin=? ")) + "[ F \"" + label + " \"];";

    // Setup: Build model, environment and check tasks
    auto env = setUpEnv();
    auto modelFormulas = buildModelFormulas(pathToModel, formulaString);
    auto mdp = std::move(modelFormulas.first);
    auto tasks = getTasks(modelFormulas.second);

    // Check max/min property
    storm::modelchecker::SparseMdpPrctlModelChecker<storm::models::sparse::Mdp<double>> checkerOriginalTask(*mdp);
    std::unique_ptr<storm::modelchecker::CheckResult> checkResult = checkerOriginalTask.check(env, tasks[0]);
    auto stateValueVector = checkResult->asExplicitQuantitativeCheckResult<double>().getValueVector();

    // Generate safety property string for permissive scheduler from initStateCheckResult:
    auto initStateCheckResult = checkResult->asExplicitQuantitativeCheckResult<double>()[*mdp->getInitialStates().begin()];
    std::string safetyProp = generateSafetyProperty(formulaString, initStateCheckResult, propertyMax, conf.prec);
    if(verbose){
        std::cout << "Safety Property: " << safetyProp << std::endl;
    }

    // Generate safety property model and formula
    auto modelSafetyProp = buildModelForSafetyProperty(pathToModel, safetyProp);
    auto safetyMdp = std::move(modelSafetyProp.first);
    auto formula = modelSafetyProp.second;

    // Produce permissive scheduler
    boost::optional<storm::ps::SubMDPPermissiveScheduler<>> permissive_scheduler = storm::ps::computePermissiveSchedulerViaSMT<>(*safetyMdp, formula);
    if(verbose){
        std::cout << "Is the permissive scheduler initialized? " << (permissive_scheduler.is_initialized() ? "Yes" : "No") << std::endl;
    }
    
    // Apply scheduler on safetyMdp to obtain submdp on which we run the simulations
    auto submdp = permissive_scheduler->apply();
    auto submdpPtr = std::make_shared<decltype(submdp)>(submdp);

    // Simulate C runs on submdp to approximate importance of states
    int l, c, delta;
    l = conf.l;
    c = conf.c;
    delta = c*conf.delta;

    MdpInfo mdpInfo;
    mdpInfo.imps = calculateImps(submdp, l, c, delta, label);
    
    // Create training data: Repeat the samples importance times
    auto valueMap = createStateActPairs<storm::models::sparse::Mdp<double>>(safetyMdp, mdpInfo);
    mdpInfo.numOfActId = safetyMdp->getChoiceOrigins()->getNumberOfIdentifiers();
    auto valueMapSubmdp = createStateActPairs<storm::models::sparse::Mdp<double, storm::models::sparse::StandardRewardModel<double>>>(submdpPtr, mdpInfo);
    if(verbose){
        std::cout << "Created value map" << std::endl;
    }

    auto result = createTrainingData(valueMap, valueMapSubmdp, mdpInfo);
    if(verbose){
        std::cout << "Created training data" << std::endl;
    }
    auto allPairs = result.first;
    auto labels = result.second;

    // DT learning
    mlpack::DecisionTree<> dt(allPairs, labels,2, dtConfig.minimumLeafSize, dtConfig.minimumGainSplit, dtConfig.maximumDepth);

    // Visualize the tree
    std::ofstream file;
    file.open ("graph.dot");
    printTreeToDot(dt, file, mdpInfo);
    file.close();
}

int main (int argc, char *argv[]) {
    // Init loggers
    storm::utility::setUp();

    // Set some settings objects.
    storm::settings::initializeAll("countexex", "countexex");
    
    // Set up CL Options
    std::string model;
    bool max=true;
    bool verbose = false;    
    config conf;
    DtConfig dtConfig;
    if(!initializeOptions(argc, argv, model, max, verbose, conf, dtConfig))
    {
        return 0;
    }

    // Start the pipeline
    pipeline(model, max, conf, dtConfig, verbose);
}