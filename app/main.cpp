#include <numeric>
#include <random>
#include <storm/modelchecker/results/CheckResult.h>
#include <storm/utility/initialize.h>
#include <storm-permissive/analysis/PermissiveSchedulers.h>
#include <storm/builder/ExplicitModelBuilder.h>
#include <storm/environment/Environment.h>
#include <storm/models/sparse/Mdp.h>
#include <storm/modelchecker/prctl/SparseMdpPrctlModelChecker.h>
#include <storm/models/sparse/StandardRewardModel.h>
#include "main.h"
#include "buildModel.h"
#include <iostream>
#include <fstream>
#undef As
#include <mlpack/methods/decision_tree/decision_tree.hpp>
#include "impCalc.h"
#include "dtreeToDot.h"
#include <boost/program_options.hpp>

namespace po = boost::program_options;

int initializeOptions(int argc, char *argv[], Options& clOptions){
    
    clOptions.conf.c = 10000;
    clOptions.conf.l = 10000;

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
    ("safetyPrec,s", po::value<int>()->default_value(16), "Set the precision for the safety property bound.")
    ("optimizer,o",po::value<std::string>()->default_value("smt"), "Choose the method for computing the permissive strategy: smt or milp. Note that for MILP, you need to have Gurobi installed.");

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

    clOptions.pathToModel = vm["model"].as<std::string>(); 

    if (vm.count("verbose")) {
        clOptions.verbose = true;
    }

    if(vm.count("propertyMax")){
        if(vm["propertyMax"].as<std::string>()=="max"){
            if(clOptions.verbose){
                std::cout << "Property: Pmax=? [ F \"goal\" ]" << std::endl;
            }
            clOptions.propertyMax = true;
        } else if(vm["propertyMax"].as<std::string>()=="min"){
            if(clOptions.verbose){
                std::cout << "Property: Pmin=? [ F \"goal\" ]" << std::endl;
            }
            clOptions.propertyMax = false;
        } else {
            std::cerr << "Error: propertyMax can take either one of the following values: max, min. For more information, type -h" << std::endl;
            return 1;
        }
    }

    if(vm.count("optimizer")){
        if (vm["optimizer"].as<std::string>()=="smt") {
            clOptions.optimizerSMT=true;        
        }else if(vm["optimizer"].as<std::string>()=="milp"){
            clOptions.optimizerSMT=false;
        } else {
            std::cerr << "Error: optimizer can take either one of the following values: smt, milp. For more information, type -h" << std::endl;
            return 1;
        }
        if(clOptions.verbose){
            std::cout << "Optimizer: " << vm["optimizer"].as<std::string>() << std::endl;
        }
    }
    
    if(clOptions.verbose){
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

    clOptions.conf.delta = vm["importanceDelta"].as<double>();
    clOptions.conf.prec = vm["safetyPrec"].as<int>();
    clOptions.dtConfig = {vm["minimumGainSplit"].as<double>(), vm["minimumLeafSize"].as<size_t>(), vm["maximumDepth"].as<size_t>()};
    return 0;
}

std::pair<ValueMap, ValueMap> createValueMapHelper(boost::optional<storm::ps::SubMDPPermissiveScheduler<>>& permissiveScheduler, Options& clOptions, MdpInfo& mdpInfo, std::string& label, std::shared_ptr<storm::models::sparse::Mdp<double>>& safetyMdp){

    if(clOptions.verbose){
        std::cout << "Is the permissive scheduler initialized? " << (permissiveScheduler.is_initialized() ? "Yes" : "No") << std::endl;
    }

    // Apply scheduler on safetyMdp to obtain submdp on which we run the simulations
    auto submdp = permissiveScheduler->apply();
    auto submdpPtr = std::make_shared<decltype(submdp)>(submdp);

    // Simulate C runs on submdp to approximate importance of states
    int l, c, delta;
    l = clOptions.conf.l;
    c = clOptions.conf.c;
    delta = c*clOptions.conf.delta;

    mdpInfo.imps = calculateImps(submdp, l, c, delta, label);

    // Create training data: Repeat the samples importance times
    auto valueMap = createStateActPairs<storm::models::sparse::Mdp<double>>(safetyMdp, mdpInfo); 
    auto valueMapSubmdp = createStateActPairs<storm::models::sparse::Mdp<double, storm::models::sparse::StandardRewardModel<double>>>(submdpPtr, mdpInfo);
    
    return std::make_pair(valueMap, valueMapSubmdp);
}

void pipeline(Options& clOptions) {

    std::string label = "goal";
    std::string formulaString = (clOptions.propertyMax ? std::string("Pmax=? ") : std::string("Pmin=? ")) + "[ F \"" + label + " \"];";

    // Setup: Build model, environment and check tasks
    auto env = setUpEnv();
    auto modelFormulas = buildModelFormulas(clOptions.pathToModel, formulaString);
    auto mdp = std::move(modelFormulas.first);
    auto tasks = getTasks(modelFormulas.second);

    // Check max/min property
    storm::modelchecker::SparseMdpPrctlModelChecker<storm::models::sparse::Mdp<double>> checkerOriginalTask(*mdp);
    std::unique_ptr<storm::modelchecker::CheckResult> checkResult = checkerOriginalTask.check(env, tasks[0]);
    auto stateValueVector = checkResult->asExplicitQuantitativeCheckResult<double>().getValueVector();

    // Generate safety property string for permissive scheduler from initStateCheckResult:
    auto initStateCheckResult = checkResult->asExplicitQuantitativeCheckResult<double>()[*mdp->getInitialStates().begin()];
    std::string safetyProp = generateSafetyProperty(formulaString, initStateCheckResult, clOptions.propertyMax, clOptions.conf.prec);
    if(clOptions.verbose){
        std::cout << "Safety Property: " << safetyProp << std::endl;
    }

    // Generate safety property model and formula
    auto modelSafetyProp = buildModelForSafetyProperty(clOptions.pathToModel, safetyProp);
    auto safetyMdp = std::move(modelSafetyProp.first);
    auto formula = modelSafetyProp.second;

    MdpInfo mdpInfo;
    mdpInfo.numOfActId = safetyMdp->getChoiceOrigins()->getNumberOfIdentifiers(); 

    std::pair<ValueMap, ValueMap> valueMapPair;
 
    // Produce permissive scheduler, compute importance and create valueMap
    if(clOptions.optimizerSMT){
        boost::optional<storm::ps::SubMDPPermissiveScheduler<>> permissiveScheduler = storm::ps::computePermissiveSchedulerViaSMT<>(*safetyMdp, formula);
        valueMapPair = createValueMapHelper(permissiveScheduler, clOptions, mdpInfo, label, safetyMdp);
    }else{
        boost::optional<storm::ps::SubMDPPermissiveScheduler<>> permissiveScheduler = storm::ps::computePermissiveSchedulerViaMILP<>(*safetyMdp, formula);
        valueMapPair = createValueMapHelper(permissiveScheduler, clOptions, mdpInfo, label, safetyMdp);
    }
    
    ValueMap valueMap = valueMapPair.first;
    ValueMap valueMapSubmdp = valueMapPair.second;    
    
    if(clOptions.verbose){
        std::cout << "Created value map" << std::endl;
    }
    
    // Create training data
    auto result = createTrainingData(valueMap, valueMapSubmdp, mdpInfo);
    if(clOptions.verbose){
        std::cout << "Created training data" << std::endl;
    }
    auto allPairs = result.first;
    auto labels = result.second;

    // DT learning
    mlpack::DecisionTree<> dt(allPairs, labels, 2, clOptions.dtConfig.minimumLeafSize, clOptions.dtConfig.minimumGainSplit, clOptions.dtConfig.maximumDepth);

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
    Options clOptions;
    clOptions.propertyMax=true;
    clOptions.verbose = false;
    clOptions.optimizerSMT =true;
    if(initializeOptions(argc, argv, clOptions))
    {
        return 1;
    }

    // Start the pipeline
    pipeline(clOptions);
}