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
#include <bits/stdc++.h>
#include "main.h"
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


#include <storm/simulator/DiscreteTimeSparseModelSimulator.h>

//#include "impCalc.hpp"



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
void model_vis(std::shared_ptr<MdpType>& model){
    model->printModelInformationToStream(std::cout);
    auto trans_M = model->getTransitionMatrix();
    trans_M.printAsMatlabMatrix(std::cout);
}

template <typename MdpType>
void print_state_act_pairs(std::shared_ptr<MdpType>& mdp){
    auto val = mdp->getStateValuations();
    std::cout << "Print (s-a)-pairs: " << std::endl;
    for(int i=0; i<mdp->getNumberOfStates(); ++i){
        auto a_count = mdp->getNumberOfChoices(i);
        for(int k=0;k<a_count;++k){
            auto l = val.at(i);
            auto start = l.begin();
            while(start!=l.end()){
                // TODO: extend to other datatypes: bool, double ... aber nur für uns also nicht unbedingt nötig
                std::cout << start.getVariable().getName() << ": " << start.getIntegerValue() << " ";
                start.operator++();
            }
            // This info should be present in the dt in the end and is represented via mdp->getChoiceOrigins()->getIdentifier(mdp->getTransitionMatrix().getRowGroupIndices()[i]+k);
            auto a = mdp->getChoiceOrigins()->getChoiceInfo(mdp->getTransitionMatrix().getRowGroupIndices()[i]+k);
            std::cout << a << " ";
            std::cout << "\n" << std::endl;
        }
    }
}

template <typename MdpType>
std::map<std::string, std::variant<std::vector<int>, std::vector<bool>, std::vector<storm::RationalNumber>>> create_state_act_pairs(std::shared_ptr<MdpType>& mdp){
    auto val = mdp->getStateValuations();
    std::map<std::string, std::variant<std::vector<int>, std::vector<bool>, std::vector<storm::RationalNumber>>> value_map;

    for(int i=0; i<mdp->getNumberOfStates(); ++i){
        auto a_count = mdp->getNumberOfChoices(i);
        for(int k=0;k<a_count;++k){
            auto l = val.at(i);
            auto start = l.begin();
            while(start!=l.end()){
                auto key = start.getVariable().getName();
                auto it = value_map.find(key);
                if(start.getVariable().hasBooleanType()){
                    auto e = start.getBooleanValue();
                    if( it == value_map.end()){
                        value_map.insert(std::make_pair(key, std::vector<bool>{e}));
                    }else{
                        auto& vector = std::get<std::vector<bool>>(it->second);
                        vector.push_back(e);
                    }
                }else if(start.getVariable().hasIntegerType()){
                    auto e = start.getIntegerValue();
                    if(it == value_map.end()){
                        std::vector<int> int_vector;
                        int_vector.push_back(e);
                        value_map.insert(std::make_pair(key, int_vector));
                    }else{
                        auto& vector = std::get<std::vector<int>>(it->second);
                        vector.push_back(e);                    }
                }else if(start.getVariable().hasRationalType()){
                    auto e = start.getRationalValue();
                    if(it == value_map.end()){
                        std::vector<storm::RationalNumber> rat_vector;
                        rat_vector.push_back(e);
                        value_map.insert(std::make_pair(key, rat_vector));
                    }else{
                        auto& vector = std::get<std::vector<storm::RationalNumber>>(it->second);
                        vector.push_back(e);
                    }
                }
                start.operator++();
            }
            // TODO: was, wenn eine Variable aus dem PRISM code "action" heißt?
            auto key = "action";
            auto elem = mdp->getChoiceOrigins()->getIdentifier(mdp->getTransitionMatrix().getRowGroupIndices()[i]+k);
            auto it = value_map.find(key);
            if(it == value_map.end()){
                std::vector<int> int_vector;
                int_vector.push_back(elem);
                value_map.insert(std::make_pair(key, int_vector));
            }else{
                auto& vector = std::get<std::vector<int>>(it->second);
                vector.push_back(elem);
            }
        }
    }
    return value_map;
}

arma::mat createMatrixFromValueMap(std::map<std::string, std::variant<std::vector<int>, std::vector<bool>, std::vector<storm::RationalNumber>>>& value_map){
    arma::mat armaData;
    for (const auto& pair : value_map) {
    // Get the vector corresponding to the key
        const std::variant<std::vector<int>, std::vector<bool>, std::vector<storm::RationalNumber>>& valueVector = pair.second;

        // Create an arma::rowvec from the vector
        arma::rowvec rowVec;
        if (const auto intVector = std::get_if<std::vector<int>>(&valueVector)) {
            rowVec = arma::conv_to<arma::rowvec>::from(*intVector);
        } else if (const auto boolVector = std::get_if<std::vector<bool>>(&valueVector)) {
            std::vector<int> intVector;
            intVector.reserve(boolVector->size());
            for (bool value : *boolVector) {
                intVector.push_back(value ? 1 : 0);
            }
            rowVec = arma::conv_to<arma::rowvec>::from(intVector);
        } else if (const auto ratVector = std::get_if<std::vector<storm::RationalNumber>>(&valueVector)) {
        // TODO: convert storm::RationalNumber to double
        //            arma::rowvec rowVec(ratVector->size());
        //            for (std::size_t i = 0; i < ratVector->size(); ++i) {
        //                rowVec(i) = storm::utility::convertNumber<double>(ratVector[i]);
        //            }
        }
    // Append the row vector to the matrix
        armaData = arma::join_vert(armaData, rowVec);
    }
    return armaData;
}

arma::Row<size_t> createDataLabels(arma::mat& all_pairs, arma::mat& strategy_pairs){
    // column-major in arma: thus each column represents a data point
    size_t numColumns = all_pairs.n_cols;
    arma::Row<size_t> labels(numColumns, arma::fill::zeros);
    for (size_t i = 0; i < numColumns; ++i) {
        // Get the i-th column of all_pairs
        auto found = 0;
        arma::vec column = all_pairs.col(i);
        for (size_t j = 0; j < strategy_pairs.n_cols; ++j) {
            arma::vec col2Cmp = strategy_pairs.col(j);
            if (arma::all(col2Cmp == column)) {
                found = 1;
                labels(i) = 1;
                break;
            }
        }
        if(!found) {
            labels(i) = 0;
        }
    }
    return labels;
}

std::pair<arma::mat, arma::Row<size_t>> createTrainingData(std::map<std::string, std::variant<std::vector<int>, std::vector<bool>, std::vector<storm::RationalNumber>>>& value_map, std::map<std::string, std::variant<std::vector<int>, std::vector<bool>, std::vector<storm::RationalNumber>>>& value_map_submdp){
    arma::mat all_pairs = createMatrixFromValueMap(value_map);
    auto strategy_pairs = createMatrixFromValueMap(value_map_submdp);
    arma::Row<size_t> labels = createDataLabels(all_pairs, strategy_pairs);
    return std::make_pair(all_pairs, labels);
}

// Helper function to recursively traverse the decision tree and generate DOT representation.
/*void GenerateDot(mlpack::DecisionTree<>& node, std::ofstream& dotFile, int& nodeCounter)
{
    if (node.Child(0) == nullptr)  // Leaf node.
    {
        dotFile << "  " << nodeCounter << " [label=\"" << node. << "\"];\n";
    }
    else  // Non-leaf node.
    {
        std::string splitRule;
        if (node.SplitDimension() != size_t(-1))
        {
            splitRule = "x" + std::to_string(node.SplitDimension()) + " <= " +
                        std::to_string(node.SplitValue());
        }
        else
        {
            splitRule = "unknown";
        }

        dotFile << "  " << nodeCounter << " [label=\"" << splitRule << "\"];\n";

        for (size_t i = 0; i < node.NumChildren(); ++i)
        {
            dotFile << "  " << nodeCounter << " -> " << ++nodeCounter << ";\n";
            GenerateDot(*node.Child(i), dotFile, nodeCounter);
        }
    }
}*/

bool pipeline(std::string const& path_to_model, std::string const& property_string = "") {

    // Setup:

    // TODO: currently works only for path_to_model = examples/die_c1.nm
    //  Properties should be in property_string in the future: lines below just for quick testing

    //        std::string formulasString = "Pmax=? [ F (l=4 & ip=1) ]; P>=x [ F \"goal1\" ]";
//    std::string formulasString = "Pmax=? [ F (l=4 & ip=1) ];";
//        std::string formulasString = "Pmax=? [ F \"goal1\"]; P<=0.5 [ F \"goal1\" ]";
//    std::string formulasString = "Pmax=? [ F s=5];";
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

    auto val = mdp->getStateValuations();
    std::cout << "Print (s-a)-pairs: " << std::endl;
    for(int i=0; i<mdp->getNumberOfStates(); ++i){
        auto a_count = mdp->getNumberOfChoices(i);
        for(int k=0;k<a_count;++k){
            auto l = val.at(i);
            auto start = l.begin();
            while(start!=l.end()){
                std::cout << start.getVariable().getName() << ": " << start.getIntegerValue() << " ";
                start.operator++();
            }

            auto a = mdp->getChoiceOrigins()->getChoiceInfo(mdp->getTransitionMatrix().getRowGroupIndices()[i]+k);
            std::cout << a << " ";
            std::cout << "\n" << std::endl;
        }
    }
    std::vector<int> action_row;
    std::map<std::string,int> action_value_map;

    // create mapping from actions to categorical values
    for(int i=0; i<mdp->getChoiceOrigins()->getNumberOfChoices();++i){
        auto res = mdp->getChoiceOrigins()->getIdentifier(i);
        std::cout << "Print choice identifier: " << mdp->getChoiceOrigins()->getIdentifier(i) << "; Print info: " << mdp->getChoiceOrigins()->getChoiceInfo(i);
    }
    mdp->getChoiceOrigins()->getNumberOfChoices();

    // Get check result for Pmax property
    auto e_opt_res = result0->asExplicitQuantitativeCheckResult<double>()[*mdp->getInitialStates().begin()];
    std::cout << "Check result from Pmax=? [ F psi]: " << e_opt_res << std::endl;
    std::cout << "Values under e-optimal strategy: " << std::endl;
    for(auto i: quantitativeResult0){
        std::cout <<  i << std::endl;
    }
    print_state_act_pairs(mdp);
    auto value_map = create_state_act_pairs<>(mdp);

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
   // boost::optional<storm::ps::SubMDPPermissiveScheduler<>>
    auto permissive_scheduler = storm::ps::computePermissiveSchedulerViaSMT<>(*mdp, formula);
    std::cout << "Is the permissive scheduler initialized? " << (permissive_scheduler.is_initialized()) << std::endl;

    //TODO:
    // permissive_scheduler: error for robot example why?
    // permissive_scheduler: infinite run for zeroconfig example?
    // How to visualize the scheduler?
    // What happens when applying scheduler (next line)?
    // Produces submdp that represents all actions that are enabled under strategy? == deterministic permissive scheduler?
    // Then we would simply need to simulate runs on this MDP by choosing actions at each state uniformly at random?

    auto submdp = permissive_scheduler->apply();
    auto submdp_ptr = std::make_shared<decltype(submdp)>(submdp);
    print_state_act_pairs(submdp_ptr);
    auto value_map_submdp = create_state_act_pairs<>(submdp_ptr);


    // Visualize submdp vs mdp
    std::cout << "Information about submdp under permissive strategy: " << std::endl;
//    model_vis(submdp_ptr);
    std::cout << "Information about mdp: " << std::endl;
//    model_vis(mdp);

    storm::modelchecker::SparseMdpPrctlModelChecker<storm::models::sparse::Mdp<double>> checker1(submdp);
    std::unique_ptr<storm::modelchecker::CheckResult> result1 = checker1.check(env,tasks[0]);
    auto quantitativeResult = result1->asExplicitQuantitativeCheckResult<double>();
    std::cout << "Check max result under permissive strategy: " << (quantitativeResult[0]) <<std::endl;


    // TODO 2. Simulate c runs under scheduler to approximate importance
    int l, C;
    l = C = 10000;
    storm::simulator::DiscreteTimeSparseModelSimulator<double> simulator(submdp);
 //   int* imps = calculateImps(simulator, submdp, l, C);

    // TODO 3. Create training data
    // TODO Repeat the samples importance times
    // TODO Label the data:
    //  Create list of all state-action pairs
    //  Label state-action pairs from the scheduler as positive examples and others as negative examples



    // TODO 4. DT learning:
    //  Implement Test visualization for storm::storage::Scheduler<double> const& scheduler
    //  How to preprocess string data
    //  How to get dt structure that we want: nodes with state action info, comparison=<>..., leaf labels

    arma::mat all_pairs = createMatrixFromValueMap(value_map);
    auto strategy_pairs = createMatrixFromValueMap(value_map_submdp);

    arma::cout << all_pairs << arma::endl;
    arma::cout << strategy_pairs << arma::endl;

    std::pair<arma::mat, arma::Row<size_t>> result = createTrainingData(value_map, value_map_submdp);
    all_pairs = result.first;
    auto labels = result.second;
    std::cout << "Labels: " << labels << std::endl;
    mlpack::DecisionTree<> dt(all_pairs,labels,2);

    /*    data::Save("dt.xml", "dt_model", dt);

    // Open the DOT file for writing.
    std::ofstream dotFile("decision_tree.dot");
    dotFile << "digraph DecisionTree {\n";

    // Generate the DOT representation recursively.
    int nodeCounter = 0;
    GenerateDot(dt, dotFile, nodeCounter);

    dotFile << "}\n";
    dotFile.close();*/

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