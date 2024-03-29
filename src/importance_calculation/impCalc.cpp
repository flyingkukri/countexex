#include "impCalc.h"
#include <cstdlib>
#include <random>
#include <storm/api/storm.h>
#include <storm-parsers/parser/PrismParser.h>
#include <storm/storage/prism/Program.h>
#include <storm/storage/sparse/PrismChoiceOrigins.h>
#include <storm/modelchecker/results/CheckResult.h>
#include <storm/utility/initialize.h>
#include <storm-parsers/parser/FormulaParser.h>
#include <storm/models/sparse/Mdp.h>
#include <storm/models/sparse/StandardRewardModel.h>

int simulateRun(storm::simulator::DiscreteTimeSparseModelSimulator<double> simulator,
                storm::models::sparse::Mdp<double, storm::models::sparse::StandardRewardModel<double>>& model, std::vector<int>& visited, int l, storm::storage::BitVector& finalStates, int currentSeed) {
    simulator.setSeed(currentSeed);
    // Reset the visited vector to zero before each simulation
    std::fill(visited.begin(), visited.end(), 0);
    int state = simulator.getCurrentState();
    for (int i = 0; i < l; ++i) {
        if(finalStates.get(state)) {
            return 1; // We assume that final states are sink states
        }
        if(model.isSinkState(state)) {
            break;
        }
        state = simulator.getCurrentState();
        visited[state] = 1;
        simulator.randomStep();
    }
    simulator.resetToInitial();
    return 0;
}

std::vector<int> calculateImps(storm::models::sparse::Mdp<double, storm::models::sparse::StandardRewardModel<double>> model, int l, int c, int delta, const std::string& label) {
    // Set seed for deterministic results
    srand(42);
    uint64_t seed; 

    storm::simulator::DiscreteTimeSparseModelSimulator<double> simulator(model);
    int nStates = model.getNumberOfStates();
    std::vector<int> imps(nStates, 0);
    std::vector<int> visited(nStates, 0);
    storm::storage::BitVector finalStates = model.getStates(label);

    for (int i = 0; i < c; ++i) {
        // simulateRun returns 1 if we are in a final state
        seed = random();
        if(simulateRun(simulator, model, visited, l, finalStates, seed)){
            for (int j = 0; j < nStates; ++j) {
                assert(visited[j] == 0 || visited[j] == 1);
                imps[j] += visited[j];
                visited[j] = 0;
            }
        }
    }

    // Set the importance of a state to zero if it is below the threshold delta
    for (int& value : imps) {
        if (value < delta) {
            value = 0;
        }
    }

    return imps;
}