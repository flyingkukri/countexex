#include "impCalc.hpp"

int simulateRun(storm::simulator::DiscreteTimeSparseModelSimulator<double> simulator, 
                 storm::models::sparse::Mdp<double> model, int *visited, int l, std::string const& label) {
    int state;
    for (int i = 0; i < l; i++) { 
        state = simulator.getCurrentState();
        visited[state] = 1;
        simulator.randomStep();
        if(model.hasLabel(label)) {
            return 1; // We assume that final states are sink states
        }
        if(model.isSinkState(state)) {
            break;
        }
    }
    simulator.resetToInitial();
    return 0;
}

int* calculateImps(storm::simulator::DiscreteTimeSparseModelSimulator<double> simulator, 
                 storm::models::sparse::Mdp<double> model, int l, int C, const std::string& label) {
    int nstates = model.getNumberOfStates();
    int *imps = new int[nstates];
    int *visited = new int[nstates];
    

    for (int i = 0; i < C; i++) {
        // simulateRun returns 1 if we are in a final state
        if(simulateRun(simulator, model, visited, l, label)){
            for (int i = 0; i < nstates; i++) {
                imps[i] += visited[i];
                visited[i] = 0;
            }
        }
    }

    delete [] visited;
    return imps;
}
