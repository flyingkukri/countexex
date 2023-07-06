#include "impCalc.hpp"

int simulateRun(storm::simulator::DiscreteTimeSparseModelSimulator<double> simulator, 
                 storm::models::sparse::Mdp<double> model, int *visited, int l) {
    int finalState = 0;
    int state;
    for (int i = 0; i < l; i++) { 
        state = simulator.getCurrentState();
        visited[state] = 1;
        simulator.randomStep();
        if(model.isSinkState(state)) {
            break;
        }
        // TODO return 1 if we are in a final state
    }
    simulator.resetToInitial();
    return finalState;
}

int* calculateImps(storm::simulator::DiscreteTimeSparseModelSimulator<double> simulator, 
                 storm::models::sparse::Mdp<double> model, int l, int C) {
    int nstates = model.getNumberOfStates();
    int *imps = new int[nstates];
    int *visited = new int[nstates];
    

    for (int i = 0; i < C; i++) {
        // simulateRun returns 1 if we are in a final state
        if(simulateRun(simulator, model, visited, l)){
            for (int i = 0; i < nstates; i++) {
                imps[i] += visited[i];
                visited[i] = 0;
            }
        }
    }

    delete [] visited;
    return imps;
}
