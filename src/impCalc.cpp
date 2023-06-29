#include "impCalc.hpp"

void simulateRun(storm::simulator::DiscreteTimeSparseModelSimulator<double> simulator, 
                 storm::models::sparse::Mdp<double> model, int *visited, int l) {
    int state;
    for (int i = 0; i < l; i++) { 
        state = simulator.getCurrentState();
        visited[state] = 1;
        simulator.randomStep();
        if(model.isSinkState(state)) {
            break;
        }
    }
    simulator.resetToInitial();
}

int* calculateImps(storm::simulator::DiscreteTimeSparseModelSimulator<double> simulator, 
                 storm::models::sparse::Mdp<double> model, int l, int C) {
    int nstates = model.getNumberOfStates();
    int *imps = new int[nstates];
    int *visited = new int[nstates];
    

    for (int i = 0; i < C; i++) {
        simulateRun(simulator, model, visited, l);
        for (int i = 0; i < nstates; i++) {
            imps[i] += visited[i];
            visited[i] = 0;
        }
    }

    delete [] visited;
    return imps;
}
