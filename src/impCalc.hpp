#include <storm/simulator/DiscreteTimeSparseModelSimulator.h>
#include <storm/models/sparse/Mdp.h>

int* calculateImps(storm::simulator::DiscreteTimeSparseModelSimulator<double> simulator, 
                 storm::models::sparse::Mdp<double> model, int l, int C);