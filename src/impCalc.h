#include <storm/simulator/DiscreteTimeSparseModelSimulator.h>
#include <storm/models/sparse/Mdp.h>

int simulateRun(storm::simulator::DiscreteTimeSparseModelSimulator<double> simulator, storm::models::sparse::Mdp<double> model, std::vector<int>& visited, int l, storm::storage::BitVector finalStates);

std::vector<int> calculateImps(storm::simulator::DiscreteTimeSparseModelSimulator<double> simulator, storm::models::sparse::Mdp<double> model, int l, int C, const std::string& label);