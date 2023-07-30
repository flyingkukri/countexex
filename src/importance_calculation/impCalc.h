#pragma once
#include <storm/simulator/DiscreteTimeSparseModelSimulator.h>
#include <storm/models/sparse/Mdp.h>
#include <vector>
#include <storm/storage/BitVector.h>
#include <string>
#include <storm/models/sparse/StandardRewardModel.h>

int simulateRun(storm::simulator::DiscreteTimeSparseModelSimulator<double> simulator, storm::models::sparse::Mdp<double, storm::models::sparse::StandardRewardModel<double>> model, std::vector<int>& visited, int l, storm::storage::BitVector finalStates);

std::vector<int> calculateImps(storm::models::sparse::Mdp<double, storm::models::sparse::StandardRewardModel<double>> model, int l, int C, int delta, const std::string& label);