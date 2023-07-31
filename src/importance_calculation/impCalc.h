#pragma once
#include <storm/simulator/DiscreteTimeSparseModelSimulator.h>
#include <storm/models/sparse/Mdp.h>
#include <vector>
#include <storm/storage/BitVector.h>
#include <string>
#include <storm/models/sparse/StandardRewardModel.h>

int simulateRun(storm::simulator::DiscreteTimeSparseModelSimulator<double> simulator, storm::models::sparse::Mdp<double, storm::models::sparse::StandardRewardModel<double>> model, std::vector<int>& visited, int l, storm::storage::BitVector finalStates);

/*
 * This function calculates the importance of each state pair. 
 * It simulates C runs each until we have either reached a state with the label goal or until we have executed l steps.
 * @param model: MDP model
 * @param l: Maximum number of steps
 * @param C: Number of runs
 * @param delta: If a state was visited in less than delta successful runs, it is given an importance of 0
 * @param label: Label given to the states that we want to reach
 * @return: Vector of size |S| containing the importance of each state
*/
std::vector<int> calculateImps(storm::models::sparse::Mdp<double, storm::models::sparse::StandardRewardModel<double>> model, int l, int C, int delta, const std::string& label);