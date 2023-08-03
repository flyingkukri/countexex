#pragma once
#include <storm/simulator/DiscreteTimeSparseModelSimulator.h>
#include <storm/models/sparse/Mdp.h>
#include <vector>
#include <storm/storage/BitVector.h>
#include <string>
#include <storm/models/sparse/StandardRewardModel.h>

/*!
 * Simulate a run of the MDP for at most l steps (or until we reach a final state
 * @param simulator: The simulator object created from the mdp
 * @param model: The mdp
 * @param visited: We will set the i-th entry of this vector to 1 if we visited the i-th state during the simulation
 * @param l: The maximum amount of steps we simulate
 * @param finalStates: A bitvector containing the final states
 * @return 1 if we reached a final state, 0 otherwise
*/
int simulateRun(storm::simulator::DiscreteTimeSparseModelSimulator<double> simulator, storm::models::sparse::Mdp<double, storm::models::sparse::StandardRewardModel<double>> model, std::vector<int>& visited, int l, storm::storage::BitVector finalStates);

/*!
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