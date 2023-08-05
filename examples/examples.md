# Examples 

The examples in this folder are taken from [storm](https://github.com/moves-rwth/storm/tree/master/resources/examples/testfiles/mdp).

## 1. Cycle
At first, we take a look at the file cycle.nm, which can be found in this folder. The mdp model file consists of three modules. Each module manipulates one of the variables x, y, z. If the respective variable is zero, then it is either increased to one or two with equal probability by choosing action b, or remains unchanged by choosing action a.  
To get a better understanding of the model, we partially created its MDP graph:
<img src="cycle_mdp-1_new.png" alt="Cycle MDP">

Assume that we are interested in reaching a state where at least one of the variables is set to two with maximal probability. Thus our property looks like this:  
    ***Pmax=? [ F "goal" ]***,  where we use ***label "goal" = ((x>=2) | (y>=2) | (z>=2));***.  


We then generate a decision tree to get a better understanding of the actions and state values, that lead to goal states:

```bash
./countexex --model "{PathToCountexex}/countexex/examples/cycle.nm" --propertyMax max
```

We get the following decision tree: 
<img src="cycle_graph-1.png" alt="Decision tree">

The decision tree teaches us that regardless of the current state, the action of performing a self-loop is irrelevant for reaching a target state. In combination with the information from the model file, we learn that state-action pairs containing the alternative action are relevant.

## 2. Coin flips

Again, we first take a look at the model file coin_flips.nm. This example models a system that counts from zero to N=20. The actions with the flip identifier are executed simultaneously for the module count and coin. Thus, either the value of x is increased by one and the variables heads and tails are set to opposite values for action flip, or both heads and tails are set to false by choosing action b.

Again, we partially visualize the corresponding MDP: 
<img src="coin_flips_mdp-1.png" alt="Coin flips MDP">

We check the following property:   
    ***Pmax=? [ F "goal" ]***,  where we use ***label "goal" = x=N;***.  

We obtain the following decision tree: 
<img src="coin_flips_graph-1.png" alt="Decision tree">

The decision tree teaches us that choosing action flip is always good. On the other hand, selecting action b if we are in a state, where tails and heads are false, is bad, as this means that we loop and don't make progress towards our goal of reaching state x=N.