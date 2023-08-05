# Examples 

## 1. Cycle
At first, we take a look at the file cycle.nm, which can be found in this folder. The mdp model file consists of three modules. Each module manipulates one of the variables x, y, z. If the respective variable is zero, then it is either increased to one or two with equal probability by choosing the first action, or remains unchanged by choosing the second action.  
To get a better understanding of the model, we partially created its MDP graph:

![Cycle mdp:](cycle_mdp.pdf)

Assume that we are interested in reaching a state where at least one of the variables is set to 2 with maximal probability. Thus our property looks like this:  
    ***Pmax=? [ F "goal" ]***,  where we use ***label "goal" = ((x>=2) | (y>=2) | (z>=2));***.  
We then generate a decision tree to get a better understanding of the actions and state values, that lead to goal states:

```bash
./countexex --model "{PathToCountexex}/countexex/examples/cycle.nm" --propertyMax max
```

We get the following decision tree: 
![Decision tree:](graph.pdf)

The decision tree teaches us that regardless of the current state, the action of performing a self-loop is irrelevant to reaching the target state. In combination with the information from the model file, we learn that state-action pairs containing the alternative action are relevant.

## 2. Coin flips

