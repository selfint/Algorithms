# NEAT
This is a simple data-oriented approach to implementing
NEAT in python. The goal is to implement all features of
the algorithm: 
* topological evolution
* speciation
* fitness sharing
* innovation history

## Implementation outline
Each agent is a list of innovations as indexes, and
a corresponding array of weights and bias values. To
evaluate the output of an agent, innovations will get
pulled from a dictionary and calculated using recursive
feed forward.

## Algorithm
1. Generate n based populations with input and output
nodes, but no connections.
2. Split population into species using genetic distance. 
3. Evaluate each genome in a separate environment.
4. Normalize fitness of genome with fitness sharing in
each species.
5. Generate new generation using mutation and 
crossover.
