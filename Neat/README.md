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

## Data Structures

### Neural Network

* Input Nodes - List of input nodes
* Hidden Nodes - List of hidden nodes
* Output Nodes - List of output nodes
* Connections - List of connections

### Innovations

* Max Node Innovation ID - Counter of node innovations
* Max Connection Innovation ID - Counter of connection innovations
* Node Innovations - List of node innovations
* Connection Innovations - List of connection innovations

### Input Node

* Innovation ID
* Output

### Hidden Node

* Innovation ID
* Bias
* Output

### Output Node

* Innovation ID
* Bias
* Output

### Connection

* Innovation ID
* Source
* Destination
* Weight
* Enabled

### Connection Innovation

* Innovation ID
* Source
* Destination

### Node Innovation

No need to keep track of past node innovations,so a simple counter will suffice to get the next node innovation ID on mutation.

### Environment

OpenAI gym environment.

## Logic

### Network Outputs

Calculates the output of each network for each environment. Runs the environments and keeps track of the rewards for each network.

#### Inputs

* Networks
* Environment States

#### Outputs

* Environment Rewards

### Network Speciation

Splits the networks into species, with a representative for each species, using genetic distance.

#### Inputs

* Networks

#### Outputs

* Species - Dict[Network, SpeciesID]
* SpeciesReps - Dict[SpeciesID, SpeciesRep]

### Network Fitness Evaluator

Evaluate the fitness of each network using the rewards from their respective environments.

#### Inputs

* Networks
* Environment Rewards

#### Outputs

* Network Fitness

### Network Fitness Normalizer

Normalize the fitness of each network in each species.

#### Inputs

* Networks
* Network Species
* Network Fitness

#### Outputs

* Normalized Network Fitness

### New Generation Generator

Generates a new generation using crossover and mutation.

#### Inputs

* Innovations
* Networks
* Network Species
* Normalized Network Fitness

### New Node Innovation

Handles node mutation generation and logging.

#### Inputs

* Innovation ID of the connection that split.

#### Outputs

* Node Innovation
