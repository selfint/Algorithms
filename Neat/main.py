from typing import List, Tuple

import gym
import numpy as np
import pygraphviz as pgv

from logics import evaluate_networks, feed_forward, new_generation, split_into_species
from structs import (
    BaseNodes,
    ConnectionDirections,
    ConnectionInnovationsMap,
    ConnectionStates,
    ConnectionWeights,
    Environments,
    NodeInnovationsMap,
)

# parameters
NETWORK_AMOUNT = 50
ENVIRONMENT_NAME = "MountainCar-v0"
GENETIC_DISTANCE_PARAMETERS = {
    "excess_constant": 1.0,
    "disjoint_constant": 1.0,
    "weight_bias_constant": 0.4,
    "large_genome_size": 20,
    "threshold": 3.0,
    "interspecies_mating_rate": 0.001,
}
MUTATION_PARAMETERS = {
    "permutation_rate": 0.7,
    "random_weight_rate": 0.1,
    "new_connection_rate": 0.05,
    "split_connection_rate": 0.03,
    "large_species": 5,
}
CROSSOVER_RATE = 0.75
GENERATIONS = 100


if __name__ == "__main__":

    # generate environments
    environments = Environments(
        [gym.make(ENVIRONMENT_NAME) for _ in range(NETWORK_AMOUNT)]
    )

    # generate empty network arrays
    networks_connection_directions = []
    networks_connection_weights = []
    networks_connection_states = []
    for _ in range(NETWORK_AMOUNT):
        networks_connection_directions.append(
            ConnectionDirections(np.array([], dtype=np.int).reshape(-1, 2))
        )
        networks_connection_weights.append(ConnectionWeights(np.array([])))
        networks_connection_states.append(ConnectionStates(np.array([])))

    # generate base nodes for environment
    test_env = environments.environments[0]

    ## input nodes are the observation space of the environment
    input_node_amount = test_env.reset().size

    ## output nodes are the
    output_node_amount = test_env.action_space.n
    base_nodes = BaseNodes(
        np.arange(input_node_amount, dtype=np.int),
        np.arange(
            input_node_amount, input_node_amount + output_node_amount, dtype=np.int
        ),
    )

    # generate innovation history maps
    global_innovation_history = ConnectionInnovationsMap(dict())
    global_node_innovation_history = NodeInnovationsMap(dict())

    # init variables
    species_reps: List[Tuple[ConnectionDirections, ConnectionWeights]] = []
    average_scores: List[float] = []
    max_scores: List[float] = []

    # train networks
    for generation in range(GENERATIONS):
        average_scores = []
        max_scores = []

        # get network rewards from environments
        networks_scores = evaluate_networks(
            environments,
            networks_connection_directions,
            networks_connection_weights,
            networks_connection_states,
            base_nodes,
            max_steps=200,
            episodes=1,
            score_exponent=1,
            render=False,
        )

        # draw best network
        best_network = networks_scores.argmax()
        best_network_connection_directions = networks_connection_directions[
            best_network
        ]
        best_network_connection_weights = networks_connection_weights[best_network]
        best_network_connection_states = networks_connection_states[best_network]
        G = pgv.AGraph(directed=True)
        for (source, dest), weight, enabled in zip(
            best_network_connection_directions.directions,
            best_network_connection_weights.weights,
            best_network_connection_states.states,
        ):
            color = "black" if not enabled else "blue" if weight > 0 else "red"
            penwidth = abs(weight) * 2

            G.add_edge(source, dest, color=color, penwidth=penwidth)
        G.draw(f"genomes/best_network_gen_{generation}.png", prog="fdp")

        # show best network perform
        # evaluate_networks(
        #     Environments([gym.make(ENVIRONMENT_NAME)]),
        #     [best_network_connection_directions],
        #     [best_network_connection_weights],
        #     [best_network_connection_states],
        #     base_nodes,
        #     max_steps=200,
        #     episodes=1,
        #     render=True,
        # )

        # generate next generation
        average_scores.append(np.average(networks_scores))
        max_scores.append(np.max(networks_scores))
        networks_species, species_reps = split_into_species(
            networks_connection_directions,
            networks_connection_weights,
            global_innovation_history,
            GENETIC_DISTANCE_PARAMETERS,
            previous_generation_species_reps=species_reps,
        )

        species_amounts = {
            species: species_amount
            for species, species_amount in zip(
                *np.unique(networks_species, return_counts=True)
            )
        }

        species_scores = {
            species: np.average(networks_scores[networks_species == species])
            for species in networks_species
        }

        print(
            f"\n-- Generation {generation} --"
            f"\nbest score: {max(networks_scores)}"
            f"\naverage score: {np.average(networks_scores)}"
            f"\nspecies: {species_amounts}"
            f"\naverage species score: {species_scores}"
            f"\nconnection innovations:\n\t{global_innovation_history}"
            f"\nnode innovations:\n\t{global_node_innovation_history}"
            "\n"
        )
        (
            networks_connection_directions,
            networks_connection_weights,
            networks_connection_states,
            global_innovation_history,
        ) = new_generation(
            networks_connection_directions,
            networks_connection_weights,
            networks_connection_states,
            base_nodes,
            networks_scores,
            networks_species,
            global_innovation_history,
            global_node_innovation_history,
            GENETIC_DISTANCE_PARAMETERS,
            MUTATION_PARAMETERS,
            CROSSOVER_RATE,
        )
