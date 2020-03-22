import numpy as np
import pytest
import gym
import matplotlib.pyplot as plt
import pygraphviz as pgv

from logics import (
    Environments,
    BaseNodes,
    ConnectionDirections,
    ConnectionWeights,
    ConnectionStates,
    ConnectionInnovationsMap,
    NodeInnovationsMap,
    feed_forward,
    evaluate_networks,
    split_into_species,
    new_generation,
)


def generate_temp_network(
    network_amount=1,
    input_amount=4,
    output_amount=2,
    max_hidden_amount=6,
    connection_amount=20,
):
    networks_connection_directions = []
    networks_connection_weights = []
    networks_connection_states = []
    all_possible_connections = (
        np.mgrid[
            -1 : input_amount + output_amount + max_hidden_amount,
            input_amount : input_amount + output_amount + max_hidden_amount,
        ]
        .reshape(2, -1)
        .T
    )
    all_chosen_connections = []
    for _ in range(network_amount):
        chosen_connections = all_possible_connections[
            np.random.choice(
                all_possible_connections.shape[0], connection_amount, replace=False
            ),
            :,
        ]
        all_chosen_connections.append(chosen_connections)
        connection_directions = ConnectionDirections(chosen_connections)
        connection_weights = ConnectionWeights(
            np.random.normal(
                loc=0, scale=0.1, size=connection_directions.directions.shape[0]
            )
        )
        connection_states = ConnectionStates(
            np.random.randint(2, size=connection_directions.directions.shape[0])
        )
        networks_connection_directions.append(connection_directions)
        networks_connection_weights.append(connection_weights)
        networks_connection_states.append(connection_states)
    base_nodes = BaseNodes(
        np.array(range(input_amount)),
        np.array(range(input_amount, input_amount + output_amount)),
    )
    all_chosen_connections = np.array(all_chosen_connections).reshape(-1, 2)
    innovation_map = dict()
    counter = 0
    for connection in all_chosen_connections:
        tuple_connection = tuple(connection)
        if tuple_connection in innovation_map:
            continue
        else:
            innovation_map[tuple_connection] = counter
            counter += 1
    global_innovation_history = ConnectionInnovationsMap(innovation_map)
    global_node_innovation_history = NodeInnovationsMap(dict())
    return (
        networks_connection_directions,
        networks_connection_weights,
        networks_connection_states,
        base_nodes,
        global_innovation_history,
        global_node_innovation_history,
    )


def test_feed_forward():
    (
        connections,
        connection_weights,
        connection_states,
        base_nodes,
        _,
        _,
    ) = generate_temp_network()
    inputs = np.random.random(size=len(base_nodes.input_nodes))
    result = feed_forward(
        inputs, connections[0], connection_weights[0], connection_states[0], base_nodes
    )
    print(result)
    assert np.sum(result) > 0


def test_evaluate_network():
    network_amount = 10
    environments = Environments(
        [gym.make("CartPole-v0") for _ in range(network_amount)]
    )
    (
        networks_connections,
        networks_connection_weights,
        networks_connection_states,
        base_nodes,
        _,
        _,
    ) = generate_temp_network(network_amount)
    result = evaluate_networks(
        environments,
        networks_connections,
        networks_connection_weights,
        networks_connection_states,
        base_nodes,
        200,
        10,
    )
    print(result)


def test_split_into_species():
    network_amount = 100
    (
        networks_connections,
        networks_connection_weights,
        _,
        _,
        global_innovation_history,
        _,
    ) = generate_temp_network(
        network_amount=network_amount, max_hidden_amount=4, connection_amount=0
    )
    genetic_distance_parameters = {
        "excess_constant": 1.0,
        "disjoint_constant": 1.0,
        "weight_bias_constant": 0.4,
        "large_genome_size": 20,
        "threshold": 3.0,
    }
    result = split_into_species(
        networks_connections,
        networks_connection_weights,
        global_innovation_history,
        genetic_distance_parameters,
    )
    print(result)


def test_new_generation():

    # parameters
    generations = 20
    network_amount = 50
    environment_name = "MountainCar-v0"
    genetic_distance_parameters = {
        "excess_constant": 1.0,
        "disjoint_constant": 1.0,
        "weight_bias_constant": 0.4,
        "large_genome_size": 20,
        "threshold": 3.0,
        "interspecies_mating_rate": 0.001,
    }
    mutation_parameters = {
        "permutation_rate": 0.7,
        "random_weight_rate": 0.1,
        "new_connection_rate": 0.05,
        "split_connection_rate": 0.03,
        "large_species": 5,
    }
    crossover_parameters = {
        "crossover_rate": 0.75,
        "disable_connection_rate": 0.75
    }

    # build environments and networks
    environments = Environments(
        [gym.make(environment_name) for _ in range(network_amount)]
    )
    (
        networks_connection_directions,
        networks_connection_weights,
        networks_connection_states,
        base_nodes,
        global_innovation_history,
        global_node_innovation_history,
    ) = generate_temp_network(
        network_amount=network_amount,
        input_amount=2,
        output_amount=3,
        max_hidden_amount=0,
        connection_amount=0,
    )

    # initialize species reps as an empty list
    species_reps = []

    # logging
    average_scores = []
    max_scores = []

    # test new generation
    for generation in range(generations):

        # evaluate networks
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

        # log scores
        average_scores.append(np.average(networks_scores))
        max_scores.append(np.max(networks_scores))

        # split into species, and get the new species reps
        networks_species, species_reps = split_into_species(
            networks_connection_directions,
            networks_connection_weights,
            global_innovation_history,
            genetic_distance_parameters,
            previous_generation_species_reps=species_reps,
        )

        # generate new networks
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
            genetic_distance_parameters,
            mutation_parameters,
            crossover_parameters,
        )

    # plot results
    plt.plot(average_scores, label="avg")
    plt.plot(max_scores, label="max")
    plt.legend()
    plt.title(f"Score history from training for {generations} generations")
    plt.show()