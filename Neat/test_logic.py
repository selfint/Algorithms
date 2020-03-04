import numpy as np
import pytest
import gym

from logics import (
    Environments,
    BaseNodes,
    ConnectionDirections,
    ConnectionWeights,
    ConnectionStates,
    feed_forward,
    evaluate_networks,
    split_into_species,
)


def generate_temp_network():
    connection_directions = np.array(
        [
            [-1, 6],
            [0, 6],
            [1, 6],
            [2, 6],
            [3, 6],
            [-1, 7],
            [0, 7],
            [1, 7],
            [2, 7],
            [3, 7],
            [-1, 4],
            [7, 4],
            [7, 5],
            [8, 4],
            [8, 5],
        ]
    )
    connection_weights = ConnectionWeights(np.random.random(connection_directions.size))
    connection_states = ConnectionStates(
        np.random.randint(2, size=connection_directions.size)
    )
    base_nodes = BaseNodes([0, 1, 2, 3], [4, 5])
    return connection_directions, connection_weights, connection_states, base_nodes


def test_feed_forward():
    (
        connections,
        connection_weights,
        connection_states,
        base_nodes,
    ) = generate_temp_network()
    inputs = np.random.random(size=len(base_nodes.input_nodes))
    result = feed_forward(
        inputs, connections, connection_weights, connection_states, base_nodes
    )
    assert np.sum(result) > 0


def test_evaluate_network():
    environments = Environments([gym.make("CartPole-v0") for _ in range(10)])
    (
        connections,
        connection_weights,
        connection_states,
        base_nodes,
    ) = generate_temp_network()
    network_amount = 10
    evaluate_networks(
        environments,
        [connections] * network_amount,
        [connection_weights] * network_amount,
        [connection_states] * network_amount,
        base_nodes,
        200,
        100,
    )


def test_split_into_species():
    connections, connection_weights, _, _ = generate_temp_network()
    genetic_distance_parameters = {
        "excess_constant": 1.0,
        "disjoint_constant": 1.0,
        "weight_bias_constant": 1.0,
        "large_genome_size": 20,
        "threshold": 0.15,
    }
    network_amount = 10
    species = split_into_species(
        [connections] * network_amount,
        [connection_weights] * network_amount,
        genetic_distance_parameters,
    )
    assert species == [0] * network_amount


print(test_split_into_species())
