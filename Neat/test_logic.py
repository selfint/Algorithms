import numpy as np
import pytest
import gym

from logics import (
    Environments,
    BaseNodes,
    ConnectionInnovation,
    ConnectionProperties,
    Nodes,
    feed_forward,
    evaluate_networks,
    split_into_species,
)


def generate_temp_network():
    connections = [
        ConnectionInnovation(0, 6),
        ConnectionInnovation(1, 6),
        ConnectionInnovation(2, 6),
        ConnectionInnovation(3, 6),
        ConnectionInnovation(0, 7),
        ConnectionInnovation(1, 7),
        ConnectionInnovation(2, 7),
        ConnectionInnovation(3, 7),
        ConnectionInnovation(7, 4),
        ConnectionInnovation(7, 5),
        ConnectionInnovation(8, 4),
        ConnectionInnovation(8, 5),
    ]
    connection_data = ConnectionProperties(
        np.random.random(size=11),
        [True, True, True, True, False, True, True, True, True, True, True],
    )
    base_nodes = BaseNodes([0, 1, 2, 3], [4, 5])
    return connections, connection_data, base_nodes


def test_feed_forward():
    connections, connection_data, base_nodes = generate_temp_network()
    inputs = np.random.random(size=len(base_nodes.input_nodes))
    assert np.sum(feed_forward(inputs, connections, connection_data, base_nodes)) > 0


def test_evaluate_network():
    environments = Environments([gym.make("CartPole-v0") for _ in range(10)])
    connections, connection_data, base_nodes = generate_temp_network()
    network_amount = 10
    evaluate_networks(
        environments,
        [connections] * network_amount,
        [connection_data] * network_amount,
        base_nodes,
        200,
        100,
    )


def test_split_into_species():
    connections, connection_data, _ = generate_temp_network()
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
        [connection_data] * network_amount,
        genetic_distance_parameters,
    )
    assert species == [0] * network_amount
