import numpy as np
import pytest
import gym

from logics import (
    Environments,
    BaseNodes,
    ConnectionInnovation,
    ConnectionProperties,
    NodeProperties,
    feed_forward,
    evaluate_networks,
)


def generate_temp_network():
    node_data = NodeProperties(
        np.random.random(size=9), [lambda x: (1.0 / (1.0 + np.exp(-x)))] * 9,
    )
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
    return connections, connection_data, node_data, base_nodes


def test_feed_forward():
    inputs = np.array([0.74])
    connections, connection_data, node_data, base_nodes = generate_temp_network()
    feed_forward(inputs, connections, connection_data, node_data, base_nodes)


def test_evaluate_network():
    environments = Environments([gym.make("CartPole-v0") for _ in range(10)])
    connections, connection_data, node_data, base_nodes = generate_temp_network()
    network_amount = 10
    rewards = evaluate_networks(
        environments,
        [connections] * network_amount,
        [connection_data] * network_amount,
        [node_data] * network_amount,
        base_nodes,
        200,
        100
    )
    print(rewards)


test_evaluate_network()
