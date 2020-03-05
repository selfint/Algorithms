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
    for _ in range(network_amount):
        connection_directions = ConnectionDirections(
            all_possible_connections[
                np.random.choice(
                    all_possible_connections.shape[0], connection_amount, replace=False
                ),
                :,
            ]
        )
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
    base_nodes = BaseNodes(list(range(input_amount)), list(range(output_amount)))
    return (
        networks_connection_directions,
        networks_connection_weights,
        networks_connection_states,
        base_nodes,
    )


def test_feed_forward():
    (
        connections,
        connection_weights,
        connection_states,
        base_nodes,
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
    ) = generate_temp_network(network_amount)
    result = evaluate_networks(
        environments,
        networks_connections,
        networks_connection_weights,
        networks_connection_states,
        base_nodes,
        200,
        100,
    )
    print(result)


def test_split_into_species():
    network_amount = 100
    networks_connections, networks_connection_weights, _, _ = generate_temp_network(
        network_amount=network_amount, max_hidden_amount=4, connection_amount=30
    )
    genetic_distance_parameters = {
        "excess_constant": 1.0,
        "disjoint_constant": 1.0,
        "weight_bias_constant": 0.4,
        "large_genome_size": 20,
        "threshold": 3.0,
    }
    result = split_into_species(
        networks_connections, networks_connection_weights, genetic_distance_parameters,
    )
    print(result)


if __name__ == "__main__":
    test_feed_forward()
    test_evaluate_network()
    test_split_into_species()

