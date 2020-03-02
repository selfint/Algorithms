"""
Contains all logical operations to that are needed to transform the data
"""
from typing import List, Dict

import numpy as np
import gym
from gym import spaces

from structs import (
    BaseNodes,
    ConnectionInnovation,
    ConnectionProperties,
    Environments,
    NodeProperties,
    NodeInnovation,
    Nodes,
)


def feed_forward(
    inputs: np.ndarray,
    connections: List[ConnectionInnovation],
    connection_data: ConnectionProperties,
    node_data: List[NodeProperties],
    base_nodes: BaseNodes,
) -> np.ndarray:
    """Calculate the output of a network using recursion

    Arguments:
        inputs {np.ndarray} -- network inputs
        connections {List[ConnectionInnovation]} -- connections between nodes
        connection_data {ConnectionProperties} -- connection weights and biases
        node_data {NodeProperties} -- node biases
        base_nodes {BaseNodes} -- input and output nodes

    Returns:
        np.ndarray -- network output
    """
    return [
        _get_node_output(
            node_index, inputs, connections, connection_data, node_data, base_nodes, []
        )
        for node_index in base_nodes.output_nodes
    ]


def _get_node_output(
    node_index: int,
    inputs: np.ndarray,
    connections: List[ConnectionInnovation],
    connection_data: ConnectionProperties,
    node_data: NodeProperties,
    base_nodes: BaseNodes,
    ignore_connections: List[ConnectionInnovation],
) -> float:
    """helper function to get output of a single node using recursion

    Arguments:
        node_index {int} -- index of node to get output of
        inputs {np.ndarray} -- network inputs
        connections {List[ConnectionInnovation]} -- connections between nodes
        connection_data {ConnectionProperties} -- connection weights and biases
        node_data {NodeProperties} -- node biases
        base_nodes {BaseNodes} -- input and output nodes
        ignore_connections {List[ConnectionInnovation]} -- connections previously
                                                           calculated

    Returns:
        float -- output of node
    """
    # input nodes are just placeholders for the network input
    if node_index in base_nodes.input_nodes:
        return inputs[node_index]

    # since input nodes don't have properties, the node_properties_index is offset by
    # the input node amount
    node_properties_index = node_index - len(base_nodes.input_nodes)
    return node_data.activations[node_properties_index](  # activation function of node
        np.sum(  # weighted sum of the outputs of all nodes outputing into node
            [
                _get_node_output(
                    connection.src,
                    inputs,
                    connections,
                    connection_data,
                    node_data,
                    base_nodes,
                    ignore_connections + [connection],  # mark connection as to-ignore
                )
                * connection_weight  # weight the output
                for connection, connection_weight, connection_enabled in zip(
                    connections, connection_data.weights, connection_data.enabled
                )
                if (
                    connection.dst == node_index  # get connections outputing into node
                    and connection
                    not in ignore_connections  # ignore accounted for connections
                    and connection_enabled  # ignore disabled connections
                )
            ]
        )
        + node_data.biases[node_properties_index]  # add node bias
    )


def transform_network_output(network_output: List[float]) -> spaces.Discrete:
    return np.argmax(network_output)


def evaluate_networks(
    environments: Environments,
    connections: List[List[ConnectionInnovation]],
    connection_data: List[ConnectionProperties],
    node_data: List[NodeProperties],
    base_nodes: BaseNodes,
    max_steps: int,
    episodes: int,
    render: bool = False,
) -> List[float]:
    """calculate the average episode reward for each network

    Arguments:
        environments {Environments} -- gym environments
        connections {List[List[ConnectionInnovation]]} -- network connections
        connection_data {List[ConnectionProperties]} -- network connection data
        node_data {List[NodeProperties]} -- network node data
        base_nodes {BaseNodes} -- input and output nodes
        max_steps {int} -- step limit for each episode
        episodes {int} -- number of episodes to test each network

    Keyword Arguments:
        render {bool} -- render episodes (default: {False})

    Returns:
        List[float] -- average network rewards over n episodes
    """
    return [
        np.average(
            [
                _get_episode_reward(
                    environment,
                    max_steps,
                    network_connections,
                    network_connection_data,
                    network_node_data,
                    base_nodes,
                    render,
                )
                for _ in range(episodes)
            ]
        )
        for (
            environment,
            network_connections,
            network_connection_data,
            network_node_data,
        ) in zip(environments.environments, connections, connection_data, node_data)
    ]


def _get_episode_reward(
    environment: gym.Env,
    max_steps: int,
    connections: List[ConnectionInnovation],
    connection_data: ConnectionProperties,
    node_data: List[NodeProperties],
    base_nodes: BaseNodes,
    render: bool = False,
) -> float:
    """helper function that runs an episode and returns the episode rewards

    Arguments:
        environment {gym.Env} -- gym environment
        max_steps {int} -- limit of steps to take in episode
        connections {List[ConnectionInnovation]} -- network connections
        connection_data {ConnectionProperties} -- network connection data
        node_data {List[NodeProperties]} -- network node data
        base_nodes {BaseNodes} -- network input and output nodes

    Returns:
        float -- network episode reward
    """

    # reset environment
    episode_reward = 0
    observation = environment.reset()

    # play through simulation
    for _ in range(max_steps):
        if render:
            environment.render()

        network_output = feed_forward(
            observation, connections, connection_data, node_data, base_nodes,
        )
        action = transform_network_output(network_output)
        observation, reward, done, _ = environment.step(action)

        episode_reward += reward

        if done:
            break

    environment.close()
    return episode_reward


def split_into_species(
    connections: List[List[ConnectionInnovation]],
    connection_data: List[ConnectionProperties],
    node_data: List[NodeProperties],
    nodes: List[Nodes],
    genetic_distance_parameters: Dict[str, float],
) -> List[int]:
    genetic_distance_threshold = genetic_distance_parameters["threshold"]
    species = []
    species_reps = []
    for (
        network_connections,
        network_connection_data,
        network_node_data,
        network_nodes,
    ) in zip(connections, connection_data, node_data, nodes):

        # check genetic distance to all species reps
        for (
            species_rep_index,
            (rep_connections, rep_connection_data, rep_nodes, rep_node_data),
        ) in enumerate(species_reps):
            if (
                _genetic_distance(
                    network_connections,
                    network_connection_data,
                    network_node_data,
                    network_nodes,
                    rep_connections,
                    rep_connection_data,
                    rep_node_data,
                    rep_nodes,
                    genetic_distance_parameters,
                )
                < genetic_distance_threshold
            ):
                species.append(species_rep_index)
                break
        else:

            # generate a new rep for new species when a network doesn't match any
            # other species rep
            species.append(len(species))
            species_reps.append(
                (
                    network_connections,
                    network_connection_data,
                    network_nodes,
                    network_node_data,
                )
            )

    return species


def _genetic_distance(
    connections_a: List[ConnectionInnovation],
    connections_data_a: ConnectionProperties,
    node_data_a: NodeProperties,
    nodes_a: Nodes,
    connections_b: List[ConnectionInnovation],
    connections_data_b: ConnectionProperties,
    node_data_b: NodeProperties,
    nodes_b: Nodes,
    genetic_distance_parameters: Dict[str, float],
) -> float:
    # TODO: split into separate functions
    # innovation index that splits disjoint and excess genes
    last_common_innovation = min(max(nodes_a.nodes), max(nodes_b.nodes))

    # get disjoint and excess amounts
    # get average bias difference
    disjoint_amount = 0
    excess_amount = 0
    bias_differences = []
    for a_node, a_node_data in zip(nodes_a.nodes, node_data_a.biases):
        if a_node in nodes_b.nodes:
            bias_differences.append(
                a_node_data - node_data_b.biases[nodes_b.nodes.index(a_node)]
            )
        elif a_node < last_common_innovation:
            disjoint_amount += 1
        else:
            excess_amount += 1
    bias_difference = np.average(bias_differences)

    for b_node in nodes_b.nodes:
        if b_node in nodes_a.nodes:
            # matching nodes were all found when iterating through nodes_a
            continue
        if b_node < last_common_innovation:
            disjoint_amount += 1
        else:
            excess_amount += 1

    # get average weight difference
    weight_differences = []
    for a_connection, a_connection_data in zip(
        connections_a, connections_data_a.weights
    ):
        if a_connection in connections_b:
            weight_differences.append(
                a_connection_data
                - connections_data_b.weights[connections_b.index(a_connection)]
            )
    weight_difference = np.average(weight_differences)

    # calculate genetic distance
    c1 = genetic_distance_parameters["excess_constant"]
    c2 = genetic_distance_parameters["disjoint_constant"]
    c3 = genetic_distance_parameters["weight_bias_constant"]
    large_genome_size = genetic_distance_parameters["large_genome_size"]
    weight_bias_distance = (abs(bias_difference) + abs(weight_difference)) / 2
    largest_genome_size = max(max(nodes_a.nodes), max(nodes_b.nodes))

    # don't normalize excess and disjoint difference in small genomes
    if largest_genome_size < large_genome_size:
        genetic_distance = (
            c1 * excess_amount + c2 * disjoint_amount + c3 * (weight_bias_distance)
        )
    else:
        genetic_distance = (
            c1 * excess_amount / largest_genome_size
            + c2 * disjoint_amount / largest_genome_size
            + c3 * (weight_bias_distance)
        )

    return genetic_distance
