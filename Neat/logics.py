"""
Contains all logical operations to that are needed to transform the data
"""
from typing import List, Dict, Tuple

import numpy as np
import gym
from gym import spaces

from structs import (
    BaseNodes,
    ConnectionInnovation,
    ConnectionWeights,
    ConnectionStates,
    ConnectionDirections,
    Environments,
    Nodes,
)


def feed_forward(
    inputs: np.ndarray,
    connections: List[ConnectionInnovation],
    connections_weights: ConnectionWeights,
    connections_states: ConnectionStates,
    base_nodes: BaseNodes,
) -> np.ndarray:
    """Calculate the output of a network using recursion

    Arguments:
        inputs {np.ndarray} -- network inputs
        connections {List[ConnectionInnovation]} -- connections between nodes
        connection_data {ConnectionProperties} -- connection weights and biases
        base_nodes {BaseNodes} -- input and output nodes

    Returns:
        np.ndarray -- network output
    """
    return [
        _get_node_output(
            node_id,
            inputs,
            connections,
            connections_weights,
            connections_states,
            base_nodes,
            ignore_connections=[],
        )
        for node_id in base_nodes.output_nodes
    ]


def _activation_function(x: float) -> float:
    """
    Sigmoid activation
    """
    return 1.0 / (1.0 + np.exp(-x))


def _get_node_output(
    node_id: int,
    inputs: np.ndarray,
    connections: List[ConnectionInnovation],
    connection_weights: ConnectionWeights,
    connection_states: ConnectionStates,
    base_nodes: BaseNodes,
    ignore_connections: List[ConnectionInnovation],
) -> float:
    """helper function to get output of a single node using recursion

    Arguments:
        node_id {int} -- index of node to get output of
        inputs {np.ndarray} -- network inputs
        connections {List[ConnectionInnovation]} -- connections between nodes
        connection_data {ConnectionProperties} -- connection weights and biases
        base_nodes {BaseNodes} -- input and output nodes
        ignore_connections {List[ConnectionInnovation]} -- connections previously
                                                           calculated

    Returns:
        float -- output of node
    """
    # input nodes are just placeholders for the network input
    if node_id in base_nodes.input_nodes:
        return inputs[node_id]

    # the output of the bias node is always 1
    if node_id == base_nodes.bias_node:
        return 1.0

    # since input nodes don't have properties, the node_properties_index is offset by
    # the input node amount
    return _activation_function(  # activation function of node
        np.sum(  # weighted sum of the outputs of all nodes outputing into node
            [
                _get_node_output(
                    connection.src,
                    inputs,
                    connections,
                    connection_weights,
                    connection_states,
                    base_nodes,
                    ignore_connections + [connection],  # mark connection as to-ignore
                )
                * connection_weight  # weight the output
                for connection, connection_weight, connection_state in zip(
                    connections, connection_weights, connection_states,
                )
                if (
                    connection.dst == node_id  # get connections outputing into node
                    and connection
                    not in ignore_connections  # ignore accounted for connections
                    and connection_state  # ignore disabled connections
                )
            ]
        )
    )


def transform_network_output_discrete(network_output: List[float]) -> spaces.Discrete:
    return np.argmax(network_output)


def evaluate_networks(
    environments: Environments,
    networks_connections: List[List[ConnectionInnovation]],
    networks_connection_weights: List[ConnectionWeights],
    networks_connection_states: List[ConnectionStates],
    base_nodes: BaseNodes,
    max_steps: int,
    episodes: int,
    render: bool = False,
) -> List[float]:
    """calculate the average episode reward for each network

    Arguments:
        environments {Environments} -- gym environments
        networks_connections {List[List[ConnectionInnovation]]} -- networks connections
        networks_connection_data {List[ConnectionProperties]} -- networks connection data
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
                    network_connection_weights,
                    network_connection_states,
                    base_nodes,
                    render,
                )
                for _ in range(episodes)
            ]
        )
        for (
            environment,
            network_connections,
            network_connection_weights,
            network_connection_states,
        ) in zip(
            environments.environments,
            networks_connections,
            networks_connection_weights,
            networks_connection_states,
        )
    ]


def _get_episode_reward(
    environment: gym.Env,
    max_steps: int,
    connections: List[ConnectionInnovation],
    connections_weights: ConnectionWeights,
    connections_states: ConnectionStates,
    base_nodes: BaseNodes,
    render: bool = False,
) -> float:
    """helper function that runs an episode and returns the episode rewards

    Arguments:
        environment {gym.Env} -- gym environment
        max_steps {int} -- limit of steps to take in episode
        connections {List[ConnectionInnovation]} -- network connections
        connection_data {ConnectionProperties} -- network connection data
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
            observation,
            connections,
            connections_weights,
            connections_states,
            base_nodes,
        )
        action = transform_network_output_discrete(network_output)
        observation, reward, done, _ = environment.step(action)

        episode_reward += reward

        if done:
            break

    environment.close()
    return episode_reward


def split_into_species(
    networks_connections: List[List[ConnectionInnovation]],
    networks_connection_weights: List[ConnectionWeights],
    genetic_distance_parameters: Dict[str, float],
) -> List[int]:
    """assign a species to each network

    Arguments:
        networks_connections {List[List[ConnectionInnovation]]} -- connections of each network
        networks_connection_data {List[ConnectionProperties]} -- data of connections of each network
        networks_nodes {List[Nodes]} -- nodes of each network
        genetic_distance_parameters {Dict[str, float]} -- hyperparameters for genetic distance

    Returns:
        List[int] -- species of each network
    """
    genetic_distance_threshold = genetic_distance_parameters["threshold"]
    species = []
    species_reps = []
    for (network_connections, network_connection_weights,) in zip(
        networks_connections, networks_connection_weights,
    ):

        # check genetic distance to all species reps
        for (species_rep_index, (rep_connections, rep_connection_data),) in enumerate(
            species_reps
        ):
            if (
                _genetic_distance(
                    network_connections,
                    network_connection_weights,
                    rep_connections,
                    rep_connection_data,
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
            species_reps.append((network_connections, network_connection_weights,))

    return species


def _genetic_distance(
    network_a_connections: List[ConnectionInnovation],
    network_a_connection_weights: ConnectionWeights,
    network_b_connections: List[ConnectionInnovation],
    network_b_connection_weights: ConnectionWeights,
    genetic_distance_parameters: Dict[str, float],
) -> float:
    """calculate the genetic distance between two networks

    Arguments:
        network_a_connections {List[List[ConnectionInnovation]]} -- connections of network a
        network_a_connections_data {List[ConnectionProperties]} -- data of connections of network a
        network_a_nodes {List[Nodes]} -- nodes of network a
        network_b_connections {List[List[ConnectionInnovation]]} -- connections of network b
        network_b_connections_data {List[ConnectionProperties]} -- data of connections of network b
        network_b_nodes {List[Nodes]} -- nodes of network b
        genetic_distance_parameters {Dict[str, float]} -- hyperparameters for genetic distance

    Returns:
        float -- genetic distance
    """
    # TODO: split into separate functions
    # TODO: only accept network weights instead of weights and enabled
    # innovation index that splits disjoint and excess genes
    # TODO: calculate last_common_innovation using connection innovations
    # last_common_innovation = min(max(network_a_nodes.nodes), max(network_b_nodes.nodes))

    # get disjoint and excess amounts
    # get average weight difference
    disjoint_amount = 0
    excess_amount = 0
    weight_differences = []

    for a_connection, a_connection_weight in zip(
        network_a_connections, network_a_connection_weights
    ):
        if a_connection in network_b_connections:
            weight_differences.append(
                a_connection_weight
                - network_b_connection_weights.weights[
                    network_b_connections.index(a_connection)
                ]
            )
        # TODO: add disjoint and excess calculation
    weight_difference = np.average(weight_differences)

    # calculate genetic distance
    c1 = genetic_distance_parameters["excess_constant"]
    c2 = genetic_distance_parameters["disjoint_constant"]
    c3 = genetic_distance_parameters["weight_bias_constant"]
    large_genome_size = genetic_distance_parameters["large_genome_size"]
    weight_difference = abs(weight_difference) / 2
    # TODO: calculate largest_genome_size using connection innovations
    largest_genome_size = 0

    # don't normalize excess and disjoint difference in small genomes
    if largest_genome_size < large_genome_size:
        genetic_distance = (
            c1 * excess_amount + c2 * disjoint_amount + c3 * (weight_difference)
        )
    else:
        genetic_distance = (
            c1 * excess_amount / largest_genome_size
            + c2 * disjoint_amount / largest_genome_size
            + c3 * (weight_difference)
        )

    return genetic_distance


def new_generation(
    networks_connections: List[List[ConnectionInnovation]],
    networks_connection_weights: List[ConnectionWeights],
    networks_connection_states: List[ConnectionStates],
    networks_nodes: List[Nodes],
    networks_scores: List[float],
    networks_species: List[int],
    genetic_distance_parameters: Dict[str, float],
) -> Tuple[
    List[List[ConnectionInnovation]],
    List[ConnectionWeights],
    List[ConnectionStates],
    List[Nodes],
]:
    """creates a new generation of networks based on the previous network's scores

    Arguments:
        networks_connections {List[List[ConnectionInnovation]]} -- connections of each network
        networks_connection_data {List[ConnectionProperties]} -- data of connections of each network
        networks_nodes {List[Nodes]} -- nodes of each network
        networks_species {List[int]} -- species of each network
        networks_scores {List[float]} -- scores of each networks from their environments
        genetic_distance_parameters {Dict[str, float]} -- hyperparameters for genetic distance

    Returns:
        Tuple[List[List[ConnectionInnovation]],List[ConnectionProperties],
              List[Nodes],] -- new generation
    """

    # normalize scores using species fitness sharing
    normalized_scores = _normalize_scores(networks_scores, networks_species)

    # use normalized scores as propabilities to select networks to parent offspings
    networks_amount = len(normalized_scores)
    networks = np.arange(networks_amount)

    # lists for new generation
    new_networks_connections = []
    new_networks_connection_weights = []
    new_networks_connection_states = []
    new_networks_nodes = []
    # generate a new network from two randomly chosen parents
    # with each parent being chosen according to its score
    # using crossover and mutation
    for parent_a, parent_b in np.random.choice(
        networks, size=(networks_amount, 2), p=normalized_scores
    ):
        (
            new_network_connections,
            new_network_connection_weights,
            new_network_connection_states,
            network_nodes,
        ) = _crossover_connections(
            networks_nodes[parent_a],
            networks_connections[parent_a],
            networks_connection_weights[parent_a],
            networks_connection_states[parent_a],
            networks_nodes[parent_b],
            networks_connections[parent_b],
            networks_connection_weights[parent_b],
            networks_connection_states[parent_b],
            genetic_distance_parameters,
        )
        new_networks_connections.append(new_network_connections)
        new_networks_connection_weights.append(new_network_connection_weights)
        new_networks_connection_states.append(new_network_connection_states)
        new_networks_nodes.append(network_nodes)

    return (
        new_networks_connections,
        new_networks_connection_weights,
        new_networks_connection_states,
        new_networks_nodes,
    )


def _crossover_connections(
    network_a_nodes: Nodes,
    network_a_connections: List[ConnectionInnovation],
    network_a_connection_weights: ConnectionWeights,
    network_a_connection_states: ConnectionStates,
    network_b_nodes: Nodes,
    network_b_connections: List[ConnectionInnovation],
    network_b_connection_weights: ConnectionWeights,
    network_b_connection_states: ConnectionStates,
    genetic_distance_parameters: Dict[str, float],
) -> Tuple[List[ConnectionInnovation], ConnectionWeights, ConnectionStates, Nodes]:
    new_network_connections = []
    new_network_connections_weights = []
    new_network_connections_enabled = []
    new_network_nodes = []

    raise NotImplementedError()


def _normalize_scores(
    networks_scores: List[float], networks_species: List[int]
) -> List[float]:
    """normalize scores using species fitness sharing

    Arguments:
        networks_species {List[int]} -- species of each network
        networks_scores {List[float]} -- scores of each networks from their environments

    Returns:
        List[float] -- normalized scores
    """

    # species amount is sorted, so each index (of species_amount) is the
    # amount of networks in that species
    species_amount = np.unique(networks_species, return_counts=True)[1]

    # normalize scores for each species
    normalized_scores = np.array(
        [
            network_score / species_amount[network_species]
            for network_score, network_species in zip(networks_scores, networks_species)
        ]
    )

    # set scores to add up to 1 as they will be used as probabilities later
    normalized_scores = normalized_scores / np.sum(normalized_scores)

    return normalized_scores
