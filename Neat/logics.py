"""
Contains all logical operations to that are needed to transform the data
"""
from typing import List, Dict, Tuple

import numpy as np
import gym
from gym import spaces

from structs import (
    BaseNodes,
    ConnectionDirections,
    ConnectionInnovationsMap,
    ConnectionWeights,
    ConnectionStates,
    ConnectionDirections,
    Environments,
)


def feed_forward(
    inputs: np.ndarray,
    connection_directions: ConnectionDirections,
    connections_weights: ConnectionWeights,
    connections_states: ConnectionStates,
    base_nodes: BaseNodes,
) -> np.ndarray:
    """Calculate the output of a network using recursion

    Arguments:
        inputs {np.ndarray} -- network inputs
        connection_directions {ConnectionDirections} -- connections between nodes
        connections_weights: {ConnectionWeights} -- connection weights
        connections_states: {ConnectionStates} -- connection states
        base_nodes {BaseNodes} -- input, output and bias nodes

    Returns:
        np.ndarray -- network output
    """
    return [
        _get_node_output(
            node_id,
            inputs,
            connection_directions,
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
    connection_directions: ConnectionDirections,
    connection_weights: ConnectionWeights,
    connection_states: ConnectionStates,
    base_nodes: BaseNodes,
    ignore_connections: List[Tuple[int, int]],
) -> float:
    """helper function to get output of a single node using recursion

    Arguments:
        node_id {int} -- id of node to get output of
        inputs {np.ndarray} -- network inputs
        connection_directions {ConnectionDirections} -- connections between nodes
        connection_weights: {ConnectionWeights} -- connection weights
        connection_states: {ConnectionStates} -- connection states
        base_nodes {BaseNodes} -- input, output and bias nodes
        ignore_connections {ConnectionDirections} -- connections previously
                                                           calculated

    Returns:
        float -- output of node
    # input nodes are just placeholders for the network input
    """
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
                    connection_src,
                    inputs,
                    connection_directions,
                    connection_weights,
                    connection_states,
                    base_nodes,
                    ignore_connections
                    + [
                        (connection_src, connection_dst,)
                    ],  # mark connection as to-ignore
                )
                * connection_weight  # weight the output
                for (
                    connection_src,
                    connection_dst,
                ), connection_weight, connection_state in zip(
                    connection_directions.directions,
                    connection_weights.weights,
                    connection_states.states,
                )
                if (
                    connection_dst == node_id  # get connections outputing into node
                    and (connection_src, connection_dst,)
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
    networks_connection_directions: List[ConnectionDirections],
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
        networks_connection_directions {List[ConnectionDirections]} -- directions of connections of each network
        networks_connection_weights {List[ConnectionWeights]} -- weights of connections of each network
        networks_connection_states {List[ConnectionStates]} -- states of connections of each network
        base_nodes {BaseNodes} -- input, output and bias nodes
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
            networks_connection_directions,
            networks_connection_weights,
            networks_connection_states,
        )
    ]


def _get_episode_reward(
    environment: gym.Env,
    max_steps: int,
    connection_directions: ConnectionDirections,
    connections_weights: ConnectionWeights,
    connections_states: ConnectionStates,
    base_nodes: BaseNodes,
    render: bool = False,
) -> float:
    """helper function that runs an episode and returns the episode rewards

    Arguments:
        environment {gym.Env} -- gym environment
        max_steps {int} -- limit of steps to take in episode
        connection_directions {ConnectionDirections} -- network connections
        connection_data {ConnectionProperties} -- network connection data
        base_nodes {BaseNodes} -- input, output and bias nodes

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
            connection_directions,
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
    networks_connection_directions: List[ConnectionDirections],
    networks_connection_weights: List[ConnectionWeights],
    global_innovation_history: ConnectionInnovationsMap,
    genetic_distance_parameters: Dict[str, float],
    previous_generation_species_reps: List[
        Tuple[ConnectionDirections, ConnectionWeights]
    ] = None,
) -> List[int]:
    """assign a species to each network

    Arguments:
        networks_connections {List[ConnectionDirections]} -- connections of each network
        networks_connection_data {List[ConnectionProperties]} -- data of connections of each network
        networks_nodes {List[Nodes]} -- nodes of each network
        genetic_distance_parameters {Dict[str, float]} -- hyperparameters for genetic distance

    Returns:
        List[int] -- species of each network
    """
    genetic_distance_threshold = genetic_distance_parameters["threshold"]
    species = []

    # if no previous generation is available, generate species reps from current generation
    species_reps: List[
        Tuple[ConnectionDirections, ConnectionWeights]
    ] = previous_generation_species_reps or []
    for (network_connection_directions, network_connection_weights,) in zip(
        networks_connection_directions, networks_connection_weights,
    ):

        # check genetic distance to all species reps
        for (
            species_rep_index,
            (rep_connection_directions, rep_connection_weights),
        ) in enumerate(species_reps):
            if (
                _genetic_distance(
                    network_connection_directions,
                    network_connection_weights,
                    rep_connection_directions,
                    rep_connection_weights,
                    global_innovation_history,
                    genetic_distance_parameters,
                )
                < genetic_distance_threshold
            ):
                species.append(species_rep_index)
                break
        else:

            # generate a new rep for new species when a network doesn't match any
            # other species rep
            species.append(len(species_reps))
            species_reps.append(
                (network_connection_directions, network_connection_weights,)
            )

    return species


def _genetic_distance(
    network_a_connection_directions: ConnectionDirections,
    network_a_connection_weights: ConnectionWeights,
    network_b_connection_directions: ConnectionDirections,
    network_b_connection_weights: ConnectionWeights,
    global_innovation_history: ConnectionInnovationsMap,
    genetic_distance_parameters: Dict[str, float],
) -> float:
    """calculate the genetic distance between two networks

    Arguments:
        network_a_connection_directions {ConnectionDirections} -- connections of network a
        network_a_connection_weights {ConnectionWeights} -- weights of connections of network a
        network_b_connection_directions {ConnectionDirections} -- connections of network b
        network_b_connections_weights {ConnectionWeights} -- data of connections of network b
        genetic_distance_parameters {Dict[str, float]} -- hyperparameters for genetic distance

    Returns:
        float -- genetic distance
    """
    # TODO: split into separate functions
    # innovation index that splits disjoint and excess genes
    # last_common_innovation = min(max(network_a_nodes.nodes), max(network_b_nodes.nodes))

    # fancy numpy trick to get the weights of all connections in a that are present in b
    # and vice-versa
    common_connections_indices_a = (
        (
            network_a_connection_directions.directions[:, None]
            == network_b_connection_directions.directions
        )
        .all(-1)
        .any(-1)
    )
    common_connections_indices_b = (
        (
            network_b_connection_directions.directions[:, None]
            == network_a_connection_directions.directions
        )
        .all(-1)
        .any(-1)
    )

    # get common connections weight values
    common_connections_weights_a = network_a_connection_weights.weights[
        common_connections_indices_a
    ]
    common_connections_weights_b = network_b_connection_weights.weights[
        common_connections_indices_b
    ]

    # get common connections and last common innovation
    common_connections_directions_value: np.ndarray = network_a_connection_directions.directions[
        common_connections_indices_a
    ]

    last_common_innovation = np.max(
        _vectorized_innovation_lookup(
            global_innovation_history, common_connections_directions_value
        )
    )

    # get uncommon connections and innovations
    uncommon_connections_directions_a = network_a_connection_directions.directions[
        np.invert(common_connections_indices_a)
    ]

    uncommon_connections_directions_b = network_b_connection_directions.directions[
        np.invert(common_connections_indices_b)
    ]

    # get uncommon connection innovation numbers
    uncommon_connection_innovations_a = _vectorized_innovation_lookup(
        global_innovation_history, uncommon_connections_directions_a
    )

    uncommon_connection_innovations_b = _vectorized_innovation_lookup(
        global_innovation_history, uncommon_connections_directions_b
    )

    # get the average distance between two connection weights
    weight_difference = np.average(
        abs(common_connections_weights_a - common_connections_weights_b)
    )

    # get disjoint and excess amounts
    a_disjoints = np.where(uncommon_connection_innovations_a < last_common_innovation)
    b_disjoints = np.where(uncommon_connection_innovations_b < last_common_innovation)
    disjoint_amount = np.sum(a_disjoints) + np.sum(b_disjoints)
    excess_amount = np.sum(np.invert(a_disjoints)) + np.sum(np.invert(b_disjoints))

    # calculate genetic distance
    c1 = genetic_distance_parameters["excess_constant"]
    c2 = genetic_distance_parameters["disjoint_constant"]
    c3 = genetic_distance_parameters["weight_bias_constant"]
    large_genome_size = genetic_distance_parameters["large_genome_size"]

    largest_genome_size = np.max(
        (
            network_a_connection_directions.directions.max(),
            network_b_connection_directions.directions.max(),
        )
    )

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


def _vectorized_innovation_lookup(
    global_innovation_history: ConnectionInnovationsMap,
    connection_directions_value: np.ndarray,
):
    tupled_common_connections = np.array(
        np.array(list(map(tuple, connection_directions_value)), dtype="i, i")
    )
    return np.vectorize(lambda c: global_innovation_history.innovations[tuple(c)])(
        tupled_common_connections
    )


def new_generation(
    networks_connection_directions: List[ConnectionDirections],
    networks_connection_weights: List[ConnectionWeights],
    networks_connection_states: List[ConnectionStates],
    networks_scores: List[float],
    networks_species: List[int],
    global_innovation_history: ConnectionInnovationsMap,
    genetic_distance_parameters: Dict[str, float],
) -> Tuple[
    List[ConnectionDirections],
    List[ConnectionWeights],
    List[ConnectionStates],
    ConnectionInnovationsMap,
]:

    # normalize scores using species fitness sharing
    normalized_scores = _normalize_scores(networks_scores, networks_species)

    # use normalized scores as propabilities to select networks to parent offspings
    networks_amount = len(normalized_scores)
    networks = np.arange(networks_amount)

    # lists for new generation
    new_networks_connections = []
    new_networks_connection_weights = []
    new_networks_connection_states = []
    # generate a new network from two randomly chosen parents
    # with each parent being chosen according to its score
    # using crossover and mutation
    for _ in range(networks_amount):
        parent_a, parent_b = np.random.choice(
            networks, size=2, p=normalized_scores, replace=False
        )
        (
            new_network_connections,
            new_network_connection_weights,
            new_network_connection_states,
        ) = _crossover(
            networks_connection_directions[parent_a],
            networks_connection_weights[parent_a],
            networks_connection_states[parent_a],
            networks_connection_directions[parent_b],
            networks_connection_weights[parent_b],
            networks_connection_states[parent_b],
            global_innovation_history,
            genetic_distance_parameters,
        )
        new_networks_connections.append(new_network_connections)
        new_networks_connection_weights.append(new_network_connection_weights)
        new_networks_connection_states.append(new_network_connection_states)

    return (
        new_networks_connections,
        new_networks_connection_weights,
        new_networks_connection_states,
    )


def _crossover(
    network_a_connections: ConnectionDirections,
    network_a_connection_weights: ConnectionWeights,
    network_a_connection_states: ConnectionStates,
    network_b_connections: ConnectionDirections,
    network_b_connection_weights: ConnectionWeights,
    network_b_connection_states: ConnectionStates,
    global_innovation_history: ConnectionInnovationsMap,
    genetic_distance_parameters: Dict[str, float],
) -> Tuple[ConnectionDirections, ConnectionWeights, ConnectionStates]:
    new_network_connection_directions = []
    new_network_connections_weights = []
    new_network_connections_enabled = []

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
