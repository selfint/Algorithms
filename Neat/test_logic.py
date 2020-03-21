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
    network_amount = 50
    environment_name = "MountainCar-v0"
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
    generations = 5
    networks_scores = np.zeros(shape=(network_amount))
    species_reps = []
    average_scores = []
    max_scores = []
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

        # draw best network
        best_network = networks_scores.argmax()
        best_network_connection_directions = networks_connection_directions[
            best_network
        ]
        best_network_connection_weights = networks_connection_weights[best_network]
        best_network_connection_states = networks_connection_states[best_network]
        G = pgv.AGraph()
        for (source, dest), weight, enabled in zip(
            best_network_connection_directions.directions,
            best_network_connection_weights.weights,
            best_network_connection_states.states,
        ):
            G.add_edge(source, dest, arrowhead="normal")
        G.draw(f"genomes/best_network_gen_{generation}.png", prog="fdp")

        # show best network perform
        evaluate_networks(
            Environments([gym.make(environment_name)]),
            [best_network_connection_directions],
            [best_network_connection_weights],
            [best_network_connection_states],
            base_nodes,
            max_steps=200,
            episodes=1,
            render=True,
        )

        # generate next generation
        average_scores.append(np.average(networks_scores))
        max_scores.append(np.max(networks_scores))
        networks_species, species_reps = split_into_species(
            networks_connection_directions,
            networks_connection_weights,
            global_innovation_history,
            genetic_distance_parameters,
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
            genetic_distance_parameters,
            mutation_parameters,
        )

    plt.plot(average_scores, label="avg")
    plt.plot(max_scores, label="max")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    test_feed_forward()
    # test_evaluate_network()
    # test_split_into_species()
    # test_new_generation()
