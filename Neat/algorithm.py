from typing import List

from structs import (
    Innovations,
    NodeInnovation,
    NeuralNetwork
)


def network_reward_evaluation(networks: List[NeuralNetwork], environments: list) -> List[float]:
    """
    Run networks in environments until they finish the simulation.

    :param networks: networks to evaluate
    :param environments: environments for each network, respectively
    :return: network scores
    """


def new_node_innovation(global_innovation_history: Innovations, split_connection_innovation_idx: int) -> NodeInnovation:
    """
    Get a node innovation for splitting a specific connections. Logs a new innovation in
    innovation history, if one occurs.

    :param global_innovation_history: innovation history of evolution
    :param split_connection_innovation_idx: innovation id of the connection that split
    :return: node innovation for splitting that connection
    """

    # return innovation from innovation history if it already happened
    for node_innovation in global_innovation_history.node_innovations:
        if (
            node_innovation.split_connection_innovation_idx
            == split_connection_innovation_idx
        ):
            return node_innovation

    # return new node innovation if not
    global_innovation_history.max_node_innovation_id += 1
    new_innovation = NodeInnovation(
        global_innovation_history.max_node_innovation_id,
        split_connection_innovation_idx,
    )
    global_innovation_history.node_innovations.append(new_innovation)
    return new_innovation
