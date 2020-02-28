"""
Contains all logical operations to that are needed to transform the data
"""
from typing import List

import numpy as np

from structs import (
    BaseNodes,
    ConnectionInnovation,
    ConnectionProperties,
    Environments,
    NodeInnovation,
    NodeProperties,
)


def feed_forward(
    inputs: np.ndarray,
    connections: List[ConnectionInnovation],
    connection_data: List[ConnectionProperties],
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
