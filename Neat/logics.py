from typing import Callable, List

import numpy as np


def feed_forward_network(
    inputs: np.ndarray,
    network_weights: np.ndarray,
    network_biases: np.ndarray,
    network_activations: List[Callable],
) -> np.ndarray:
    """
    Calculate the output of a network given an input

    Arguments:
        inputs {np.ndarray} -- network input
        network_weights {np.ndarray} -- weights connecting each layer to previous
                                      layer
        network_biases {np.ndarray} -- biases in each layer for each node

    Returns:
        np.ndarray -- network output
    """
    previous_layer_output = inputs
    for layer_weights, layer_biases, layer_activation in zip(
        network_weights, network_biases, network_activations
    ):
        previous_layer_output = layer_activation(
            np.sum(previous_layer_output * layer_weights.T + layer_biases, axis=1)
        )

    return previous_layer_output


if __name__ == "__main__":

    # test running a five layer mlp with 2 nodes in each layer
    test_inputs = np.array([0, 1])
    test_network_weights = np.random.random(size=(4, 2, 2)) * 2 - 1
    test_network_biases = np.random.random(size=(4, 2)) * 2 - 1
    test_network_activations = [lambda x: 1.0 / (1.0 + np.exp(-x))] * 2
    print(
        feed_forward_network(
            test_inputs,
            test_network_weights,
            test_network_biases,
            test_network_activations,
        )
    )
