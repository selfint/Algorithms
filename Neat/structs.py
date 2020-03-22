from typing import Dict, Iterator, List, NamedTuple, Tuple

import gym
import numpy as np


class Environments(NamedTuple):
    environments: List[gym.Env]


class ConnectionWeights(NamedTuple):
    weights: np.ndarray

    def __iter__(self) -> Iterator[float]:
        return iter(self.weights)


class ConnectionStates(NamedTuple):
    states: np.ndarray

    def __iter__(self) -> Iterator[int]:
        return iter(self.states)


class ConnectionDirections(NamedTuple):
    directions: np.ndarray

    def __iter__(self) -> Iterator[int]:
        return iter(self.directions)


class ConnectionInnovationsMap(NamedTuple):
    """maps a connection direction to an innovation number"""

    innovations: Dict[Tuple[int, int], int]


class NodeInnovationsMap(NamedTuple):
    """
    maps a split connection to the new node number that
    represents splitting that connection
    """

    innovations: Dict[Tuple[int, int], int]


class BaseNodes(NamedTuple):
    input_nodes: np.ndarray
    output_nodes: np.ndarray
    bias_node: int = -1


# TODO: replace as much classes as possible with a custom type
