from typing import Dict, Iterator, List, NamedTuple, Tuple

import gym
import numpy as np


class Environments(NamedTuple):
    # TODO: delete strcut, just use List[gym.Env]
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


class ConnectionInnovationsMap(NamedTuple):
    """maps a connection direction to an innovation number"""

    innovations: Dict[Tuple[int, int], int]


class NodeInnovation(NamedTuple):
    split_connection_innovation_index: int


class BaseNodes(NamedTuple):
    input_nodes: List[int]
    output_nodes: List[int]
    bias_node: int = -1
