from typing import List, NamedTuple, Iterator

import gym


class Environments(NamedTuple):
    # TODO: delete strcut, just use List[gym.Env]
    environments: List[gym.Env]


class ConnectionWeights(NamedTuple):
    weights: List[float]

    def __iter__(self) -> Iterator[float]:
        return iter(self.weights)


class ConnectionStates(NamedTuple):
    states: List[bool]

    def __iter__(self) -> Iterator[bool]:
        return iter(self.states)


class Nodes(NamedTuple):
    # TODO: delete strcut, just use List[int]
    nodes: List[int]


class ConnectionInnovation(NamedTuple):
    src: int
    dst: int


class NodeInnovation(NamedTuple):
    split_connection_innovation_index: int


class BaseNodes(NamedTuple):
    input_nodes: List[int]
    output_nodes: List[int]
    bias_node: int = -1
