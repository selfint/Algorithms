from typing import NamedTuple, List, Callable
import gym


class Environments(NamedTuple):
    environments: List[gym.Env]


class ConnectionProperties(NamedTuple):
    weights: List[float]
    enabled: List[bool]


class NodeProperties(NamedTuple):
    biases: List[float]
    activations: List[Callable]


class Nodes(NamedTuple):
    index: List[int]


class ConnectionInnovation(NamedTuple):
    src: int
    dst: int


class NodeInnovation(NamedTuple):
    split_connection_innovation_index: int


class BaseNodes(NamedTuple):
    input_nodes: List[int]
    output_nodes: List[int]
