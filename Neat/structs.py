from typing import NamedTuple, List
import numpy as np


class Environments(NamedTuple):
    environments: list


class Connections(NamedTuple):
    weights: List[float]
    enabled: List[bool]


class Nodes(NamedTuple):
    index: int
    biases: List[float]


class ConnectionInnovation(NamedTuple):
    src: int
    dst: int


class NodeInnovation(NamedTuple):
    split_connection_innovation_index: int


class BaseNodes(NamedTuple):
    input_nodes: List[int]
    output_nodes: List[int]
