from typing import List, Tuple, NamedTuple


class ConnectionInnovation(NamedTuple):
    idx: int
    src: int
    dst: int


class NodeInnovation(NamedTuple):
    idx: int
    split_connection_innovation_idx: int


class Innovations:
    def __init__(self):
        self.max_node_innovation_id: int = 0
        self.max_connection_innovation_id: int = 0
        self.node_innovations: List[NodeInnovation] = []
        self.connection_innovations: List[ConnectionInnovation] = []


class InputNode:
    def __init__(self, innovation_id: int):
        self.innovation_id = innovation_id
        self.output_value: float = 0


class HiddenNode:
    def __init__(self, innovation_id: int, bias: float):
        self.innovation_id = innovation_id
        self.bias = bias
        self.input_value: float = 0
        self.output_value: float = 0


class OutputNode:
    def __init__(self, innovation_id: int, bias: float):
        self.innovation_id = innovation_id
        self.bias = bias
        self.input_value: float = 0
        self.output_value: float = 0


class Connection:
    def __init__(
        self,
        innovation_id: int,
        source: int,
        destination: int,
        enabled: bool,
        weight: float,
    ):
        self.innovation_id = innovation_id
        self.source = source
        self.destination = destination
        self.enabled = enabled
        self.weight = weight


class NeuralNetwork:
    def __init__(
        self,
        input_nodes: List[InputNode],
        hidden_nodes: List[HiddenNode],
        output_nodes: List[OutputNode],
        connections: List[Connection],
        innovations: Innovations,
    ):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.connections = connections
        self.innovations = innovations
