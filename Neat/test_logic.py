import numpy as np
import pytest

from logics import (
    BaseNodes,
    ConnectionInnovation,
    ConnectionProperties,
    NodeProperties,
    feed_forward,
)


def test_feed_forward():
    # inputs: ndarray,
    # connections: List[ConnectionInnovation],
    # connection_data: ConnectionProperties,
    # node_data: NodeProperties,
    # base_nodes: BaseNodes,

    inputs = np.array([0.74])
    node_data = NodeProperties(
        np.random.random(size=6),
        [lambda x: (1.0 / (1.0 + np.exp(-x)))] * 3 + [lambda x: max(0, x)] * 3,
    )
    connections = [
        ConnectionInnovation(0, 1),
        ConnectionInnovation(0, 2),
        ConnectionInnovation(1, 3),
        ConnectionInnovation(2, 3),
        ConnectionInnovation(3, 4),
        ConnectionInnovation(3, 5),
    ]
    connection_data = ConnectionProperties(
        np.random.random(size=6), [True, True, True, True, False, True]
    )
    base_nodes = BaseNodes([0], [4, 5])
    feed_forward(inputs, connections, connection_data, node_data, base_nodes)

