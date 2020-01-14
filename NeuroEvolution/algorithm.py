import numpy as np
from typing import Union, Tuple, List


class Model:
    def __init__(
        self,
        input_shape: Union[int, Tuple[int]],
        hidden_layers: List[int],
        output_shape: Union[int, Tuple[int]],
        activation=lambda x: max(0, x),
        output=lambda x: (1.0 / (1.0 + np.exp(-x))),
    ):
        """Generate a model with feed forward capabilites
        
        Arguments:
            input_shape {Union[int, Tuple[int]]} -- shape of expected input
            hidden_layers {List[int]} -- dimensions of hidden layers, if any
            output_shape {Union[int, Tuple[int]]} -- shape of output
        
        Keyword Arguments:
            activation {function} -- activation function for hidden layers 
                                     (default: {lambdax:max(0, x)})
            output {function} -- activaiton function for output layer 
                                 (default: {lambdax:(1.0 / (1.0 + np.exp(-x)))})
        """
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.weights: list = []
        self.biases: list = []

        previous_layer_size = self.input_shape
        for layer_dimension in hidden_layers:
            self.weights.append(np.random.normal(
                size=(layer_dimension, previous_layer_size)
            ))
            self.biases.append(np.random.normal(size=layer_dimension))

        # TODO: generate weights
        # TODO: generate biases
        # TODO: add feed forward method

    def act(self, observation) -> List[float]:
        pass


class NeuroEvolution:

    # TODO: add mutation method
    # TODO: add crossover method
    def __init__(
        self,
        population_size: int,
        input_shape: Union[int, Tuple[int]],
        hidden_layers: List[int],
        output_shape: Union[int, Tuple[int]],
    ):
        self.population_size = population_size
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.population = [
            self.generate_model(input_shape, hidden_layers, output_shape)
            for _ in range(population_size)
        ]

    def get_actions(self, observations: list):
        """Get the actions for each agent according to each observation
        
        Arguments:
            observations {list} -- list of gym observations
        """
        actions = []
        for agent, observation in zip(self.population, observations):
            actions.append(agent.act(observation))

    def generate_model(
        self,
        input_shape: Union[int, Tuple[int]],
        hidden_layers: List[int],
        output_shape: Union[int, Tuple[int]],
    ) -> Model:
        return Model(input_shape, hidden_layers, output_shape)

if __name__ == "__main__":
    ne = NeuroEvolution(10, 5, [2, 3], 4)