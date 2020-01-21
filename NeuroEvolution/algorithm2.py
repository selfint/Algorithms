from collections import namedtuple
from typing import Tuple, List

import numpy as np
import gym


class NeuroEvolution:
    def __init__(
        self,
        amount: int,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        hidden_dimensions: List[int],
        mutation_rate: float = 0.001,
    ):
        self.agents = range(amount)
        self.input_shape = input_shape
        self.hidden_dimensions = hidden_dimensions
        self.output_shape = output_shape
        self.agent_outputs = [None for _ in self.agents]
        self.mutation_rate = mutation_rate

        # generate agents
        self.agent_weights = []
        self.agent_biases = []
        input_layer = int(np.prod(self.input_shape))
        output_layer = int(np.prod(self.output_shape))
        layers = [input_layer] + hidden_dimensions + [output_layer]
        for _ in self.agents:
            new_weights = [
                np.random.normal(size=(layers[i], layers[i - 1]))
                for i in range(1, len(layers))
            ]
            new_biases = [
                np.random.normal(size=layer_size) for layer_size in layers[1:]
            ]
            self.agent_weights.append(new_weights)
            self.agent_biases.append(new_biases)

    def calculate_outputs(self, inputs: List[np.ndarray]):
        """
        calculate the output of each agent
        """
        for agent, input_, weights, biases in zip(
            self.agents, inputs, self.agent_weights, self.agent_biases
        ):
            previous_layer_output: np.ndarray = input_.reshape(-1, 1)
            for layer_weights, layer_biases in zip(weights, biases):
                previous_layer_output = (
                    np.sum(previous_layer_output.T * layer_weights, axis=1)
                    + layer_biases
                )

            self.agent_outputs[agent] = previous_layer_output.reshape(self.output_shape)

    def new_generation(self, agent_fitness_levels: List[float]):
        """
        spawn a new generation using crossover and mutation judging agents by their fitness
        """
        new_weights = []
        new_biases = []
        normalized_fitness_levels = np.array(agent_fitness_levels)
        normalized_fitness_levels = (
            normalized_fitness_levels / normalized_fitness_levels.sum()
        )

        # generate new weights and biases
        for agent in self.agents:

            # choose two random parents
            a, b = np.random.choice(
                self.agents, size=2, replace=True, p=normalized_fitness_levels
            )
            agent_weights = self.agent_weights[a][:]
            agent_biases = self.agent_biases[a][:]
            for i, layer in enumerate(agent_weights):
                for j in range(len(layer)):
                    if np.random.random() < 0.5:
                        agent_weights[i][j] = self.agent_weights[b][i][j]
                    if np.random.random() < self.mutation_rate:
                        agent_weights[i][j] = np.random.normal()
            new_weights.append(agent_weights)
            new_biases.append(agent_biases)
        self.agent_weights = new_weights
        self.agent_biases = new_biases


if __name__ == "__main__":
    # env setup
    ENV_NAME = "CartPole-v0"
    episodes = 50
    agents = 10
    environments = [gym.make(ENV_NAME) for _ in range(agents)]
    observations = [env.reset() for env in environments][0]
    hidden_layers = [5]
    neuro = NeuroEvolution(
        agents,
        environments[0].observation_space.shape,
        environments[0].action_space.n,
        hidden_layers,
        0.001,
    )
    for i in range(episodes):

        # get agent actions
        neuro.calculate_outputs(observations)
        observations = []
        rewards = []
        reset = True
        for action, environment in zip(neuro.agent_outputs, environments):
            observation, reward, done, _ = environment.step(np.argmax(action))
            rewards.append(reward)
            observations.append(observation)

            if not done:
                reset = False

        if reset:
            for env in environments:
                env.reset()

    # close environments
    for env in environments:
        env.close()
