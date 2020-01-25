from collections import namedtuple
from typing import Tuple, List
import matplotlib.pyplot as plt
import numpy as np
import gym


class NeuroEvolution:

    agent_outputs: List[np.ndarray]
    agent_weights: List[List[np.ndarray]]
    agent_biases: List[List[np.ndarray]]

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
        self.agent_outputs = [np.zeros(shape=self.output_shape) for _ in self.agents]
        self.mutation_rate = mutation_rate

        # generate agents
        self.agent_weights = []
        self.agent_biases = []
        input_layer = int(np.prod(self.input_shape))
        output_layer = int(np.prod(self.output_shape))
        layers = [input_layer] + hidden_dimensions + [output_layer]
        for _ in self.agents:
            new_weights = []
            new_biases = []
            for i in range(1, len(layers)):
                new_weights.append(np.random.normal(size=(layers[i], layers[i - 1])))
                new_biases.append(np.random.normal(size=layers[i]))
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

    def new_generation(self, agent_fitness_levels: np.ndarray):
        """
        spawn a new generation using crossover and mutation judging agents by their fitness
        """
        new_weights = []
        new_biases = []
        normalized_fitness_levels = (
            agent_fitness_levels / agent_fitness_levels.sum()
        )

        # generate new weights and biases
        for _ in self.agents:

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
    episodes = 10000
    episode_steps = 210
    agents = 50
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

    episode_avg_rewards = []

    for episode in range(episodes):

        rewards = np.ones(shape=len(neuro.agents))
        env_states = [False for env in environments]
        for step in range(episode_steps):

            # get agent actions
            neuro.calculate_outputs(observations)
            observations = []
            reset = True
            for (agent_index, action), (env_index, environment) in zip(
                enumerate(neuro.agent_outputs), enumerate(environments)
            ):

                # don't act in environment if simulation is done
                if env_states[env_index]:
                    observations.append(np.zeros(shape=neuro.input_shape))
                    continue

                observation, reward, done, _ = environment.step(np.argmax(action))

                rewards[agent_index] += reward
                observations.append(observation)

                if done:
                    env_states[env_index] = True
                else:
                    reset = False

            if reset:
                break

        for env in environments:
            env.reset()
        average_rewards = np.average(rewards)
        if episode % 20 == 0:
            print(f"episode {episode}>= avg={average_rewards} max={np.max(rewards)}")
        episode_avg_rewards.append(average_rewards)
        modified_rewards = np.power(rewards, 2)
        neuro.new_generation(modified_rewards)

    # close environments
    for env in environments:
        env.close()

    plt.plot(episode_avg_rewards)
    plt.show()
