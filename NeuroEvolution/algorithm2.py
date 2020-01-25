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
        layers = [input_layer] + self.hidden_dimensions + [output_layer]
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

    def new_generation(
        self, agent_fitness_levels: np.ndarray, keep_champion: bool = False
    ):
        """
        spawn a new generation using crossover and mutation judging agents by their fitness
        """
        new_weights = []
        new_biases = []
        normalized_fitness_levels: np.ndarray = agent_fitness_levels / agent_fitness_levels.sum()

        if keep_champion:
            new_weights.append(self.agent_weights[normalized_fitness_levels.argmax()])
            new_biases.append(self.agent_biases[normalized_fitness_levels.argmax()])

        # generate new weights and biases
        while len(new_biases) < len(self.agents):

            # choose two random parents
            a, b = np.random.choice(
                self.agents, size=2, replace=False, p=normalized_fitness_levels
            )
            agent_weights = self.agent_weights[a][:]
            agent_biases = self.agent_biases[a][:]
            for i, layer in enumerate(agent_weights):
                for j in range(len(layer)):
                    if np.random.random() < self.mutation_rate:
                        agent_weights[i][j] = np.random.normal()
                    elif np.random.random() < 0.5:
                        agent_weights[i][j] = self.agent_weights[b][i][j]
            new_weights.append(agent_weights)
            new_biases.append(agent_biases)
        self.agent_weights = new_weights
        self.agent_biases = new_biases


def training_loop(
    env_name: str,
    episodes: int,
    episode_steps: int,
    agents: int,
    hidden_layers: List[int],
    mutation_rate: float,
    keep_champion: bool,
):
    environments = [gym.make(env_name) for _ in range(agents)]
    observations = [env.reset() for env in environments]
    neuro = NeuroEvolution(
        agents,
        environments[0].observation_space.shape,
        environments[0].action_space.n,
        hidden_layers,
        mutation_rate,
    )

    avg_rewards = []
    max_rewards = []

    # training loop
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
        max_reward = np.max(rewards)
        avg_rewards.append(average_rewards)
        max_rewards.append(max_reward)
        rewards = np.power(rewards, 1)
        neuro.new_generation(rewards, keep_champion=keep_champion)

    # close environments
    for env in environments:
        env.close()

    return avg_rewards, max_rewards


if __name__ == "__main__":

    from multiprocessing import Pool
    from itertools import product, repeat

    # env setup
    ENV_NAME = "CartPole-v0"
    EPISODES = 300
    EPISODE_STEPS = 210
    AGENTS = 50
    TRIALS = 10
    HIDDEN_LAYERS = [5]
    MUTATION_RATE = 0.0000
    KEEP_CHAMPION = True

    for trial_run, (episode_avg_rewards, episode_max_rewards) in enumerate(
        Pool(TRIALS).starmap(
            training_loop,
            repeat(
                (
                    ENV_NAME,
                    EPISODES,
                    EPISODE_STEPS,
                    AGENTS,
                    HIDDEN_LAYERS,
                    MUTATION_RATE,
                    KEEP_CHAMPION,
                ),
                times=TRIALS,
            ),
        )
    ):

        plt.plot(
            [
                np.average(episode_max_rewards[:i])
                for i in range(len(episode_max_rewards))
            ],
            label=f"max_trend_{trial_run}",
        )
    plt.legend()
    plt.show()
