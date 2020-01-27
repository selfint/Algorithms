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
        keep_champion: bool = False,
        survival_rate: float = 0.0,
    ):
        # each agent is represented as in index, instead of an object
        self.agents = range(amount)
        self.input_shape = input_shape
        self.hidden_dimensions = hidden_dimensions
        self.output_shape = output_shape
        self.agent_outputs = [np.zeros(shape=self.output_shape) for _ in self.agents]
        self.mutation_rate = mutation_rate
        self.keep_champion = keep_champion
        self.survival_rate = survival_rate

        # generate agent weights and biases using a normal distribution
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
        calculate the output of each agent with respect to the inputs
        """

        # get the weights and biases for each agent and calculate the output
        # for the inputs using forward propagation
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
        new_generation_weights = []
        new_generation_biases = []
        normalized_fitness_levels: np.ndarray = agent_fitness_levels / agent_fitness_levels.sum()

        # keep the best agent from the previous generation
        if self.keep_champion:
            new_generation_weights.append(self.agent_weights[normalized_fitness_levels.argmax()])
            new_generation_biases.append(self.agent_biases[normalized_fitness_levels.argmax()])

        # keep survival_rate * 100 % of agents from the previous generation
        if self.survival_rate:
            new_weights_and_biases = np.random.choice(
                self.agents,
                replace=False,
                size=int(len(self.agents) * self.survival_rate),
                p=normalized_fitness_levels,
            )
            new_generation_weights.extend(list(np.array(self.agent_weights)[new_weights_and_biases]))
            new_generation_biases.extend(list(np.array(self.agent_biases)[new_weights_and_biases]))

        # generate new weights and biases for each new agent
        while len(new_generation_biases) < len(self.agents):

            # choose two random parents
            parent_a, parent_b = np.random.choice(
                self.agents, size=2, replace=False, p=normalized_fitness_levels
            )

            # generate new agent weights and biases
            new_agent_weights = self.agent_weights[parent_a][:]
            new_agent_biases = self.agent_biases[parent_a][:]
            for i in range(len(new_agent_weights)):
                for j in range(len(new_agent_weights[i])):
                    for k in range(len(new_agent_weights[i][j])):
                        if np.random.random() < self.mutation_rate:
                            new_agent_weights[i][j][k] = np.random.normal()
                        elif np.random.random() < 0.5:
                            new_agent_weights[i][j][k] = self.agent_weights[parent_b][i][j][k]

                    if np.random.random() < self.mutation_rate:
                        new_agent_biases[i][j] = np.random.normal()
                    elif np.random.random() < 0.5:
                        new_agent_biases[i][j] = self.agent_biases[parent_b][i][j]
            new_generation_weights.append(new_agent_weights)
            new_generation_biases.append(new_agent_biases)

        # set generation to new generation
        self.agent_weights = new_generation_weights
        self.agent_biases = new_generation_biases


def training_loop(
    env_name: str,
    episodes: int,
    episode_steps: int,
    agents: int,
    hidden_layers: List[int],
    mutation_rate: float,
    keep_champion: bool,
    survival_rate: float,
):

    # initialize environments
    environments = [gym.make(env_name) for _ in range(agents)]
    observations = [env.reset() for env in environments]

    # build neuro evolution trainer
    neuro = NeuroEvolution(
        agents,
        environments[0].observation_space.shape,
        environments[0].action_space.n,
        hidden_layers,
        mutation_rate,
        keep_champion,
        survival_rate,
    )

    # logging
    avg_rewards = []
    max_rewards = []

    # training loop
    for episode in range(episodes):

        # initialize rewards at 1 to avoid 0 division errors
        episode_rewards = np.ones(shape=len(neuro.agents))
        env_states = [False for _ in environments]
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

                # take action and log reward and new observation
                observation, reward, done, _ = environment.step(np.argmax(action))
                episode_rewards[agent_index] += reward
                observations.append(observation)

                # don't reset simulation till all environments are done
                if done:
                    env_states[env_index] = True
                else:
                    reset = False

            if reset:
                break

        # reset environments and get initial observations
        observations = [env.reset() for env in environments]

        # log average and max rewards for all agents in this episode
        average_rewards = np.average(episode_rewards)
        max_reward = np.max(episode_rewards)
        avg_rewards.append(average_rewards)
        max_rewards.append(max_reward)

        # generate new generation with respect to the episode rewards
        neuro.new_generation(episode_rewards)

    # close environments
    for env in environments:
        env.close()

    return avg_rewards, max_rewards


if __name__ == "__main__":

    from multiprocessing import Pool
    from itertools import product, repeat

    # env and hyper parameters setup
    ENV_NAME = "CartPole-v0"
    EPISODES = 100
    EPISODE_STEPS = 210
    AGENTS = 10
    TRIALS = 5
    HIDDEN_LAYERS = []
    MUTATION_RATE = 0.01
    KEEP_CHAMPION = False
    SURVIVAL_RATE = 0.4

    # run trainer and get avg and max rewards for each episode during training
    # use Pool to run trainers in parallel
    for trial_run, (training_avg_rewards, training_max_rewards) in enumerate(
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
                    SURVIVAL_RATE
                ),
                times=TRIALS,
            ),
        )
    ):

        # add training results to plot
        plt.plot(training_max_rewards, label=f"max_reward_{trial_run}")
        plt.plot(training_avg_rewards, label=f"avg_reward_{trial_run}")

    # plot training results
    plt.legend()
    plt.show()
