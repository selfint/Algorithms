from multiprocessing import Pool
from itertools import repeat
import matplotlib.pyplot as plt
import numpy as np
import gym
from typing import List


from NeuroEvolution.algorithm import NeuroEvolution

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
                    SURVIVAL_RATE,
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
