from typing import List

import gym
import matplotlib.pyplot as plt
import numpy as np

# env setup
ENV_NAME = "CartPole-v0"
env = gym.make(ENV_NAME)
observation = env.reset()

# hyper params
AGENTS = 10  # number of agents
INPUT_SHAPE = observation.shape
HIDDEN_LAYERS: List[int] = [5, 5]
OUTPUT_SHAPE = env.action_space.n
EPISODES = 10

# progress tracking variables
total_reward = 0
rewards = []
avg_rewards = []

# run training
for _ in range(EPISODES):
    # env.render()
    for agent_index in range(AGENTS):
        action = env.action_space.sample()
        next_observation, reward, done, info = env.step(action)
        observation = next_observation
        total_reward += reward

        if done:
            observation = env.reset()
            rewards.append(total_reward)
            avg_rewards.append(np.average(rewards))
            total_reward = 0
env.close()

# plot algorithm success
plt.plot(rewards)
plt.plot(avg_rewards)
plt.show()
