import gym
import random
import numpy as np

env = gym.make("Taxi-v3").env

# Hyperparameters
alpha = 0.1
gamma = 0.6
epsilon = 0.1

q_table = np.zeros([env.observation_space.n, env.action_space.n])

# For plotting metrics
all_epochs = []
all_penalties = []

total_reward = 0
for i in range(1, 100001):
    state = env.reset()

    epochs, penalties, reward, = 0, 0, 0
    done = False

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample() # Explore action space
        else:
            action = np.argmax(q_table[state]) # Exploit learned values

        next_state, reward, done, info = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value
        total_reward += reward

        if reward == -10:
            penalties += 1

        state = next_state
        epochs += 1

    if i % 100 == 0:
        print("Episode: %04d | Average Reward: %03f" % (i, total_reward / 100))
        total_reward = 0

print("Training finished.\n")

