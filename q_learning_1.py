import numpy as np
import gym
import random
import tqdm


class Model:
    def __init__(self, observation_space, action_space):
        self.q_table = np.zeros([observation_space, action_space])

    def predict(self, state):
        # Get the best action given a state
        return self.q_table[state]

    def fit(self, state, action, new_value):
        # Get the best action given a state
        self.q_table[state][action] = new_value

class EpsilonGreedyTrainer:
    def __init__(self, env, model, epsilon=0.1, epochs=100001, discount_factor=0.5, alpha=0.1):
        self.env = env
        self.model = model
        self.epsilon = epsilon
        self.epochs = epochs
        self.discount_factor = discount_factor
        self.alpha = alpha

    def train(self):
        total_reward = 0
        for e in range(self.epochs):
            # Reset the environment
            state = self.env.reset()
            done = False
            while not done:
                # Epsilon greedy action sampling.
                if random.uniform(0, 1) < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.model.predict(state))

                next_state, reward, done, info = self.env.step(action)
                # learning algorithm:
                # Q(s, a) = Q(s, a) + alpha * [R(s, a) + discount_factor * np.max(Q'(s', a')) - Q(s, a)]
                action_values = self.model.predict(next_state)
                
                q_value = np.max(self.model.predict(state))
                next_max = np.max(action_values) - q_value
                #new_q_value = reward if done else q_value + self.alpha * (reward + self.discount_factor * next_max)
                new_q_value = reward if done else q_value + self.alpha * (reward + self.discount_factor * next_max)
                self.model.fit(state, action, new_q_value)
                total_reward += reward

                state = next_state
            if e % 100 == 0:
                print("Episode: %04d | Average Reward: %03f" % (e, total_reward / 100))
                total_reward = 0

def main():
    env = gym.make("Taxi-v3").env
    model = Model(env.observation_space.n, env.action_space.n)
    trainer = EpsilonGreedyTrainer(env, model)
    trainer.train()

if __name__ == "__main__":
    main()
