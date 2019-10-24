import numpy as np
import gym
import random
import tqdm
import torch.nn.functional as F
import torch
from collections import deque
import math

from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class MemoryDataset(Dataset):
    def __init__(self, maxlen=1000):
        self.samples = deque(maxlen=maxlen)

    def add_sample(self, sample):
        self.samples.append(sample)

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

class Model(nn.Module):
    def __init__(self, observation_space, action_space, hidden_layers=32):
        super(Model, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.network = nn.Sequential(
            nn.Linear(observation_space, hidden_layers),
            nn.ReLU(),
            nn.Linear(hidden_layers, action_space),
        )
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.xavier_uniform(self.network[0].weight)
        torch.nn.init.xavier_uniform(self.network[2].weight)

    def forward(self, states, labels=None):
        # One hot encoding
        # TODO
        output = {}
        predictions = self.network(states)
        output['predictions'] = predictions

        if not labels is None:
            output['loss'] = F.mse_loss(predictions, labels)
        return output


class EpsilonGreedyTrainer:
    def __init__(
        self,
        env,
        model,
        epsilon=0.1,
        episodes=100000,
        epsilon_decay=0.99995,
        epsilon_min=0.001,
        gamma=0.9,
        lr=0.01,
        momentum=0.5,
    ):
        self.env = env
        self.model = model
        self.epsilon = epsilon
        self.episodes = episodes
        self.gamma = gamma
        self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        self.dataset = MemoryDataset()
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def collate_fn(self, data):
        states, actions, rewards, next_states, dones = zip(*data)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        # This isn't propertly vectorized
        # reward if done else reward + self.gamma * np.max(self.model.predict(next_state)[0])
        next_state_output = self.model(next_states)['predictions']
        target_values = rewards + (1. - dones) * torch.max(next_state_output, 1)[0] * self.gamma

        output = self.model(states)
        predictions = output['predictions']
        predictions.scatter_(1, actions.view(-1, 1), target_values.view(-1, 1))
        return states, predictions

    def train(self):
        data_loader = DataLoader(
            self.dataset,
            collate_fn=self.collate_fn,
            batch_size=64,
            shuffle=True,
        )
        xs, ys = iter(data_loader).next()
        self.optimizer.zero_grad()
        output = self.model(xs, ys)
        output['loss'].backward()
        self.optimizer.step()

    def get_epsilon(self, e):
        return max(self.epsilon_min, self.epsilon * self.epsilon_decay ** (e + 1))
        #return max(self.epsilon_min, self.epsilon / math.log2(e + 2))
        #return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((e + 1) * self.epsilon_decay)))

    def run(self):
        scores = deque(maxlen=100)
        for e in tqdm.tqdm(range(self.episodes)):
            # Reset the environment
            state = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                # Epsilon greedy action sampling.
                if random.uniform(0, 1) < self.get_epsilon(e):
                    action = self.env.action_space.sample()
                else:
                    predictions = self.model(torch.FloatTensor([state]))['predictions']
                    action = torch.argmax(predictions[0]).item()

                next_state, reward, done, info = self.env.step(action)
                self.dataset.add_sample((state, action, reward, next_state, done))
                state = next_state
                total_reward += reward
            scores.append(total_reward)

            if e % 100 == 0:
                print("Average reward: {} | Epsilon: {}".format(np.mean(scores), self.get_epsilon(e)))
            self.train()

def main():
    env = gym.make("CartPole-v0")
    model = Model(env.observation_space.shape[0], env.action_space.n)
    trainer = EpsilonGreedyTrainer(env, model)
    trainer.run()

if __name__ == "__main__":
    main()
