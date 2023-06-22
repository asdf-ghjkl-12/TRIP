import torch
import numpy as np
import gym
import os
import random
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import AgentConfig, EnvConfig
from memory import ReplayMemory
from network import MlpPolicy
from ops import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent(AgentConfig, EnvConfig):
    def __init__(self):
        self.env = gym.make(self.env_name)
        self.action_size = self.env.action_space.n  # 2 for cartpole
        self.memory = ReplayMemory(memory_size=self.memory_size, action_size=self.action_size, per=self.per)
        if self.train_cartpole:
            self.policy_network = MlpPolicy(action_size=self.action_size , input_size=self.obs_size+self.latent_size).to(device)
            self.target_network = MlpPolicy(action_size=self.action_size, input_size=self.obs_size+self.latent_size).to(device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        self.loss = 0
        self.criterion = nn.MSELoss()
        
        self.z = torch.zeros(self.latent_size, device=device)

    def new_random_game(self):
        self.env.reset()
        action = self.env.action_space.sample()
        screen, reward, terminal, _, info = self.env.step(action)
        return screen, reward, action, terminal

    def train(self):
        episode = 0
        step = 0
        reward_history = []

        if not os.path.exists("./GIF/"):
            os.makedirs("./GIF/")

        # A new episode
        while step < self.max_step:
            start_step = step
            episode += 1
            episode_length = 0
            total_episode_reward = 0
            frames_for_gif = []

            self.gif = True if episode % self.gif_every == 0 else False

            # Get initial state
            state, reward, action, terminal = self.new_random_game()
            current_state = state
            # current_state = np.stack((state, state, state, state))

            # A step in an episode
            while episode_length < self.max_episode_length:
                step += 1
                episode_length += 1

                # Choose action
                action = random.randrange(self.action_size) if np.random.rand() < self.epsilon else \
                    torch.argmax(self.policy_network(torch.FloatTensor(current_state).to(device))).item()

                # print(current_state)
                # print(self.policy_network(torch.FloatTensor(current_state).to(device)))

                # Act
                state, reward, terminal, _, _ = self.env.step(action)
                new_state = state
                # new_state = np.concatenate((current_state[1:], [state]))

                reward = -1 if terminal else reward

                if self.gif:
                    frames_for_gif.append(new_state)

                self.memory.add(current_state, reward, action, terminal, new_state)

                current_state = new_state
                total_episode_reward += reward

                self.epsilon_decay()

                if step > self.start_learning and step % self.train_freq == 0:
                    self.minibatch_learning()

                if terminal:
                    last_episode_reward = total_episode_reward
                    last_episode_length = step - start_step
                    reward_history.append(last_episode_reward)

                    print('episode: %.2f, total step: %.2f, last_episode length: %.2f, last_episode_reward: %.2f, '
                          'loss: %.4f, eps = %.2f' % (episode, step, last_episode_length, last_episode_reward,
                                                      self.loss, self.epsilon))

                    self.env.reset()

                    if self.gif:
                        generate_gif(last_episode_length, frames_for_gif, total_episode_reward, "./GIF/", episode)

                    break

            if episode % self.reset_step == 0:
                self.target_network.load_state_dict(self.policy_network.state_dict())

            if episode % self.plot_every == 0:
                plot_graph(reward_history)

            # self.env.render()

        self.env.close()

    def minibatch_learning(self):
        state_batch, reward_batch, action_batch, terminal_batch, next_state_batch = self.memory.sample(self.batch_size)
        
        state_batch = torch.tensor(state_batch).float().to(device)
        reward_batch = torch.tensor(reward_batch).float().to(device)
        action_batch = torch.tensor(action_batch).long().to(device)  # Assuming action_batch should be long tensor
        terminal_batch = torch.tensor(terminal_batch).float().to(device)
        next_state_batch = torch.tensor(next_state_batch).float().to(device)

        y_batch = torch.FloatTensor().to(device)
        for i in range(self.batch_size):
            if terminal_batch[i]:
                # y_batch = torch.cat((y_batch, torch.FloatTensor([reward_batch[i]])), 0)
                y_batch = torch.cat((y_batch, reward_batch[i].unsqueeze(0)), 0)
            else:
                # next_state_q = torch.max(self.target_network(torch.FloatTensor(next_state_batch[i])))
                next_state_q = torch.max(self.target_network(torch.cat([next_state_batch[i]])))
                # y = torch.FloatTensor([reward_batch[i] + self.gamma * next_state_q])
                y = reward_batch[i] + self.gamma * next_state_q
                y_batch = torch.cat((y_batch, y.unsqueeze(0)), 0)

        current_state_q = torch.max(self.policy_network(state_batch), dim=1)[0]

        self.loss = self.criterion(current_state_q, y_batch).mean()

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def epsilon_decay(self):
        self.epsilon *= self.epsilon_decay_rate
        self.epsilon = max(self.epsilon, self.epsilon_minimum)


def plot_graph(reward_history):
    df = pd.DataFrame({'x': range(len(reward_history)), 'y': reward_history})
    plt.style.use('seaborn-darkgrid')
    palette = plt.get_cmap('Set1')
    num = 0
    for column in df.drop('x', axis=1):
        num += 1
        plt.plot(df['x'], df[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)
    plt.title("CartPole", fontsize=14)
    plt.xlabel("episode", fontsize=12)
    plt.ylabel("score", fontsize=12)

    plt.savefig('score.png')
