from copy import deepcopy

import numpy
import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
from torch import optim

from replay_buffers import BasicBuffer
from DQN.dqn_model import ConvDQN, DQN


class DQNAgent:

    def __init__(self, env, use_conv=True):
        self.env = env
        self.learning_rate = 0.01
        self.gamma = 0.9
        self.replay_buffer = BasicBuffer(max_size=100000)
        self.epsilon = 1  # 探索率
        self.epsilon_decay = 0.9995  # 衰减因子
        self.epsilon_min = 0.1  # 探索率最小值
        self.update_rate = 100  # 网络更新频率
        self.steps = 0
        self.path = "C:\\fafa\\coding\\python\\dqn\\model_train"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.use_conv = use_conv
        if self.use_conv:
            self.model = ConvDQN(env.observation_space.shape, env.action_space.n).to(self.device)
        else:
            self.model = DQN(env.observation_space.shape, env.action_space.n).to(self.device)

        self.model_prime = deepcopy(self.model)
        # self.optimizer = torch.optim.Adam(self.model.parameters())
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.MSE_loss = nn.MSELoss()

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()

        state = torch.FloatTensor(state).float().unsqueeze(0).to(self.device)
        qvals = self.model.forward(state)       # tensor([[ 0.0078,  0.1858,  0.0379, -0.2320]])
        action = np.argmax(qvals.cpu().detach().numpy())
        return action

    def compute_loss(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.tensor(numpy.array(states), dtype=torch.float).to(self.device)
        actions = torch.tensor(numpy.array(actions), dtype=torch.long).to(self.device)
        rewards = torch.tensor(numpy.array(rewards), dtype=torch.float).to(self.device)
        next_states = torch.tensor(numpy.array(next_states), dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float)

        if self.use_conv:
            states = states.view(len(states), 1, 4)
            next_states = next_states.view(len(states), 1, 4)

        curr_Q = self.model.forward(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        next_Q = self.model_prime.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        expected_Q = rewards.squeeze(1) + self.gamma * max_next_Q

        loss = self.MSE_loss(curr_Q, expected_Q)
        return loss

    def update(self, batch_size):
        batch = self.replay_buffer.sample(batch_size)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save(self.model.state_dict(), path + "/value_model.pt")
        torch.save(self.model_prime.state_dict(), path + "/target_model.pt")

    def load(self, path):
        self.model.load_state_dict(torch.load(path + "/value_model.pt"))
        self.model_prime.load_state_dict(torch.load(path + "/target_model.pt"))