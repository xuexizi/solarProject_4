from __future__ import print_function
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np


class MemoryDNN:
    def __init__(
            self,
            net,
            learning_rate=0.01,
            training_interval=10,
            batch_size=100,
            memory_size=1000,
    ):

        self.net = net
        self.lr = learning_rate
        self.training_interval = training_interval
        self.batch_size = batch_size
        self.memory_size = memory_size

        # store all binary actions
        self.enumerate_actions = []

        # stored # memory entry
        self.memory_counter = 1

        # store training cost
        self.cost_his = []

        # initialize zero memory [h, m]
        self.memory = np.zeros((self.memory_size, self.net[0] + self.net[-1]))

        # construct memory network
        self._build_net()

    def _build_net(self):
        self.model = nn.Sequential(
            nn.Linear(self.net[0], self.net[1]),
            nn.ReLU(),
            nn.Linear(self.net[1], self.net[2]),
            nn.ReLU(),
            nn.Linear(self.net[2], self.net[3]),
            nn.Sigmoid()
        )

    def remember(self, nn_input, result):
        # replace the old memory with new memory
        idx = self.memory_counter % self.memory_size
        self.memory[idx, :] = np.hstack((nn_input, result))
        self.memory_counter += 1

    def encode(self, nn_input, result):
        self.remember(nn_input, result)
        if self.memory_counter % self.training_interval == 0:
            self.learn()

    def learn(self):
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        in_train = torch.Tensor(batch_memory[:, 0: self.net[0]])  # in_train 是输入数据
        out_train = torch.Tensor(batch_memory[:, self.net[0]:])  # out_train 是目标标签

        # train the DNN
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.09, 0.999), weight_decay=0.0001)
        criterion = nn.BCELoss()
        self.model.train()
        optimizer.zero_grad()
        predict = self.model(in_train)
        loss = criterion(predict, out_train)
        loss.backward()
        optimizer.step()

        cost = loss.item()
        if cost < 0:
            print(cost)
        assert (cost >= 0)
        self.cost_his.append(cost)

    def decode(self, nn_input):
        nn_input = torch.Tensor(nn_input)
        self.model.eval()
        pred = self.model(nn_input).detach().numpy()
        return pred

    def save_model(self):
        torch.save({'model': self.model.state_dict()}, 'model_name.pth')

    def read_model(self):
        state_dict = torch.load('model_name.pth')
        self.model.load_state_dict(state_dict['model'])

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)) * self.training_interval, self.cost_his)
        plt.ylabel('Training Loss')
        plt.xlabel('Time Frames')
        plt.show()
