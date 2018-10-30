#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class QNetwork(nn.Module):
    'Policy Model for solving RL tasks with deep learning.'

    def __init__(self, state_size: int, action_size: int, seed: int):
        '''Initializes parameters and builds the model.

        Parameters
        ----------
            state_size: int
                Dimensions of each observation
            action_size: int
                Dimensions of the actions
            seed: int
                Random seed
        '''
        super().__init__()

        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        self.state_size = state_size

        self.fc1 = nn.Linear(state_size, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, action_size)

    def forward(self, state):
        'Runs the state through the network to generate action values.'

        x = F.leaky_relu(self.fc1(state))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)

        return x

class AdvantageNetwork(nn.Module):
    'Policy Model with Dueling Networks for solving RL tasks with deep learning.'


    def __init__(self, state_size: int, action_size: int, seed: int):
        '''Initializes parameters and builds the model.

        Parameters
        ----------
            state_size: int
                Dimensions of each observation
            action_size: int
                Dimensions of the actions
            seed: int
                Random seed
        '''
        super().__init__()

        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        self.state_size = state_size

        self.fc1 = nn.Linear(state_size, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 64)

        self.value = nn.Linear(64, 1)
        self.advantage = nn.Linear(64, action_size)

    def forward(self, state):
        'Runs the state through the network to generate action values.'

        x = F.leaky_relu(self.fc1(state))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))

        value = self.value(x)
        advantage = self.advantage(x)

        return value + (advantage - advantage.mean())


class VisualQNetwork(nn.Module):
    'Policy Model with Dueling Networks for solving RL tasks with deep learning.'

    def __init__(self, state_size: tuple, action_size: int, seed: int):
        '''Initializes parameters and builds the model.

        Parameters
        ----------
            state_size: tuple
                Dimensions of each observation
            action_size: int
                Dimensions of the actions
            seed: int
                Random seed
        '''
        super().__init__()

        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        self.state_size = state_size

        _, self.channels, self.stack_size, self.height, self.width = state_size

        self.conv1 = nn.Conv2d(self.stack_size, 128, kernel_size=(3, 3), stride=(3, 3))
        self.conv2 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(3, 3))
        self.conv3 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(3, 3))
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(3, 3))

        self.h1, self.w1, _ = self._conv_size(self.height, self.width, self.stack_size, 3, 0, 1, 3)
        self.h2, self.w2, _ = self._conv_size(self.h1, self.w1, self.stack_size, 3, 0, 1, 3)
        self.h3, self.w3, _ = self._conv_size(self.h2, self.w2, 0, 3, 0, 1, 3)
        self.h4, self.w4, _ = self._conv_size(self.h3, self.w3, 0, 3, 0, 1, 3)

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(self.h4 * self.w4 * 512, 1024)
        self.fc2 = nn.Linear(1024, action_size)

    def _conv_size(self, h, w, d, kernel_size, padding, dilation, stride):
        return math.floor((h + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1), \
                math.floor((w + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1), \
                math.floor((d + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

    def conv_forward(self, state):
        'Runs the state through the convolutional layers of the network.'

        x = F.relu(self.bn1(self.conv1(state)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        return x.view(x.size()[0], -1)


    def forward(self, state):
        'Runs the state through the network to generate action values.'

        x = self.conv_forward(state)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class VisualAdvantageNetwork(VisualQNetwork):
    'Policy Model with Dueling Networks for solving RL tasks with deep learning.'

    def __init__(self, state_size: tuple, action_size: int, seed: int):
        '''Initializes parameters and builds the model.

        Parameters
        ----------
            state_size: tuple
                Dimensions of each observation
            action_size: int
                Dimensions of the actions
            seed: int
                Random seed
        '''
        super().__init__(state_size, action_size, seed)

        self.value_fc = nn.Linear(self.h4 * self.w4 * 512, 1024)
        self.advantage_fc = nn.Linear(self.h4 * self.w4 * 512, 1024)

        self.value = nn.Linear(1024, 1)
        self.advantage = nn.Linear(1024, action_size)

    def forward(self, state):
        'Runs the state through the network to generate action values.'

        x = self.conv_forward(state)
        value = self.value(F.relu(self.value_fc(x)))
        advantage = self.advantage(F.relu(self.advantage_fc(x)))

        return value + (advantage - advantage.mean())


class DDPGActor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(DDPGActor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.tanh(self.fc3(x))


class DDPGCritic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=256, fc2_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(DDPGCritic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = F.relu(self.fcs1(state))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
