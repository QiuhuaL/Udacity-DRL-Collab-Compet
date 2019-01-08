import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=512, fc2_units=256, fc3_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        if state.dim() == 1:
            state = torch.unsqueeze(state, 0)
        x = F.relu(self.fc1(state))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return torch.tanh(self.fc4(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, input_size, action_size, seed, fc1_units=512, fc2_units=256, fc3_units=64):
        """Initialize parameters and build model.
        Params
        ======
            input_size (int): dimension of the input
            seed (int): Random seed
            fc1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        # for MADPGG, the input size is (state_size + action_size) * Num_agents (=2 for Unity Tennis Environment)
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4_1 = nn.Linear(fc3_units, action_size)  # for agent 1
        self.fc4_2 = nn.Linear(fc3_units, action_size)  # for agent 2
        self.q1 = nn.Linear(action_size, 1)  # for agent 1
        self.q2 = nn.Linear(action_size, 1)  # for agent 1
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4_1.weight.data.uniform_(*hidden_init(self.fc4_1))
        self.fc4_2.weight.data.uniform_(*hidden_init(self.fc4_2))
        self.q1.weight.data.uniform_(-3e-3, 3e-3)
        self.q2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states_actions):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # input is the concatenation of states and actions from multiple (two) agents
        if states_actions.dim() == 1:
            states_actions = torch.unsqueeze(states_actions, 0)
        x = F.relu(self.fc1(states_actions))
        x = self.bn1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x1 = F.relu(self.fc4_1(x))
        x2 = F.relu(self.fc4_2(x))
        qv1 = self.q1(x1)
        qv2 = self.q2(x2)
        # return the q value for agent 1 and agent 2
        return torch.cat((qv1, qv2), dim=1)
