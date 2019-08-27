import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from gym.spaces.utils import flatdim, flatten, unflatten

K_EPOCH = 3
HIDDEN_SIZE = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PovEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(1024, 512)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc(x))
        return x


class ItemEncoder(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.do1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 128)
        self.do2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(128, 128)
    
    def forward(self, x):
        x = F.relu(self.do1(self.fc1(x)))
        x = F.relu(self.do2(self.fc2(x)))
        x = F.relu(self.fc3(x))
        return x


class ValueDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512 + 128, 512)
        self.do2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 256)
        self.do2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.do1(self.fc1(x)))
        x = F.relu(self.do2(self.fc2(x)))
        x = self.fc3(x)
        return x


class ActionDecoder(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.fc1 = nn.Linear(512 + 128, 512)
        self.do1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 256)
        self.do2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(256, out_features)

    def forward(self, x):
        x = F.relu(self.do1(self.fc1(x)))
        x = F.relu(self.do2(self.fc2(x)))
        x = F.softmax(self.fc3(x), dim=-1)
        return x


class Policy(nn.Module):
    def __init__(self, n_item, n_action):
        super().__init__()
        self.pov = PovEncoder()
        self.item = ItemEncoder(n_item)
        self.val = ValueDecoder()
        self.action = ActionDecoder(n_action)

    def embed(self, p, i):
        hp = self.pov(p)
        hi = self.item(i)
        return torch.cat((hp, hi), -1)

    def val(self, p, i):
        h = self.embed(p, i)
        x = self.val(h)
        return x

    def act(self, p, i):
        h = self.embed(p, i)
        x = self.action(h)
        return x


class Agent:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

        obs_dim = flatdim(self.observation_space['equipped_items']) + \
            flatdim(self.observation_space['inventory'])

        self.policy = Policy(obs_dim, self.action_space.n).to(device)

        # self.memory = []

    def act(self, obs):
        pov = obs['pov'].astype(np.float) / 255
        item = np.concatenate([
            flatten(self.observation_space['equipped_items'], obs['equipped_items']),
            flatten(self.observation_space['inventory'], obs['inventory'])
        ])
        pov = torch.tensor([pov], device=device).float()
        item = torch.tensor([item], device=device).float()

        act = self.policy.act(pov, item)

        m = Categorical(act)
        action_idx = m.sample()
        action = np.zeros(self.action_space.n, dtype=np.int)
        action[action_idx.item()] = 1

        return unflatten(self.action_space, action)


    def add_data(self, s, a, r, ns):
        pass

    def train(self):
        pass
