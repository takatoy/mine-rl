import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PovEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(1024, 512)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ItemEncoder(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
    
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, nvec, item_dim):
        super().__init__()
        self.nvec = nvec
        n_action = np.sum(nvec)
        self.pov = PovEncoder()
        self.item = ItemEncoder(item_dim)
        self.fc1 = nn.Linear(512 + 64, 256)
        self.fc2 = nn.Linear(256, n_action)

    def forward(self, p, i):
        hp = F.leaky_relu(self.pov(p))
        hi = F.leaky_relu(self.item(i))
        x = torch.cat((hp, hi), -1)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        xs = torch.split(x, self.nvec.tolist(), dim=-1)
        xs = [F.softmax(x, dim=-1) for x in xs]
        return xs


class Critic(nn.Module):
    def __init__(self, item_dim):
        super().__init__()
        self.pov = PovEncoder()
        self.item = ItemEncoder(item_dim)
        self.fc1 = nn.Linear(512 + 64, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, p, i):
        hp = F.leaky_relu(self.pov(p))
        hi = F.leaky_relu(self.item(i))
        x = torch.cat((hp, hi), -1)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, nvec, item_dim):
        super().__init__()
        n_action = np.sum(nvec)
        self.pov = PovEncoder()
        self.item = ItemEncoder(item_dim)
        self.fc1 = nn.Linear(512 + 64 + n_action, 512)
        self.do1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 256)
        self.do2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, p, i, a):
        x = self.pov(p)
        y = self.item(i)
        x = torch.cat([x, y, a], -1)
        x = F.leaky_relu(self.do1(self.fc1(x)))
        x = F.leaky_relu(self.do2(self.fc2(x)))
        x = self.fc3(x)
        return x
