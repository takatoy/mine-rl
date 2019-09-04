import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from gym.spaces.utils import flatdim, flatten, unflatten

MEMORY_CAPACITY = 512
BATCH_SIZE = 128
GAMMA = 0.99
LAMBDA = 0.95
C_1 = 0.5
C_2 = 0.01
EPS_CLIP = 0.1
K_EPOCH = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Memory:
    def __init__(self, capacity):
        self.cursor = 0
        self.capacity = capacity
        self.is_max = False
        self.data = []

    def push(self, datum):
        if self.is_max:
            self.data[self.cursor] = datum
        else:
            self.data.append(datum)
        
        self.cursor += 1
        if self.cursor == self.capacity:
            self.is_max = True
        self.cursor %= self.capacity

    def sample(self, batch_size):
        high = self.capacity if self.is_max else self.cursor
        idx = random.sample(range(0, high), batch_size)
        batches = []
        for i in idx:
            batches.append(self.data[i])
        return batches


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
        self.do1 = nn.Dropout(p=0.5)
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
        self.value = ValueDecoder()
        self.action = ActionDecoder(n_action)

    def embed(self, p, i):
        hp = self.pov(p)
        hi = self.item(i)
        return torch.cat((hp, hi), -1)

    def val(self, p, i):
        h = self.embed(p, i)
        x = self.value(h)
        return x

    def act(self, p, i):
        h = self.embed(p, i)
        x = self.action(h)
        return x


class Discriminator(nn.Module):
    def __init__(self, n_item, n_action):
        super().__init__()
        self.pov = PovEncoder()
        self.item = ItemEncoder(n_item)
        self.fc1 = nn.Linear(640 + n_action, 512)
        self.do1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 256)
        self.do2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, p, i, a):
        x = self.pov(p)
        y = self.item(i)
        x = torch.cat([x, y, a], -1)
        x = F.relu(self.do1(self.fc1(x)))
        x = F.relu(self.do2(self.fc2(x)))
        x = torch.sigmoid(self.fc3(x))
        return x


class Agent:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

        item_dim = flatdim(self.observation_space['equipped_items']) + \
            flatdim(self.observation_space['inventory'])

        self.policy = Policy(item_dim, self.action_space.n).to(device)
        self.policy_optim = torch.optim.Adam(self.policy.parameters())
        self.discriminator = Discriminator(item_dim, self.action_space.n).to(device)
        self.discriminator_optim = torch.optim.Adam(self.discriminator.parameters())

        self.memory = Memory(capacity=MEMORY_CAPACITY)

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

        self.last_lp = None

    def act(self, obs):
        pov, item = self.preprocess(obs)
        pov = torch.tensor([pov], device=device).float()
        item = torch.tensor([item], device=device).float()

        act = self.policy.act(pov, item)

        m = Categorical(act)
        action = m.sample()
        self.last_lp = m.log_prob(action).item()

        return action.item()

    def preprocess(self, obs):
        pov = obs['pov'].astype(np.float) / 255
        item = np.concatenate([
            flatten(self.observation_space['equipped_items'], obs['equipped_items']),
            flatten(self.observation_space['inventory'], obs['inventory'])
        ])
        return pov, item

    def add_data(self, obs, action, reward, n_obs, done):
        pov, item = self.preprocess(obs)
        n_pov, n_item = self.preprocess(n_obs)
        action = flatten(self.action_space, action)
        action = np.argmax(action)  # to one-hot
        datum = [pov, item, action, reward, n_pov, n_item, done, self.last_lp]
        self.memory.push(datum)

    def train_discriminator(self, expert_states, expert_actions):
        n = len(expert_states)

        exp_povs = []
        exp_items = []
        povs = []
        items = []
        for i, s in enumerate(expert_states):
            pov, item = self.preprocess(s)
            if i < n // 2:
                exp_povs.append(pov)
                exp_items.append(item)
            else:
                povs.append(pov)
                items.append(item)
        exp_povs = torch.tensor(exp_povs, dtype=torch.float, device=device)
        exp_items = torch.tensor(exp_items, dtype=torch.float, device=device)
        povs = torch.tensor(povs, dtype=torch.float, device=device)
        items = torch.tensor(items, dtype=torch.float, device=device)

        exp_actions = []
        for i in range(n // 2):
            exp_actions.append(flatten(self.action_space, expert_actions[i]))
        exp_actions = torch.tensor(exp_actions, dtype=torch.float, device=device)
        exp_labels = torch.full((exp_actions.size(0), 1), 1, device=device)

        actions = self.policy.act(povs, items)
        labels = torch.full((actions.size(0), 1), 0, device=device)

        # discriminator
        probs = self.discriminator(exp_povs, exp_items, exp_actions)
        loss = self.bce_loss(probs, exp_labels)
        probs = self.discriminator(povs, items, actions.detach())
        loss += self.bce_loss(probs, labels)
        print('discriminator loss: {}'.format(loss))

        self.discriminator_optim.zero_grad()
        loss.backward()
        self.discriminator_optim.step()

    def bonus_reward(self, state, action):
        pov, item = self.preprocess(state)
        pov = torch.tensor([pov], device=device).float()
        item = torch.tensor([item], device=device).float()
        action = torch.tensor([flatten(self.action_space, action)], device=device).float()
        return -torch.log(self.discriminator(pov, item, action)).item()

    def train(self):
        self.policy.train()

        pov, item, action, reward, n_pov, n_item, done_mask, olp = self.make_batches()

        for _ in range(K_EPOCH):
            s_val = self.policy.val(pov, item)
            td_target = reward.unsqueeze(-1) + GAMMA * self.policy.val(n_pov, n_item) * done_mask
            delta = td_target - s_val
            delta = delta.detach().numpy()

            adv = []
            A = 0.0
            for d in delta[::-1]:
                A = GAMMA * LAMBDA * A + d[0]
                adv.insert(0, [A])
            adv = torch.tensor(adv, dtype=torch.float, device=device)

            prob = self.policy.act(pov, item)
            m = Categorical(prob)
            action = action.squeeze(-1)
            lp = m.log_prob(action)
            entropy = m.entropy().unsqueeze(-1)
            ratio = torch.exp(lp - olp.detach()).unsqueeze(-1)

            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * adv
            loss = -torch.min(surr1, surr2) \
                + C_1 * self.mse_loss(s_val, td_target.detach()) \
                - C_2 * entropy

            self.policy_optim.zero_grad()
            loss.mean().backward()
            self.policy_optim.step()

    def make_batches(self):
        samples = self.memory.sample(BATCH_SIZE)
        samples = list(zip(*samples))  # transpose

        pov = torch.tensor(samples[0], dtype=torch.float, device=device)
        item = torch.tensor(samples[1], dtype=torch.float, device=device)
        action = torch.tensor(samples[2], dtype=torch.int, device=device)
        reward = torch.tensor(samples[3], dtype=torch.float, device=device)
        n_pov = torch.tensor(samples[4], dtype=torch.float, device=device)
        n_item = torch.tensor(samples[5], dtype=torch.float, device=device)
        
        done_mask = np.where(np.array(samples[6]), 0, 1)
        done_mask = torch.from_numpy(done_mask).float().to(device)
        done_mask = done_mask.unsqueeze(-1)

        lp = torch.tensor(samples[7], dtype=torch.float, device=device)

        return pov, item, action, reward, n_pov, n_item, done_mask, lp
