import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_

from gym.spaces.utils import flatdim, flatten, unflatten

########## params ##########
GAMMA = 0.99
LAMBDA = 0.95
C_1 = 1.0
C_2 = 0.01
EPS_CLIP = 0.2
K_EPOCH = 3
BONUS_RATIO = 1e-7
CLIPPING_VALUE = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 256
############################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_printoptions(profile="full")


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
        x = F.leaky_relu(self.fc(x))
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
        x = F.leaky_relu(self.fc3(x))
        return x


class ValueDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512 + 64, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ActionDecoder(nn.Module):
    def __init__(self, out_features):
        super().__init__()
        self.fc1 = nn.Linear(512 + 64, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, out_features)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = F.softmax(self.fc4(x), dim=-1)
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


class Agent:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

        self.item_dim = flatdim(self.observation_space['equipped_items']) + \
            flatdim(self.observation_space['inventory'])

        self.policy = Policy(self.item_dim, self.action_space.n).to(device)
        self.discriminator = Discriminator(self.item_dim, self.action_space.n).to(device)

        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        self.discriminator_optim = torch.optim.Adam(self.discriminator.parameters(), lr=LEARNING_RATE)

        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCELoss()

        self.last_lp = None

        self.memory = []

    def act(self, obs, action=None):
        pov, item = self.preprocess(obs)
        pov = torch.tensor([pov], device=device).float()
        item = torch.tensor([item], device=device).float()

        act = self.policy.act(pov, item)

        m = Categorical(act)
        action = torch.tensor(action, device=device) if action is not None else m.sample()
        self.last_lp = m.log_prob(action).item()

        return action.item()

    def preprocess(self, obs):
        pov = obs['pov'].astype(np.float) / 255.0
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
        self.memory.append(datum)

    def is_memory_empty(self):
        return len(self.memory) == 0

    def train_discriminator(self, expert_states, expert_actions):
        self.discriminator.train()

        n = len(expert_states)
        if n % 2 == 1:
            expert_states = expert_states[:-1]
            expert_actions = expert_actions[:-1]
            n -= 1

        # shuffle data
        expert_states = np.array(expert_states)
        expert_actions = np.array(expert_actions)
        idx = np.random.permutation(n)
        expert_states = expert_states[idx]
        expert_actions = expert_actions[idx]

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
        actions = self.policy.act(povs, items)

        # train discriminator with WGAN-gp
        exp_pred = self.discriminator(exp_povs, exp_items, exp_actions)
        exp_loss = exp_pred.mean()

        fake_pred = self.discriminator(povs, items, actions.detach())
        fake_loss = fake_pred.mean()

        # gradient penalty
        alpha1 = torch.rand(n // 2, *self.observation_space['pov'].shape).to(device)
        alpha2 = torch.rand(n // 2, self.item_dim).to(device)
        alpha3 = torch.rand(n // 2, self.action_space.n).to(device)
        pov_interpolates = (alpha1 * exp_povs + ((1 - alpha1) * povs)).detach().requires_grad_()
        item_interpolates = (alpha2 * exp_items + ((1 - alpha2) * items)).detach().requires_grad_()
        action_interpolates = (alpha3 * exp_actions + ((1 - alpha3) * actions)).detach().requires_grad_()
        pred = self.discriminator(pov_interpolates, item_interpolates, action_interpolates)
        gradients = autograd.grad(outputs=pred,
                                  inputs=[pov_interpolates, item_interpolates, action_interpolates],
                                  grad_outputs=torch.ones(pred.size()).to(device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(n // 2, -1)
        gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
        gradient_penalty = ((gradients_norm - 1.0) ** 2).mean() * 10

        loss = fake_loss - exp_loss + gradient_penalty

        self.discriminator_optim.zero_grad()
        loss.backward()
        clip_grad_norm_(self.policy.parameters(), CLIPPING_VALUE)
        self.discriminator_optim.step()

        return loss.item()

    def bonus_reward(self, state, action, n_state):
        pov, item = self.preprocess(state)
        pov = torch.tensor([pov], device=device).float()
        item = torch.tensor([item], device=device).float()
        action = torch.tensor([flatten(self.action_space, action)], device=device).float()
        pred = self.discriminator(pov, item, action)
        reward = pred * BONUS_RATIO
        return reward.item()

    def train_policy(self):
        pov, item, action, reward, n_pov, n_item, done_mask, olp = \
            self.make_batches(self.memory[:BATCH_SIZE])

        total_value = 0.0
        total_ppo_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        for _ in range(K_EPOCH):
            s_val = self.policy.val(pov, item)
            total_value += s_val.mean().item()
            td_target = reward.unsqueeze(-1) + GAMMA * self.policy.val(n_pov, n_item) * done_mask
            delta = td_target - s_val
            delta = delta.detach().cpu().numpy()

            adv = []
            A = 0.0
            for d in delta[::-1]:
                A = GAMMA * LAMBDA * A + d[0]
                adv.insert(0, [A])
            adv = torch.tensor(adv, dtype=torch.float, device=device)
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            prob = self.policy.act(pov, item)
            m = Categorical(prob)

            lp = m.log_prob(action)
            ratio = torch.exp(lp - olp.detach()).unsqueeze(-1)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - EPS_CLIP, 1.0 + EPS_CLIP) * adv
            ppo_loss = -torch.min(surr1, surr2)
            total_ppo_loss += ppo_loss.mean().item()

            value_loss = C_1 * self.mse_loss(s_val, td_target.detach())
            total_value_loss += value_loss.mean().item()

            entropy = m.entropy().unsqueeze(-1)
            entropy_loss = -C_2 * entropy
            total_entropy += entropy.mean().item()

            loss = ppo_loss + value_loss + entropy_loss

            self.policy_optim.zero_grad()
            loss.mean().backward()
            clip_grad_norm_(self.policy.parameters(), CLIPPING_VALUE)
            self.policy_optim.step()
        del self.memory[:BATCH_SIZE]

        return total_value / K_EPOCH, total_ppo_loss / K_EPOCH, total_value_loss / K_EPOCH, total_entropy / K_EPOCH

    def make_batches(self, data):
        samples = list(zip(*data))  # transpose

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

    def save_model(self, path='train/checkpoint.pth'):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'policy_optim_state_dict': self.policy_optim.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'discriminator_optim_state_dict': self.discriminator_optim.state_dict()
        }, path)

    def load_model(self, path='train/checkpoint.pth'):
        checkpoint = torch.load(path, map_location=device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy_optim.load_state_dict(checkpoint['policy_optim_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.discriminator_optim.load_state_dict(checkpoint['discriminator_optim_state_dict'])
