import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_

from gym.spaces.utils import flatdim, flatten

from .model import Actor, Critic, Discriminator

########## params ##########
GAMMA = 0.99
LAMBDA = 0.95
C_1 = 1.0
C_2 = 0.01
EPS_CLIP = 0.2
K_EPOCH = 3
BONUS_RATIO = 1e-7
CLIPPING_VALUE = 1.0
LEARNING_RATE = 0.0001
BATCH_SIZE = 256
############################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_printoptions(profile="full")


class Agent:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space

        self.item_dim = flatdim(self.observation_space['equipped_items']) + \
            flatdim(self.observation_space['inventory'])

        self.actor = Actor(self.action_space.nvec, self.item_dim).to(device)
        self.critic = Critic(self.item_dim).to(device)
        self.discriminator = Discriminator(self.action_space.nvec, self.item_dim).to(device)

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=LEARNING_RATE)
        self.discriminator_optim = torch.optim.Adam(self.discriminator.parameters(), lr=LEARNING_RATE)

        self.mse_loss = nn.MSELoss(reduction='none')
        self.bce_loss = nn.BCELoss()

        self.last_lp = None

        self.memory = []

    def act(self, obs, action=None):
        pov, item = self.preprocess(obs)
        pov = torch.tensor([pov], device=device).float()
        item = torch.tensor([item], device=device).float()

        act = self.actor(pov, item)

        ms = [Categorical(a) for a in act]
        if action is not None:
            action = torch.tensor(action, device=device).unsqueeze(-1)
        else:
            action = [m.sample() for m in ms]
        self.last_lp = np.sum([m.log_prob(a).item() for m, a in zip(ms, action)]).item()

        return [a.item() for a in action]

    def preprocess(self, obs):
        pov = obs['pov'].astype(np.float) / 255.0
        item = np.concatenate([
            flatten(self.observation_space['equipped_items'], obs['equipped_items']),
            flatten(self.observation_space['inventory'], obs['inventory'])
        ])
        return pov, item

    def action_to_onehot(self, action):
        action = np.concatenate([np.eye(self.action_space.nvec[i])[a] for i, a in enumerate(action)])
        return action

    def add_data(self, obs, action, reward, n_obs, done):
        pov, item = self.preprocess(obs)
        n_pov, n_item = self.preprocess(n_obs)
        datum = [pov, item, action, reward, n_pov, n_item, done, self.last_lp]
        self.memory.append(datum)

    def is_memory_empty(self):
        return len(self.memory) == 0

    def train_discriminator(self, expert_states, expert_actions):
        self.discriminator.train()

        n = len(expert_states)
        if n % 2 == 1:
            # To even number
            expert_states = expert_states[:-1]
            expert_actions = expert_actions[:-1]
            n -= 1
        if n == 0:
            return 0.0

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
            exp_actions.append(self.action_to_onehot(expert_actions[i]))
        exp_actions = torch.tensor(exp_actions, dtype=torch.float, device=device)
        act = self.actor(povs, items)
        ms = [Categorical(a) for a in act]
        act = torch.cat([
                torch.eye(self.action_space.nvec[i].item())[m.sample()] for i, m in enumerate(ms)
            ], dim=1).to(device)

        # train discriminator with WGAN-gp
        exp_pred = self.discriminator(exp_povs, exp_items, exp_actions)
        exp_loss = exp_pred.mean()

        fake_pred = self.discriminator(povs, items, act.detach())
        fake_loss = fake_pred.mean()

        # gradient penalty
        n_action = np.sum(self.action_space.nvec)
        alpha1 = torch.rand(n // 2, *self.observation_space['pov'].shape).to(device)
        alpha2 = torch.rand(n // 2, self.item_dim).to(device)
        alpha3 = torch.rand(n // 2, n_action).to(device)
        pov_interpolates = (alpha1 * exp_povs + ((1 - alpha1) * povs)).detach().requires_grad_()
        item_interpolates = (alpha2 * exp_items + ((1 - alpha2) * items)).detach().requires_grad_()
        action_interpolates = (alpha3 * exp_actions + ((1 - alpha3) * act)).detach().requires_grad_()
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
        clip_grad_norm_(self.discriminator.parameters(), CLIPPING_VALUE)
        self.discriminator_optim.step()

        return loss.item()

    def bonus_reward(self, state, action, n_state):
        pov, item = self.preprocess(state)
        pov = torch.tensor([pov], device=device).float()
        item = torch.tensor([item], device=device).float()
        action = torch.tensor([self.action_to_onehot(action)], device=device).float()
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
            s_val = self.critic(pov, item)
            total_value += s_val.mean().item()
            td_target = reward + GAMMA * self.critic(n_pov, n_item) * done_mask
            delta = td_target - s_val
            delta = delta.detach().cpu().numpy()
            value_loss = C_1 * self.mse_loss(s_val, td_target.detach())
            total_value_loss += value_loss.mean().item()

            self.critic_optim.zero_grad()
            value_loss.mean().backward()
            clip_grad_norm_(self.critic.parameters(), CLIPPING_VALUE)
            self.critic_optim.step()

            adv = []
            A = 0.0
            for d in delta[::-1]:
                A = GAMMA * LAMBDA * A + d[0]
                adv.insert(0, [A])
            adv = torch.tensor(adv, dtype=torch.float, device=device)
            # adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            probs = self.actor(pov, item)
            ms = [Categorical(prob) for prob in probs]
            lp = torch.cat([m.log_prob(a).unsqueeze(0) for m, a in zip(ms, action)]).transpose(0, 1)
            lp = torch.sum(lp, 1, keepdim=True)
            ratio = torch.exp(lp - olp.detach())
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - EPS_CLIP, 1.0 + EPS_CLIP) * adv
            ppo_loss = -torch.min(surr1, surr2)
            total_ppo_loss += ppo_loss.mean().item()

            entropy = torch.cat([m.entropy().unsqueeze(0) for m in ms]).transpose(0, 1)
            entropy = torch.sum(entropy, 1, keepdim=True)
            entropy_loss = -C_2 * entropy
            total_entropy += entropy.mean().item()

            loss = ppo_loss + entropy_loss

            self.actor_optim.zero_grad()
            loss.mean().backward()
            clip_grad_norm_(self.actor.parameters(), CLIPPING_VALUE)
            self.actor_optim.step()

        del self.memory[:BATCH_SIZE]

        return (total_value / K_EPOCH, total_ppo_loss / K_EPOCH,
                total_value_loss / K_EPOCH, total_entropy / K_EPOCH)

    def make_batches(self, data):
        samples = list(zip(*data))  # transpose

        pov = torch.tensor(samples[0], dtype=torch.float, device=device)
        item = torch.tensor(samples[1], dtype=torch.float, device=device)
        action = torch.tensor(samples[2], dtype=torch.int, device=device).transpose(0, 1)
        reward = torch.tensor(samples[3], dtype=torch.float, device=device).unsqueeze(-1)
        n_pov = torch.tensor(samples[4], dtype=torch.float, device=device)
        n_item = torch.tensor(samples[5], dtype=torch.float, device=device)
        
        done_mask = np.where(np.array(samples[6]), 0, 1)
        done_mask = torch.from_numpy(done_mask).float().to(device)
        done_mask = done_mask.unsqueeze(-1)

        lp = torch.tensor(samples[7], dtype=torch.float, device=device).unsqueeze(-1)

        return pov, item, action, reward, n_pov, n_item, done_mask, lp

    def save_model(self, path='train/checkpoint.pth'):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'actor_optim_state_dict': self.actor_optim.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_optim_state_dict': self.critic_optim.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'discriminator_optim_state_dict': self.discriminator_optim.state_dict()
        }, path)

    def load_model(self, path='train/checkpoint.pth'):
        checkpoint = torch.load(path, map_location=device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_optim.load_state_dict(checkpoint['actor_optim_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_optim.load_state_dict(checkpoint['critic_optim_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.discriminator_optim.load_state_dict(checkpoint['discriminator_optim_state_dict'])
