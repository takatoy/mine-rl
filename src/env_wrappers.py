from collections import OrderedDict
import copy
import time
import random
import pickle

import gym
import numpy as np
from gym.wrappers import Monitor
from gym.wrappers.monitoring.stats_recorder import StatsRecorder


class FrameSkip(gym.Wrapper):
    """Return every `skip`-th frame and repeat given action during skip.

    Note that this wrapper does not "maximize" over the skipped frames.
    """
    def __init__(self, env, skip=4, enable_rendering=False):
        super().__init__(env)
        self._skip = skip
        self.enable_rendering = enable_rendering

    def step(self, action):
        # action that need to be taken just once
        tmp_action = copy.deepcopy(action)
        for k in ['place', 'equip', 'craft', 'nearbyCraft', 'nearbySmelt']:
            tmp_action[k] = 0
        tmp_action['camera'] = np.array([0, 0])

        total_reward = 0.0
        for i in range(self._skip):
            a = action if i == 0 else tmp_action
            obs, reward, done, info = self.env.step(a)
            total_reward += reward
            if self.enable_rendering:
                self.env.render()
            if done:
                break
        return obs, total_reward, done, info


full_obs = {
    "equipped_items": {
        "mainhand": {
            "damage": 0,
            "maxDamage": 0,
            "type": 0
        }
    },
    "inventory": {
        "coal": 0,
        "cobblestone": 0,
        "crafting_table": 0,
        "dirt": 0,
        "furnace": 0,
        "iron_axe": 0,
        "iron_ingot": 0,
        "iron_ore": 0,
        "iron_pickaxe": 0,
        "log": 0,
        "planks": 0,
        "stick": 0,
        "stone": 0,
        "stone_axe": 0,
        "stone_pickaxe": 0,
        "torch": 0,
        "wooden_axe": 0,
        "wooden_pickaxe": 0
    },
    "pov": None
}


def get_full_obs(obs):
    s = copy.deepcopy(full_obs)
    if 'equipped_items' in obs.keys() and \
            'mainhand' in obs['equipped_items'].keys():
        for k in obs['equipped_items']['mainhand'].keys():
            s['equipped_items']['mainhand'][k] = obs['equipped_items']['mainhand'][k]
    if 'inventory' in obs.keys():
        for k in obs['inventory'].keys():
            s['inventory'][k] = obs['inventory'][k]
    s['pov'] = obs['pov']
    return s


def get_full_obs_space():
    with open('src/obs_space.pickle', 'rb') as f:
        obs_space = pickle.load(f)
    return obs_space


class ObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = get_full_obs_space()

    def observation(self, observation):
        observation = get_full_obs(observation)
        if type(observation['equipped_items']['mainhand']['type']) is str:
            observation['equipped_items']['mainhand']['type'] = 8
        return observation


class MoveAxisWrapper(gym.ObservationWrapper):
    """Move axes of observation ndarrays."""
    def __init__(self, env, source, destination):
        super().__init__(env)

        self.source = source
        self.destination = destination

        low = np.moveaxis(self.observation_space.spaces['pov'].low, self.source, self.destination)
        high = np.moveaxis(self.observation_space.spaces['pov'].high, self.source, self.destination)
        self.observation_space.spaces['pov'] = gym.spaces.Box(low=low, high=high, dtype=self.observation_space.spaces['pov'].dtype)

    def observation(self, observation):
        observation['pov'] = np.moveaxis(observation['pov'], self.source, self.destination)
        return observation


class CombineActionWrapper(gym.ActionWrapper):
    """Combine MineRL env's "exclusive" actions.

    "exclusive" actions will be combined as:
        - "forward", "back" -> noop/forward/back (Discrete(3))
        - "left", "right" -> noop/left/right (Discrete(3))
        - "sneak", "sprint" -> noop/sneak/sprint (Discrete(3))
        - "attack", "place", "equip", "craft", "nearbyCraft", "nearbySmelt"
            -> noop/attack/place/equip/craft/nearbyCraft/nearbySmelt (Discrete(n))
    The combined action's names will be concatenation of originals, i.e.,
    "forward_back", "left_right", "snaek_sprint", "attack_place_equip_craft_nearbyCraft_nearbySmelt".
    """
    def __init__(self, env):
        super().__init__(env)

        self.wrapping_action_space = self.env.action_space

        def combine_exclusive_actions(keys):
            """
            Dict({'forward': Discrete(2), 'back': Discrete(2)})
            =>
            new_actions: [{'forward':0, 'back':0}, {'forward':1, 'back':0}, {'forward':0, 'back':1}]
            """
            new_key = '_'.join(keys)
            valid_action_keys = [k for k in keys if k in self.wrapping_action_space.spaces]
            noop = {a: 0 for a in valid_action_keys}
            new_actions = [noop]

            for key in valid_action_keys:
                space = self.wrapping_action_space.spaces[key]
                for i in range(1, space.n):
                    op = copy.deepcopy(noop)
                    op[key] = i
                    new_actions.append(op)
            return new_key, new_actions

        self._maps = {}
        for keys in (
                ('forward', 'back'), ('left', 'right'), ('sneak', 'sprint'),
                ('attack', 'place', 'equip', 'craft', 'nearbyCraft', 'nearbySmelt')):
            new_key, new_actions = combine_exclusive_actions(keys)
            self._maps[new_key] = new_actions

        # discretize camera
        self._maps['camera'] = [
            {'camera': np.array([0, 0])},
            {'camera': np.array([10., 0])},
            {'camera': np.array([-10., 0])},
            {'camera': np.array([0, 10.])},
            {'camera': np.array([0, -10.])}
        ]

        self.noop = [0, 0, 0, 0, 0, 0]
        self.keys = [
            'forward_back',
            'left_right',
            'jump',
            'sneak_sprint',
            'camera',
            'attack_place_equip_craft_nearbyCraft_nearbySmelt'
        ]

        self.action_space = gym.spaces.MultiDiscrete([
            len(self._maps['forward_back']),
            len(self._maps['left_right']),
            self.wrapping_action_space.spaces['jump'].n,
            len(self._maps['sneak_sprint']),
            len(self._maps['camera']),
            len(self._maps['attack_place_equip_craft_nearbyCraft_nearbySmelt'])
        ])

    def action(self, action):
        if not self.action_space.contains(action):
            raise ValueError('action {} is invalid for {}'.format(action, self.action_space))

        original_space_action = OrderedDict()
        for i, v in enumerate(action):
            k = self.keys[i]
            if k in self._maps:
                a = self._maps[k][v]
                original_space_action.update(a)
            else:
                original_space_action[k] = v

        return original_space_action


class SerialDiscreteCombineActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.wrapping_action_space = self.env.action_space

        self.noop = OrderedDict([
            ('forward_back', 0),
            ('left_right', 0),
            ('jump', 0),
            ('sneak_sprint', 0),
            ('camera', np.zeros((2, ), dtype=np.float32)),
            ('attack_place_equip_craft_nearbyCraft_nearbySmelt', 0),
        ])

        # get each discrete action
        self._actions = [self.noop]
        for key in self.noop:
            if key == 'camera':
                op = copy.deepcopy(self.noop)
                op[key] = np.array([10., 0], dtype=np.float32)
                self._actions.append(op)
                op = copy.deepcopy(self.noop)
                op[key] = np.array([-10., 0], dtype=np.float32)
                self._actions.append(op)
                op = copy.deepcopy(self.noop)
                op[key] = np.array([0, 10.], dtype=np.float32)
                self._actions.append(op)
                op = copy.deepcopy(self.noop)
                op[key] = np.array([0, -10.], dtype=np.float32)
                self._actions.append(op)
                op = copy.deepcopy(self.noop)
            elif key == 'sneak_sprint':
                for i in [1, 2]:
                    for j in ['forward_back', 'left_right']:
                        for k in [1, 2]:
                            op = copy.deepcopy(self.noop)
                            op[key] = i
                            op[j] = k
                            self._actions.append(op)
            else:
                for a in range(1, self.wrapping_action_space.spaces[key].n):
                    op = copy.deepcopy(self.noop)
                    op[key] = a
                    self._actions.append(op)

        n = len(self._actions)
        self.action_space = gym.spaces.Discrete(n)

    def action(self, action):
        if not self.action_space.contains(action):
            raise ValueError('action {} is invalid for {}'.format(action, self.action_space))

        original_space_action = self._actions[action]
        return original_space_action


mapping = {
    'forward': [0, 1],
    'back': [0, 2],
    'left': [1, 1],
    'right': [1, 2],
    'jump': [2, 1],
    'sneak': [3, 1],
    'sprint': [3, 2],
    'camera': [[4, 1], [4, 2], [4, 3], [4, 4]],
    'attack': [5, 1],
    'place': [[5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7]],
    'equip': [[5, 8], [5, 9], [5, 10], [5, 11], [5, 12], [5, 13], [5, 14]],
    'craft': [[5, 15], [5, 16], [5, 17], [5, 18]],
    'nearbyCraft': [[5, 19], [5, 20], [5, 21], [5, 22], [5, 23], [5, 24], [5, 25]],
    'nearbySmelt': [[5, 26], [5, 27]]
}


def _data_action_wrapper(action):
    wrapped = [0, 0, 0, 0, 0, 0]
    for k, v in action.items():
        if k in ['forward', 'back', 'left', 'right', 'jump', 'attack'] and v == 1:
            wrapped[mapping[k][0]] = mapping[k][1]
        elif k in ['place', 'equip', 'craft', 'nearbyCraft', 'nearbySmelt'] and v != 0:
            wrapped[mapping[k][v - 1][0]] = mapping[k][v - 1][1]
        elif k == 'camera':
            if v[0] > 10.0:
                wrapped[mapping[k][0][0]] = mapping[k][0][1]
            if v[0] < -10.0:
                wrapped[mapping[k][1][0]] = mapping[k][1][1]
            if v[1] > 10.0:
                wrapped[mapping[k][2][0]] = mapping[k][2][1]
            if v[1] < -10.0:
                wrapped[mapping[k][3][0]] = mapping[k][3][1]
    return wrapped


def data_action_wrapper(data_actions):
    actions = []
    for i in range(len(data_actions['attack'])):
        a = {}
        for k, v in data_actions.items():
            a[k] = v[i]
        actions.append(_data_action_wrapper(a))
    return actions


def data_state_wrapper(data_states):
    states = []
    for i in range(len(data_states['pov'])):
        s = copy.deepcopy(full_obs)
        for k in s['equipped_items']['mainhand'].keys():
            if 'equipped_items' in data_states.keys() and \
                    'mainhand' in data_states['equipped_items'].keys() and \
                    k in data_states['equipped_items']['mainhand'].keys():
                s['equipped_items']['mainhand'][k] = data_states['equipped_items']['mainhand'][k][i]
        for k in s['inventory'].keys():
            if 'inventory' in data_states.keys():
                s['inventory'][k] = data_states['inventory'][k][i]
        if 'pov' in s.keys():
            s['pov'] = np.moveaxis(data_states['pov'][i], -1, 0)
        states.append(s)
    return states
