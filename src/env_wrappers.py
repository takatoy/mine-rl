from collections import OrderedDict
import copy
import time
import random

import gym
import numpy as np
from gym.wrappers import Monitor
from gym.wrappers.monitoring.stats_recorder import StatsRecorder


class FrameSkip(gym.Wrapper):
    """Return every `skip`-th frame and repeat given action during skip.

    Note that this wrapper does not "maximize" over the skipped frames.
    """
    def __init__(self, env, skip=4):
        super().__init__(env)

        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class ObsWrapper(gym.ObservationWrapper):
    """Obtain 'pov' value (current game display) of the original observation."""
    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
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

        self.noop = OrderedDict([
            ('forward_back', 0),
            ('left_right', 0),
            ('jump', 0),
            ('sneak_sprint', 0),
            ('camera', np.zeros((2, ), dtype=np.float32)),
            ('attack_place_equip_craft_nearbyCraft_nearbySmelt', 0),
        ])

        self.action_space = gym.spaces.Dict({
            'forward_back':
                gym.spaces.Discrete(len(self._maps['forward_back'])),
            'left_right':
                gym.spaces.Discrete(len(self._maps['left_right'])),
            'jump':
                self.wrapping_action_space.spaces['jump'],
            'sneak_sprint':
                gym.spaces.Discrete(len(self._maps['sneak_sprint'])),
            'camera':
                self.wrapping_action_space.spaces['camera'],
            'attack_place_equip_craft_nearbyCraft_nearbySmelt':
                gym.spaces.Discrete(len(self._maps['attack_place_equip_craft_nearbyCraft_nearbySmelt']))
        })

    def action(self, action):
        if not self.action_space.contains(action):
            raise ValueError('action {} is invalid for {}'.format(action, self.action_space))

        original_space_action = OrderedDict()
        for k, v in action.items():
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
                op[key] = np.array([0, -10], dtype=np.float32)
                self._actions.append(op)
                op = copy.deepcopy(self.noop)
                op[key] = np.array([0, 10], dtype=np.float32)
                self._actions.append(op)
                op = copy.deepcopy(self.noop)
                op[key] = np.array([-10, 0], dtype=np.float32)
                self._actions.append(op)
                op = copy.deepcopy(self.noop)
                op[key] = np.array([10, 0], dtype=np.float32)
                self._actions.append(op)
                op = copy.deepcopy(self.noop)
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


def action_wrapper(action):
    # discrete actions
    mapping = {
        'forward': 1,
        'back': 2,
        'left': 3,
        'right': 4,
        'jump': 5,
        'sneak': 6,
        'sprint': 7,
        'camera': [8, 9, 10, 11],
        'attack': 12,
        'place': [13, 14, 15, 16, 17, 18],
        'equip': [19, 20, 21, 22, 23, 24, 25],
        'craft': [26, 27, 28, 29],
        'nearbyCraft': [30, 31, 32, 33, 34, 35, 36],
        'nearbySmelt': [37, 38]
    }
    candidate = []
    for k, v in action.items():
        if k in ['forward', 'back', 'left', 'right', 'jump', 'sneak', 'sprint', 'attack'] and v == 1:
            candidate.append(mapping[k])
        elif k in ['place', 'equip', 'craft', 'nearbyCraft', 'nearbySmelt'] and v != 0:
            candidate.append(mapping[k][v - 1])
        elif k == 'camera':
            if v[1] < -3.0:
                candidate.append(mapping[k][0])
            if v[1] > 3.0:
                candidate.append(mapping[k][1])
            if v[0] < -3.0:
                candidate.append(mapping[k][2])
            if v[0] > 3.0:
                candidate.append(mapping[k][3])

    if len(candidate) == 0:
        candidate.append(0)

    action = random.choice(candidate)
    return action


def data_wrapper(data_states, data_actions):
    actions = []
    for i in range(len(data_actions['attack'])):
        a = {}
        for k, v in data_actions.items():
            a[k] = v[i]
        actions.append(action_wrapper(a))
    
    states = []
    for i in range(len(data_states['pov'])):
        s = {
            'equipped_items': {
                'mainhand': {}
            },
            'inventory': {}
        }
        for k, v in data_states['equipped_items']['mainhand'].items():
            s['equipped_items']['mainhand'][k] = v[i]
        for k, v in data_states['inventory'].items():
            s['inventory'][k] = v[i]
        s['pov'] = np.moveaxis(data_states['pov'][i], -1, 0)
        states.append(s)
    
    return states, actions
