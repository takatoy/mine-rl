# Simple env test.
import json
import select
import time
import logging
import os

import aicrowd_helper
import gym
import minerl

import coloredlogs
coloredlogs.install(logging.DEBUG)

from src.agent import Agent
from src.env_wrappers import CombineActionWrapper, SerialDiscreteCombineActionWrapper, FrameSkip, MoveAxisWrapper, ObsWrapper

# All the evaluations will be evaluated on MineRLObtainDiamond-v0 environment
MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLObtainDiamond-v0')
MINERL_MAX_EVALUATION_EPISODES = int(os.getenv('MINERL_MAX_EVALUATION_EPISODES', 5))

FRAME_SKIP = 4

def main():
    """
    This function will be called for training phase.
    """
    # Sample code for illustration, add your code below to run in test phase.
    # Load trained model from train/ directory
    env = gym.make(MINERL_GYM_ENV)
    if FRAME_SKIP > 0:
        env = FrameSkip(env)
    env = ObsWrapper(env)
    env = MoveAxisWrapper(env, -1, 0)
    env = CombineActionWrapper(env)

    agent = Agent(env.observation_space, env.action_space)
    agent.load_model()

    for _ in range(MINERL_MAX_EVALUATION_EPISODES):
        obs = env.reset()
        done = False
        netr = 0
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            netr += reward
            env.render()

    env.close()

if __name__ == "__main__":
    main()
