# Simple env test.
import json
import select
import time
import logging
import os

import aicrowd_helper
import gym
import minerl
from utility.parser import Parser

import coloredlogs
coloredlogs.install(logging.DEBUG)

from collections import OrderedDict
from src.agent import Agent
from src.env_wrappers import CombineActionWrapper, SerialDiscreteCombineActionWrapper, FrameSkip, MoveAxisWrapper, ObsWrapper, data_action_wrapper, data_state_wrapper
from torch.utils.tensorboard import SummaryWriter

# All the evaluations will be evaluated on MineRLObtainDiamond-v0 environment
MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLObtainDiamond-v0')
# You need to ensure that your submission is trained in under MINERL_TRAINING_MAX_STEPS steps
MINERL_TRAINING_MAX_STEPS = int(os.getenv('MINERL_TRAINING_MAX_STEPS', 8000000))
# You need to ensure that your submission is trained by launching less than MINERL_TRAINING_MAX_INSTANCES instances
MINERL_TRAINING_MAX_INSTANCES = int(os.getenv('MINERL_TRAINING_MAX_INSTANCES', 5))
# You need to ensure that your submission is trained within allowed training time.
# Round 1: Training timeout is 15 minutes
# Round 2: Training timeout is 4 days
MINERL_TRAINING_TIMEOUT = int(os.getenv('MINERL_TRAINING_TIMEOUT_MINUTES', 4*24*60))
# The dataset is available in data/ directory from repository root.
MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', 'data/')

# Optional: You can view best effort status of your instances with the help of parser.py
# This will give you current state like number of steps completed, instances launched and so on. Make your you keep a tap on the numbers to avoid breaching any limits.
parser = Parser('performance/',
                allowed_environment=MINERL_GYM_ENV,
                maximum_instances=MINERL_TRAINING_MAX_INSTANCES,
                maximum_steps=MINERL_TRAINING_MAX_STEPS,
                raise_on_error=False,
                no_entry_poll_timeout=600,
                submission_timeout=MINERL_TRAINING_TIMEOUT*60,
                initial_poll_timeout=600)

TRAIN_PER = 128
TRAIN_DISCRIM_EPOCH = 3

def main():
    writer = SummaryWriter()

    data = minerl.data.make(MINERL_GYM_ENV, data_dir=MINERL_DATA_ROOT)

    env = gym.make('MineRLObtainDiamondDense-v0')
    env = ObsWrapper(env)
    env = MoveAxisWrapper(env, -1, 0)
    env = CombineActionWrapper(env)
    env = SerialDiscreteCombineActionWrapper(env)

    agent = Agent(env.observation_space, env.action_space)

    for s, a, r, ns, d in data.sarsd_iter(num_epochs=20, max_sequence_len=128):
        s = data_state_wrapper(s)
        ns = data_state_wrapper(ns)
        a = data_action_wrapper(a)
        for state, action, reward, n_state, done in zip(s, a, r, ns, d):
            agent.act(state, action)
            agent.add_data(state, action, reward, n_state, done)
        agent.train()

    data_provider = data.sarsd_iter(num_epochs=-1, max_sequence_len=128)

    for i in range(256):
        s, a, _, _, _ = data_provider.__next__()
        s = data_state_wrapper(s)
        a = data_action_wrapper(a)
        agent.train_discriminator(s, a)

    net_steps = 0
    n_episode = 0
    while True:
        obs = env.reset()
        done = False
        netr = 0
        nobs = None
        step = 0
        while not done:
            action = agent.act(obs)
            nobs, reward, done, info = env.step(action)
            netr += reward
            reward += agent.bonus_reward(obs, action, nobs)
            agent.add_data(obs, action, reward, nobs, done)
            obs = nobs

            # To get better view in your training phase, it is suggested
            # to register progress continuously, example when 54% completed
            # aicrowd_helper.register_progress(0.54)

            # To fetch latest information from instance manager, you can run below when you want to know the state
            #>> parser.update_information()
            #>> print(parser.payload)
            # .payload: provide AIcrowd generated json
            # Example: {'state': 'RUNNING', 'score': {'score': 0.0, 'score_secondary': 0.0}, 'instances': {'1': {'totalNumberSteps': 2001, 'totalNumberEpisodes': 0, 'currentEnvironment': 'MineRLObtainDiamond-v0', 'state': 'IN_PROGRESS', 'episodes': [{'numTicks': 2001, 'environment': 'MineRLObtainDiamond-v0', 'rewards': 0.0, 'state': 'IN_PROGRESS'}], 'score': {'score': 0.0, 'score_secondary': 0.0}}}}
            # .current_state: provide indepth state information avaiable as dictionary (key: instance id)

            step += 1
            net_steps += 1

            if step % TRAIN_PER == 0:
                discrim_loss = 0.0
                state_discrim_loss = 0.0
                for i in range(TRAIN_DISCRIM_EPOCH):
                    s, a, _, _, _ = data_provider.__next__()
                    s, a = data_wrapper(s, a)
                    discrim_loss += agent.train_discriminator(s, a)
                    state_discrim_loss += agent.train_state_discriminator(s)
                writer.add_scalar('Loss/Discriminator', discrim_loss / TRAIN_DISCRIM_EPOCH, net_steps)
                writer.add_scalar('Loss/StateDiscriminator', state_discrim_loss / TRAIN_DISCRIM_EPOCH, net_steps)

                policy_loss = agent.train()
                writer.add_scalar('Loss/Policy', policy_loss, net_steps)
                agent.save_model()

            if net_steps >= MINERL_TRAINING_MAX_STEPS:
                break

        policy_loss = agent.train()
        agent.save_model()

        writer.add_scalar('Reward', netr, n_episode)
        n_episode += 1

        if net_steps >= MINERL_TRAINING_MAX_STEPS:
            break

    agent.save_model()

    aicrowd_helper.register_progress(1)
    env.close()


if __name__ == "__main__":
    main()
