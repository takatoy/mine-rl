# Simple env test.
import json
import select
import time
import logging
import os
import pickle

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

########## params ##########
TRAIN_INTERVAL = 0  # 0 for one episode
TRAIN_FROM_EXPERT_EPOCH = 1000
FRAME_SKIP = 4
DATA_BATCH_SIZE = 256
############################

def train_from_expert(agent, data_source):
    for _ in range(TRAIN_FROM_EXPERT_EPOCH):
        s, a, r, ns, d = data_source.__next__()
        s = data_state_wrapper(s)
        ns = data_state_wrapper(ns)
        a = data_action_wrapper(a)
        for state, action, reward, n_state, done in zip(s, a, r, ns, d):
            agent.act(state, action)
            agent.add_data(state, action, reward, n_state, done)
        agent.train_discriminator(s, a)
        agent.train_policy()
        agent.save_model()

def main():
    writer = SummaryWriter()

    env = gym.make('MineRLObtainDiamondDense-v0')
    if FRAME_SKIP > 0:
        env = FrameSkip(env, FRAME_SKIP)
    env = ObsWrapper(env)
    env = MoveAxisWrapper(env, -1, 0)
    env = CombineActionWrapper(env)

    agent = Agent(env.observation_space, env.action_space)
    data = minerl.data.make('MineRLTreechop-v0', data_dir=MINERL_DATA_ROOT)
    data_source = data.sarsd_iter(num_epochs=-1, max_sequence_len=DATA_BATCH_SIZE)

    # data_2 = minerl.data.make('MineRLObtainDiamond-v0', data_dir=MINERL_DATA_ROOT)
    # data_2_source = data.sarsd_iter(num_epochs=-1, max_sequence_len=128)

    # behavioral cloning
    train_from_expert(agent, data_source)

    net_steps = 0
    n_episode = 0
    while True:
        obs = env.reset()
        done = False
        netr = 0
        net_bonus_r = 0
        nobs = None
        step = 0
        while not done:
            action = agent.act(obs)
            nobs, reward, done, info = env.step(action)
            netr += reward
            reward += agent.bonus_reward(obs, action, nobs)
            net_bonus_r += reward
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

            if (TRAIN_INTERVAL != 0 and step % TRAIN_INTERVAL == 0) or done:
                total_discrim_loss = 0.0
                total_value = total_ppo_loss = total_value_loss = total_entropy = 0
                n_epoch = 0
                while not agent.is_memory_empty():
                    s, a, _, _, _ = data_source.__next__()
                    s = data_state_wrapper(s)
                    a = data_action_wrapper(a)
                    total_discrim_loss += agent.train_discriminator(s, a)
                    value, ppo_loss, value_loss, entropy = agent.train_policy()

                    total_value += value
                    total_ppo_loss += ppo_loss
                    total_value_loss += value_loss
                    total_entropy += entropy
                    n_epoch += 1

                writer.add_scalar('Train/Value', value / n_epoch, net_steps)
                writer.add_scalar('Train/PolicyLoss', ppo_loss / n_epoch, net_steps)
                writer.add_scalar('Train/ValueLoss', value_loss / n_epoch, net_steps)
                writer.add_scalar('Train/Entropy', entropy / n_epoch, net_steps)
                writer.add_scalar('Train/DiscriminatorLoss', total_discrim_loss / n_epoch, net_steps)
                agent.save_model()

        writer.add_scalar('Reward/ExternalReward', netr, n_episode)
        writer.add_scalar('Reward/TotalReward', net_bonus_r, n_episode)
        n_episode += 1

        agent.save_model()

    agent.save_model()

    aicrowd_helper.register_progress(1)
    env.close()


if __name__ == "__main__":
    main()
