#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import logging
import argparse
from collections import deque

import torch
import numpy as np

from .agent import Agent, DDPGAgent
from .util import load_environment, UnityEnvironmentWrapper, get_state
from .train import reset_deque


def evalddpg(env, checkpointfn='checkpoint-%d.pth'):
    'Function that evalutes DDPG Agents.'
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]

    action_size = brain.vector_action_space_size
    n_agents = len(env_info.agents)
    states = get_state(env_info, n_agents=n_agents)
    state_size = len(states) if n_agents == 1 else len(states[0])

    agents = [DDPGAgent.load(checkpointfn % i) for i in range(n_agents)]
    env_info = env.reset(train_mode=False)[brain_name]
    states = get_state(env_info, n_agents=n_agents)

    scores = np.zeros(n_agents)

    input('Press Enter to continue')

    while True:
        actions = np.vstack([agent.act(state) for agent, state in zip(agents, states)])

        env_info = env.step(actions)[brain_name]
        next_states = get_state(env_info, n_agents=n_agents)
        rewards = env_info.rewards
        dones = env_info.local_done

        for i in range(n_agents):
            agents[i].step(states[i], actions[i], rewards[i], next_states[i], dones[i], learning=False)

        states = next_states
        scores += rewards

        if any(dones):
            break

    logging.info(
        'Final scores: %s', scores
    )

    return scores


def main():
    parser = argparse.ArgumentParser(description='Evaluates a learned agent')
    parser.add_argument('--checkpoint', dest='checkpoint', action='store',
                        default='checkpoint-%d.pth')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    env = load_environment()
    evalddpg(env, checkpointfn=args.checkpoint)

if __name__ == '__main__':
    main()
