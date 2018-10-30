#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import logging
import argparse
from collections import deque

import torch
import numpy as np

from .agent import Agent
from .util import load_environment, UnityEnvironmentWrapper, get_state
from .util import STACK_SIZE, show_agent
from .train import reset_deque


def evaldqn(env, checkpointfn='checkpoint.pth'):
    """Function that uses Deep Q Networks to learn environments.

    Parameters
    ----------
        env: Environment
            execution environment
        checkpointfn: str
            Name of the file to load network parameters
    """
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]

    action_size = brain.vector_action_space_size
    state = env_info.vector_observations[0]
    state_size = len(state)

    if state_size == 0:
        use_visual = True
        initial_state = get_state(env_info, use_visual)
        state_size = list(initial_state.shape)
        state_size.insert(2, STACK_SIZE)
        state_size = tuple(state_size)

    agent = Agent.load(checkpointfn, use_visual=True)

    state_deque = reset_deque(initial_state)
    env_info = env.reset(train_mode=False)[brain_name]
    state = get_state(env_info, use_visual)
    state_deque.append(state)

    score = 0
    first = True
    while True:
        state = np.stack(state_deque, axis=-1) \
                .squeeze(axis=0).transpose(0, -1, 1, 2)

        action = agent.act(state)
        env_info = env.step(action)[brain_name]

        next_state = get_state(env_info, use_visual)
        state_deque.append(next_state)
        next_state = np.stack(state_deque, axis=-1) \
                .squeeze(axis=0).transpose(0, -1, 1, 2)

        reward = env_info.rewards[0]
        done = env_info.local_done[0]

        show_agent(state, next_state, action)
        if first:
            input('Press enter to continue')
            first = False

        agent.step(
            state,
            action,
            reward,
            next_state,
            done,
        )

        score += reward
        if done:
            break
    logging.info(
        'Final score: %g', score
    )
    return score

def main():
    parser = argparse.ArgumentParser(description='Evaluates a learned agent')
    parser.add_argument('--checkpoint', dest='checkpoint', action='store',
                        default='checkpoint.pth')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    env = load_environment()
    evaldqn(env, checkpointfn=args.checkpoint)

if __name__ == '__main__':
    main()
