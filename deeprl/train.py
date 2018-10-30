#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import logging
import argparse
from collections import deque

import torch
import numpy as np

from .agent import DDPGAgent as Agent, BUFFER_SIZE, BATCH_SIZE
from .util import load_environment, UnityEnvironmentWrapper, get_state
from .util import FRAME_SKIP

BATCH = 0
CHANNELS = 1
DEPTH = 2
HEIGHT = 3
WIDTH = 4
STACK_SIZE = 1


def reset_deque(state, stack_size=STACK_SIZE):
    state_deque = deque(maxlen=stack_size)

    for _ in range(stack_size):
        state_deque.append(np.zeros(
            state.shape
        ))

    return state_deque


def ddpg(env, n_episodes=2000, max_t=1000, checkpointfn='checkpoint.pth', load_checkpoint=False,
         update_every=2, n_updates=1, solution_threshold=30.0):
    '''Runs DDPG in an environment'''

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]

    action_size = brain.vector_action_space_size
    n_agents = len(env_info.agents)
    states = get_state(env_info, n_agents=n_agents)
    state_size = len(states) if n_agents == 1 else len(states[0])

    if load_checkpoint:
        try:
            agents = [Agent.load(checkpointfn) for _ in range(n_agents)]
        except Exception:
            logging.exception('Failed to load checkpoint. Ignoring...')
            agents = [Agent(state_size, action_size, 0) for _ in range(n_agents)]
    else:
        agents = [Agent(state_size, action_size, 0) for _ in range(n_agents)]

    for i_episode in range(1, n_episodes + 1):
        for agent in agents:
            agent.episode += 1

        env_info = env.reset(train_mode=True)[brain_name]
        states = get_state(env_info, n_agents=n_agents)
        scores = np.zeros(n_agents)

        for t in range(max_t):
            actions = np.vstack([agent.act(state) for agent, state in zip(agents, states)])

            env_info = env.step(actions)[brain_name]
            next_states = get_state(env_info, n_agents=n_agents)
            rewards = env_info.rewards
            dones = env_info.local_done

            for i in range(n_agents):
                agents[i].step(states[i], actions[i], rewards[i], next_states[i], dones[i], learning=False)

            if len(agent.memory) > BATCH_SIZE and t and t % update_every == 0:
                for i in range(n_agents):
                    for j in range(n_updates):
                        experiences = agents[i].memory.sample()
                        agents[i].learn(experiences)

            states = next_states
            scores += rewards

            if any(dones):
                break

        # Store scores for all agents
        [agent.scores.append(score) for agent, score in zip(agents, scores)]

        avg_score = np.mean([agent.scores[-100:] for agent in agents])

        logging.debug(
            'Episode {}\tAverage Score: {:.3f}\tCurrent (avg) Score: {:.3f}'
                .format(i_episode, avg_score, np.mean(scores))
        )

        if i_episode % 100 == 0:
            logging.info(
                'Episode {}\tAverage Score: {:.3f}'
                    .format(i_episode, avg_score)
            )
            logging.info(
                'Saving checkpoint file...'
            )
            [agent.save(checkpointfn % i) for i, agent in enumerate(agents)]

        if np.mean(avg_score) >= solution_threshold:
            logging.info(
                'Environment solved in {:d} episodes!'
                    .format(i_episode - 99)
            )
            logging.info(
                'Saving checkpoint file at %s', checkpointfn
            )
            [agent.save(checkpointfn % i) for i, agent in enumerate(agents)]
            if i_episode > 100:
                break

    return agents


def dqn(env, n_episodes=1001, max_t=1200 * FRAME_SKIP, eps_start=1.0,
        eps_end=0.001, eps_decay=0.995, solution_threshold=13.0,
        checkpointfn='checkpoint.pth', load_checkpoint=False,
        reload_every=None):
    """Function that uses Deep Q Networks to learn environments.

    Parameters
    ----------
        n_episodes: int
            maximum number of training episodes
        max_t: int
            maximum number of timesteps per episode
        eps_start: float
            starting value of epsilon, for epsilon-greedy action selection
        eps_end: float
            minimum value of epsilon
        eps_decay: float
            multiplicative factor (per episode) for decreasing epsilon
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
    else:
        use_visual = False
        initial_state = get_state(env_info, use_visual)
        state_size = initial_state.shape[0]

    if load_checkpoint:
        try:
            agent = Agent.load(checkpointfn, use_visual)
        except Exception:
            logging.exception('Failed to load checkpoint. Ignoring...')
            agent = Agent(state_size, action_size, 0, use_visual)
    else:
        agent = Agent(state_size, action_size, 0, use_visual)

    if agent.episode:
        eps = (eps_start * eps_decay) ** agent.episode
    else:
        eps = eps_start

    for i_episode in range(agent.episode, n_episodes):
        state_deque = reset_deque(initial_state)

        env_info = env.reset(train_mode=True)[brain_name]
        state = get_state(env_info, use_visual)
        state_deque.append(state)

        score = 0
        for t in range(max_t):
            if use_visual:
                state = np.stack(state_deque, axis=-1) \
                        .squeeze(axis=0).transpose(0, -1, 1, 2) \
                        .squeeze(axis=0)
            else:
                state = np.array(state_deque).squeeze()

            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]

            next_state = get_state(env_info, use_visual)
            state_deque.append(next_state)
            next_state = np.stack(state_deque, axis=-1) \
                    .squeeze(axis=0).transpose(0, -1, 1, 2) \
                    .squeeze(axis=0)

            reward = env_info.rewards[0]
            done = env_info.local_done[0]

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
        agent.scores.append(score)
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        agent.episode += 1

        logging.debug(
            'Episode {}\tAverage Score: {:.3f}\tCurrent Score: {:.3f}\tEpsilon: {:.4f}'
            .format(i_episode, np.mean(agent.scores[-100:]), score, eps)
        )
        if (i_episode + 1) % 100 == 0:
            logging.info(
                'Episode {}\tAverage Score: {:.3f}'
                .format(i_episode, np.mean(agent.scores[-100:]))
            )
            logging.info(
                'Saving checkpoint file...'
            )
            agent.save(checkpointfn)
        if np.mean(agent.scores[-100:]) >= solution_threshold:
            logging.info(
                'Environment solved in {:d} episodes!'
                .format(i_episode - 99)
            )
            logging.info(
                'Saving checkpoint file at %s', checkpointfn
            )
            agent.save(checkpointfn)
            break
        if reload_every and i_episode and (i_episode + 1) % reload_every == 0:
            env.close()
            reload_process()

    return agent


def reload_process():
    if '--load-checkpoint' not in sys.argv:
        sys.argv.append('--load-checkpoint')
    os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    os.execv('/proc/self/exe', 'python -m deeprl.train'.split() + sys.argv[1:])


def main():
    parser = argparse.ArgumentParser(description='Trains a learning agent')
    parser.add_argument('--checkpoint', dest='checkpoint', action='store',
                        default='checkpoint-%d.pth')
    parser.add_argument('--load-checkpoint', dest='load_chkpt', action='store_true',
                        default=False)
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.DEBUG)

    env = UnityEnvironmentWrapper(load_environment())
    ddpg(env, n_episodes=3000,
        checkpointfn=args.checkpoint, load_checkpoint=args.load_chkpt)

if __name__ == '__main__':
    main()
