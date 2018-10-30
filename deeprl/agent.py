#!/usr/bin/env python
# -*- coding: utf-8 -*-


import copy
import random
import logging
import numpy as np
from collections import namedtuple, deque

import torch
import torch.optim as optim
import torch.nn.functional as F

from .model import DDPGActor, DDPGCritic
from .model import AdvantageNetwork as QNetwork
from .model import VisualAdvantageNetwork as VisualQNetwork

BUFFER_SIZE = int(1e6)  # Replay buffer size
BATCH_SIZE = 128        # Minibatch size
GAMMA = 0.99            # Discount factor
TAU = 1e-3              # target parameters soft update
LR = 5e-4               # learning rate
UPDATE_EVERY = 1        # how often to update the network
LR_ACTOR = 1e-3         # learning rate of the actor
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay

DDPG_MEMORY = None      # Un-initialized memory buffer

Experience = namedtuple('Experience',
                        field_names='state action reward next_state done'.split())


def randargmax(a):
    return np.random.choice(np.flatnonzero(a == a.max()))


class DeviceAwareClass(object):
    'Sets appropriate device for computation based on GPU availability.'

    if torch.cuda.is_available():
        logging.info(
            'Using first CUDA device for computation. '
            'Tune visibility by setting the CUDA_VISIBLE_DEVICES env var.'
        )
        device = torch.device('cuda:0')
    else:
        logging.info(
            'Solely using CPU for computation.'
        )
        device = torch.device('cpu')


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


class BaseAgent(DeviceAwareClass):
    def step(self, state, action, reward, next_state, done, learning=True, tau=TAU, gamma=GAMMA):
        # Save experience in replay buffer
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps {{{
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # Only sample experiences if they can fill a batch
            # (makes sure we don't try to learn at the very beginning)
            if learning and len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, gamma, tau)
        # }}}

    def learn(self, experiences=None, gamma=None, tau=None):
        abstract

    def soft_update(self, local_model, target_model, tau):
        '''Soft update model parameters.

        θ_target = τ*θ_local + (1 - τ)*θ_target

        Goes slowly from the local weights to the target weights. Is this
        a contraction? 🤔

        Parameters
        ----------
            local_model: PyTorch model
                Model from which weights will be copied
            target_model: PyTorch model
                Model to which weights will be copied
            tau: float
                step size
        '''

        for target, local in zip(target_model.parameters(), local_model.parameters()):
            target.data.copy_(tau * local.data + (1.0 - tau) * target.data)


class DQNAgent(BaseAgent):
    'Interacts with and learns from the environment'

    def __init__(self, state_size: int, action_size: int, seed: int, use_visual=False, update_every=UPDATE_EVERY):
        '''Initializes an Agent object.

        Parameters
        ----------
            state_size: int
                Dimension of each state
            action_size: int
                Dimension of each action
            seed: int
                Random seed to use
        '''
        self.state_size = state_size
        self.action_size = action_size
        if seed is None:
            seed = 1234
        self.seed = random.seed(seed)
        self.use_visual = use_visual
        self.episode = 0
        self.scores = []
        self.update_every = update_every

        # Q-Network(s) {{{
        if use_visual:
            self.qnetwork_local = VisualQNetwork(state_size, action_size, seed).to(self.device)
            self.qnetwork_target = VisualQNetwork(state_size, action_size, seed).to(self.device)
        else:
            self.qnetwork_local = QNetwork(state_size, action_size, seed).to(self.device)
            self.qnetwork_target = QNetwork(state_size, action_size, seed).to(self.device)
        # }}}

        # Gradient descent optimizer {{{
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        # }}}

        # Replay memory {{{
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # }}}

        # Current time step
        self.t_step = 0

    def act(self, state, epsilon=0.0):
        '''Returns actions for a given state as per current policy.

        Parameters
        ----------
            state: array_like
                current state the agent is in
            epsilon: float
                probability of selecting random actions in epsilon-greedy
        '''

        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        # Policy evaluation {{{
        self.qnetwork_local.eval()
        with torch.no_grad():
            # We don't want to compute gradients here, since we're evaluating
            # our policy. 
            action_values = self.qnetwork_local.forward(state)
        self.qnetwork_local.train()
        # }}}

        # Epsilon-greedy action selection
        if random.random() > epsilon:
            return np.array([randargmax(action_values.cpu().data.numpy())])
        else:
            return np.array([random.choice(np.arange(self.action_size))])

    def learn(self, experiences, gamma=GAMMA, tau=TAU):
        '''Updates value parameters using given batch of experience tuples.

        Parameters
        ----------
            experiences: Tuple[torch.Tensor]
                tuple of (s, a, r, s', done) tuples
            gamma: float
                Discount factor
        '''
        states, actions, rewards, next_states, dones = experiences

        # Estimates the TD target R + γ max_a q(S′, a, w−) {{{
        targets_next = self.qnetwork_target.forward(next_states).detach().max(1)[0].unsqueeze(1)
        # By definition, all future rewards after reaching a terminal states are zero.
        # Hence, we use the `dones` booleans to properly assign value to states.
        targets = rewards.reshape((-1, 1)) + (gamma * targets_next * (1 - dones.reshape((-1, 1))))
        # }}}

        # Now we get what our current policy thinks are the values of the actions
        # we've taken in the past
        estimated = self.qnetwork_local.forward(states).gather(1, actions)

        # We want to minimize the MSE (although this part is super confusing in
        # the DQN paper. Some people say this should be the Huber loss, but that's
        # not what I understood from the code attached to the paper.
        # For more context:
        # https://stackoverflow.com/a/43720168
        loss = F.mse_loss(estimated, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Updating the target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, tau)

    @staticmethod
    def load(path, use_visual=False):
        checkpoint = torch.load(path)
        cls = globals()[checkpoint['class']]
        model = cls(checkpoint['state_size'], checkpoint['action_size'],
                    checkpoint['seed'], use_visual)
        model.qnetwork_local.load_state_dict(checkpoint['state_dict'])
        model.qnetwork_target.load_state_dict(checkpoint['target_state_dict'])
        model.episode = checkpoint['episode']
        model.scores = checkpoint['scores']
        return model

    def save(self, path):
        checkpoint = {
            'class': self.__class__.__name__,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'seed': self.seed,
            'state_dict': self.qnetwork_local.state_dict(),
            'target_state_dict': self.qnetwork_target.state_dict(),
            'episode': self.episode,
            'scores': self.scores,
        }
        torch.save(checkpoint, path)

    def reset(self):
        pass


class ReplayBuffer(DeviceAwareClass):
    'Fixed-size buffer to store experience tuples.'

    def __init__(self, action_size, buffer_size, batch_size, seed):
        '''Initializes a ReplayBuffer object.

        Parameters
        ----------
            action_size: int
                dimention of each action
            buffer_size: int
                maximum size of buffer
            batch_size: int
                size of each training batch
            seed: int
                random seed
        '''

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = Experience

    def __len__(self):
        return len(self.memory)

    def add(self, state, action, reward, next_state, done):
        'Adds a new experience to memory.'

        self.memory.append(
            self.experience(state, action, reward, next_state, done)
        )

    def sample(self):
        'Randomly sample a batch of experiences from memory.'

        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])
        ).float().to(self.device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])
        ).float().to(self.device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])
        ).float().to(self.device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])
        ).float().to(self.device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)
        ).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        'Returns the current size of the buffer.'

        return len(self.memory)



class DoubleDQNAgent(DQNAgent):
    def learn(self, experiences, gamma=GAMMA, tau=TAU):
        '''Updates value parameters using given batch of experience tuples.

        Parameters
        ----------
            experiences: Tuple[torch.Tensor]
                tuple of (s, a, r, s', done) tuples
            gamma: float
                Discount factor
        '''
        states, actions, rewards, next_states, dones = experiences

        # Estimates the TD target R + γ Q(s′, argmax_a' Q(s', a', w) w−) {{{

        # Estimating argmax_a Q(s', a, w)
        argmax_q_next_state = self.qnetwork_local.forward(next_states).detach().argmax(dim=1).unsqueeze(1)

        q_next_state = self.qnetwork_target.forward(next_states).gather(1, argmax_q_next_state)

        # By definition, all future rewards after reaching a terminal states are zero.
        # Hence, we use the `dones` booleans to properly assign value to states.
        targets = rewards.reshape((-1, 1)) + (gamma * q_next_state * (1 - dones.reshape((-1, 1))))
        # }}}

        # Now we get what our current policy thinks are the values of the actions
        # we've taken in the past
        estimated = self.qnetwork_local.forward(states).gather(1, actions)

        # We want to minimize the MSE (although this part is super confusing in
        # the DQN paper. Some people say this should be the Huber loss, but that's
        # not what I understood from the code attached to the paper.
        # For more context:
        # https://stackoverflow.com/a/43720168
        loss = F.mse_loss(estimated, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Updating the target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, tau)


class DDPGAgent(BaseAgent):
    def __init__(self, state_size: int, action_size: int, seed: int, use_visual=None):
        '''Initializes a DDPG Agent.

        :param state_size (int): dimension of each state
        :param action_size (int):  dimension of each action
        :param seed (int): random seed
        '''

        # Current time step
        self.t_step = 0

        self.scores = []
        self.episode = 0
        self.seed = seed
        self.update_every = 1

        self.state_size = state_size
        self.action_size = action_size

        # Actor Network (w/ Target Network)
        self.actor_local = DDPGActor(state_size, action_size, seed).to(self.device)
        self.actor_target = DDPGActor(state_size, action_size, seed).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = DDPGCritic(state_size, action_size, seed).to(self.device)
        self.critic_target = DDPGCritic(state_size, action_size, seed).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, seed)

        global DDPG_MEMORY
        if DDPG_MEMORY is None:
            DDPG_MEMORY = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Replay memory
        self.memory = DDPG_MEMORY

        # Make sure we start with the same set of weights.
        # This happens to be good for this environment
        self.soft_update(self.critic_local, self.critic_target, 1.0)
        self.soft_update(self.actor_local, self.actor_target, 1.0)

    def act(self, state, add_noise=True):
        '''Returns actions for given state as per current policy.'''
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()

        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()

        return np.clip(action, -1, 1)

    def learn(self, experiences, gamma=GAMMA, tau=TAU):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        :param experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
        :param gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, tau)
        self.soft_update(self.actor_local, self.actor_target, tau)

    def reset(self):
        self.noise.reset()

    @staticmethod
    def load(path):
        checkpoint = torch.load(path)
        cls = globals()[checkpoint['class']]
        model = cls(checkpoint['state_size'], checkpoint['action_size'],
                    checkpoint['seed'])
        model.actor_local.load_state_dict(checkpoint['actor_local'])
        model.critic_local.load_state_dict(checkpoint['critic_local'])
        model.actor_target.load_state_dict(checkpoint['actor_target'])
        model.critic_target.load_state_dict(checkpoint['critic_target'])
        model.episode = checkpoint['episode']
        model.scores = checkpoint['scores']
        return model

    def save(self, path):
        checkpoint = {
            'class': self.__class__.__name__,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'seed': self.seed,
            'actor_local': self.actor_local.state_dict(),
            'critic_local': self.critic_local.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'episode': self.episode,
            'scores': self.scores,
        }
        torch.save(checkpoint, path)

Agent = DDPGAgent
