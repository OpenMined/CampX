#!/usr/bin/env python

# Based on
# # Practical PyTorch: Playing GridWorld with Reinforcement Learning (Actor-Critic with REINFORCE)

# ## Resources

# ## Requirements

import sys
import numpy as np
import random, math
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import torch.distributions as distrib

from helpers import *

# Configuration

# Discounted reward factor
gamma = 0.95

hidden_size = 32
learning_rate = 0.005
weight_decay = 0.0001

log_every = 200

# import sconce
# job = sconce.Job('rl2', {
#     'gamma': gamma,
#     'learning_rate': learning_rate,
# })
# job.log_every = log_every
# job.plot_every = 500

DROP_MAX = 0.3
DROP_MIN = 0.05
DROP_OVER = 200000

# ## The Grid World, Agent and Environment

# ### The Grid

class Grid():
    '''Define the grid interactions.'''
    MIN_PLANT_VALUE = -1
    MAX_PLANT_VALUE = 0.5
    GOAL_VALUE = 10
    EDGE_VALUE = -10
    VISIBLE_RADIUS = 1
    
    def __init__(self, grid_size=8, n_plants=15):
        self.grid_size = grid_size
        self.n_plants = n_plants

    def reset(self):
        padded_size = self.grid_size + 2 * VISIBLE_RADIUS
        self.grid = np.zeros((padded_size, padded_size)) # Padding for edges

        # Edges
        self.grid[0:VISIBLE_RADIUS, :] = EDGE_VALUE
        self.grid[-1*VISIBLE_RADIUS:, :] = EDGE_VALUE
        self.grid[:, 0:VISIBLE_RADIUS] = EDGE_VALUE
        self.grid[:, -1*VISIBLE_RADIUS:] = EDGE_VALUE

        # Randomly placed plants
        for i in range(self.n_plants):
            plant_value = random.random() * (MAX_PLANT_VALUE - MIN_PLANT_VALUE) + MIN_PLANT_VALUE
            ry = random.randint(0, self.grid_size-1) + VISIBLE_RADIUS
            rx = random.randint(0, self.grid_size-1) + VISIBLE_RADIUS
            self.grid[ry, rx] = plant_value

        # Goal in one of the corners
        S = VISIBLE_RADIUS
        E = self.grid_size + VISIBLE_RADIUS - 1
        gps = [(E, E), (S, E), (E, S), (S, S)]
        gp = gps[random.randint(0, len(gps)-1)]
        self.grid[gp] = GOAL_VALUE

    def visible(self, pos):
        y, x = pos
        return self.grid[y-VISIBLE_RADIUS:y+VISIBLE_RADIUS+1, x-VISIBLE_RADIUS:x+VISIBLE_RADIUS+1]

class Agent:
    '''How the agent interacts.'''
    def reset(self):
        self.health = 1 # START_HEALTH

    def act(self, action):
        # Move according to action: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        y, x = self.pos
        if action == 0: y -= 1
        elif action == 1: x += 1
        elif action == 2: y += 1
        elif action == 3: x -= 1
        self.pos = (y, x)
        self.health += -0.02 # STEP_VALUE # Gradually getting hungrier

class Environment:
    '''Definition of the gridworld environment.'''
    def __init__(self):
        self.grid = Grid()
        self.agent = Agent()

    def reset(self):
        """Start a new episode by resetting grid and agent"""
        self.grid.reset()
        self.agent.reset()
        c = int(self.grid.grid_size / 2)
        self.agent.pos = (c, c)

        self.t = 0
        self.history = []
        self.record_step()

        return self.visible_state

    def record_step(self):
        """Add the current state to history for display later"""
        grid = np.array(self.grid.grid)
        grid[self.agent.pos] = self.agent.health * 0.5 # Agent marker faded by health
        visible = np.array(self.grid.visible(self.agent.pos))
        self.history.append((grid, visible, self.agent.health))

    @property
    def visible_state(self):
        """Return the visible area surrounding the agent, and current agent health"""
        visible = self.grid.visible(self.agent.pos)
        y, x = self.agent.pos
        yp = (y - VISIBLE_RADIUS) / self.grid.grid_size
        xp = (x - VISIBLE_RADIUS) / self.grid.grid_size
        extras = [self.agent.health, yp, xp]
        return np.concatenate((visible.flatten(), extras), 0)

    def step(self, action):
        """Update state (grid and agent) based on an action"""
        self.agent.act(action)

        # Get reward from where agent landed, add to agent health
        value = self.grid.grid[self.agent.pos]
        self.grid.grid[self.agent.pos] = 0
        self.agent.health += value

        # Check if agent won (reached the goal) or lost (health reached 0)
        won = value == GOAL_VALUE
        lost = self.agent.health <= 0
        done = won or lost

        # Rewards at end of episode
        if won:
            reward = 1
        elif lost:
            reward = -1
        else:
            reward = 0 # Reward will only come at the end
            # reward = value # Try this for quicker learning

        # Save in history
        self.record_step()
        return self.visible_state, reward, done

# ## Actor-Critic network

class Policy(nn.Module):
    def __init__(self, hidden_size):
        super(Policy, self).__init__()

        visible_squares = (VISIBLE_RADIUS * 2 + 1) ** 2
        input_size = visible_squares + 1 + 2 # Plus agent health, y, x

        self.inp = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, 4 + 1, bias=False) # For both action and expected value

    def forward(self, x):
        x = x.view(1, -1)
        x = F.tanh(x) # Squash inputs
        x = F.relu(self.inp(x))
        x = self.out(x)

        # Split last five outputs into probs and value
        probs = x[:,:4]
        value = x[:,4]
        return probs, value

# ## Selecting actions

def select_action(e, state):
    drop = interpolate(e, DROP_MAX, DROP_MIN, DROP_OVER)
    # print('drop', drop)
    state = Variable(torch.from_numpy(state).float())
    # print('state', state)
    probs, value = policy(state)          # Forward state through network
    # print('probs', probs, 'value', value)
    probs = F.dropout(probs, drop, True) # Dropout for exploration
    # print('probs drop', probs)
    # Must specify the dimension of the softmax
    probs = F.softmax(probs, dim=1)
    # print('probs softmax', probs)

    # replace this
    # action = probs.multinomial()          # Sample an action
    # print(action)
    # NOTE: categorical is equivalent to what used to be called multinomial
    m = distrib.Categorical(probs)
    # print('m', m)
    action = m.sample()
    # print('action.data', action.data[0])
    return action, value, m

# ## Playing through an episode

def run_episode(e):
    state = env.reset()
    actions = []
    values = []
    rewards = []
    ms = []
    done = False

    while not done:
        action, value, m = select_action(e, state)
        next_state, reward, done = env.step(action.data[0])
        actions.append(action)
        values.append(value)
        rewards.append(reward)
        ms.append(m)

    return actions, values, rewards, ms

# ## Using REINFORCE with a value baseline

mse = nn.MSELoss()

def finish_episode(e, actions, values, rewards, ms):

    # Calculate discounted rewards, going backwards from end
    discounted_rewards = []
    R = 0
    for r in rewards[::-1]:
        R = r + gamma * R
        discounted_rewards.insert(0, R)
    discounted_rewards = torch.Tensor(discounted_rewards)

    # Use REINFORCE on chosen actions and associated discounted rewards
    value_loss = 0
    for action, value, reward, m in zip(actions, values,
                                        discounted_rewards, ms):
        # Treat critic value as baseline
        # Try to perform better than baseline
        reward_diff = reward - value.data[0]
        # REPLACE THIS LINE WITH THE CORRECT REINFORCE
        # action.reinforce(reward_diff)

        # Reinforce the selected action
        loss = -m.log_prob(action) * reward_diff
        loss.backward(retain_graph=True)

        # Compare with actual reward
        value_loss += mse(value, Variable(torch.Tensor([reward])))

    # Backpropagate
    optimizer.zero_grad()
    # value_loss.backward()

    ## REPLACE THE FOLLOWING
    nodes = [value_loss] + actions
    # No gradients for reinforced values
    gradients = [torch.ones(1)] + [None for _ in actions]
    # print('nodes and gradients', nodes + gradients)
    print('gradients', gradients, type(gradients))
    print('nodes', nodes, type(nodes))
    autograd.backward(nodes, gradients)
    optimizer.step()

    return discounted_rewards, value_loss

env = Environment()
policy = Policy(hidden_size=hidden_size)
optimizer = optim.Adam(policy.parameters(), lr=learning_rate, weight_decay=weight_decay)

reward_avg = SlidingAverage('reward avg', steps=log_every)
value_loss_avg = SlidingAverage('value loss avg', steps=log_every)

e = 0

while reward_avg < 1.0:
    actions, values, rewards, ms = run_episode(e)
    final_reward = rewards[-1]

    discounted_rewards, value_loss = finish_episode(e, actions, values, rewards, ms)

    # job.record(e, final_reward) # REMOVE
    reward_avg.add(final_reward)
    value_loss_avg.add(value_loss.data[0])

    if e % log_every == 0:
        print('[epoch=%d]' % e, reward_avg, value_loss_avg)

    e += 1

