"""Unit tests for CampX."""

import config
import logging
import unittest
import numpy as np

# Import Reinforcement Learning Utilities
import rl

# Environments
from envs.EnvCatcher import EnvCatcher

# Agents
from agents.RandomAgent import RandomAgent
from agents.SarsaAgent import SarsaAgent
from agents.QLearningAgent import QLearningAgent

# Define a number of test episodes.
n_eps = 200


class RandomAgentTest(unittest.TestCase):
  
  def setUp(self):
    # Set a random seed for consistency in agent AND environment.
    if config.RANDOM_SEED is not None:
        np.random.seed(config.RANDOM_SEED)

    # Make environment.
    self.env = EnvCatcher(grid_size=config.GRID_SIZE, 
                     env_type='episodic', 
                     verbose=False, 
                     random_seed=config.RANDOM_SEED)

    # Make agent.
    self.agent = RandomAgent(actions=list(range(self.env.action_space)))

  def runTest(self):
    logging.info('RandomAgentTest')
    ep_returns = rl.run_loop(self.agent, self.env, (config.GRID_SIZE-1)*n_eps) 
    self.assertEqual(len(ep_returns), n_eps)


class QLearningAgentTest(unittest.TestCase):
  def setUp(self):
    # Set a random seed for consistency in agent AND environment.
    if config.RANDOM_SEED is not None:
        np.random.seed(config.RANDOM_SEED)

    # Make environment.
    self.env = EnvCatcher(grid_size=config.GRID_SIZE, 
                     env_type='episodic', 
                     verbose=False, 
                     random_seed=config.RANDOM_SEED)

    # Make agent.
    self.agent = QLearningAgent(actions=list(range(self.env.action_space)), 
                                learning_rate=config.LEARNING_RATE,
                                discount_factor=config.DISCOUNT_FACTOR, 
                                epsilon=config.EPSILON)

  def runTest(self):
    logging.info('QLearningAgentTest')
    ep_returns = rl.run_loop(self.agent, self.env, (config.GRID_SIZE-1)*n_eps) 
    self.assertEqual(len(ep_returns), n_eps)
    # Test for learning.
    self.assertEqual(0.2,np.average(ep_returns[-config.REPORT_EVERY_N:]))

class SarsaAgentTest(unittest.TestCase):
  
  def setUp(self):
    # Set a random seed for consistency in agent AND environment.
    if config.RANDOM_SEED is not None:
        np.random.seed(config.RANDOM_SEED)

    # Make environment.
    self.env = EnvCatcher(grid_size=config.GRID_SIZE, 
                     env_type='episodic', 
                     verbose=False, 
                     random_seed=config.RANDOM_SEED)

    # Make agent.
    self.agent = SarsaAgent(actions=list(range(self.env.action_space)), 
                            learning_rate=config.LEARNING_RATE,
                            discount_factor=config.DISCOUNT_FACTOR, 
                            epsilon=config.EPSILON)

  def runTest(self):
    logging.info('SarsaAgentTest')
    ep_returns = rl.run_loop(self.agent, self.env, (config.GRID_SIZE-1)*n_eps) 
    self.assertEqual(len(ep_returns), n_eps)
    # Test for learning.
    self.assertEqual(0.2,np.average(ep_returns[-config.REPORT_EVERY_N:]))

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()