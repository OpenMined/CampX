"""Unit tests for CampX."""

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

# Define the test configuration.
N_EPS = 200
MAX_NUM_STEPS = 1800
GRID_SIZE = 10
REPORT_EVERY_N = 100
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.98
EPSILON = 0.1
RANDOM_SEED = 42


class RandomAgentTest(unittest.TestCase):

    def setUp(self):
        # Set a random seed for consistency in agent AND environment.
        if RANDOM_SEED is not None:
            np.random.seed(RANDOM_SEED)

        # Make environment.
        self.env = EnvCatcher(grid_size=GRID_SIZE,
                              env_type='episodic',
                              verbose=False,
                              random_seed=RANDOM_SEED)

        # Make agent.
        self.agent = RandomAgent(actions=list(range(self.env.action_space)))

    def runTest(self):
        logging.info('RandomAgentTest')
        ep_returns = rl.run_loop(self.agent, self.env, (GRID_SIZE - 1) * N_EPS)
        self.assertEqual(len(ep_returns), N_EPS)


class QLearningAgentTest(unittest.TestCase):
    def setUp(self):
        # Set a random seed for consistency in agent AND environment.
        if RANDOM_SEED is not None:
            np.random.seed(RANDOM_SEED)

            # Make environment.
        self.env = EnvCatcher(grid_size=GRID_SIZE,
                              env_type='episodic',
                              verbose=False,
                              random_seed=RANDOM_SEED)

        # Make agent.
        self.agent = QLearningAgent(actions=list(range(self.env.action_space)),
                                    learning_rate=LEARNING_RATE,
                                    discount_factor=DISCOUNT_FACTOR,
                                    epsilon=EPSILON)

    def runTest(self):
        logging.info('QLearningAgentTest')
        ep_returns = rl.run_loop(self.agent, self.env, (GRID_SIZE - 1) * N_EPS)
        self.assertEqual(len(ep_returns), N_EPS)
        # Test for learning.
        self.assertEqual(0.2, np.average(ep_returns[-REPORT_EVERY_N:]))


class SarsaAgentTest(unittest.TestCase):
    def setUp(self):
        # Set a random seed for consistency in agent AND environment.
        if RANDOM_SEED is not None:
            np.random.seed(RANDOM_SEED)

        # Make environment.
        self.env = EnvCatcher(grid_size=GRID_SIZE,
                              env_type='episodic',
                              verbose=False,
                              random_seed=RANDOM_SEED)

        # Make agent.
        self.agent = SarsaAgent(actions=list(range(self.env.action_space)),
                                learning_rate=LEARNING_RATE,
                                discount_factor=DISCOUNT_FACTOR,
                                epsilon=EPSILON)

    def runTest(self):
        logging.info('SarsaAgentTest')
        ep_returns = rl.run_loop(self.agent, self.env, (GRID_SIZE - 1) * N_EPS)
        self.assertEqual(len(ep_returns), N_EPS)
        # Test for learning.
        self.assertEqual(0.2, np.average(ep_returns[-REPORT_EVERY_N:]))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()
