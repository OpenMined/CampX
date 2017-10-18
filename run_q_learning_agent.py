"""Run Q-learning on Catch."""

import rl
import random
import logging
import numpy as np

import config
from envs.EnvCatcher import EnvCatcher
from agents.QLearningAgent import QLearningAgent


def main():
  # Set log level.
  logging.basicConfig(level=logging.DEBUG)

  # Set a random seed for consistency in agent AND environment.
  if config.RANDOM_SEED is not None:
      np.random.seed(config.RANDOM_SEED)

  # Make environment.
  env = EnvCatcher(grid_size=config.GRID_SIZE, 
                   env_type='episodic', 
                   verbose=False, 
                   random_seed=config.RANDOM_SEED)

  # Make agent.
  agent = QLearningAgent(actions=list(range(env.action_space)), 
                         learning_rate=config.LEARNING_RATE,
                         discount_factor=config.DISCOUNT_FACTOR, 
                         epsilon=config.EPSILON)

  # Run the RL Loop.
  rl.run_loop(agent, env, config.MAX_NUM_STEPS, config.REPORT_EVERY_N)

if __name__ == '__main__':
  main()