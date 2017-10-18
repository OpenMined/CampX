"""Run Q-learning on Catch."""

import rl
import random
import logging
import numpy as np

from EnvCatcher import EnvCatcher
from agents.QLearningAgent import QLearningAgent


# Set experimental parameters.
MAX_NUM_STEPS = 100000
GRID_SIZE = 24
REPORT_EVERY_N = 100

def main():
  # Set log level.
  logging.basicConfig(level=logging.DEBUG)

  # Set a random seed for consistency in agent AND environment.
  random_seed = None
  if random_seed is not None:
      np.random.seed(random_seed)

  # Make environment.
  env = EnvCatcher(grid_size=GRID_SIZE, 
                   env_type='episodic', 
                   verbose=False, 
                   random_seed=random_seed)

  # Make agent.
  agent = QLearningAgent(actions=list(range(env.action_space)), 
                         learning_rate=0.05,
                         discount_factor=0.9, 
                         epsilon=0.1)

  rl.run_loop(agent, env, MAX_NUM_STEPS, REPORT_EVERY_N)

if __name__ == '__main__':
  main()