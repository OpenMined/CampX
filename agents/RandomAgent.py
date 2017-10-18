import numpy as np
from collections import defaultdict


class RandomAgent:
    def __init__(self, actions):
        self.actions = actions
        self._stats = {}

    def learn(self, observation, action, reward, next_observation, next_action):
      """No learning for a random agent."""
      return False

    def get_action(self, observation):
      """Return random action from environmental action space."""
      return np.random.choice(self.actions)

    def get_statistics(self):
        return self._stats