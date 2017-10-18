import numpy as np
from collections import defaultdict


class QLearningAgent:
    def __init__(self, actions, learning_rate, discount_factor, epsilon):
        self.actions = actions
        self.learning_rate = learning_rate
        # Discount factor is gamma 
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = defaultdict(lambda: list(np.zeros(len(actions))))
        # Various statistics.
        self._stats = {}

    def convert_observation_to_q_key(self, observation):
      """Convert state to observation for agent."""
      return str(list(observation[:])) 

    # Update q_table with sample.
    def learn(self, observation, action, reward, next_observation, next_action):
      observation = self.convert_observation_to_q_key(observation)
      next_observation = self.convert_observation_to_q_key(next_observation)
      
      # Get the current and new q_values.
      q_value = self.q_table[observation][action]
      next_q_value = self.q_table[next_observation][next_action]
      
      # Calculate the temporal difference error.
      td_error = reward + self.gamma * next_q_value - q_value
      
      # Update q_table.
      self.q_table[observation][action] += self.learning_rate * td_error

      # Update statistics for reporting.
      self._stats['td-error'] = td_error

    # Get action for the state according to the q function table
    # agent pick action of epsilon-greedy policy.
    def get_action(self, observation):
      observation = self.convert_observation_to_q_key(observation)
      # Epsilon greedy policy.
      if np.random.rand() < self.epsilon:
        # Take random action.
        action = np.random.choice(self.actions)
      else:
        # Take greedy action.
        action = np.argmax(self.q_table[observation])
      return action

    def get_statistics(self):
        return self._stats