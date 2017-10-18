""" Reinforcement Learning Run Loop""" 

import logging
import numpy as np


def print_statistics(episode_returns, episode, agent, report_every_n):
  """Print results during learning"""
  for name, value in agent.get_statistics().items():
       logging.debug('Agent ' + name + ': ' + str(value))
  logging.info('Episode #{} finished. Return: {}'.format(episode, 
              episode_returns[-1]))
  logging.info('Average return so far: {}'.format(np.average(episode_returns)))
  if len(episode_returns) > report_every_n:
        b = np.average(episode_returns[-report_every_n:])
        logging.info('Average return over the previous ' 
          + str(report_every_n) + ' episodes: {}'.format(b))


def run_loop(agent, env, max_num_steps, report_every_n):
  """Run an agent for a defined number of steps."""
  t = 0
  episode = 0
  episode_return = 0
  episode_returns = []

  # Reset the environment.
  observation = env.reset()
      
  while t < max_num_steps:
    # Get action from agent policy.
    action = agent.get_action(observation)
    
    # Take action in environment.
    next_observation, reward, done, info = env.step(action)
    
    # Accumulate reward.
    episode_return += reward

    # Provide agent learning information.
    agent.learn(observation, action, reward, next_observation)

    # Update the current observation from the next_observation.
    observation = next_observation
    
    # Environment will provide a done flag, learning should handle it.
    if done:
      episode += 1
      episode_returns.append(episode_return)
      # Reset the environment.
      observation = env.reset()
      print_statistics(episode_returns, episode, agent, report_every_n)
      episode_return = 0
  return episode_returns