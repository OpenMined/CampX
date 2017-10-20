""" Reinforcement Learning Run Loop"""

import os
import logging
import numpy as np


def save_episode_returns(filename, episode_returns):
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.savetxt(os.path.join(output_dir, filename),
               episode_returns, delimiter=",")


def print_statistics(episode_returns, episode, agent, report_every_n=100):
    """Print results during learning"""
    for name, value in agent.get_statistics().items():
        logging.debug('Agent ' + name + ': ' + str(value))
    logging.info('Episode #{} finished. Return: {}'
                 .format(episode, episode_returns[-1]))
    logging.info('Average return so far: {}'
                 .format(np.average(episode_returns)))
    if len(episode_returns) > report_every_n:
        b = np.average(episode_returns[-report_every_n:])
        logging.info('Average return over the previous ' + str(report_every_n) +
                     ' episodes: {}'.format(b))


def run_loop(agent, env, max_num_steps, report_every_n=100):
    """Run an agent for a defined number of steps."""
    t = 0
    episode = 0
    episode_return = 0
    episode_returns = []

    # Reset the environment.
    observation = env.reset()
    reward = 0

    while t < max_num_steps:
        # Get action from agent policy.
        action = agent.get_action(observation)

        # Take action in environment.
        next_observation, reward, done, info = env.step(action)

        # Accumulate reward.
        episode_return += reward

        # Get the next action.
        next_action = agent.get_action(next_observation)

        # Provide agent learning information.
        agent.learn(observation, action, reward, next_observation, next_action)

        # Update the current observation from the next_observation.
        observation = next_observation

        # Update the current action from the next_action.
        action = next_action

        # Increment the step counter.
        t += 1

        # Environment will provide a done flag, learning should handle it.
        if done:
            # Increment the episode counter.
            episode += 1
            # Append episode return to collection.
            episode_returns.append(episode_return)
            # Reset the environment.
            observation = env.reset()
            print_statistics(episode_returns, episode, agent, report_every_n)
            # Reset episode return and reward.
            episode_return = 0
            reward = 0
    return episode_returns
