import random
import numpy as np

from EnvCatcher import EnvCatcher

# set experimental parameters
max_num_episodes = 10
max_num_steps = 100

# can set a random seed for consistency in agent AND environment
random_seed = None

if random_seed is not None:
    np.random.seed(random_seed)

# initialize the environment
env = EnvCatcher(grid_size=10, env_type='episodic', verbose=False, 
                 max_num_steps=100, random_seed=random_seed)

total_reward_by_episode = []

for i_episode in range(max_num_episodes):

    # for each episode reset the environment and the episode reward
    observation = env.reset()
    ep_reward = 0
    
    # perform the episode for the max number of steps
    for t in range(max_num_steps):
    
        # random action policy
        action = np.random.randint(env.action_space)
        
        # take action in environment
        observation, reward, done, info = env.step(action)
        
        # accumulate reward
        ep_reward += reward

        # environment will provide a done flag, learning should handle it
        if done:
            print("ep: {}, steps: {}, ep_reward_total: {}".format(i_episode, t+1, 
                  ep_reward))
            total_reward_by_episode.append(ep_reward)
            break

# print details of experiment
print('episode rewards', total_reward_by_episode)
print('sum of episode rewards', np.sum(total_reward_by_episode))
print('average episodic reward', np.sum(total_reward_by_episode)/float(len(total_reward_by_episode)))