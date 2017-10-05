import random
import numpy as np

from EnvCatcher import EnvCatcher

# set experimental parameters
max_num_episodes = 10
max_num_steps = 100

# can set a random seed for consistency in agent AND environment
random_seed = 42
# set a random seed
np.random.seed(random_seed)

# initialize the environment
env = EnvCatcher(grid_size=10, env_type='episodic', verbose=True, 
                 max_num_steps=100, random_seed=random_seed)

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
            print("ep: {}, steps: {}, r_total: {}".format(i_episode, t+1, 
                  ep_reward))
            break