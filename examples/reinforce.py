import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

from helpers import VISIBLE_RADIUS, Grid, Agent, Environment


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

# select the environment
env = gym.make('CartPole-v0')
env.seed(args.seed)
reward_threshold = env.spec.reward_threshold
input_size = 4
output_size = 2

# use the gridworld environment
# env = Environment()
# visible_squares = (VISIBLE_RADIUS * 2 + 1) ** 2
# # Plus agent health, y, x
# input_size = visible_squares + 1 + 2 
# # For both action and expected value
# output_size = 4+1
# reward_threshold = 100000 # env.spec.reward_threshold

torch.manual_seed(args.seed) # args.seed)


class Policy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(input_size, hidden_size)
        self.affine2 = nn.Linear(hidden_size, output_size)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=0)

hidden_size = 32
policy = Policy(input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size)
optimizer = optim.Adam(policy.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = Variable(torch.from_numpy(state).float())
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    # print(action)
    return action # .item()


def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    '''Main run code.'''
    # Initialize the running reward to track task completion.
    running_reward = 0
    for i_episode in count(1):
        state = env.reset()
        # Don't loop forever
        for t in range(10000):
            action = select_action(state)
            state, reward, done, _ = env.step(action.data[0])
            if args.render and (i_episode % 100 == 0):
                env.render()
            policy.rewards.append(reward)
            if done:
                break

        running_reward = running_reward * 0.99 + t * 0.01
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast ep. length: {:5d}\tAv. reward: {:.2f}'.format(
                i_episode, t, running_reward))
        if running_reward > reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()