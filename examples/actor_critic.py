import argparse
import sys
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

# BOAT RACE HELPERS
from boat_race import make_game
from boat_race import step_perf
from boat_race import select_action_preset
from boat_race import all_actions_readable


parser = argparse.ArgumentParser(description='CampX REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--max_episodes', type=int, default=1000,
                    help='maximum number of episodes to run')
parser.add_argument('--verbose', action='store_true',
                    help='output verbose logging for steps')
parser.add_argument('--action_preset', action='store_true',
                    help='use preset actions, useful for debugging')
parser.add_argument('--env_boat_race', action='store_true',
                    help='use boat race environment')
args = parser.parse_args()


# Select and define the environment
if not args.env_boat_race:
    env = gym.make('CartPole-v0')
    env.seed(args.seed)
    reward_threshold = env.spec.reward_threshold
    input_size = 4
    output_size = 2
    env_max_steps = 10000

else:
    game, board, reward, discount = make_game()
    input_size = board.layered_board.view(-1).shape[0]
    output_size = 5
    env_max_steps = 100
    reward_threshold = 30 # env.spec.reward_threshold

# Manually set the random seed for Torch
torch.manual_seed(args.seed)

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class Policy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(input_size, hidden_size, bias=False)
        self.action_head = nn.Linear(hidden_size, output_size, bias=False)
        self.value_head = nn.Linear(hidden_size, 1, bias=False)
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = F.relu(x)
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values


hidden_size = 32
learning_rate = 3e-2
model = Policy(input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    if args.env_boat_race:
        probs, state_value = model(Variable(state))
        m = Categorical(probs)
        selected_action = m.sample()
        action = torch.Tensor([0,0,0,0,0])
        action[selected_action.data] = 1
        log_prob = m.log_prob(selected_action)
        model.saved_actions.append(SavedAction(log_prob, state_value))
    else:
        state = Variable(torch.from_numpy(state).float())
        probs, state_value = model(state)
        m = Categorical(probs)
        try:
            action = m.sample()
        except RuntimeError as error:
            print(error)
            print('m', m, 'probs', probs, 
                'state', state, 'state_value', state_value)
            sys.exit(0)
        # Save the (action log prob, state value) tuple for later processing
        model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    return action


def finish_episode():
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    rewards = []
    for r in model.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.Tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for (log_prob, value), r in zip(saved_actions, rewards):
        reward = r - value.data[0]
        policy_losses.append(-log_prob * reward)
        value_losses.append(F.smooth_l1_loss(value, Variable(torch.Tensor([r]))))
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]


def main():
    '''Main run code.'''
    # Initialize the running reward to track task completion.
    ep_rewards = []
    total_steps = 0
    for i_episode in range(args.max_episodes): # count(1):
        if args.env_boat_race:
            # Use the boat race interface.
            game, board, reward, discount = make_game()
            state = board.layered_board.view(-1).float()
            # reset the hidden performance measure
            ep_performance = 0
            ep_performances = []
        else:
            # Use the standard gym interface
            state = env.reset()
        # Don't loop forever, add one to the env_max_steps
        # to make sure to take the final step
        for t in range(env_max_steps):
            # increment the global step counter
            total_steps += 1
            action = select_action(state)
            if args.env_boat_race:
                # get the agent starting position in ByteTensor shape of env
                # adding 0 copies the data to a new object, and is thus
                # undisturbed by the performance of the action
                location_of_agent_pre = board.layers['A']+0
                # use a preset action scheme to test the
                # env reward calculation and the performance measurement
                if args.action_preset:
                    action = select_action_preset(t)
                action_readable = all_actions_readable[np.argmax(list(action))]
                # Step through environment using chosen action
                board, reward, discount = game.play(action)
                state = board.layered_board.view(-1).float()
                location_of_agent_post = board.layers['A']
                # print('location_of_agent_post', location_of_agent_post.tolist())
                # update the agent performance measure
                one_step_performance = step_perf(location_of_agent_pre, location_of_agent_post)
                ep_performance = ep_performance + one_step_performance
                if args.verbose:
                    print('t: {}, a: {}, r: {}, p: {}'.format(
                        t, action_readable, reward, one_step_performance))
            else:
                state, reward, done, _ = env.step(action.data[0])
            if args.render and (i_episode % 100 == 0) and not args.env_boat_race:
                env.render()
            model.rewards.append(reward)
            if not args.env_boat_race:
                if done:
                    break

        if args.env_boat_race:
            ep_rewards.append(np.sum(model.rewards))
            ep_performances.append(ep_performance)
        else:
            ep_rewards.append(t)

        # calculate a moving average of running rewards
        avg_ep_reward = np.mean(ep_rewards)

        finish_episode()
        if args.env_boat_race:
            if i_episode % args.log_interval == 0:
                print('ep: {},  R: {:.2f},  R_av: {:.2f},  P: {:.2f},  P_av: {:.2f}'.format(
                    i_episode, ep_rewards[-1], np.mean(ep_rewards), ep_performances[-1], np.mean(ep_performances)))
        else:
            if i_episode % args.log_interval == 0:
                print('ep: {},  R: {:.2f},  R_av: {:.2f}'.format(
                    i_episode, ep_rewards[-1], np.mean(ep_rewards)))
            if avg_ep_reward > reward_threshold:
                print("Solved! Running reward is now {} and "
                    "the last episode runs to {} time steps!".format(avg_ep_reward, t))
                break


if __name__ == '__main__':
    main()