import random
import argparse
import sys
import gym
import csv
import time
import numpy as np
from itertools import count
import os

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
parser.add_argument('--log_interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--max_episodes', type=int, default=100,
                    help='maximum number of episodes to run')
parser.add_argument('--env_max_steps', type=int, default=100,
                    help='maximum steps in each episodes to run')
parser.add_argument('--num_runs', type=int, default=5,
                    help='number of runs to perform')
parser.add_argument('--exp_name_prefix', type=str, default='default_exp_name_prefix',
                    help='prefix to name of experiment')
parser.add_argument('--verbose', action='store_true',
                    help='output verbose logging for steps')
parser.add_argument('--action_preset', action='store_true',
                    help='use preset actions, useful for debugging')
parser.add_argument('--env_boat_race', action='store_true',
                    help='use boat race environment')
parser.add_argument('--sassy', action='store_true',
                    help='secret agent in secret environment')
args = parser.parse_args()


class Policy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(input_size, hidden_size, bias=False)
        self.affine2 = nn.Linear(hidden_size, output_size, bias=False)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.affine1(x)
        x = F.relu(x)
        x = self.affine2(x)
        action_scores = F.softmax(x, dim=0)
        return action_scores


def select_action(state):
    if args.env_boat_race:
        probs = policy(Variable(state))
        m = Categorical(probs)
        selected_action = m.sample()
        action = torch.Tensor([0,0,0,0,0])
        action[selected_action.data] = 1
        log_prob = m.log_prob(selected_action)
        policy.saved_log_probs.append(log_prob)
    else:
        state = Variable(torch.from_numpy(state).float())
        probs = policy(state)
        m = Categorical(probs)
        try:
            action = m.sample()
        except RuntimeError as error:
            print(error)
            print('m', m, 'probs', probs, 'state', state)
            sys.exit(0)
        policy.saved_log_probs.append(m.log_prob(action))
    return action


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
    return policy_loss


def main(run_id='default_id', exp_log_file_writer='default_exp_log_file_writer'):
    '''Main run code.'''
    # Initialize the running reward to track task completion.
    ep_rewards = []
    total_steps = 0
    ep_start_time = time.time()
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
        last_time = time.time()
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
                    print('t(ms): {}, t: {}, a: {}, r: {}, p: {}'.format(
                         round(1000 * (time.time() - last_time), 2), t, action_readable, reward, one_step_performance))
                    last_time = time.time()
            else:
                state, reward, done, _ = env.step(action.data[0])
            if args.render and (i_episode % 100 == 0) and not args.env_boat_race:
                env.render()
            policy.rewards.append(reward)
            if not args.env_boat_race:
                if done:
                    break

        # collect relevant metrics for reporting
        if args.env_boat_race:
            ep_rewards.append(np.sum(policy.rewards))
            ep_performances.append(ep_performance)
        else:
            ep_rewards.append(t)

        # calculate the policy loss, update the model
        # clear saved rewards and log probs
        policy_loss = finish_episode()
        ep_report_time = round(time.time() - ep_start_time, 2)
        ep_start_time = time.time()

        # Logging and reporting
        if args.env_boat_race:
            ep_fields = [run_id, total_steps, ep_report_time, 
                         i_episode, round(policy_loss.data[0],2),
                         ep_rewards[-1], np.mean(ep_rewards[-5:]), 
                         ep_performances[-1], np.mean(ep_performances)]
            exp_log_file_writer.writerow(ep_fields)
            if i_episode % args.log_interval == 0:
                print('id: {}, t(s): {}, ep: {}, L: {}, R: {:.2f}, R_av_5: {:.2f}, P: {:.2f}, P_av: {:.2f}'.format(
                    run_id, ep_report_time, i_episode, round(policy_loss.data[0],2),
                    ep_rewards[-1], np.mean(ep_rewards[-5:]), ep_performances[-1], np.mean(ep_performances)))
        else:
            if i_episode % args.log_interval == 0:
                print('t(s): {}, ep: {}, R: {:.2f}, R_av_5: {:.2f}'.format(
                    ep_report_time, i_episode, ep_rewards[-1], np.mean(ep_rewards[-5:])))

            # calculate a moving average of running rewards
            avg_ep_reward = np.mean(ep_rewards)
            if avg_ep_reward > reward_threshold:
                print("Solved! Running reward is now {} and "
                    "the last episode runs to {} time steps!".format(avg_ep_reward, t))
                break


if __name__ == '__main__':
    # Select and define the environment
    if not args.env_boat_race:
        env = gym.make('CartPole-v0')
        env.seed(args.seed)
        reward_threshold = env.spec.reward_thresholsd
        input_size = 4
        output_size = 2
        env_max_steps = 10000
    else:
        game, board, reward, discount = make_game()
        input_size = board.layered_board.view(-1).shape[0]
        output_size = 5
        env_max_steps = args.env_max_steps
        reward_threshold = 30 # env.spec.reward_threshold
        if args.sassy:
            import syft as sy

            hook = sy.TorchHook(verbose=True)
            me = hook.local_worker
            me.is_client_worker = True
            bob = sy.VirtualWorker(id="bob", hook=hook, is_client_worker=False)
            alice = sy.VirtualWorker(id="alice", hook=hook, is_client_worker=False)
            james = sy.VirtualWorker(id="james", hook=hook, is_client_worker=False)
            me.add_worker(bob)
            me.add_workers([bob, alice, james])
            bob.add_workers([me, alice, james])
            alice.add_workers([me, bob, james])
            james.add_workers([me, bob, alice])

    eps = np.finfo(np.float32).eps.item()

    # Build an output file for processing results
    logging_dir = 'logs/'
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    with open('logs/'+args.exp_name_prefix +
              '_n{}_steps{}_eps{}_sassy{}'.format(args.num_runs,
                                          args.env_max_steps,
                                          args.max_episodes,
                                          int(args.sassy)) +'.csv', mode='w') as exp_log_file:
        # write the header row
        fieldnames = ['id', 'step', 't(s)', 'ep', 'L', 'R', 'R_av_5', 'P', 'P_av']
        exp_log_file_writer = csv.writer(exp_log_file, delimiter=',',
                                         quotechar='"', quoting=csv.QUOTE_MINIMAL)
        exp_log_file_writer.writerow(fieldnames)
        for run_id in range(args.num_runs):
            # Manually set the random seed for Torch
            torch.manual_seed(args.seed + (run_id * random.randint(1,args.seed)))

            hidden_size = 32
            learning_rate = 1e-2
            policy = Policy(input_size=input_size,
                            hidden_size=hidden_size,
                            output_size=output_size)
            optimizer = optim.Adam(policy.parameters(),
                lr=learning_rate)

            # Share the weight data with campx sassy protocol
            if args.env_boat_race and args.sassy:
                W = policy.affine1.weight.data
                W = W.fix_precision().share(bob, alice)
                W2 = policy.affine2.weight.data
                W2 = W2.fix_precision().share(bob, alice)

            main(run_id=str(run_id), exp_log_file_writer=exp_log_file_writer)