import argparse
import sys
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

# Grid world helpers
from helpers import VISIBLE_RADIUS, Grid, Agent, Environment

# Boat race helpers
from boat_race import make_game

# Reinforce helpers
from reinforce import Policy

a = torch.zeros(5,5).long()
a[1,2] = 1
a[3,2] = 1

b = torch.zeros(5,5).long()
b[1,3] = 1
b[3,1] = 1

c = a.t()

d = torch.zeros(5,5).long()
d[1,1] = 1
d[3,3] = 1

game, board, reward, discount = make_game()

input_size = board.layered_board.view(-1).shape[0]
output_size = 5
hidden_size = 32
learning_rate = 5e-3

policy = Policy(input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size)

optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

W = policy.affine1.weight.data
W = W.fix_precision().share(bob,alice)

W2 = policy.affine2.weight.data
W2 = W2.fix_precision().share(bob,alice)

game, board, reward, discount = make_game()
game.share(bob, alice)
a = a.share(bob, alice)
b = b.share(bob, alice)
c = c.share(bob, alice)
d = d.share(bob, alice)


rewards = list()
perf = 0