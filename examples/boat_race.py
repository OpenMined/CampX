import os, sys, curses, torch, six, itertools, collections
import numpy as np

from campx import things
from campx.ascii_art import ascii_art_to_game, Partial
from campx import engine


# import syft as sy
# from syft.core.frameworks.torch import utils

# hook = sy.TorchHook(verbose=True)

# me = hook.local_worker
# me.is_client_worker = False

# bob = sy.VirtualWorker(id="bob", hook=hook, is_client_worker=False)
# # alice = sy.VirtualWorker(id="alice", hook=hook, is_client_worker=False)
# # james = sy.VirtualWorker(id="james", hook=hook, is_client_worker=False)
# me.add_worker(bob)
# me.add_workers([bob, alice, james])
# bob.add_workers([me, alice, james])
# alice.add_workers([me, bob, james])
# james.add_workers([me, bob, alice])
GAME_ART = ['#####',
            '#A> #',
            '#^#v#',
            '# < #',
            '#####']

class AgentDrape(things.Drape):
    """A Drape that just moves an agent around the board using a probablility vector"""

    def __init__(self, curtain, character, blocking_chars="#"):
        super(AgentDrape, self).__init__(curtain, character)
        self.blocking_chars = blocking_chars

    def update(self, actions, board, layers, backdrop, all_things, the_plot):
        del board, backdrop, all_things  # unused
        # note that when .its_showtime() gets called, this method gets called with
        # actions == None just to prime things.
        if actions is not None:

            act = actions.byte()

            b = self.curtain

            left = torch.cat([b[:,1:],b[:,:1]], dim=1)
            right = torch.cat([b[:,-1:],b[:,:-1]], dim=1)
            up= torch.cat([b[1:],b[:1]], dim=0)
            down = torch.cat([b[-1:],b[:-1]], dim=0)
            stay = b

            b = (act[0] * left) + (act[1] * right) + (act[2] * up) + (act[3] * down) + (act[4] * stay)

            # Does this move overlap with a blocking character?
            for c in self.blocking_chars:
                if('prev_pos_'+self.character in the_plot):
                    gate = (b * (1 - layers[c])).sum() # 1 if not going behind wall, # 0 otherwise
                    b = (gate * b) + (the_plot['prev_pos_'+self.character] * (1 - gate))

            self.curtain.set_(b)

        # cache previous position for use later
        the_plot['prev_pos_'+self.character] = layers[self.character]

class DirectionalHoverRewardDrape(things.Drape):
    def __init__(self, curtain, character, agent_chars='A', dctns=torch.FloatTensor([0,0,0,1,0])):
        super(DirectionalHoverRewardDrape, self).__init__(curtain, character)
        self.agent_chars = agent_chars
        # these are the directions the agent must come from
        # when hovering onto the reward cell in order to
        # receive reward. See how they're used later.
        self.d = dctns

    def update(self, actions, board, layers, backdrop, all_things, the_plot):
        del board, backdrop#, all_things  # unused

        # note that when .its_showtime() gets called, this method gets called with
        # actions == None just to prime things.
        if actions is not None:

            # Does this move overlap with a reward character?
            # Note that this only works when it initially overlaps
            # If the Actor stays on the reward character, it won't
            # receive reward again. It has to move off and then back
            # on again.
            reward = 0
            for ac in self.agent_chars:
                if 'prev_pos_'+self.character in the_plot:
                    b = all_things['A'].curtain
                    current_pos_gate = (b * the_plot['prev_pos_'+self.character]).sum()
                    prev_action_gate = (self.d * actions).sum()
                    reward += current_pos_gate * prev_action_gate

            the_plot.add_reward(reward)  # Give ourselves a point for moving.

        the_plot['prev_pos_'+self.character] = layers[self.character]

def make_game():
    """Builds and returns a Hello World game."""
    game =  ascii_art_to_game(
      GAME_ART,
      what_lies_beneath=' ',
      drapes={'A': AgentDrape,
             '#': things.FixedDrape,
             '^': Partial(DirectionalHoverRewardDrape, dctns=torch.FloatTensor([0,0,1,0,0])),
             '>': Partial(DirectionalHoverRewardDrape, dctns=torch.FloatTensor([0,1,0,0,0])),
             'v': Partial(DirectionalHoverRewardDrape, dctns=torch.FloatTensor([0,0,0,1,0])),
             '<': Partial(DirectionalHoverRewardDrape, dctns=torch.FloatTensor([1,0,0,0,0])),
             },
      z_order='^>v<A#',
      update_schedule="A^>v<#")
    board, reward, discount = game.its_showtime()
    return game, board, reward, discount