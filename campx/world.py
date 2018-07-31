import torch
from . import things
import collections

class Engine(object):
    
    def __init__(self, rows, cols):
        self._rows = rows
        self._cols = cols

        self._sprites_and_drapes = collections.OrderedDict()

    def add_sprite(self, character, position, sprite_class, *args, **kwargs):
        # Construct the game board dimensions for the benefit of this sprite.
        corner = things.Sprite.Position(self._rows, self._cols)

        # Build and save the drape.
        sprite = sprite_class(corner, position, character, *args, **kwargs)
        self._sprites_and_drapes[character] = sprite

        
    def set_agent(self, agent):
        self.agents = agent
        
    def reset(self):

        for thing in self.things:
            thing.create_world(self.rows, self.cols)
            
        self.agent.create_world(self.rows, self.cols)
        
    def render(self):
        dims = list()
        for cat, things in self.cat2things.items():
            dims.append(sum(map(lambda x:x.w,things)))
        dims.append(self.agent.w)
        return torch.cat(dims)

    def reward(self):
        return (sum(map(lambda x:x.render_reward_mask(),self.things)) * self.agent.w).sum()
        
    def update(self, y):
        left = torch.cat([self.agent.w[:,1:],self.agent.w[:,:1]], dim=1)
        right = torch.cat([self.agent.w[:,-1:],self.agent.w[:,:-1]], dim=1)
        up= torch.cat([self.agent.w[1:],self.agent.w[:1]], dim=0)
        down = torch.cat([self.agent.w[-1:],self.agent.w[:-1]], dim=0)
        stay = self.agent.w

        new_state = (y[0] * left) + (y[1] * right) + (y[2] * up) + (y[3] * down) + (y[4] * stay)
        self.agent.w = new_state

    def step(self, action):
        action = torch.FloatTensor([0,1,0,0,0])
        self.update(action)
        reward = self.reward()
        observation = self.render()
        return observation, reward
