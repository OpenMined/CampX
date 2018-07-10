import torch

class AbstractThing(object):
    
    def __init__(self, init_x=None, init_y=None, identity=1):
        self.init_x = init_x
        self.init_y = init_y
        self.identity = identity
        
    def create_world(self, rows, cols):
        w = torch.zeros(rows, cols)
        w[self.init_x][self.init_y] += self.identity
        self.w = w
        return w

class Thing(AbstractThing):
    def __init__(self, x=None, y=None, hover_reward=1, category="apple"):
        super().__init__(init_x=x, init_y=y)
        self.hover_reward = hover_reward
        self.category = category
        
    def render_reward_mask(self):
        return self.w * self.hover_reward