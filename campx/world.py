import torch

class TensorWorld(object):
    
    def __init__(self, rows, cols, things=None, agent=None):
        self.rows = rows
        self.cols = cols
        
        if(things is None):
            self.things = list()
        else:
            self.things = things

        self.agent = agent
        self.cat2things = {}
    
    def add_thing(self, thing):
        self.things.append(thing)
        if(thing.category not in self.cat2things):
            self.cat2things[thing.category] = set()
        self.cat2things[thing.category].add(thing)
        
    def set_agent(self, agent):
        self.agents = agent
        
    def start(self):

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