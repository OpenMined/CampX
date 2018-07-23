import torch

class TensorWorld(object):
    
    def __init__(self, rows, cols, channels=1, things=None, agent=None):
        self.rows = rows
        self.cols = cols
        
        if things is None:
            self.things = list()
        else:
            self.things = things
        if agent is not None:
            self.agent = self.add_agent(agent)
        self.thing_registry = {}

        self.reset()

    def step(self, action):
        self.update(action)
        for thing in self.things:
            thing.update(action)
        reward = self._get_reward()
        # TODO
        # done = self._get_done()
        # info = self._gather_info() # needs to match
        done, info = None, None
        return reward, done, info, successor
    
    def register_thing(self, thing):
        self.things.append(thing)
        if thing.kind not in self.thing_registry:
            self.thing_registry[thing.kind] = list()
        self.thing_registry[thing.kind].append(thing)
        
    def add_nonagent(self, thing):
        self.register_thing(thing)
        self.world[]

    def add_agent(self, agent):
        self.agents = agent
        
    def reset(self):
        self.world = self.create_world()

    def create_world(self):
        return torch.zeros(self.rows, self.cols, self.channels)

    def render(self):
        dims = list()
        for cat, things in self.thing_registry.items():
            dims.append(sum(map(lambda x:x.world,things)))
        dims.append(self.world)
        return torch.cat(dims)

    def _get_reward(self):
        return (sum(map(lambda x:x.render_reward_mask(), self.things)) * self.world).sum()

    def update(self, action):
        left = torch.cat([self.agent.world[:,1:],self.agent.world[:,:1]], dim=1)
        right = torch.cat([self.agent.world[:,-1:],self.agent.world[:,:-1]], dim=1)
        up= torch.cat([self.agent.world[1:],self.agent.world[:1]], dim=0)
        down = torch.cat([self.agent.world[-1:],self.agent.world[:-1]], dim=0)
        stay = self.world

        new_state = (action[0] * left) + (action[1] * right) + (action[2] * up) + (action[3] * down) + (action[4] * stay)
        self.world = new_state

    def step(self, action):
        action_tensor = torch.zeroes(5)
        action_tensor[action] = 1
        self.update(action_tensor)
        reward = self._get_reward()
        observation = self.render()
        return observation, reward, None

