import torch
from abc import ABC

class AbstractThing(ABC):
    def __init__(self, init_x=None, init_y=None, identity=1):
        self.x = init_x
        self.y = init_y
        self.identity = identity

class EnvThing(AbstractThing, ABC):
    def __init__(self, x=None, y=None, hover_reward=1):
        super().__init__(init_x=x, init_y=y)
        self.hover_reward = hover_reward
    
    @abstractmethod
    def update(self):
        pass

    def render_reward_mask(self):
        return self.world * self.hover_reward

    def create_world(self, tensor_world):
        world = torch.zeros(tensor_world.rows, tensor_world.cols)
        world[self.init_x][self.init_y] = self.identity
        self.world = world
        return world

class WrapAgent(AbstractThing):
    """Wraps an agent for compatibility with safe-grid-agents
    (see https://github.com/jvmancuso/safe-grid-agents)
    """
    def __init__(self, agent, init_x=None, init_y=None):
        super(WrapAgent, self).__init__(init_x, init_y, 1) # assuming agent identity is 1

    def __call__(self, agent):
        agent.x = self.x
        agent.y = self.y
        agent.identity = self.identity
        self = agent
