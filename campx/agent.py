from .things import AbstractThing

class Agent(AbstractThing):
    def __init__(self, init_x=None, init_y=None):
        super().__init__(init_x=init_x, init_y=init_y)