from unittest import TestCase

import torch

from campx.things import Thing
from campx.agent import Agent
from campx.engine import TensorWorld

class TestTensorWorld(TestCase):
    def test_world_init(self):

        world = TensorWorld(3,3, agent=Agent(1,1))
        world.add_thing(Thing(0,0, hover_reward=1, category="apple")) # good thing
        world.add_thing(Thing(1,2, hover_reward=-1, category="snake")) # bad thing
        world.add_thing(Thing(1,0, hover_reward=-1, category="snake")) # bad thing

        world.start()

        data = world.render()
        
        _data = torch.FloatTensor([[1.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0],
                                   [1.0, 0.0, 1.0],
                                   [0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0],
                                   [0.0, 0.0, 0.0]])

        assert (data == _data).all()

        assert world.reward() == 0

        # move right (onto a snake)
        pred = torch.FloatTensor([0,1,0,0,0])
        world.update(pred)

        assert world.reward() == -1