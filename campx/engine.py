import torch
from . import things
import collections


class Engine(object):

    def __init__(self, rows, cols, occlusion_in_layers=True):
        self._rows = rows
        self._cols = cols
        self._backdrop = None
        self._sprites_and_drapes = collections.OrderedDict()
        self._update_groups = collections.defaultdict(list)
        self._showtime = False

    def add_sprite(self, character, position, sprite_class, *args, **kwargs):
        self._runtime_error_if_called_during_showtime('add_sprite')
        self._value_error_if_characters_are_bad(character, mandatory_len=1)
        self._runtime_error_if_characters_claimed_already(character)
        if not issubclass(sprite_class, things.Sprite):
            raise TypeError('sprite_class arguments to Engine.add_sprite must be a '
                            'subclass of Sprite')
        if (not 0 <= position[0] < self._rows or
                not 0 <= position[1] < self._cols):
            raise ValueError('Position {} does not fall inside a {}x{} game board.'
                             ''.format(position, self._rows, self._cols))

        # Construct the game board dimensions for the benefit of this sprite.
        corner = things.Sprite.Position(self._rows, self._cols)

        # Construct a new position for the sprite.
        position = things.Sprite.Position(*position)

        # Build and save the drape.
        sprite = sprite_class(corner, position, character, *args, **kwargs)
        self._sprites_and_drapes[character] = sprite
        self._update_groups[self._current_update_group].append(sprite)

        return sprite

    def set_agent(self, agent):
        self.agents = agent

    def reset(self):

        for thing in self.things:
            thing.create_world(self.rows, self.cols)

        self.agent.create_world(self.rows, self.cols)

    def render(self):
        dims = list()
        for cat, things in self.cat2things.items():
            dims.append(sum(map(lambda x: x.w, things)))
        dims.append(self.agent.w)
        return torch.cat(dims)

    def reward(self):
        return (sum(map(lambda x: x.render_reward_mask(), self.things)) * self.agent.w).sum()

    def update(self, y):
        left = torch.cat([self.agent.w[:, 1:], self.agent.w[:, :1]], dim=1)
        right = torch.cat([self.agent.w[:, -1:], self.agent.w[:, :-1]], dim=1)
        up = torch.cat([self.agent.w[1:], self.agent.w[:1]], dim=0)
        down = torch.cat([self.agent.w[-1:], self.agent.w[:-1]], dim=0)
        stay = self.agent.w

        new_state = (y[0] * left) + (y[1] * right) + (y[2] * up) + (y[3] * down) + (y[4] * stay)
        self.agent.w = new_state

    def step(self, action):
        action = torch.FloatTensor([0, 1, 0, 0, 0])
        self.update(action)
        reward = self.reward()
        observation = self.render()
        return observation, reward

    def _runtime_error_if_called_during_showtime(self, method_name):
        if self._showtime:
            raise RuntimeError('{} should not be called after its_showtime() '
                               'has been called'.format(method_name))

    def _value_error_if_characters_are_bad(self, characters, mandatory_len=None):
        if mandatory_len is not None and len(characters) != mandatory_len:
            raise ValueError(
                '{}, a string of length {}, was used where a string of length {} was '
                'required'.format(repr(characters), len(characters), mandatory_len))
        for char in characters:
            try:  # This test won't catch all non-ASCII characters; if
                ord(char)  # someone uses a unicode string, it'll pass. But that's
            except TypeError:  # hard to do by accident.
                raise ValueError('Character {} is not an ASCII character'.format(char))

    def _runtime_error_if_characters_claimed_already(self, characters):
        for char in characters:
            if self._backdrop and char in self._backdrop.palette:
                raise RuntimeError('Character {} is already being used by '
                                   'the backdrop'.format(repr(char)))
            if char in self._sprites_and_drapes:
                raise RuntimeError('Character {} is already being used by a sprite '
                                   'or a drape'.format(repr(char)))

    def update_group(self, group_name):
        """Change the update group for subsequent `add_sprite`/`add_drape` calls.
        The `Engine` consults `Sprite`s and `Drape`s for board updates in an order
        determined first by the update group name, then by the order in which the
        `Sprite` or `Drape` was added to the `Engine`. See the `Engine` constructor
        docstring for more details.
        It's fine to return to an update group after leaving it.
        Args:
          group_name: name of the new current update group.
        Raises:
          RuntimeError: if gameplay has already begun.
        """
        self._runtime_error_if_called_during_showtime('update_group')
        self._current_update_group = group_name

    def add_prefilled_drape(
            self, character, prefill, drape_class, *args, **kwargs):
        """Add a `Drape` to this `Engine`, with a custom initial mask.
        Much the same as `add_drape`, this method also allows callers to "prefill"
        the drape's `curtain` with an arbitrary mask. This method is mainly intended
        for use by the `ascii_art` tools; most `Drape` subclasses should fill their
        `curtain` on their own in the constructor (or in `update()`).
        Args:
        character: The ASCII character that this `Drape` directs the `Engine`
          to paint on the game board.
        prefill: 2-D `bool_` numpy array of the same dimensions as this `Engine`.
          The `Drape`'s curtain will be initialised with this pattern.
        drape_class: A subclass of `Drape` to be constructed by this method.
        *args: Additional positional arguments for the `drape_class` constructor.
        **kwargs: Additional keyword arguments for the `drape_class` constructor.
        Returns:
        the newly-created `Drape`.
        Raises:
        RuntimeError: if gameplay has already begun, or if any characters in
          `characters` has already been claimed by a preceding call to the
          `set_backdrop` or `add` methods.
        TypeError: if `drape_class` is not a `Drape` subclass.
        ValueError: if `character` is not a single ASCII character.
        """
        self._runtime_error_if_called_during_showtime('add_prefilled_drape')
        self._value_error_if_characters_are_bad(character, mandatory_len=1)
        self._runtime_error_if_characters_claimed_already(character)
        # Construct a new curtain for the drape.
        curtain = torch.ByteTensor(np.zeros((self._rows, self._cols), dtype=np.bool_))
        # Fill the curtain with the prefill data.
        curtain.set_(prefill)

        # Build and save the drape.
        drape = drape_class(curtain, character, *args, **kwargs)
        self._sprites_and_drapes[character] = drape
        self._update_groups[self._current_update_group].append(drape)

        return drape

    def set_z_order(self, z_order):
        """Set the z-ordering of all `Sprite`s and `Drape`s in this engine.
        Specify the complete order in which all `Sprite`s and `Drape`s should have
        their characters painted onto the game board. This method is available
        during game set-up only.
        Args:
          z_order: an ordered collection of all of the characters corresponding to
              all `Sprite`s and `Drape`s registered with this `Engine`.
        Raises:
          RuntimeError: if gameplay has already begun.
          ValueError: if the set of characters in `z_order` does not match the
              set of characters corresponding to all `Sprite`s and `Drape`s
              registered with this `Engine`.
        """
        self._runtime_error_if_called_during_showtime('set_z_order')
        if (set(z_order) != set(self._sprites_and_drapes.keys()) or
                len(z_order) != len(self._sprites_and_drapes)):
            raise ValueError('The z_order argument {} to Engine.set_z_order is not a '
                             'proper permutation of the characters corresponding to '
                             'Sprites and Drapes in this game, which are {}.'.format(
                repr(z_order), self._sprites_and_drapes.keys()))
        new_sprites_and_drapes = collections.OrderedDict()
        for character in z_order:
            new_sprites_and_drapes[character] = self._sprites_and_drapes[character]
        self._sprites_and_drapes = new_sprites_and_drapes
