import torch
from . import things
import collections
import numpy as np


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


    def set_prefilled_backdrop(
          self, characters, prefill, backdrop_class, *args, **kwargs):
        """Add a `Backdrop` to this `Engine`, with a custom initial pattern.
        Much the same as `set_backdrop`, this method also allows callers to
        "prefill" the background with an arbitrary pattern. This method is mainly
        intended for use by the `ascii_art` tools; most `Backdrop` subclasses should
        fill their `curtain` on their own in the constructor (or in `update()`).
        This method does NOT check to make certain that `prefill` contains only
        ASCII values corresponding to characters in `characters`; your `Backdrop`
        class should ensure that only valid characters are present in the curtain
        after the first call to its `update` method returns.
        Args:
          characters: A collection of ASCII characters that the `Backdrop` is
              allowed to use. (A string will work as an argument here.)
          prefill: 2-D `uint8` numpy array of the same dimensions as this `Engine`.
              The `Backdrop`'s curtain will be initialised with this pattern.
          backdrop_class: A subclass of `Backdrop` (including `Backdrop` itself)
              that will be constructed by this method.
          *args: Additional positional arguments for the `backdrop_class`
              constructor.
          **kwargs: Additional keyword arguments for the `backdrop_class`
              constructor.
        Returns:
          the newly-created `Backdrop`.
        Raises:
          RuntimeError: if gameplay has already begun, if `set_backdrop` has already
              been called for this engine, or if any characters in `characters` has
              already been claimed by a preceding call to the `add` method.
          TypeError: if `backdrop_class` is not a `Backdrop` subclass.
          ValueError: if `characters` are not ASCII characters.
        """
        self._runtime_error_if_called_during_showtime('set_prefilled_backdrop')
        self._value_error_if_characters_are_bad(characters)
        self._runtime_error_if_characters_claimed_already(characters)
        if self._backdrop:
          raise RuntimeError('A backdrop of type {} has already been supplied to '
                             'this Engine.'.format(type(self._backdrop)))
        if not issubclass(backdrop_class, things.Backdrop):
          raise TypeError('backdrop_class arguments to Engine.set_backdrop must '
                          'either be a Backdrop class or one of its subclasses.')

        # Construct a new curtain and palette for the Backdrop.
        curtain = torch.LongTensor(np.zeros((self._rows, self._cols), dtype=np.uint8))
        palette = Palette(characters)

        # Fill the curtain with the prefill data.
        # np.copyto(dst=curtain, src=prefill, casting='equiv')
        curtain.set_(prefill)

        # Build and set the Backdrop.
        self._backdrop = backdrop_class(curtain, palette, *args, **kwargs)

        return self._backdrop

class Palette(object):
  """A helper class for turning human-readable characters into numerical values.
  Classes like `Backdrop` need to assign certain `uint8` values to cells in the
  game board. Since these values are typically printable ASCII characters, this
  assignment can be both cumbersome (e.g. `board[r][c] = ord('j')`) and error-
  prone (what if 'j' isn't a valid value for the Backdrop to use?).
  A `Palette` object (which you can give a very short name, like `p`) is
  programmed with all of the valid characters for your Backdrop. Those that are
  valid Python variable names become attributes of the object, whose access
  yields the corresponding ASCII ordinal value (e.g. `p.j == 106`). Characters
  that are not legal Python names, like `#`, can be converted through lookup
  notation (e.g. `p['#'] == 35`). However, any character that was NOT programmed
  into the `Palette` object yields an `AttributeError` or and `IndexError`
  respectively.
  Finally, this class also supports a wide range of aliases for characters that
  are not valid variable names. There is a decent chance that the name you give
  to a symbolic character is there; for example, `p.hash == p['#'] == 35`. If
  it's not there, consider adding it...
  """

  _ALIASES = dict(
      backtick='`', backquote='`', grave='`',
      tilde='~',
      zero='0', one='1', two='2', three='3', four='4',
      five='5', six='6', seven='7', eight='8', nine='9',
      bang='!', exclamation='!', exclamation_point='!', exclamation_pt='!',
      at='@',
      # regrettably, £ is not ASCII.
      hash='#', octothorpe='#', number_sign='#', pigpen='#', pound='#',
      dollar='$', dollar_sign='$', buck='$', mammon='$',
      percent='%', percent_sign='%', food='%',
      carat='^', circumflex='^', trap='^',
      and_sign='&', ampersand='&',
      asterisk='*', star='*', splat='*',
      lbracket='(', left_bracket='(', lparen='(', left_paren='(',
      rbracket=')', right_bracket=')', rparen=')', right_paren=')',
      dash='-', hyphen='-',
      underscore='_',
      plus='+', add='+',
      equal='=', equals='=',
      lsquare='[', left_square_bracket='[',
      rsquare=']', right_square_bracket=']',
      lbrace='{', lcurly='{', left_brace='{', left_curly='{',
      left_curly_brace='{',
      rbrace='}', rcurly='}', right_brace='}', right_curly='}',
      right_curly_brace='}',
      pipe='|', bar='|',
      backslash='\\', back_slash='\\', reverse_solidus='\\',
      semicolon=';',
      colon=':',
      tick='\'', quote='\'', inverted_comma='\'', prime='\'',
      quotes='"', double_inverted_commas='"', quotation_mark='"',
      zed='z',
      comma=',',
      less_than='<', langle='<', left_angle='<', left_angle_bracket='<',
      period='.', full_stop='.',
      greater_than='>', rangle='>', right_angle='>', right_angle_bracket='>',
      question='?', question_mark='?',
      slash='/', solidus='/',
  )

  def __init__(self, legal_characters):
    """Construct a new `Palette` object.
    Args:
      legal_characters: An iterable of characters that users of this `Palette`
          are allowed to use. (A string like "#.o " will work.)
    Raises:
      ValueError: a value inside `legal_characters` is not a single character.
    """
    for char in legal_characters:
      if len(char) != 1:
        raise ValueError('Palette constructor requires legal characters to be '
                         'actual single charaters. "{}" is not.'.format(char))
    self._legal_characters = set(legal_characters)

  def __getattr__(self, name):
    return self._actual_lookup(name, AttributeError)

  def __getitem__(self, key):
    return self._actual_lookup(key, IndexError)

  def __contains__(self, key):
    # It is intentional, but probably not so important (as long as there are no
    # single-character aliases) that we do not do an aliases lookup for key.
    return key in self._legal_characters

  def __iter__(self):
    return iter(self._legal_characters)

  def _actual_lookup(self, key, error):
    """Helper: perform character validation and conversion to numeric value."""
    if key in self._ALIASES: key = self._ALIASES[key]
    if key in self._legal_characters: return ord(key)
    raise error(
        '{} is not a legal character in this Palette; legal characters '
        'are {}.'.format(key, list(self._legal_characters)))
