# coding=utf8

# Copyright 2017 the pycolab Authors
# Copyright 2018 the campx Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from . import things
from . import rendering
from . import plot
import collections
import numpy as np
import six
import syft as sy
from syft.core.frameworks.torch import utils

class Engine(object):

    def __init__(self, rows, cols, occlusion_in_layers=True):
        self._rows = rows
        self._cols = cols
        self._backdrop = None
        self._sprites_and_drapes = collections.OrderedDict()
        self._update_groups = collections.defaultdict(list)
        self._showtime = False
        self._game_over = False
        self._occlusion_in_layers = occlusion_in_layers
        self._the_plot = plot.Plot()


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

    def send(self, location):
        for k, tensor in self._the_plot.items():
            self._the_plot[k] = tensor.send(location)

        self._board.board.send(location)

        self._board.layered_board.send(location)

        for k, tensor in self._board.layers.items():
            if (not isinstance(tensor.child, sy._PointerTensor)):
                tensor.send(location)

        for a in self._update_groups:
            for b in a:
                for c in b:
                    if (not isinstance(c, str)):
                        for item in c.__dict__.values():
                            if (utils.is_tensor(item)):
                                if (not isinstance(item.child, sy._PointerTensor)):
                                    item.send(location)

        self._backdrop.curtain.send(location)

    def share(self, *workers):
        for k, tensor in self._the_plot.items():
            self._the_plot[k] = tensor.long().share(*workers)

        self._board.board.share(*workers)

        self._board.layered_board.share(*workers)

        for k, tensor in self._board.layers.items():
            if (not isinstance(tensor.child, sy._SNNTensor)):
                self._board.layers[k] = tensor.long().share(*workers)

        for a in self._update_groups:
            for b in a:
                for c in b:
                    if (not isinstance(c, str)):
                        for key, item in c.__dict__.items():
                            if (utils.is_tensor(item)):
                                if (not isinstance(item.child, sy._SNNTensor)):
                                    c.__dict__[key] = item.long().share(*workers)

        self._backdrop.curtain.share(*workers)

    def play(self, actions):
        """Perform another game iteration, applying player actions.
        Receives an action (or actions) from the player (or players). Consults the
        `Backdrop` and all `Sprite`s and `Drape`s for updates in response to those
        actions, and derives a new observation from them to show the user. Also
        collects reward(s) for the last action and determines whether the episode
        has terminated.
        Args:
          actions: Actions supplied by the external agent(s) in response to the last
              board. Could be a scalar, could be an arbitrarily nested structure
              of... stuff, it's entirely up to the game you're making. When the game
              begins, however, it is guaranteed to be None. Used for the `update()`
              method of the `Backdrop` and all `Sprite`s and `Layer`s.
        Returns:
          A three-tuple with the following members:
            * A `rendering.Observation` object containing single-array and
              multi-array feature-map representations of the game board.
            * An reward given to the player (or players) for having performed
              `actions` in response to the last observation. This reward can be any
              type---it all depends on what the `Backdrop`, `Sprite`s, and `Drape`s
              have communicated to the `Plot`. If none have communicated anything at
              all, this will be None.
            * A reinforcement learning discount factor value. By default, it will be
              1.0 if the game is still ongoing; if the game has just terminated
              (before the player got a chance to do anything!), `discount` will be
              0.0 unless the game has chosen to supply a non-standard value to the
              `Plot`'s `terminate_episode` method.
        Raises:
          RuntimeError: if this method has been called before the `Engine` has
              been finalised via `its_showtime()`, or if this method has been called
              after the episode has terminated.
        """
        if not self._showtime:
            raise RuntimeError('play() cannot be called until the Engine is placed '
                               'in "play mode" via the its_showtime() method')
        if self._game_over:
            raise RuntimeError('play() was called after the episode handled by this '
                               'Engine has terminated')

        # Update Backdrop and all Sprites and Drapes.
        self._update_and_render(actions)

        # Apply all plot directives that the Backdrop, Sprites, and Drapes have
        # submitted to the Plot during the update.
        reward, discount, should_rerender = self._apply_and_clear_plot()

        # If directives in the Plot changed our state in any way that would change
        # the appearance of the observation (e.g. changing the z-order), we'll have
        # to re-render it before we return it.
        if should_rerender: self._render()

        # Return first-frame rendering to the user.
        return self._board, reward, discount

    def _update_and_render(self, actions):
        """Perform all game entity updates and render the next observation.
        This private method is the heart of the `Engine`: as dictated by the update
        order, it consults the `Backdrop` and all `Sprite`s and `Layer`s for
        updates, then renders the game board (`self._board`) based on those updates.
        Args:
          actions: Actions supplied by the external agent(s) in response to the last
              board. Could be a scalar, could be an arbitrarily nested structure
              of... stuff, it's entirely up to the game you're making. When the game
              begins, however, it is guaranteed to be None. Used for the `update()`
              method of the `Backdrop` and all `Sprite`s and `Layer`s.
        """
        assert self._board, (
            '_update_and_render() called without a prior rendering of the board')

        # A new frame begins!
        self._the_plot.frame += 1

        # We start with the backdrop; it doesn't really belong to an update group,
        # or it belongs to the first update group, depending on how you look at it.
        self._the_plot.update_group = None
        self._backdrop.update(actions,
                              self._board.board, self._board.layers,
                              self._sprites_and_drapes, self._the_plot)


        # Now we proceed through each of the update groups in the prescribed order.
        for update_group, entities in self._update_groups:
            # First, consult each item in this update group for updates.
            self._the_plot.update_group = update_group

            for entity in entities:


                entity.update(actions,
                              self._board.board, self._board.layers,
                              self._backdrop, self._sprites_and_drapes, self._the_plot)

            # Next, repaint the board to reflect the updates from this update group.
            self._render()

    def _apply_and_clear_plot(self):
        """Apply directives to this `Engine` found in its `Plot` object.
        These directives are requests from the `Backdrop` and all `Drape`s and
        `Sprite`s for the engine to alter its global state or its interaction with
        the player (or players). They include requests to alter the z-order,
        terminate the game, or report some kind of reward. For more information on
        these directives, refer to `Plot` object documentation.
        After collecting and applying these directives to the `Engine`s state, all
        are cleared in preparation for the next game iteration.
        Returns:
          A 2-tuple with the following elements:
            * A reward value summed over all of the rewards that the `Backdrop` and
              all `Drape`s and `Sprite`s requested be reported to the player (or
              players), or None if nobody specified a reward. Otherwise, this reward
              can be any type; it all depends on what the `Backdrop`, `Drape`s, and
              `Sprite`s have provided.
            * A boolean value indicating whether the `Engine` should re-render the
              observation before supplying it to the user. This is necessary if any
              of the Plot directives change the `Engine`'s state in ways that would
              change the appearance of the observation, like changing the z-order.
        Raises:
          RuntimeError: a z-order change directive in the Plot refers to a `Sprite`
              or `Drape` that does not exist.
        """
        directives = self._the_plot._get_engine_directives()  # pylint: disable=protected-access

        # So far, there's no reason to re-render the observation.
        should_rerender = False

        # We don't expect to have too many z-order changes, so this slow, simple
        # algorithm will probably do the trick.
        for move_this, in_front_of_that in directives.z_updates:
            # We have a z-order change, so re-rendering is necessary.
            should_rerender = True

            # Make sure that the characters in the z-order change directive correspond
            # to actual `Sprite`s and `Drape`s.
            if move_this not in self._sprites_and_drapes:
                raise RuntimeError(
                    'A z-order change directive said to move a Sprite or Drape '
                    'corresponding to character {}, but no such Sprite or Drape '
                    'exists'.format(repr(move_this)))
            if in_front_of_that is not None:
                if in_front_of_that not in self._sprites_and_drapes:
                    raise RuntimeError(
                        'A z-order change directive said to move a Sprite or Drape in '
                        'front of a Sprite or Drape corresponding to character {}, but '
                        'no such Sprite or Drape exists'.format(repr(in_front_of_that)))

            # Each directive means replacing the entire self._sprites_and_drapes dict.
            new_sprites_and_drapes = collections.OrderedDict()

            # Retrieve the Sprite or Drape that we are directed to move.
            moving_sprite_or_drape = self._sprites_and_drapes[move_this]

            # This special case handles circumstances where a Sprite or Drape is moved
            # all the way to the back of the z-order.
            if in_front_of_that is None:
                new_sprites_and_drapes[move_this] = moving_sprite_or_drape

            # Copy all Sprites or Drapes into the new sprites_and_drapes OrderedDict,
            # inserting the moving entity in front of the one it's meant to occulude.
            for character, entity in six.iteritems(self._sprites_and_drapes):
                if character == move_this: continue
                new_sprites_and_drapes[character] = entity
                if character == in_front_of_that:
                    new_sprites_and_drapes[move_this] = moving_sprite_or_drape

            # Install the OrderedDict just made as the new z-order and catalogue
            # of Sprites and Drapes.
            self._sprites_and_drapes = new_sprites_and_drapes

        # The Backdrop or one of the Sprites or Drapes may have directed the game
        # to end. Update our game-over flag.
        self._game_over = directives.game_over
        # Collect the sum of all rewards from this latest game iteration, in
        # preparation to return it to the player.
        reward = directives.summed_reward
        # Get the discount value from the latest game iteration.
        discount = directives.discount
        # Reset the Plot for the next game iteration, should there be one.
        self._the_plot._clear_engine_directives()  # pylint: disable=protected-access
        return reward, discount, should_rerender

    def _render(self):
        """Render a new game board.
        Computes a new rendering of the game board, and assigns it to `self._board`,
        based on the current contents of the `Backdrop` and all `Sprite`s and
        `Drape`s. Uses properties of those objects to obtain those contents; no
        computation should be done on their part.
        Each object is "painted" on the board in a prescribed order: the `Backdrop`
        first, then the `Sprite`s and `Drape`s according to the z-order (the order
        in which they appear in `self._sprites_and_drapes`
        """
        self._renderer.clear()
        self._renderer.paint_all_of(self._backdrop.curtain)

        for character, entity in six.iteritems(self._sprites_and_drapes):
            # print(character)
            # if (hasattr(self, '_board')):
                # import syft
                # if (isinstance(self._board.layers['A'].child, syft._PointerTensor)):
                    # print((self._board.layers['A'] + 0).get())
                # else:
                    # print(self._board.layers['A'])

            # By now we should have checked fairly carefully that all entities in
            # _sprites_and_drapes are Sprites or Drapes.
            if isinstance(entity, things.Sprite) and entity.visible:
                self._renderer.paint_sprite(character, entity.position)
            elif isinstance(entity, things.Drape):
                self._renderer.paint_drape(character, entity.curtain)
        # Done with all the layers; render the board!
        self._board = self._renderer.render()

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
        curtain = torch.ByteTensor(np.zeros((self._rows, self._cols)))
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

    def its_showtime(self):
        """Finalise `Engine` set-up and compute the first observation of the game.
        Switches the `Engine` from set-up mode, where more `Sprite`s and `Drape`s
        can be added, to "play" mode, where gameplay iterates via the `play` method.
        After this permanent modal switch, no further calls to `add_drape` or
        `add_sprite` can be made.
        Once in "play" mode, consults the `Backdrop` and all `Sprite`s and `Drape`s
        for updates, and uses these to compute the episode's first observation.
        Returns:
          A three-tuple with the following members:
            * A `rendering.Observation` object containing single-array and
              multi-array feature-map representations of the game board.
            * An initial reward given to the player (or players) (before it/they
              even gets/get a chance to play!). This reward can be any type---it all
              depends on what the `Backdrop`, `Sprite`s, and `Drape`s have
              communicated to the `Plot`. If none have communicated anything at all,
              this will be None.
            * A reinforcement learning discount factor value. By default, it will be
              1.0 if the game is still ongoing; if the game has just terminated
              (before the player got a chance to do anything!), `discount` will be
              0.0 unless the game has chosen to supply a non-standard value to the
              `Plot`'s `terminate_episode` method.
        Raises:
          RuntimeError: if this method is called more than once, or if no
              `Backdrop` class has ever been provided to the Engine.
        """
        self._runtime_error_if_called_during_showtime('its_showtime')

        # It's showtime!
        self._showtime = True

        # Now that all the Sprites and Drapes are known, convert the update groups
        # to a more efficient structure.
        self._update_groups = [(key, self._update_groups[key])
                               for key in sorted(self._update_groups.keys())]

        # And, I guess we promised to do this:
        self._current_update_group = None

        # Construct the game's observation renderer.
        chars = set(self._sprites_and_drapes.keys()).union(self._backdrop.palette)
        if self._occlusion_in_layers:
            self._renderer = rendering.BaseObservationRenderer(
                self._rows, self._cols, chars)
        else:
            self._renderer = rendering.BaseUnoccludedObservationRenderer(
                self._rows, self._cols, chars)

        # Render a "pre-initial" board rendering from all of the data in the
        # Engine's Backdrop, Sprites, and Drapes. This rendering is only used as
        # input to these entities to collect their updates for the very first frame;
        # as it accesses data members inside the entities directly, it doesn't
        # actually run any of their code (unless implementers ignore notes that say
        # "Final. Do not override.").
        self._render()

        # The behaviour of this method is now identical to play() with None actions.
        return self.play(None)

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
