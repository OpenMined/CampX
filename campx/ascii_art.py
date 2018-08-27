from . import things
from .engine import Engine

import torch
import six
import itertools

import numpy as np


def ascii_art_to_long_tensor(art):
    """Construct a numpy array of dtype `uint8` from an ASCII art diagram.
    This function takes ASCII art diagrams (expressed as lists or tuples of
    equal-length strings) and derives 2-D numpy arrays with dtype `uint8`.
    Args:
      art: An ASCII art diagram; this should be a list or tuple whose values are
          all strings containing the same number of ASCII characters.
    Returns:
      A 2-D numpy array as described.
    Raises:
      ValueError: `art` wasn't an ASCII art diagram, as described; this could be
        because the strings it is made of contain non-ASCII characters, or do not
        have constant length.
      TypeError: `art` was not a list of strings.
    """
    error_text = (
        'the argument to ascii_art_to_uint8_nparray must be a list (or tuple) '
        'of strings containing the same number of strictly-ASCII characters.')
    try:
        art = np.vstack(np.fromstring(line, dtype=np.uint8) for line in art)
    except ValueError as e:
        raise ValueError('{} (original error from numpy: {})'.format(error_text, e))
    except TypeError as e:
        if isinstance(art, (list, tuple)) and not all(
                isinstance(row, six.string_types) for row in art):
            error_text += ' Did you pass a list of list of single characters?'
        raise TypeError('{} (original error from numpy: {})'.format(error_text, e))
    if np.any(art > 127): raise ValueError(error_text)
    return torch.LongTensor(art)


def ascii_art_to_game(art,
                      what_lies_beneath,
                      sprites={},
                      drapes={},
                      backdrop=things.Backdrop,
                      update_schedule=None,
                      z_order=None,
                      occlusion_in_layers=True):
    ### 1. Set default arguments, normalise arguments, derive various things ###

    # Partial() holds arbitrary arguments (args and kwargs) that you want to pass
    # to CampX objects like sprites, drapes, and backdrops.
    sprites = {char: sprite if isinstance(sprite, Partial) else Partial(sprite)
               for char, sprite in six.iteritems(sprites)}
    drapes = {char: drape if isinstance(drape, Partial) else Partial(drape)
              for char, drape in six.iteritems(drapes)}
    # Likewise, turn a bare Backdrop class into an argument-free Partial.
    if not isinstance(backdrop, Partial): backdrop = Partial(backdrop)

    # Compile characters corresponding to all Sprites and Drapes.
    non_backdrop_characters = set()
    non_backdrop_characters.update(sprites.keys())
    non_backdrop_characters.update(drapes.keys())
    if update_schedule is None: update_schedule = list(non_backdrop_characters)

    # If update_schedule is a string (someone wasn't reading the docs!),
    # gracefully convert it to a list of single-character strings.
    if isinstance(update_schedule, str): update_schedule = list(update_schedule)

    # If update_schedule is not a list-of-lists already, convert it to be one.
    if all(isinstance(item, str) for item in update_schedule):
        update_schedule = [update_schedule]

    ### 2. Argument checking and derivation of more... things ###

    # The update schedule (flattened) is the basis for the default z-order.
    try:
        flat_update_schedule = list(itertools.chain.from_iterable(update_schedule))
    except TypeError:
        raise TypeError('if any element in update_schedule is an iterable (like a '
                        'list), all elements in update_schedule must be')
    if set(flat_update_schedule) != non_backdrop_characters:
        raise ValueError('if specified, update_schedule must list each sprite and '
                         'drape exactly once.')

    # The default z-order is derived from there.
    if z_order is None: z_order = flat_update_schedule
    if set(z_order) != non_backdrop_characters:
        raise ValueError('if specified, z_order must list each sprite and drape '
                         'exactly once.')

    # All this checking is rather strict, but as this function is likely to be
    # popular with new users, it will help to fail with a helpful error message
    # now rather than an incomprehensible stack trace later.
    if isinstance(what_lies_beneath, str) and len(what_lies_beneath) != 1:
        raise ValueError(
            'what_lies_beneath may either be a single-character ASCII string or '
            'a list of ASCII-character strings')

        # Note that the what_lies_beneath check works for characters and lists both.
        try:
            _ = [ord(character) for character in ''.join(what_lies_beneath)]
            _ = [ord(character) for character in non_backdrop_characters]
            _ = [ord(character) for character in z_order]
            _ = [ord(character) for character in flat_update_schedule]
        except TypeError:
            raise ValueError(
                'keys of sprites, keys of drapes, what_lies_beneath (or its entries), '
                'values in z_order, and (possibly nested) values in update_schedule '
                'must all be single-character ASCII strings.')

    if non_backdrop_characters.intersection(''.join(what_lies_beneath)):
        raise ValueError(
            'any character specified in what_lies_beneath must not be one of the '
            'characters used as keys in the sprites or drapes arguments.')

    ### 3. Convert all ASCII art to numpy arrays ###

    art = ascii_art_to_long_tensor(art)

    # In preparation for masking out sprites and drapes from the ASCII art array
    # (to make the background), do similar for what_lies_beneath.
    if isinstance(what_lies_beneath, str):
        what_lies_beneath = torch.zeros_like(art) + ord(what_lies_beneath)
    else:
        what_lies_beneath = ascii_art_to_long_tensor(what_lies_beneath)
        if art.shape != what_lies_beneath.shape:
            raise ValueError(
                'if not a single ASCII character, what_lies_beneath must be ASCII '
                'art whose shape is the same as that of the ASCII art in art.')

    ### 4. Other miscellaneous preparation ###

    # This dict maps the characters associated with Sprites and Drapes to an
    # identifier for the update group to which they belong. The sorted order of
    # the identifiers matches the group ordering in update_schedule, but is
    # otherwise generic.
    update_group_for = {}
    for i, update_group in enumerate(update_schedule):
        group_id = '{:05d}'.format(i)
        update_group_for.update({character: group_id for character in update_group})

    ### 5. Construct engine; populate with Sprites and Drapes ###

    game = Engine(*art.shape, occlusion_in_layers=occlusion_in_layers)

    # Sprites and Drapes are added according to the depth-first traversal of the
    # update schedule.
    for character in flat_update_schedule:
        # Switch to this character's update group.
        game.update_group(update_group_for[character])
        # Find locations where this character appears in the ASCII art.
        mask = art == ord(character)

        if character in drapes:
            # Add the drape to the Engine.
            partial = drapes[character]
            game.add_prefilled_drape(character, mask,
                                     partial.pycolab_thing,
                                     *partial.args, **partial.kwargs)

        if character in sprites:
            # Get the location of the sprite in the ASCII art, if there was one.
            row, col = np.where(mask)
            if len(row) > 1:
                raise ValueError('sprite character {} can appear in at most one place '
                                 'in art.'.format(character))
            # If there was a location, convert it to integer values; otherwise, 0,0.
            # gpylint doesn't know how implicit bools work with numpy arrays...
            row, col = (int(row), int(col)) if len(row) > 0 else (0, 0)  # pylint: disable=g-explicit-length-test

            # Add the sprite to the Engine.
            partial = sprites[character]
            game.add_sprite(character, (row, col),
                            partial.pycolab_thing,
                            *partial.args, **partial.kwargs)

        # Clear out the newly-added Sprite or Drape from the ASCII art.
        art[mask] = what_lies_beneath[mask]

    ### 6. Impose specified Z-order ###
    game.set_z_order(z_order)

    ### 7. Add the Backdrop to the engine ###

    game.set_prefilled_backdrop(
        characters=''.join(chr(c) for c in np.unique(art)),
        prefill=art,
        backdrop_class=backdrop.pycolab_thing,
        *backdrop.args, **backdrop.kwargs)

    return game

class Partial(object):
  """Holds a pycolab "thing" and its extra constructor arguments.
  In a spirit similar to `functools.partial`, a `Partial` object holds a
  subclass of one of the pycolab game entities described in `things.py`, along
  with any "extra" arguments required for its constructor (i.e. those besides
  the constructor arguments specified by the `things.py` base class
  constructors).
  `Partial` instances can be used to pass `Sprite`, `Drape` and `Backdrop`
  subclasses *and* their necessary "extra" constructor arguments to
  `ascii_art_to_game`.
  """

  def __init__(self, pycolab_thing, *args, **kwargs):
    """Construct a new Partial object.
    Args:
      pycolab_thing: a `Backdrop`, `Sprite`, or `Drape` subclass (note: not an
          object, the class itself).
      *args: "Extra" positional arguments for the `pycolab_thing` constructor.
      **kwargs: "Extra" keyword arguments for the `pycolab_thing` constructor.
    Raises:
      TypeError: `pycolab_thing` was not a `Backdrop`, a `Sprite`, or a `Drape`.
    """
    if not issubclass(pycolab_thing,
                      (things.Backdrop, things.Sprite, things.Drape)):
      raise TypeError('the pycolab_thing argument to ascii_art.Partial must be '
                      'a Backdrop, Sprite, or Drape subclass.')

    self.pycolab_thing = pycolab_thing
    self.args = args
    self.kwargs = kwargs