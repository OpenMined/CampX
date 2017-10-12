import numpy as np


class EnvCatcher(object):
  """EnvCatcher
  The basket catches the falling fruit. 
  Agent controls movement (left, stay, right) of basket 
  Square grid of size 'grid_size'. 
  Fruits fall from top and return +1 reward if caught and -1 else.

  Environment is either episodic to end after each fruit, 
  or continuous to continue for a maximum number of steps.

  Verbose will print out perstep information from dictionary info.
  """
  def __init__(self, 
               grid_size=24,
               env_type='episodic',
               verbose=False,
               max_num_steps=100,
               random_seed=None):

      self.grid_size = grid_size
      
      # three actions (left, stay, right)
      self.action_space = 3

      # set a step limit for the continuous case
      # also keep track of the steps taken
      self.step_limit = max_num_steps
      self.steps_taken = 0

      # set the width of the paddle
      self.paddle_width = 2

      self.env_type = env_type 
      self.verbose = verbose

      self.info = {}

      if random_seed is not None:
        # set a random seed
        np.random.seed(random_seed)

      # reset the environment
      self.reset()

  def _update_state(self, action):
    state = self.state

    # map action from action space to environment space
    mapped_action = action-1
    fruit_pos_row, fruit_pos_col, basket = state[0]

    # move the basket, unless you have reached a wall
    new_basket = min(max(1, basket + mapped_action), (self.grid_size-self.paddle_width))
    
    # move the fruit
    fruit_pos_row += 1

    # regenerate fruit when it leaves the screen
    if fruit_pos_row == self.grid_size:
      # new fruit location at the top row
      fruit_pos_row = 0
      fruit_pos_col = np.random.randint(0, self.grid_size-1, size=1)

    # reshape the output
    out = np.asarray([fruit_pos_row, fruit_pos_col, new_basket])
    out = out[np.newaxis]

    assert len(out.shape) == 2
    self._state = out

  def _draw_state(self):
    im_size = (self.grid_size, ) * 2
    state = self.state[0]
    canvas = np.zeros(im_size)
    # index is [row, col]
    canvas[state[0], state[1]] = 1           # draw fruit
    canvas[-1, state[2]-1:state[2] + 2] = 1  # draw basket
    return canvas

  def _reward(self):
    # split the state into fruit row/col position and basket position
    fruit_row, fruit_col, basket = self.state[0]
    
    # determine if the baseket catches the fruit
    if fruit_row == self.grid_size-1:
        if abs(fruit_col - basket) <= 1:
            return 1
        else:
            return -1
    else:
        return 0

  def _observe(self):
    return self._draw_state()

  def step(self, action):
    self.steps_taken += 1
    self._update_state(action)
    reward = self._reward()
    
    if self.env_type == 'continuous':
      # continuous catcher ends when the maximum step limit is reached
      game_over = self.continuous_is_over
    elif self.env_type == 'episodic':
      # episodic catcher ends when fruit reaches the bottom
      game_over = self.episodic_is_over
    
    # build information dictionary
    self.info['s'] = self.state[0]
    self.info['steps_taken'] = self.steps_taken
    self.info['a'] = action-1
    self.info['reward'] = reward
    self.info['game_over'] = game_over

    obs = self._observe()

    if self.verbose:
      print(self.info)
    
    return obs, reward, game_over, self.info

  def reset(self):
    # start the fruit and basket in random positions
    fruit_pos_col = np.random.randint(0, self.grid_size-1, size=1)
    # random position for the left hand side of the basket 
    basket_pos_col = np.random.randint(0, self.grid_size-self.paddle_width, size=1)

    # always start fruit on row 0
    self._state = np.asarray([0, fruit_pos_col, basket_pos_col])[np.newaxis]
    self.steps_taken = 0
    obs = self._observe()
    return obs

  @property
  def state(self):
    return self._state

  @property
  def episodic_is_over(self):
    # episodic, each episode ends when the fruit reaches the bottom
    if self.state[0, 0] == self.grid_size-1:
        return True
    else:
        return False

  @property
  def continuous_is_over(self):
    # episodic, each episode ends when the fruit reaches the bottom
    if self.step_limit == self.steps_taken:
        return True
    else:
        return False