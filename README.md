# CampX

Secret Agent Training Facility

# Agents

* Q-learning - run_q_learning_agent.py
* SARSA - run_sarsa_agent.py
* Random - run_random_agent.py

![COMPARISON](https://github.com/OpenMined/CampX/blob/master/images/group_comparison.png "Comaprison of Average Returns")

# Environments

## EnvCatcher 
Implementation of the Catch/Catcher environment [[1]](#references)

# Configuration

All configuration values are set in config.py.

# Run

```python
python run_random_agent.py
```

# Visualize Results

## Interactive Notebook
```sh
jupyter notebook interactive_results_visualization.ipynb
```

## Script

```python
python make_results_figure.py
```

# Results

## Q-learning Batch Process

20 separate runs batch processed by running ```sh ./par_q_learn.sh```.

Maximum number of steps = 1000000 ((43478 episodes), average over 100 episodes, grid size = 24, learning rate = 0.1, discount factor = 0.98, epsilon = 0.1, random seed = None.

![QLEARN](/images/qlearn_groupfig.png "Q-learning Average Returns")

## SARSA

20 separate runs batch processed by running ```sh ./par_sarsa.sh```.

Maximum number of steps = 1000000 ((43478 episodes), average over 100 episodes, grid size = 24, learning rate = 0.1, discount factor = 0.98, epsilon = 0.1, random seed = None.

## Random

20 separate runs batch processed by running ```sh ./par_random.sh```.

![RANDOM](/images/random_groupfig.png "Random Average Returns")

# References

[1] - Mnih, Volodymyr, Nicolas Heess, and Alex Graves. "Recurrent models of visual attention." Advances in neural information processing systems. 2014.
