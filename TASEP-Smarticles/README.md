# TASEP-Smarticles

[![PyPI version](https://badge.fury.io/py/smarttasep.svg)](https://badge.fury.io/py/smarttasep)![GitHub license](https://img.shields.io/github/license/jonasmaertens/TASEP)

![](https://raw.githubusercontent.com/jonasmaertens/TASEP/main/TASEP-Smarticles/vids/smarticle-title.gif)

## Overview

The **SmartTasep** package implements a Totally Asymmetric Simple Exclusion Process (TASEP) using reinforcement
learning (RL) agents. In this simulation, particles perceive their environment at each time step, and a neural network
decides their next action (move forward, up, or down) to maximize the total current. The repository employs Double Deep
Q Learning (DDQN) with a policy network and a target network that is updated using a soft update mechanism. Experience
replay is used to sample from a buffer of past experiences, facilitating learning.

## Features

- TASEP simulation with RL agents
- Double Deep Q Learning (dDQN)
- Policy network and target network with soft updates
- (Prioritized) Experience replay for improved learning
- Real-time training visualization with pygame and matplotlib
- Saving trained networks
- Loading and running pretrained simulations
- Testing learned policies interactively
- Customizable reward functions

## Files

Classes in TASEP-Smarticles:

- `DQN.py`: Neural network RL agent class
- `GridEnvironment.py`: Environment class that computes states, transitions, and rewards
- `Trainer.py`: Wrapper class for training the network, saving, loading and running simulations
- `Playground.py`: Class for testing learned policies interactively

## Setup

1. Install the package using pip:

   ```bash
   pip install smarttasep

## Using the package

1. Import the package:

   ```python
   from smarttasep import Trainer, EnvParams, Hyperparams, Playground
   ```
   
2. Run a training session:
   ```python
   envParams = EnvParams(render_mode="human",
                          length=128,
                          width=24,
                          moves_per_timestep=200,
                          window_height=200,
                          observation_distance=3,
                          distinguishable_particles=True,
                          initial_state_template="checkerboard",
                          social_reward=True,
                          use_speeds=True,
                          sigma=10,
                          allow_wait=True,
                          invert_speed_observation=True)
    hyperparams = Hyperparams(BATCH_SIZE=32,
                              GAMMA=0.85,
                              EPS_START=0.9,
                              EPS_END=0.05,
                              EPS_DECAY=100_000,
                              TAU=0.005,
                              LR=0.001,
                              MEMORY_SIZE=500_000)

    trainer = Trainer(envParams, 
                        hyperparams, 
                        reset_interval=100_000, ﬂ
                        total_steps=500_000, 
                        plot_interval=4000, 
                        new_model=True)

    trainer.train_and_save()
   
   ```

3. Load a pretrained model and run a simulation:
   ```python
   trainer = trainer.load() # shows a list of available models and prompts the user to choose one
   trainer.run()
   ```
   ![](https://raw.githubusercontent.com/jonasmaertens/TASEP/main/TASEP-Smarticles/vids/speed_gradient.gif)
   
4. Test a learned policy interactively:
   ```python
   Playground() # shows a list of available models and prompts the user to choose one
   ```
   ![](https://raw.githubusercontent.com/jonasmaertens/TASEP/main/TASEP-Smarticles/plots/playground/playground_demo.gif)
   

## Author
Jonas Märtens

GitHub: https://github.com/jonasmaertens

## Version
Current Version: 0.1.0

## License
This project is licensed under the MIT License - see the LICENSE.md file for details