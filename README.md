# TASEP-Smarticles

![GitHub release (latest by date)](https://img.shields.io/github/v/release/jonasmaertens/TASEP?style=flat)
![GitHub license](https://img.shields.io/github/license/jonasmaertens/TASEP)
![GitHub last commit](https://img.shields.io/github/last-commit/jonasmaertens/TASEP)

## Overview

**TASEP-Smarticles** is a repository that implements a Totally Asymmetric Simple Exclusion Process (TASEP) using reinforcement learning (RL) agents. In this simulation, particles perceive their environment at each time step, and a neural network decides their next action (move forward, up, or down) to maximize the total current. The repository employs Double Deep Q Learning (DDQN) with a policy network and a target network that is updated using a soft update mechanism. Experience replay is used to sample from a buffer of past experiences, facilitating learning.

## Features

- TASEP simulation with RL agents
- Double Deep Q Learning (DDQN)
- Policy network and target network with soft updates
- Experience replay for improved learning
- Training visualization with Pygame
- Saving the trained network
- Loading and running pretrained simulations

## Files

- `DQN.py`: Neural network RL agent class
- `GridEnvironment.py`: Environment class that computes states, transitions, and rewards
- `Memory.py`: Class for storing and sampling from experiences
- `Trainer.py`: Wrapper class for training the network
- `example_training.py`: Sample training script that plots progress, visualizes with Pygame, and saves the network at the end
- `example_run_pretrained.py`: Sample script for loading and running a pretrained network

## Setup

1. Install PyTorch with CUDA support (if available):

   - [PyTorch Installation](https://pytorch.org/get-started/locally/)

   Note: If you're on a Mac, CPU is probably faster than Metal Performance Shaders (MPS) ([PyTorch on Apple Silicon](https://developer.apple.com/metal/pytorch/))

2. Install the following Python packages using Conda or Pip:

   - torchrl (you might have to install this before pytorch depending on the version) 
   - gymnasium
   - matplotlib
   - tqdm
   - pygame
   - numpy

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/jonasmaertens/TASEP
   cd TASEP

2. Install the required dependencies as mentioned in the Setup section.

3. Run the training script:

    ```bash
    cd TASEP-Smarticles
    python example_training.py

## Author
Jonas Märtens
GitHub: https://github.com/jonasmaertens
Email: ***REMOVED***

## Version
Current Version: 0.1.0