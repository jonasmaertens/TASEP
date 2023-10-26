# %% Imports && Setup
import datetime

import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from DQN import DQN
from Memory import ReplayMemory
from GridEnvironment import GridEnv, EnvParams

from typing import TypedDict, Optional


class Hyperparams(TypedDict):
    """
    Attributes:
        BATCH_SIZE: The number of transitions sampled from the replay buffer
        GAMMA: The discount factor
        EPS_START: The starting value of epsilon
        EPS_END: The final value of epsilon
        EPS_DECAY: The rate of exponential decay of epsilon, higher means a slower decay
        TAU: The update rate of the target network
        LR: The learning rate of the ``AdamW`` optimizer
        MEMORY_SIZE: The size of the replay buffer
    """
    BATCH_SIZE: int
    GAMMA: float
    EPS_START: float
    EPS_END: float
    EPS_DECAY: int
    TAU: float
    LR: float
    MEMORY_SIZE: int


class Trainer:
    def __init__(self, env_params: EnvParams, hyperparams: Hyperparams, reset_interval: int = None,
                 total_steps: int = 100000,
                 render_start: int = None, do_plot: bool = True, plot_interval: int = 10000):
        """
        :type env_params: EnvParams
        :type reset_interval: int
        :type total_steps: int
        :type render_start: int
        :type do_plot: bool
        :type plot_interval: int
        :param env_params: The parameters for the environment. Of type GridEnv.EnvParams
        :param hyperparams: The hyperparameters for the agent. Of type Trainer.Hyperparams
        :param reset_interval: The interval at which the environment should be reset
        :param total_steps: The total number of steps to train for
        :param render_start: The step at which the environment should be rendered in human mode
        :param do_plot: Whether to plot the current value of the current at regular intervals
        :param plot_interval: The interval at which to plot the current value
        """
        self.env_params = env_params
        self.hyperparams = hyperparams
        self.reset_interval = reset_interval
        self.total_steps = total_steps
        self.render_start = render_start
        self.do_plot = do_plot
        self.plot_interval = plot_interval

        self.currents = []
        self.timesteps = []
        if self.env_params["distinguishable_particles"]:
            self.last_states: dict[int, tuple[np.ndarray, torch.Tensor, torch.Tensor]] = dict()
            self.mover: Optional[int] = None
        # if GPU is to be used, CUDA for NVIDIA, MPS for Apple Silicon (not used bc slow)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.env = self._init_env()
        self.state: Optional[torch.Tensor] = None  # Initialized in self.reset_env() in self._init_model()
        self.policy_net, self.target_net, self.optimizer, self.memory, self.criterion = self._init_model()
        self.steps_done = 0

    def _init_env(self) -> GridEnv:
        gym.envs.registration.register(
            id='GridEnv',
            entry_point='GridEnvironment:GridEnv',
        )
        GridEnv.metadata["render_fps"] = 144
        env: GridEnv | gym.Env = gym.make("GridEnv", **self.env_params)
        return env

    def reset_env(self):
        if self.env_params["distinguishable_particles"]:
            (state, mover), info = self.env.reset()
            self.mover = mover
            self.state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            (state, _), info = self.env.reset()
            self.state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _init_model(self) -> tuple[DQN, DQN, optim.AdamW, ReplayMemory, nn.SmoothL1Loss]:
        # Get number of actions from environment action space
        n_actions = self.env.action_space.n
        # Get the number of observed grid cells
        self.reset_env()
        n_observations = self.state.size()[1]

        # Init model, optimizer, replay memory
        policy_net = DQN(n_observations, n_actions).to(self.device)
        target_net = DQN(n_observations, n_actions).to(self.device)
        target_net.load_state_dict(policy_net.state_dict())

        optimizer = optim.AdamW(policy_net.parameters(), lr=self.hyperparams['LR'], amsgrad=True)
        memory = ReplayMemory(self.hyperparams['MEMORY_SIZE'])
        criterion = nn.SmoothL1Loss()

        return policy_net, target_net, optimizer, memory, criterion

    def _get_current_eps(self) -> float:
        """
        Returns the current epsilon value for the epsilon-greedy policy
        Epsilon decays exponentially from EPS_START to EPS_END over EPS_DECAY steps
        """
        return self.hyperparams['EPS_END'] + (self.hyperparams['EPS_START'] - self.hyperparams['EPS_END']) * \
            math.exp(-1. * self.steps_done / self.hyperparams['EPS_DECAY'])

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Selects an action using an epsilon-greedy policy
        :param state: Current state
        """
        sample = random.random()
        eps_threshold = self._get_current_eps()
        # self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # max(1) gives max along axis 1 ==> max of every row
                # return tensor with first row = max values and second row = indices of max values
                # we want the indices of the max values, so we use [1]
                # we also need to reshape the tensor to be 1x1 instead of 1
                # ==> pick action with the largest expected reward
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.hyperparams['BATCH_SIZE']:
            return
        batch = self.memory.sample(self.hyperparams['BATCH_SIZE'])

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.hyperparams['BATCH_SIZE'], device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.hyperparams['GAMMA']) + reward_batch

        # Compute Huber loss
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # clear the .grad attribute of the weights tensor
        self.optimizer.zero_grad()

        # compute the gradient of loss w.r.t. all the weights
        # computation graph is embedded in the loss tensor because it is created
        # from the action values tensor which is computed by forward pass though
        # the network. Gradient values are stored in the .grad attribute of the
        # weights tensor
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)

        # use the gradient values in the .grad attribute of the weights tensor
        # to update the weight values
        self.optimizer.step()

    def train(self):
        """
        Trains the agent
        """
        #print(f"Training for {self.total_steps} timesteps on {self.device}")
        # Training loop
        for self.steps_done in (pbar := tqdm(range(self.total_steps), unit="steps", leave=False)):
            just_reset = False
            # Reset the environment if the reset interval has been reached
            if self.steps_done % self.reset_interval == 0 and self.steps_done != 0:
                self.reset_env()
                just_reset = True

            # Reset the environment to human render mode if the render start has been reached
            if self.steps_done == self.render_start:
                self.env_params["render_mode"] = "human"
                self.env: GridEnv | gym.Env = gym.make("GridEnv", **self.env_params)
                self.reset_env()
                just_reset = True

            # Select action for current state
            action = self.select_action(self.state)

            if self.env_params["distinguishable_particles"]:
                (next_observation, next_mover), reward, terminated, truncated, info = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)
                next_state = torch.tensor(next_observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                if next_mover in self.last_states:
                    self.memory.push(self.last_states[next_mover][0], self.last_states[next_mover][1],
                                     next_state, self.last_states[next_mover][2])
                self.last_states[self.mover] = (self.state, action, reward)
                self.mover = next_mover
            else:
                (react_observation, next_observation), reward, terminated, truncated, info = self.env.step(
                    action.item())  # Perform action and get reward, reaction and next state
                next_state = torch.tensor(next_observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                reward = torch.tensor([reward], device=self.device)
                react_state = torch.tensor(react_observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                # Store the transition in memory
                self.memory.push(self.state, action, react_state, reward)

            # Move to the next state
            self.state = next_state

            # Perform one step of the optimization (on the policy network)
            self.optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * self.hyperparams["TAU"] + \
                                             target_net_state_dict[
                                                 key] * (1 - self.hyperparams["TAU"])
            self.target_net.load_state_dict(target_net_state_dict)

            if self.steps_done % self.plot_interval == 0 and not just_reset and self.steps_done != 0:
                self.currents.append(info['current'])
                self.timesteps.append(self.steps_done)
                pbar.set_description(f"Eps.: {self._get_current_eps():.2f}, Current: {self.currents[-1]:.2f}")
                if self.do_plot:
                    plt.plot(self.timesteps, self.currents)
                    plt.show(block=False)
                    plt.pause(0.01)

    def save(self, file: str = None, append_timestamp=True):
        if file is None:
            file = f"models/policy_net_trained_{self.total_steps}_steps.pt"
        if append_timestamp:
            file = file.replace(".pt", f"_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pt")
        if "models/" not in file:
            file = f"models/{file}"
        torch.save(self.policy_net.state_dict(), file)

    def save_plot(self, file: str = None, append_timestamp=True):
        if file is None:
            file = f"plots/plot_{self.total_steps}_steps.png"
        if append_timestamp:
            file.replace(".png", f"_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png")
        if "plots/" not in file:
            file = f"plots/{file}"
        plt.plot(self.timesteps, self.currents)
        plt.savefig(file)
        plt.cla()
        plt.close()
