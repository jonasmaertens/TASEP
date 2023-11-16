import json
import os
import datetime
import time
from collections import namedtuple

import gymnasium as gym
import math
import random
import numpy as np
from tabulate import tabulate, TableFormat

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict
from tqdm import tqdm

from DQN import DQN
from GridEnvironment import GridEnv, EnvParams

from typing import TypedDict, Optional

from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer

import matplotlib

backend = matplotlib.get_backend()
if backend == 'MacOSX':
    import PyQt6.QtCore
    matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


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


def choose_model() -> int:
    """
    Prints a table of all models and prompts the user to select one
    """
    from GridEnvironment import default_env_params
    with open("models/all_models.json", "r") as f:
        all_models = json.load(f)
    # create table with all models
    table = []
    for key, value in all_models.items():
        # replace unset params with default values
        for param in default_env_params:
            if param not in value["env_params"]:
                value["env_params"][param] = default_env_params[param]
        table.append([key,
                      value["total_steps"],
                      value["hyperparams"]["BATCH_SIZE"],
                      value["hyperparams"]["GAMMA"],
                      value["hyperparams"]["EPS_START"],
                      value["hyperparams"]["EPS_END"],
                      value["hyperparams"]["EPS_DECAY"],
                      value["hyperparams"]["TAU"],
                      value["hyperparams"]["LR"],
                      value["hyperparams"]["MEMORY_SIZE"],
                      value["env_params"]["length"],
                      value["env_params"]["width"],
                      value["env_params"]["observation_distance"],
                      value["env_params"]["distinguishable_particles"],
                      value["env_params"]["use_speeds"],
                      value["env_params"]["sigma"],
                      value["env_params"]["average_window"],
                      value["env_params"]["allow_wait"],
                      value["env_params"]["social_reward"],
                      value["env_params"]["invert_speed_observation"],
                      value["env_params"]["speed_observation_threshold"],
                      value["random_density"],
                      value["env_params"]["punish_inhomogeneities"],
                      value["env_params"]["speed_gradient_reward"],
                      value["env_params"]["speed_gradient_linearity"]
                      ])
    tabulate.MIN_PADDING = 0
    Line = namedtuple("Line", ["begin", "hline", "sep", "end"])
    DataRow = namedtuple("DataRow", ["begin", "sep", "end"])
    # noinspection PyTypeChecker
    grid = TableFormat(
        lineabove=Line("╒", "═", "╤", "╕"),
        linebelowheader=Line("╞", "═", "╪", "╡"),
        linebetweenrows=Line("├", "─", "┼", "┤"),
        linebelow=Line("╘", "═", "╧", "╛"),
        headerrow=DataRow("│", "│", "│"),
        datarow=DataRow("│", "│", "│"),
        padding=0,
        with_header_hide=None,
    )
    print(tabulate(table, headers=[
        "id", "tot_step", "BATCH", "γ", "ε_0", "ε_end", "ε_dec", "τ",
        "LR", "MEM", "len", "width", "r_obs", "disting.",
        "speeds",
        "σ", "avg_wdw", "wait", "horn", "inv_spd", "spd_thld", "rnd_ρ", "inhom", "spd_grad", "lnrty"
    ], tablefmt=grid))
    # prompt user to select a model
    model_id = int(input("Enter model id: "))
    return model_id


def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry(f"+{x}+{y}")
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)


class Trainer:
    def __init__(self, env_params: EnvParams, hyperparams: Hyperparams | None = None, reset_interval: int = None,
                 total_steps: int = 100000, render_start: int = None, do_plot: bool = True, plot_interval: int = 10000,
                 model: str | None = None, progress_bar: bool = True, wait_initial: bool = False,
                 random_density: bool = False):
        """

        **The TASEP-Smarticles Trainer class**

        ==================================

        Description
        -----------
        The Trainer class can be used to train networks and run simulations on the 2D TASEP "GridEnv" environment.
        For training, the agent uses a Deep Q-Network (DQN) with a replay buffer and a target network. The agent
        interacts with the environment using an epsilon-greedy policy. The value of epsilon decays over a specified
        number of timesteps. The agent can be trained for a specified number of steps and the current value of the
        current can be plotted at regular intervals. The agent can also be run for a specified number of steps
        without training. During training or running, the environment can be reset at regular intervals. The
        environment can also be reset to human render mode at a specified step.

        =======

        Usage
        -----
        **Training**
            To train an agent, create a Trainer object providing the environment_params, hyperparams and Trainer params.
            Then call the ``train_and_save()`` method. The ``train_and_save()`` method will train the agent and save the
            model, plot and currents to the models directory.

            **Example:**

            >>> from Trainer import Trainer, Hyperparams, EnvParams
            >>> envParams = EnvParams(
                                    render_mode=None,
                                    length=128,
                                    width=32,
                                    moves_per_timestep=400,
                                    window_height=400,
                                    observation_distance=3,
                                    distinguishable_particles=True,
                                    initial_state_template=None,
                                    use_speeds=True,
                                    sigma=5,
                                    average_window=5000,
                                    allow_wait=True,
                                    social_reward=0.6)
            >>> hyperparams = Hyperparams(
                                    BATCH_SIZE=256,
                                    GAMMA=0.99,
                                    EPS_START=0.9,
                                    EPS_END=0.01,
                                    EPS_DECAY=240000,
                                    TAU=0.005,
                                    LR=0.005,
                                    MEMORY_SIZE=900_000)
            >>> trainer = Trainer(
                                    envParams,
                                    hyperparams,
                                    reset_interval=80000,
                                    total_steps=1_500_000,
                                    do_plot=True,
                                    plot_interval=5000,
                                    random_density=True)
            >>> trainer.train_and_save()

        **Running**
            To run a simulation with a pre-trained agent, use the ``load()`` method to load a model and then call the
            ``run()`` method. The ``load()`` method takes the id of the model to load as an argument. If no id is
            provided, the user will be prompted to select a model from a table of all models.

            **Example:**

            >>> from Trainer import Trainer
            >>> trainer = Trainer.load(
                    do_plot=True,
                    render_start=0,
                    total_steps=400000,
                    moves_per_timestep=400,
                    window_height=300,
                    average_window=3000)
            >>> trainer.run()

        =======

        Args:
            env_params: The parameters for the environment. Of type GridEnv.EnvParams
            hyperparams: The hyperparameters for the agent. Of type Trainer.Hyperparams
            reset_interval: The interval at which the environment should be reset
            total_steps: The total number of steps to train for
            render_start: The step at which the environment should be rendered in human mode
            do_plot: Whether to plot the current value of the current at regular intervals
            plot_interval: The interval at which to plot the current value
            progress_bar: Whether to show a progress bar
            wait_initial: Whether to wait 30 seconds before starting the simulation
            random_density: Whether to use a random density for the initial state
        """
        self.env_params = env_params
        self.wait_initial = wait_initial
        if "initial_state_template" in self.env_params or "initial_state" in self.env_params:
            random_density = False
        self.random_density = random_density
        self.hyperparams = hyperparams
        self.progress_bar = progress_bar
        self.model = model
        self.reset_interval = reset_interval
        self.total_steps = total_steps
        self.render_start = render_start
        self.do_plot = do_plot
        self.plot_interval = plot_interval
        if plot_interval is not None and "average_window" not in env_params:
            self.env_params["average_window"] = plot_interval
        if plot_interval is None and "average_window" in env_params:
            self.plot_interval = env_params["average_window"]

        self.currents = []
        self.rewards = []
        self.timesteps = []
        if "distinguishable_particles" in env_params and self.env_params["distinguishable_particles"]:
            self.last_states: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = dict()
            self.mover: Optional[int] = None
        # if GPU is to be used, CUDA for NVIDIA, MPS for Apple Silicon (not used bc slow)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.env = self._init_env()
        self.state: Optional[torch.Tensor] = None  # Initialized in self.reset_env() in self._init_model()
        self.policy_net, self.target_net, self.optimizer, self.memory, self.criterion = self._init_model()
        self.steps_done = 0

        # set up matplotlib figure with two axes
        dpi = 300
        # noinspection PyUnresolvedReferences
        window_width = self.env.unwrapped.window_width
        window_width_inches = window_width / dpi
        window_height_inches = window_width_inches / 2.5
        plt.rcParams["font.size"] = 3.5
        plt.style.use("dark_background")
        plt.rcParams["toolbar"] = "None"
        plt.rcParams['lines.linewidth'] = 0.5
        fig, self.ax_current = plt.subplots(figsize=(window_width_inches, window_height_inches), dpi=dpi)
        move_figure(fig, 0, 0)
        if "sigma" in self.env_params:
            self.ax_current.set_title(f"Current and reward over time for sigma = {self.env_params['sigma']}")
        else:
            self.ax_current.set_title("Current and reward over time")
        self.ax_reward = self.ax_current.twinx()
        self.ax_current.set_xlabel("Time")
        self.ax_current.set_ylabel("Current")
        self.ax_reward.set_ylabel("Reward")
        self.ax_current.tick_params(axis="y", labelcolor="blue")
        self.ax_reward.tick_params(axis="y", labelcolor="red")
        self.ax_current.yaxis.label.set_color("blue")
        self.ax_reward.yaxis.label.set_color("red")
        plt.tight_layout()
        plt.subplots_adjust(right=0.9)
        plt.subplots_adjust(left=0.1)
        plt.show(block=False)
        plt.ion()

    @classmethod
    def load(cls, model_id: int = None, sigma: float = None, total_steps: int = None, average_window=None, do_plot=None,
             wait_initial=None, render_start=None, window_height=None, moves_per_timestep=None, progress_bar=True) -> "Trainer":
        """
        Loads a model from the models directory and returns a Trainer object with the loaded model. Specify extra args
        to override the values in the loaded model.
        Args:
            model_id: The id of the model to load. If None, the user will
                be prompted to select a model from a table of all models
            sigma: Overrides the sigma value in the loaded model
            total_steps: Overrides the total_steps value in the loaded model
            average_window: Overrides the average_window value in the loaded model
            do_plot: Overrides the do_plot value in the loaded model
            wait_initial: Overrides the wait_initial value in the loaded model
            render_start: Overrides the render_start value in the loaded model
            window_height: Overrides the window_height value in the loaded model
            moves_per_timestep: Overrides the moves_per_timestep value in the loaded model
        """
        # load all_models.json
        with open("models/all_models.json", "r") as f:
            all_models = json.load(f)
        if model_id is None:
            model_id = choose_model()
        # load model
        with open(f"models/by_id/{model_id}/hyperparams.json", "r") as f:
            hyperparams = json.load(f)
        with open(f"models/by_id/{model_id}/env_params.json", "r") as f:
            env_params = json.load(f)
        if sigma is not None:
            env_params["sigma"] = sigma
        tot_steps = total_steps if total_steps else all_models[str(model_id)]["total_steps"]
        if average_window is not None:
            env_params["average_window"] = average_window
            plot_interval = average_window
        else:
            plot_interval = all_models[str(model_id)]["env_params"]["average_window"]
        if window_height is not None:
            env_params["window_height"] = window_height
        if moves_per_timestep is not None:
            env_params["moves_per_timestep"] = moves_per_timestep
        render_start = render_start if render_start is not None else all_models[str(model_id)]["render_start"]
        do_plot = do_plot if do_plot is not None else True
        wait_initial = wait_initial if wait_initial is not None else False
        trainer = cls(env_params, hyperparams, model=f"models/by_id/{model_id}/policy_net.pt",
                      total_steps=tot_steps, do_plot=do_plot, progress_bar=progress_bar, plot_interval=plot_interval,
                      wait_initial=wait_initial, render_start=render_start)
        return trainer

    def _init_env(self) -> GridEnv:
        if "GridEnv" not in gym.envs.registry:
            gym.envs.registration.register(
                id='GridEnv',
                entry_point='GridEnvironment:GridEnv',
            )
        GridEnv.metadata["render_fps"] = 144
        env: GridEnv | gym.Env = gym.make("GridEnv", **self.env_params)
        return env

    def reset_env(self):
        if self.env_params["distinguishable_particles"]:
            (state, mover), info = self.env.reset(random_density=self.random_density)
            self.mover = mover
            self.state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            (state, _), info = self.env.reset(random_density=self.random_density)
            self.state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

    def _init_model(self) -> tuple[DQN, DQN, optim.AdamW, TensorDictReplayBuffer, nn.SmoothL1Loss]:
        # Get number of actions from environment action space
        try:
            n_actions = self.env.action_space.n
        except AttributeError:  # GridEnvJIT doesn't have an action space
            n_actions = self.env.action_space
        # Get the number of observed grid cells
        self.reset_env()
        n_observations = self.state.size()[1]

        # Init model, optimizer, replay memory
        if self.model is not None:
            policy_net = DQN(n_observations, n_actions).to(self.device)
            try:
                policy_net.load_state_dict(torch.load(self.model))
            except FileNotFoundError:
                self.model = os.path.join(os.getcwd(), self.model)
                policy_net.load_state_dict(torch.load(self.model))
            target_net = DQN(n_observations, n_actions).to(self.device)
            target_net.load_state_dict(policy_net.state_dict())
        else:
            policy_net = DQN(n_observations, n_actions).to(self.device)
            target_net = DQN(n_observations, n_actions).to(self.device)
            target_net.load_state_dict(policy_net.state_dict())

        if self.hyperparams:
            optimizer = optim.AdamW(policy_net.parameters(), lr=self.hyperparams['LR'], amsgrad=True)
            memory = TensorDictReplayBuffer(
                storage=LazyTensorStorage(self.hyperparams['MEMORY_SIZE'], device=self.device),
                batch_size=self.hyperparams['BATCH_SIZE'])
            criterion = nn.SmoothL1Loss()
        else:
            optimizer = None
            memory = None
            criterion = None

        return policy_net, target_net, optimizer, memory, criterion

    def _get_current_eps(self) -> float:
        """
        Returns the current epsilon value for the epsilon-greedy policy
        Epsilon decays exponentially from EPS_START to EPS_END over EPS_DECAY steps
        """
        return self.hyperparams['EPS_END'] + (self.hyperparams['EPS_START'] - self.hyperparams['EPS_END']) * \
            math.exp(-1. * self.steps_done / self.hyperparams['EPS_DECAY'])

    def _select_action(self, state: torch.Tensor, eps_greedy=True) -> torch.Tensor:
        """
        Selects an action using an epsilon-greedy policy
        :param state: Current state
        :param eps_greedy: Whether to use epsilon-greedy policy or greedy policy
        """
        if not eps_greedy:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        sample = random.random()
        eps_threshold = self._get_current_eps()
        if sample > eps_threshold:
            with torch.no_grad():
                # max(1) gives max along axis 1 ==> max of every row
                # return tensor with first row = max values and second row = indices of max values
                # we want the indices of the max values, so we use [1]
                # we also need to reshape the tensor to be 1x1 instead of 1
                # ==> pick action with the largest expected reward
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[np.random.randint(3)]], device=self.device, dtype=torch.long)

    def _optimize_model(self):
        if len(self.memory) < self.hyperparams['BATCH_SIZE']:
            return
        batch = self.memory.sample()

        non_final_next_states = batch["next_state"]
        state_batch = batch["state"]
        action_batch = batch["action"]
        reward_batch = batch["reward"]

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        with torch.no_grad():
            next_state_values = self.target_net(non_final_next_states).max(1)[0]
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
        Trains the agent for the specified number of steps. Does not save the model.
        """
        # Training loop
        for self.steps_done in (
                pbar := tqdm(range(self.total_steps), unit="steps", leave=False, disable=not self.progress_bar)):
            just_reset = False
            # Reset the environment if the reset interval has been reached
            if self.reset_interval and self.steps_done % self.reset_interval == 0 and self.steps_done != 0:
                self.reset_env()
                just_reset = True

            # Reset the environment to human render mode if the render start has been reached
            if self.steps_done == self.render_start:
                self.env_params["render_mode"] = "human"
                self.env: GridEnv | gym.Env = gym.make("GridEnv", **self.env_params)
                self.reset_env()
                just_reset = True

            # Select action for current state
            action = self._select_action(self.state)

            if self.env_params["distinguishable_particles"]:
                (next_observation, next_mover), reward, terminated, truncated, info = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device, dtype=torch.float32)
                next_state = torch.tensor(next_observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                if next_mover in self.last_states:
                    transition = TensorDict({
                        "state": self.last_states[next_mover][0],
                        "action": self.last_states[next_mover][1],
                        "next_state": next_state,
                        "reward": self.last_states[next_mover][2],
                    }, batch_size=1, device=self.device)
                    self.memory.extend(transition)
                self.last_states[self.mover] = (self.state, action, reward)
                self.mover = next_mover
            else:
                (react_observation, next_observation), reward, terminated, truncated, info = self.env.step(
                    action.item())  # Perform action and get reward, reaction and next state
                next_state = torch.tensor(next_observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                reward = torch.tensor([reward], device=self.device, dtype=torch.float32)
                react_state = torch.tensor(react_observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                # Store the transition in memory
                transition = TensorDict({
                    "state": self.state,
                    "action": action,
                    "next_state": next_state,
                    "reward": reward,
                    "react_state": react_state
                }, batch_size=1, device=self.device)
                self.memory.extend(transition)

            # Move to the next state
            self.state = next_state

            # Perform one step of the optimization (on the policy network)
            self._optimize_model()

            # Update the target network
            self._soft_update()

            if self.steps_done % self.plot_interval == 0 and not just_reset and self.steps_done != 0:
                self.currents.append(info['current'])
                self.rewards.append(info['avg_reward'])
                self.timesteps.append(self.steps_done)
                pbar.set_description(
                    f"Eps.: {self._get_current_eps():.2f}, Current: {self.currents[-1]:.2f}, rho={self.env.unwrapped.density:.2f}")
                if self.do_plot:
                    # plot on same figure but different axes
                    self.ax_current.plot(self.timesteps, self.currents, color="blue")
                    self.ax_reward.plot(self.timesteps, self.rewards, color="red")

    def _soft_update(self):
        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                self.hyperparams['TAU'] * policy_param.data + (1 - self.hyperparams['TAU']) * target_param.data)

    def run(self):
        """
        Runs the simulation for the specified number of steps. Does not train the agent. Epsilon-greedy policy is
        disabled.
        """
        for self.steps_done in (
                pbar := tqdm(range(self.total_steps), unit="steps", leave=False, disable=not self.progress_bar)):
            # TODO: Avoid code duplication
            just_reset = False
            if self.wait_initial and self.steps_done == 101:
                plt.show(block=False)
                plt.pause(0.1)
                self.env.render()
                time.sleep(30)

            # Reset the environment if the reset interval has been reached
            if self.reset_interval and self.steps_done % self.reset_interval == 0 and self.steps_done != 0:
                self.reset_env()
                just_reset = True

            # Reset the environment to human render mode if the render start has been reached
            if self.steps_done == self.render_start:
                self.env_params["render_mode"] = "human"
                self.env: GridEnv | gym.Env = gym.make("GridEnv", **self.env_params)
                self.reset_env()
                just_reset = True

            action = self._select_action(self.state, eps_greedy=False)

            if self.env_params["distinguishable_particles"]:
                (next_observation, _), _, terminated, truncated, info = self.env.step(action.item())
                next_state = torch.tensor(next_observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            else:
                (_, next_observation), _, terminated, truncated, info = self.env.step(
                    action.item())  # Perform action and get reward, reaction and next state
                next_state = torch.tensor(next_observation, dtype=torch.float32, device=self.device).unsqueeze(0)

            self.state = next_state

            if self.steps_done % self.plot_interval == 0 and not just_reset and self.steps_done != 0:
                self.currents.append(info['current'])
                self.rewards.append(info['avg_reward'])
                self.timesteps.append(self.steps_done)
                pbar.set_description(f"Current: {self.currents[-1]:.2f}")
                if self.do_plot:
                    self.ax_current.plot(self.timesteps, self.currents, color="blue")
                    self.ax_reward.plot(self.timesteps, self.rewards, color="red")

    def save(self):
        # create models directory if it doesn't exist
        if not os.path.exists(os.path.dirname("models/")):
            os.makedirs(os.path.dirname("models/"))
        # create by_id directory if it doesn't exist
        if not os.path.exists(os.path.dirname("models/by_id/")):
            os.makedirs(os.path.dirname("models/by_id/"))
        # create all_models.json if it doesn't exist
        if not os.path.exists("models/all_models.json"):
            with open("models/all_models.json", "w") as f:
                f.write("{}")
            model_id = 0
            all_models = {}
        else:
            # load all_models.json
            with open("models/all_models.json", "r") as f:
                all_models = json.load(f)
            # create unique model id, one higher than the highest existing id
            model_id = max([int(key) for key in all_models.keys()]) + 1
        # create model directory
        os.makedirs(os.path.dirname(f"models/by_id/{model_id}/"))
        # save model policy net and target net
        torch.save(self.policy_net.state_dict(), f"models/by_id/{model_id}/policy_net.pt")
        torch.save(self.target_net.state_dict(), f"models/by_id/{model_id}/target_net.pt")
        # save hyperparams
        with open(f"models/by_id/{model_id}/hyperparams.json", "w") as f:
            json.dump(self.hyperparams, f, indent=4)
        # save env params
        with open(f"models/by_id/{model_id}/env_params.json", "w") as f:
            json.dump(self.env_params, f, indent=4)
        # save currents
        np.save(f"models/by_id/{model_id}/currents.npy", self.currents)
        # save timesteps
        np.save(f"models/by_id/{model_id}/timesteps.npy", self.timesteps)
        # save plot
        plt.plot(self.timesteps, self.currents)
        plt.xlabel("Time")
        plt.ylabel("Current")
        plt.title(f"Current over time for model {model_id}")
        plt.savefig(f"models/by_id/{model_id}/plot.png")
        plt.cla()
        plt.close()
        # save model path and metadata to all_models.json
        all_models[str(model_id)] = {
            "path": f"models/by_id/{model_id}/",
            "hyperparams": self.hyperparams,
            "env_params": self.env_params,
            "total_steps": self.total_steps,
            "plot_interval": self.plot_interval,
            "render_start": self.render_start,
            "reset_interval": self.reset_interval,
            "random_density": self.random_density,
            "model_id": model_id,
            "timestamp": datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        }
        with open("models/all_models.json", "w") as f:
            json.dump(all_models, f, indent=4)
        print(f"Saved model with id {model_id} to models/by_id/{model_id}/")
        print(f"Load model with `trainer = Trainer.load({model_id})`")

    def _save(self, file: str = None, append_timestamp=True):
        if file is None:
            file = f"models/policy_net_trained_{self.total_steps}_steps.pt"
        if append_timestamp:
            file = file.replace(".pt", f"_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pt")
        if "models/" not in file:
            file = f"models/{file}"
        # create directory if it doesn't exist
        if not os.path.exists(os.path.dirname(file)):
            os.makedirs(os.path.dirname(file))
        torch.save(self.policy_net.state_dict(), file)

    def _save_plot(self, file: str = None, append_timestamp=True):
        if file is None:
            file = f"plots/plot_{self.total_steps}_steps.png"
        if append_timestamp:
            file.replace(".png", f"_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png")
        if "plots/" not in file:
            file = f"plots/{file}"
        # create directory if it doesn't exist
        if not os.path.exists(os.path.dirname(file)):
            os.makedirs(os.path.dirname(file))
        plt.plot(self.timesteps, self.currents)
        plt.savefig(file)
        plt.cla()
        plt.close()

    def _save_currents(self, file: str = None, append_timestamp=True):
        if file is None:
            file = f"data/currents_{self.total_steps}_steps.npy"
        if append_timestamp:
            file.replace(".npy", f"_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.npy")
        if "data/" not in file:
            file = f"data/{file}"
        # create directory if it doesn't exist
        if not os.path.exists(os.path.dirname(file)):
            os.makedirs(os.path.dirname(file))
        np.save(file, self.currents)
        np.save(file.replace(".npy", "_timesteps.npy"), self.timesteps)

    def train_and_save(self):
        """
        Trains the agent for the specified number of steps and saves the model, plot and currents to the models
        directory.
        """
        self.train()
        self.save()
