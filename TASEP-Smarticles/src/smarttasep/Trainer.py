import json
import os
import datetime
import time
from collections import namedtuple, OrderedDict
from typing import Optional

import gymnasium as gym
import math
import random
import numpy as np
from tabulate import tabulate, TableFormat

import torch
import torch.nn as nn
import torch.optim as optim
from torchrl.data import LazyTensorStorage, TensorDictPrioritizedReplayBuffer, TensorDictReplayBuffer
from tensordict import TensorDict
from tqdm import tqdm

from .DQN import DQN
from .GridEnvironment import GridEnv
from .TrainerInterface import TrainerInterface, Hyperparams, EnvParams  # noqa: PyUnresolvedReferences

import matplotlib

_backend = matplotlib.get_backend()
if _backend == 'MacOSX':
    import PyQt6.QtCore  # noqa: F401

    matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt  # noqa: E402


class Trainer(TrainerInterface):
    def __init__(self, env_params, hyperparams=None, reset_interval=None,
                 total_steps=100000, render_start=None, do_plot=True, plot_interval=10000,
                 model=None, progress_bar=True, wait_initial=False,
                 random_density=False, new_model=None, different_models=False, num_models=3, prio_exp_replay=False,
                 hidden_layer_sizes=None, activation_function=None):
        self.env_params = env_params
        self.wait_initial = wait_initial
        self.prio_exp_replay = prio_exp_replay
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation_function = activation_function
        if "initial_state_template" in self.env_params or "initial_state" in self.env_params:
            random_density = False
        self.random_density = random_density
        if different_models and not env_params["use_speeds"]:
            raise ValueError("different_models can only be True if use_speeds is True in env_params")
        self.diff_models = different_models
        self.num_models = num_models
        self.hyperparams = hyperparams
        self.new_model = new_model
        self.progress_bar = progress_bar
        if model is not None and not isinstance(model, list | str):
            raise ValueError("model must be a list of strings or a string (path to model)")
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
        self.all_rewards = np.zeros(self.total_steps)
        if "distinguishable_particles" in env_params and self.env_params["distinguishable_particles"]:
            self.last_states: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = dict()
            self.mover: Optional[int] = None
        # if GPU is to be used, CUDA for NVIDIA, MPS for Apple Silicon (not used bc slow)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.env = self._init_env()
        self.state: Optional[torch.Tensor] = None  # Initialized in self.reset_env() in self._init_model()
        if self.diff_models:
            self.policy_nets, self.target_nets, self.optimizers, self.memories, self.criteria = self._init_model()
        else:
            self.policy_net, self.target_net, self.optimizer, self.memory, self.criterion = self._init_model()
        self.steps_done = 0

        if self.do_plot:
            self.setup_plot()
            plt.show(block=False)
            plt.ion()

    def setup_plot(self):
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
        # self.fig, self.axs = plt.subplots(2, 3, figsize=(window_width_inches, window_height_inches), dpi=dpi)
        self.move_figure(fig, 0, 0)
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
        plt.subplots_adjust(right=0.89)
        plt.subplots_adjust(left=0.11)
        # force scientific notation on all axes
        self.ax_current.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        self.ax_current.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        self.ax_reward.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    @classmethod
    def load(cls, model_id=None, sigma=None, total_steps=None, average_window=None, do_plot=None,
             wait_initial=None, render_start=None, window_height=None, moves_per_timestep=None, progress_bar=True,
             new_model=None, hidden_layer_sizes=None, activation_function=None, env_params=None):
        # load all_models.json
        with open("models/all_models.json", "r") as f:
            all_models = json.load(f)
        if model_id is None:
            model_id = cls.choose_model()
        path = all_models[str(model_id)]["path"]
        # load model
        with open(
                os.path.join(path, "hyperparams.json"), "r") as f:
            hyperparams = json.load(f)
        with open(os.path.join(path, "env_params.json"), "r") as f:
            env_params = json.load(f) if env_params is None else env_params
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
        new_model = all_models[str(model_id)]["new_model"] if "new_model" in all_models[str(model_id)] else False
        diff_models = all_models[str(model_id)]["diff_models"]
        num_models = all_models[str(model_id)]["num_models"]
        if diff_models:
            model = [os.path.join(path, f"policy_net_{net_id}.pt") for net_id in range(num_models)]
        else:
            model = os.path.join(path, f"policy_net.pt")
        trainer = cls(env_params, hyperparams, model=model, total_steps=tot_steps, do_plot=do_plot,
                      progress_bar=progress_bar, plot_interval=plot_interval, wait_initial=wait_initial,
                      render_start=render_start, new_model=new_model, different_models=diff_models,
                      num_models=num_models)
        return trainer

    @staticmethod
    def choose_model():
        from .GridEnvironment import default_env_params
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

    @staticmethod
    def move_figure(f, x, y):
        backend = matplotlib.get_backend()
        window = f.canvas.manager.window  # noqa
        if backend == 'TkAgg':
            window.wm_geometry(f"+{x}+{y}")
        elif backend == 'WXAgg':
            window.SetPosition((x, y))
        else:
            # This works for QT and GTK
            # You can also use window.setGeometry
            window.move(x, y)


    def _init_env(self):
        if "GridEnv" not in gym.envs.registry:
            gym.envs.registration.register(
                id='GridEnv',
                entry_point='smarttasep:GridEnv',
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

    def _init_model(self):
        # Get number of actions from environment action space
        try:
            n_actions = self.env.action_space.n
        except AttributeError:  # GridEnvJIT doesn't have an action space
            n_actions = self.env.action_space
        # Get the number of observed grid cells
        self.reset_env()
        n_observations = self.state.size()[1]

        # Determine the number of models to initialize
        num_models = self.num_models if self.diff_models else 1
        models = self.model if isinstance(self.model, list) else [self.model] * num_models

        # Initialize dictionaries for storing models, optimizers, memories, and criteria
        policy_net = {}
        target_net = {}
        optimizer = {}
        memory = {}
        criterion = {}

        # Initialize models, optimizers, memories, and criteria
        for net_id, model in enumerate(models):
            new_policy_net, new_target_net = self._init_dqn(n_observations, n_actions, model)
            policy_net[net_id] = new_policy_net
            target_net[net_id] = new_target_net
            if self.hyperparams:
                optimizer[net_id] = optim.AdamW(policy_net[net_id].parameters(), lr=self.hyperparams['LR'],
                                                amsgrad=True)
                if self.prio_exp_replay:
                    memory[net_id] = TensorDictPrioritizedReplayBuffer(
                        alpha=0.7,
                        beta=0.5,
                        priority_key="td_error",
                        storage=LazyTensorStorage(self.hyperparams['MEMORY_SIZE'], device=self.device),
                        batch_size=self.hyperparams['BATCH_SIZE'],
                        prefetch=4)
                else:
                    memory[net_id] = TensorDictReplayBuffer(
                        storage=LazyTensorStorage(self.hyperparams['MEMORY_SIZE'], device=self.device),
                        batch_size=self.hyperparams['BATCH_SIZE'],
                        prefetch=4)
                criterion[net_id] = nn.SmoothL1Loss(reduction="none")

        # If only one model is initialized, extract it from the dictionary
        if not self.diff_models:
            policy_net, target_net = policy_net[0], target_net[0]
            optimizer, memory, criterion = optimizer[0], memory[0], criterion[0]

        return policy_net, target_net, optimizer, memory, criterion

    def _init_dqn(self, n_observations, n_actions, model):
        policy_net = DQN(n_observations, n_actions, new_model=self.new_model, hidden_sizes=self.hidden_layer_sizes,
                         activation_function=self.activation_function).to(self.device)
        if model is not None:
            try:
                policy_net.load_state_dict(torch.load(model))
            except RuntimeError:
                # map old dict keys to new dict keys
                old_state_dict = torch.load(model)
                new_state_dict = OrderedDict()
                for k, v in old_state_dict.items():
                    k = k.replace("layer1", "model.0").replace("layer2", "model.2").replace("layer3", "model.4")
                    new_state_dict[k] = v
                policy_net.load_state_dict(new_state_dict)
            except FileNotFoundError:
                model = os.path.join(os.getcwd(), model)
                policy_net.load_state_dict(torch.load(model))
        target_net = DQN(n_observations, n_actions, self.new_model, self.hidden_layer_sizes).to(self.device)
        target_net.load_state_dict(policy_net.state_dict())
        return policy_net, target_net

    def _get_current_eps(self):
        return self.hyperparams['EPS_END'] + (self.hyperparams['EPS_START'] - self.hyperparams['EPS_END']) * \
            math.exp(-1. * self.steps_done / self.hyperparams['EPS_DECAY'])

    def _select_action(self, state, eps_greedy=True, model_id=None):
        if not eps_greedy:
            with torch.no_grad():
                if model_id is not None:
                    return self.policy_nets[model_id](state).max(1)[1].view(1, 1)
                else:
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
                if model_id is not None:
                    return self.policy_nets[model_id](state).max(1)[1].view(1, 1)
                else:
                    return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            n_actions = self.env.action_space.n
            return torch.tensor([[np.random.randint(n_actions)]], device=self.device, dtype=torch.long)

    def _optimize_model(self, model_id=None):
        memory = self.memory if model_id is None else self.memories[model_id]

        if len(memory) < self.hyperparams['BATCH_SIZE']:
            return

        if self.diff_models:
            batch = self.memories[model_id].sample()
            policy_net = self.policy_nets[model_id]
            target_net = self.target_nets[model_id]
            optimizer = self.optimizers[model_id]
            criterion = self.criteria[model_id]
        else:
            batch = self.memory.sample()
            policy_net = self.policy_net
            target_net = self.target_net
            optimizer = self.optimizer
            criterion = self.criterion

        non_final_next_states = batch["next_state"]
        state_batch = batch["state"]
        action_batch = batch["action"]
        reward_batch = batch["reward"]

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        with torch.no_grad():
            next_state_values = target_net(non_final_next_states).max(1)[0]

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.hyperparams['GAMMA']) + reward_batch

        # Compute td_error
        if self.prio_exp_replay:
            weight_batch = batch["_weight"]
            with torch.no_grad():
                td_error = torch.abs(state_action_values - expected_state_action_values.unsqueeze(1))
                batch["td_error"] = td_error
                memory.update_tensordict_priority(batch)

            # Compute Huber loss
            loss = (criterion(state_action_values, expected_state_action_values.unsqueeze(1)) *
                    weight_batch.unsqueeze(1)).mean()
        else:
            # Compute Huber lossx
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1)).mean()

        # clear the .grad attribute of the weights tensor
        optimizer.zero_grad()

        # compute the gradient of loss w.r.t. all the weights
        # computation graph is embedded in the loss tensor because it is created
        # from the action values tensor which is computed by forward pass though
        # the network. Gradient values are stored in the .grad attribute of the
        # weights tensor
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)

        # use the gradient values in the .grad attribute of the weights tensor
        # to update the weight values
        optimizer.step()

    def train(self):
        # Training loop
        for self.steps_done in (
                pbar := tqdm(range(self.total_steps), unit="steps", leave=False, disable=not self.progress_bar)):
            just_reset = False
            just_reset = self._check_env_reset(just_reset)

            model_id = self._get_model_id()

            # Select action for current state
            action = self._select_action(self.state, model_id=model_id)

            if self.env_params["distinguishable_particles"]:
                (next_observation, next_mover), reward, terminated, truncated, info = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device, dtype=torch.float32)
                next_state = torch.tensor(next_observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                self.last_states[self.mover] = (self.state, action, reward)
                if next_mover in self.last_states:
                    transition = TensorDict({
                        "state": self.last_states[next_mover][0],
                        "action": self.last_states[next_mover][1],
                        "next_state": next_state,
                        "reward": self.last_states[next_mover][2],
                    }, batch_size=1, device=self.device)
                    if self.diff_models:
                        self.memories[model_id].extend(transition)
                    else:
                        self.memory.extend(transition)
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

            # Store reward in all_rewards array
            self.all_rewards[self.steps_done] = reward.item()

            # Move to the next state
            self.state = next_state

            # Perform one step of the optimization (on the policy network)
            self._optimize_model(model_id=model_id)

            # Update the target network
            self._soft_update(model_id=model_id)

            if self.steps_done % self.plot_interval == 0 and not just_reset and self.steps_done != 0:
                self.currents.append(info['current'])
                self.rewards.append(info['avg_reward'])
                self.timesteps.append(self.steps_done)
                pbar.set_description(
                    f"Eps.: {self._get_current_eps():.2f}, Current: {self.currents[-1]:.2f}, rho={self.env.unwrapped.density:.2f}")  # noqa: PyUnresolvedReferences
                if self.do_plot:
                    # plot on same figure but different axes
                    self.ax_current.plot(self.timesteps, self.currents, color="blue")
                    self.ax_reward.plot(self.timesteps, self.rewards, color="red")

    def _check_env_reset(self, just_reset):
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
        return just_reset

    def _soft_update(self, model_id=None):
        if model_id is not None:
            for target_param, policy_param in zip(self.target_nets[model_id].parameters(),
                                                  self.policy_nets[model_id].parameters()):
                target_param.data.copy_(
                    self.hyperparams['TAU'] * policy_param.data + (1 - self.hyperparams['TAU']) * target_param.data)
        else:
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(
                    self.hyperparams['TAU'] * policy_param.data + (1 - self.hyperparams['TAU']) * target_param.data)

    def run(self, reset_stats=False):
        if reset_stats:
            self.currents = []
            self.rewards = []
            self.timesteps = []
            self.all_rewards = np.zeros(self.total_steps)
            self.reset_env()

        for self.steps_done in (
                pbar := tqdm(range(self.total_steps), unit="steps", leave=False, disable=not self.progress_bar)):
            just_reset = False
            if self.wait_initial and self.steps_done == 1:
                plt.show(block=False)
                plt.pause(0.1)
                self.env.render()
                time.sleep(self.wait_initial)

            just_reset = self._check_env_reset(just_reset)

            model_id = self._get_model_id()
            action = self._select_action(self.state, eps_greedy=False, model_id=model_id)
            if self.env_params["distinguishable_particles"]:
                (next_observation, _), reward, terminated, truncated, info = self.env.step(action.item())
                next_state = torch.tensor(next_observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            else:
                (_, next_observation), reward, terminated, truncated, info = self.env.step(
                    action.item())  # Perform action and get reward, reaction and next state
                next_state = torch.tensor(next_observation, dtype=torch.float32, device=self.device).unsqueeze(0)

            # Store reward in all_rewards array
            self.all_rewards[self.steps_done] = reward

            self.state = next_state

            if self.steps_done % self.plot_interval == 0 and not just_reset and self.steps_done != 0:
                self.currents.append(info['current'])
                self.rewards.append(info['avg_reward'])
                self.timesteps.append(self.steps_done)
                pbar.set_description(f"Current: {self.currents[-1]:.2f}")
                if self.do_plot:
                    # clear previous plot
                    for artist in self.ax_current.lines + self.ax_reward.lines:
                        artist.remove()
                    self.ax_current.plot(self.timesteps, self.currents, color="blue")
                    self.ax_reward.plot(self.timesteps, self.rewards, color="red")
                    # plot average current in green
                    self.ax_current.plot(self.timesteps,
                                         np.ones(len(self.timesteps)) * np.mean(self.currents[-100:]),
                                         color="green")

    def _get_model_id(self):
        if self.diff_models:
            obs_dist = self.env_params["observation_distance"]
            state = self.state.reshape(2 * obs_dist + 1, 2 * obs_dist + 1)
            speed = state[obs_dist, obs_dist]
            if self.env_params["invert_speed_observation"]:
                speed = 1 - speed + self.env_params["speed_observation_threshold"]
            model_id = int(speed * self.num_models)
            if model_id == self.num_models:  # shouldn't happen, but does sometimes due to rounding errors
                model_id -= 1
        else:
            model_id = None
        return model_id

    def save(self, path="by_id"):
        # create models directory if it doesn't exist
        if not os.path.exists(os.path.dirname("models/")):
            os.makedirs(os.path.dirname("models/"))
        # create by_id directory if it doesn't exist
        if not os.path.exists(os.path.dirname(f"models/{path}/")):
            os.makedirs(os.path.dirname(f"models/{path}/"))
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
        os.makedirs(os.path.dirname(f"models/{path}/{model_id}/"))
        # save model policy net and target net
        if self.diff_models:
            for net_id, net in self.policy_nets.items():
                torch.save(net.state_dict(), f"models/{path}/{model_id}/policy_net_{net_id}.pt")
            for net_id, net in self.target_nets.items():
                torch.save(net.state_dict(), f"models/{path}/{model_id}/target_net_{net_id}.pt")
        else:
            torch.save(self.policy_net.state_dict(), f"models/{path}/{model_id}/policy_net.pt")
            torch.save(self.target_net.state_dict(), f"models/{path}/{model_id}/target_net.pt")
        # save hyperparams
        with open(f"models/{path}/{model_id}/hyperparams.json", "w") as f:
            json.dump(self.hyperparams, f, indent=4)
        # save env params
        with open(f"models/{path}/{model_id}/env_params.json", "w") as f:
            json.dump(self.env_params, f, indent=4)
        # save currents
        np.save(f"models/{path}/{model_id}/currents.npy", self.currents)
        # save timesteps
        np.save(f"models/{path}/{model_id}/timesteps.npy", self.timesteps)
        # save rewards
        np.save(f"models/{path}/{model_id}/rewards.npy", self.rewards)
        # save all_rewards
        np.save(f"models/{path}/{model_id}/all_rewards.npy", self.all_rewards)
        # save plot
        self._save_plot(f"models/{path}/{model_id}/plot.png", append_timestamp=False)
        # save model path and metadata to all_models.json
        all_models[str(model_id)] = {
            "path": f"models/{path}/{model_id}/",
            "hyperparams": self.hyperparams,
            "env_params": self.env_params,
            "total_steps": self.total_steps,
            "plot_interval": self.plot_interval,
            "render_start": self.render_start,
            "reset_interval": self.reset_interval,
            "random_density": self.random_density,
            "model_id": model_id,
            "timestamp": datetime.datetime.now().strftime('%Y%m%d%H%M%S'),
            "new_model": self.new_model,
            "diff_models": self.diff_models,
            "num_models": self.num_models
        }
        with open("models/all_models.json", "w") as f:
            json.dump(all_models, f, indent=4)
        print(f"Saved model with id {model_id} to models/{path}/{model_id}/")
        print(f"Load model with `trainer = Trainer.load({model_id})`")
        return model_id

    def _save_plot(self, file=None, append_timestamp=True):
        if file is None:
            file = f"plots/plot_{self.total_steps}_steps.png"
        if append_timestamp:
            file.replace(".png", f"_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.png")
        if "plots/" not in file:
            file = f"plots/{file}"
        # create directory if it doesn't exist
        if not os.path.exists(os.path.dirname(file)):
            os.makedirs(os.path.dirname(file))
        if not self.do_plot:
            self.setup_plot()
        self.ax_current.plot(self.timesteps, self.currents, color="blue")
        self.ax_reward.plot(self.timesteps, self.rewards, color="red")
        plt.savefig(file)
        plt.cla()
        plt.close()

    def train_and_save(self, path="by_id"):
        self.train()
        return self.save(path)

    def save_run_data_for_plot(self, name):
        # create directory if it doesn't exist
        os.makedirs(f"data/{name}/")
        # save currents
        np.save(f"data/{name}/currents.npy", self.currents)
        # save timesteps
        np.save(f"data/{name}/timesteps.npy", self.timesteps)

