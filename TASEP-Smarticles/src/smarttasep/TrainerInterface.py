from abc import ABC, abstractmethod
from typing import TypedDict, Iterable

from .GridEnvironment import EnvParams
from .DQN import DQN
import torch.optim as optim
import torch.nn as nn
from .GridEnvironment import GridEnv
from torchrl.data import LazyTensorStorage, TensorDictPrioritizedReplayBuffer
from torch import Tensor


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


class TrainerInterface(ABC):
    @abstractmethod
    def __init__(self, env_params: EnvParams, hyperparams: Hyperparams | None = None, reset_interval: int = None,
                 total_steps: int = 100000, render_start: int = None, do_plot: bool = True, plot_interval: int = 10000,
                 model: str | None = None, progress_bar: bool = True, wait_initial: bool = False,
                 random_density: bool = False, new_model: bool = False, hidden_layer_sizes: Iterable[int] = None,
                 different_models: bool = False,
                 num_models: int = 1, prio_exp_replay: bool = True, activation_function=None):
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
            new_model: Whether to use the new model
            hidden_layer_sizes: The sizes of the hidden layers of the policy network
            activation_function: The activation function of the network
            different_models: Whether to use different models for different speeds
            num_models: The number of models to use if different_models is True
            prio_exp_replay: Whether to use prioritized experience replay
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, model_id: int = None, sigma: float = None, total_steps: int = None, average_window=None, do_plot=None,
             wait_initial=None, render_start=None, window_height=None, moves_per_timestep=None, progress_bar=True,
             new_model=None, hidden_layer_sizes=None, activation_function=None) -> "Trainer":
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
            progress_bar: Whether to show a progress bar
            new_model: Whether to use the new model
            hidden_layer_sizes: The sizes of the hidden layers of the policy network
            activation_function: The activation function of the network
        """
        pass

    @staticmethod
    @abstractmethod
    def choose_model() -> int:
        """
        Displays a table of all models in the models directory and prompts the user to select a model.
        """
        pass

    @staticmethod
    @abstractmethod
    def move_figure(f, x: int, y: int):
        """
        Moves the figure f to the position (x, y)
        """
        pass

    @abstractmethod
    def _init_env(self) -> GridEnv:
        pass

    @abstractmethod
    def reset_env(self):
        pass

    @abstractmethod
    def _init_model(self) -> tuple[DQN, DQN, optim.AdamW, TensorDictPrioritizedReplayBuffer, nn.SmoothL1Loss]:
        pass

    @abstractmethod
    def _get_current_eps(self) -> float:
        """
        Returns the current epsilon value for the epsilon-greedy policy
        Epsilon decays exponentially from EPS_START to EPS_END over EPS_DECAY steps
        """
        pass

    @abstractmethod
    def _select_action(self, state: Tensor, eps_greedy=True) -> Tensor:
        """
        Selects an action using an epsilon-greedy policy
        :param state: Current state
        :param eps_greedy: Whether to use epsilon-greedy policy or greedy policy
        """
        pass

    @abstractmethod
    def _optimize_model(self):
        pass

    @abstractmethod
    def _check_env_reset(self, just_reset: bool) -> bool:
        pass

    @abstractmethod
    def train(self):
        """
        Trains the agent for the specified number of steps. Does not save the model.
        """
        pass

    @abstractmethod
    def _soft_update(self):
        """Soft update of the target network's weights
        θ′ ← τ θ + (1 −τ )θ′"""
        pass

    @abstractmethod
    def run(self):
        """
        Runs the simulation for the specified number of steps. Does not train the agent. Epsilon-greedy policy is
        disabled.
        """
        pass

    @abstractmethod
    def save(self) -> int:
        """
        Saves the model, plot and currents to the models directory, and returns the id of the model.

        Returns:
            The id of the model
        """
        pass

    @abstractmethod
    def _save_plot(self, file: str = None, append_timestamp=True):
        pass

    @abstractmethod
    def train_and_save(self) -> int:
        """
        Trains the agent for the specified number of steps and saves the model, plot and currents to the models
        directory.

        Returns:
            The id of the model
        """
        pass
