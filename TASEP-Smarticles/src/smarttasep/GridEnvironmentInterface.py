from abc import ABC, abstractmethod
import numpy as np

from typing import SupportsFloat, TypeVar, Any, TypedDict, Optional, TypeAlias, NotRequired

ObsType = TypeVar("ObsType")
WholeObsType: TypeAlias = tuple[ObsType, ObsType] | tuple[ObsType, int]


class EnvParams(TypedDict):
    """
    Environment Params for GridEnvironment
    Attributes:
        render_mode (str, optional): The mode in which the environment is rendered. Defaults to None. Can be "human"
                or "rgb_array".
            length (int, optional): The length of the grid. Defaults to 64.
            width (int, optional): The number of "lanes". Defaults to 16.
            moves_per_timestep (int, optional): The number of moves per timestep. Defaults to 5.
            window_height (int, optional): The height of the PyGame window. Defaults to 256.
            observation_distance (int, optional): The agent's observation radius. Defaults to 3.
            initial_state (np.ndarray, optional): The initial state of the grid. Defaults to None.
            initial_state_template (np.ndarray, optional): The template for the initial state of the grid. Defaults to
                None. Can be "checkerboard" or "everyThird".
            distinguishable_particles (bool, optional): Whether the particles are distinguishable. Defaults to False.
                If True, a transition is stored when after same agent is picked again. s' then includes the movements
                of the other agents.
            use_speeds (bool, optional): Whether agents should have different speeds. Defaults to False.
            sigma (float, optional): The standard deviation of the truncated normal distribution to draw speeds from.
                Defaults to None.
            average_window (int, optional): The size of the time averaging period. Defaults to 1000.
            allow_wait (bool, optional): Whether to allow the agents to wait. Defaults to False.
            social_reward (float, optional): If specified, agents get a negative reward for moving into a cell with a
                particle behind it. When using speeds, the reward is scaled the speed of the particle behind.
                Defaults to None.
            density (float, optional): The density of the grid. Defaults to 0.5. Used for random initial states when
                initial_state and initial_state_template are None.
            invert_speed_observation (bool, optional): If True, higher speeds are represented by lower values in the
                observation. Defaults to False.
            speed_observation_threshold (float, optional): The value that a particle with speed 1 should have in the
                observation. Defaults to 0.35.
            punish_inhomogeneities (bool, optional): Whether to punish speed inhomogeneity in the observation.
                Defaults to False.
            speed_gradient_reward (bool, optional): Whether to encourage a vertical speed gradient in the system.
            speed_gradient_linearity (float, optional): The linearity of the speed gradient reward. Defaults to 0.1.
            inh_rew_idx (int, optional): The index of the reward formula that should be used for the inhomogeneity reward
            binary_speeds: Whether to sample speeds from a binary distribution instead of a truncated normal distribution
            choices: How many options to choose from when sampling speeds from a "binary" distribution
            inflate_speeds: Whether to map the speeds in the observations to a range of 0 to 1
    """
    render_mode: NotRequired[str | None]
    length: int
    width: int
    moves_per_timestep: NotRequired[int]
    window_height: NotRequired[int]
    observation_distance: int
    initial_state: NotRequired[np.ndarray[np.uint8 | np.int32]]
    initial_state_template: NotRequired[str]
    distinguishable_particles: bool
    use_speeds: bool
    sigma: NotRequired[float]
    average_window: NotRequired[int]
    allow_wait: NotRequired[bool]
    social_reward: NotRequired[float]
    density: NotRequired[float]
    invert_speed_observation: NotRequired[bool]
    speed_observation_threshold: NotRequired[float]
    punish_inhomogeneities: NotRequired[bool]
    speed_gradient_reward: NotRequired[bool]
    speed_gradient_linearity: NotRequired[float]
    inh_rew_idx: NotRequired[int]
    forward_reward: NotRequired[float]
    binary_speeds: NotRequired[bool]
    choices: NotRequired[int]
    inflate_speeds: NotRequired[bool]


class GridEnvInterface(ABC):
    @abstractmethod
    def __init__(self, render_mode: str = None,
                 length: int = 64,
                 width: int = 16,
                 moves_per_timestep: int = 5,
                 window_height: int = 256,
                 observation_distance: int = 3,
                 initial_state: np.ndarray = None,
                 initial_state_template: str = None,
                 distinguishable_particles: bool = False,
                 use_speeds: bool = False,
                 sigma: float = None,
                 average_window: int = 1000,
                 allow_wait: bool = False,
                 social_reward: bool = None,
                 density: float = 0.5,
                 invert_speed_observation: bool = False,
                 speed_observation_threshold: float = 0.35,
                 punish_inhomogeneities: bool = False,
                 speed_gradient_reward: bool = False,
                 speed_gradient_linearity: float = 0.1,
                 inh_rew_idx: int = -1,
                 forward_reward: float = 1.5,
                 binary_speeds: bool = False,
                 choices: int = 2,
                 inflate_speeds: bool = False):
        """
        The GridEnvironment class implements a grid environment with particles that can move forward, up,
        down or wait. It is a 2D version of the TASEP (Totally Asymmetric Simple Exclusion Process) model for use
        with smarticles (smart particles). The environment is implemented as a gym environment. The reward structure
        is as follows:
            - +1 for moving forward
            - 0 for moving up or down
            - 0 for waiting
            - -1 for trying to move into an occupied cell
            - -social_reward for moving into a cell with a particle behind it (only if social_reward is specified)

        Args:
            render_mode (str, optional): The mode in which the environment is rendered. Defaults to None. Can be "human"
                or "rgb_array".
            length (int, optional): The length of the grid. Defaults to 64.
            width (int, optional): The number of "lanes". Defaults to 16.
            moves_per_timestep (int, optional): The number of moves per timestep. Defaults to 5.
            window_height (int, optional): The height of the PyGame window. Defaults to 256.
            observation_distance (int, optional): The agent's observation radius. Defaults to 3.
            initial_state (np.ndarray, optional): The initial state of the grid. Defaults to None.
            initial_state_template (np.ndarray, optional): The template for the initial state of the grid. Defaults to
                None. Can be "checkerboard" or "everyThird".
            distinguishable_particles (bool, optional): Whether the particles are distinguishable. Defaults to False.
                If True, a transition is stored when after same agent is picked again. s' then includes the movements
                of the other agents.
            use_speeds (bool, optional): Whether agents should have different speeds. Defaults to False.
            sigma (float, optional): The standard deviation of the truncated normal distribution to draw speeds from.
                Defaults to None.
            average_window (int, optional): The size of the time averaging period. Defaults to 1000.
            allow_wait (bool, optional): Whether to allow the agents to wait. Defaults to False.
            social_reward (float, optional): If specified, agents get a negative reward for moving into a cell with a
                particle behind it. When using speeds, the reward is scaled the speed of the particle behind.
                Defaults to None.
            density (float, optional): The density of the grid. Defaults to 0.5. Used for random initial states when
                initial_state and initial_state_template are None.
            invert_speed_observation (bool, optional): If True, higher speeds are represented by lower values in the
                observation. Defaults to False.
            speed_observation_threshold (float, optional): The value that a particle with speed 1 should have in the
                observation. Defaults to 0.35.
            punish_inhomogeneities (bool, optional): Whether to punish speed inhomogeneity in the observation.
                Defaults to False.
            speed_gradient_reward (bool, optional): Whether to encourage a vertical speed gradient in the system.
            speed_gradient_linearity (float, optional): The linearity of the speed gradient reward. Defaults to 0.1.
            inh_rew_idx (int, optional): The index of the reward formula that should be used for the inhomogeneity reward
            forward_reward (float, optional): The reward for moving forward. Defaults to 1.5.
            binary_speeds: Whether to sample speeds from a binary distribution instead of a truncated normal distribution
            choices: How many options to choose from when sampling speeds from a "binary" distribution
            inflate_speeds: Whether to map the speeds in the observations to a range of 0 to 1
        """
        pass

    @abstractmethod
    def _setup_spec(self):
        pass

    @abstractmethod
    def _get_obs(self, new_mover: bool = True) -> np.ndarray:
        """
        Returns a new observation.
        Args:
            new_mover (bool, optional): Whether to select a new agent or not. Defaults to True.
        """
        pass

    @property
    @abstractmethod
    def n(self) -> int:
        """
        Returns the number of particles in the system.
        """
        pass

    @property
    @abstractmethod
    def rho(self) -> float:
        """
        Returns the density of particles in the system.
        """
        pass

    @abstractmethod
    def _get_current(self) -> float:
        """
        Returns the current through the system averaged over the last `self.average_window` timesteps.
        """
        pass

    @abstractmethod
    def _get_info(self) -> dict[str, float]:
        """
        Returns a dictionary with information about the current state of the environment.

        Returns:
            dict: A dictionary containing the following keys:
                - "current": The current state of the environment.
        """
        pass

    @abstractmethod
    def render(self) -> Optional[np.ndarray]:
        """
        Renders the current state of the grid environment if render_mode is "rgb_array".

        Returns:
            If render_mode is "rgb_array", returns a numpy array representing the rendered frame.
        """
        pass

    @abstractmethod
    def _render_frame(self) -> Optional[np.ndarray]:
        """
        Renders the current state of the environment as an RGB array or on a PyGame window.

        Returns:
            If render_mode is "rgb_array", returns an RGB array of the current state of the environment.
            If render_mode is "human", renders the current state of the environment on a PyGame window.
        """
        pass

    @abstractmethod
    def reset(self, seed: int = None, options: dict = None, random_density: bool = False) -> tuple[
        WholeObsType, dict[str, Any]]:
        pass

    @abstractmethod
    def _move_if_possible(self, position: tuple) -> bool:
        """
        Moves the agent to the specified position if possible.
        Returns:
            bool: True if the agent moved, False otherwise.
        """
        pass

    @abstractmethod
    def step(self, action: int) -> tuple[WholeObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        pass

    @abstractmethod
    def _update_current(self):
        """
        Updates the current if the average window time is greater than the average window.
        Note that time averaging only equals ensemble averaging if the steady state has been reached.
        """
        pass

    @abstractmethod
    def _update_current_initial(self):
        """
        Updates the current if the avg_window_time has not yet been reached. Note that this is only used for convenience.
        Time averaging only equals ensemble averaging if the steady state has been reached.
        """
        pass

    @abstractmethod
    def _perform_action(self, action: int) -> int | float:
        """
        Performs the specified action and returns the reward.
        Args:
            action (int): The action to perform.

        Returns:
            int | float: The reward for the action.
        """
        pass

    @abstractmethod
    def _calculate_social_reward(self, row: int) -> float:
        """
        Calculates the social reward for moving into the specified row/lane.
        Args:
            row: The row/lane to calculate the social reward for.
        Returns:
            float: The social reward for moving into the specified row/lane.
        """
        pass

    @abstractmethod
    def _calc_inhomo_reward(self, obs: np.ndarray) -> float:
        """
        For each particle in the observation, calculate the absolute difference in speed between that particle and the
        particle in the center, divided by the distance between the particle and the center. The reward is the
        negative sum of these values.
        """
        pass

    @abstractmethod
    def _speed_gradient(self, x: float) -> float:
        """
        Maps speed to their scaling factor.
        Args:
            x: The speed value to calculate the gradient for.
        """
        pass

    @abstractmethod
    def _calculate_speed_gradient_reward(self) -> float:
        """
        Calculates the reward for the speed gradient.
        Returns:
            float: The reward for the speed gradient.
        """
        pass

    @abstractmethod
    def _move_forward(self) -> int:
        """
        Moves the agent forward if possible and return the reward.
        Returns:
            int: The reward for moving forward. 1 if the agent moved, -1 otherwise.
        """
        pass

    @abstractmethod
    def _move_up_down(self, x_pos) -> int:
        """
        Moves the agent up or down if possible and return the reward.
        Args:
            x_pos: The row/lane to move to.
        Returns:
            int: The reward for moving up or down. 0 if the agent moved, -1 otherwise.
        """
        pass

    @abstractmethod
    def _render_if_human(self):
        """
        Renders the environment if render_mode is "human" and the number of timesteps is a multiple of
        moves_per_timestep.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Closes the environment.
        """
        pass
