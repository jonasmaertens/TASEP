import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces
from Hasel import hsl2rgb

from typing import SupportsFloat, TypeVar, Any, TypedDict, Optional, TypeAlias, NotRequired

ObsType = TypeVar("ObsType")
WholeObsType: TypeAlias = tuple[ObsType, ObsType] | tuple[ObsType, int]


class EnvParams(TypedDict):
    """
    Environment Params for GridEnvironment
    Attributes:
        render_mode: The mode to render the environment in, either "human" or "rgb_array"
        length: The length of the grid
        width: The number of "lanes"
        moves_per_timestep: The number of moves per timestep
        window_height: The height of the PyGame window
        observation_distance: The distance from the agent to the edge of the observation square
        initial_state_template: The template to use for the initial state, either "checkerboard" or "everyThird"
        distinguishable_particles: Whether particles should be distinguishable or not
        use_speeds: Whether to use speeds or not
        sigma: The standard deviation of the normal distribution to draw speeds from
        average_window: The number of timesteps to average over for the current. If None, the whole episode is averaged
        allow_wait: Whether to allow the agent to wait or not
        social_reward: The absolute value of the negative reward for moving into a cell with a particle behind it
        density: The density of particles in the system [0, 1]

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


def truncated_normal_single(mean, std_dev) -> float:
    """
    Generates a single sample from a truncated normal distribution with the given mean and standard deviation.
    The sample is truncated to the range [0, 1].

    Args:
        mean (float): The mean of the normal distribution.
        std_dev (float): The standard deviation of the normal distribution.

    Returns:
        float: A single sample from the truncated normal distribution.
    """
    sample = -1
    while sample < 0 or sample > 1:
        sample = np.random.normal(mean, std_dev)
    return sample


def truncated_normal(mean, std_dev, size) -> np.ndarray:
    """
    Generates an array of random numbers from a truncated normal distribution.

    Args:
        mean (float): Mean of the normal distribution.
        std_dev (float): Standard deviation of the normal distribution.
        size (int): Number of samples to generate.

    Returns:
        np.ndarray: Array of random numbers from the truncated normal distribution.
    """
    samples = np.zeros(size, dtype=np.float32)
    for i in range(size):
        samples[i] = truncated_normal_single(mean, std_dev)
    return samples


class GridEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 144,
                "initial_state_templates": ["checkerboard", "everyThird"]}

    def __init__(self, render_mode: str = None, length: int = 64, width: int = 16, moves_per_timestep: int = 5,
                 window_height: int = 256,
                 observation_distance: int = 3, initial_state: np.ndarray = None, initial_state_template: str = None,
                 distinguishable_particles: bool = False, use_speeds: bool = False, sigma: float = None,
                 average_window: int = 1000,
                 allow_wait: bool = False,
                 social_reward: bool = None, density: float = 0.5):
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
            render_mode (str, optional): The mode in which the environment is rendered. Defaults to None. Can be "human" or "rgb_array".
            length (int, optional): The length of the grid. Defaults to 64.
            width (int, optional): The number of "lanes". Defaults to 16.
            moves_per_timestep (int, optional): The number of moves per timestep. Defaults to 5.
            window_height (int, optional): The height of the PyGame window. Defaults to 256.
            observation_distance (int, optional): The agent's observation radius. Defaults to 3.
            initial_state (np.ndarray, optional): The initial state of the grid. Defaults to None.
            initial_state_template (np.ndarray, optional): The template for the initial state of the grid. Defaults to None. Can be "checkerboard" or "everyThird".
            distinguishable_particles (bool, optional): Whether the particles are distinguishable. Defaults to False. If True, a transition is stored when after same agent is picked again. s' then includes the movements of the other agents.
            use_speeds (bool, optional): Whether agents should have different speeds. Defaults to False.
            sigma (float, optional): The standard deviation of the truncated normal distribution to draw speeds from. Defaults to None.
            average_window (int, optional): The size of the time averaging period. Defaults to 1000.
            allow_wait (bool, optional): Whether to allow the agents to wait. Defaults to False.
            social_reward (float, optional): If specified, agents get a negative reward for moving into a cell with a particle behind it. When using speeds, the reward is the speed of the particle behind divided by 2. Defaults to None.
            density (float, optional): The density of the grid. Defaults to 0.5. Used for random initial states when initial_state and initial_state_template are None.
        """
        self.state: Optional[np.ndarray[np.uint8 | np.int32]] = None
        self.social_reward = social_reward
        self.density = density
        self.current_mover: Optional[np.ndarray] = None
        self.allow_wait = allow_wait
        self.length = length if initial_state is None else initial_state.shape[1]  # The length of the grid
        self.width = width if initial_state is None else initial_state.shape[0]  # The number of "lanes"
        self.window_height = window_height  # The height of the PyGame window
        self.window_width = self.window_height * self.length / self.width  # The width of the PyGame window
        self.pix_square_size = (self.window_height / self.width)
        self.total_forward = 0
        self.total_timesteps = 0
        self.average_window = average_window
        self.avg_window_time = 0
        self.avg_window_forward = 0
        self._current = 0
        self.moves_per_timestep = moves_per_timestep
        self.distinguishable_particles = distinguishable_particles
        self.use_speeds = use_speeds
        if self.use_speeds:
            self.distinguishable_particles = True
            if sigma is None:
                raise ValueError("sigma must be specified if use_speeds is True")
        self.sigma = sigma
        self.initial_state_template = initial_state_template
        self.initial_state = initial_state

        # The agent perceives part of the surrounding grid, it has a square view
        # with viewing distance `self.observation_size`
        self.obs_dist = observation_distance

        # With distinguishable particles, the observation is a tuple of the binary
        # observation space and the number of the agent
        if self.distinguishable_particles:
            self.use_dtype = np.float32 if self.use_speeds else np.int32
            single_obs = spaces.Box(low=0, high=1, shape=((self.obs_dist * 2 + 1) ** 2,), dtype=self.use_dtype)
            self.observation_space: spaces.Tuple[ObsType,] = spaces.Tuple(
                (single_obs, spaces.Discrete(self.length * self.width)))
        elif self.use_speeds:  # binary observation space -> speed observation space
            self.use_dtype = np.float32
            single_obs = spaces.Box(low=0, high=1, shape=((self.obs_dist * 2 + 1) ** 2,), dtype=self.use_dtype)
            self.observation_space: spaces.Tuple[ObsType,] = spaces.Tuple(
                (single_obs, spaces.Box(low=0, high=1, shape=(1,), dtype=self.use_dtype)))
        else:
            self.use_dtype = np.uint8
            single_obs = spaces.Box(low=0, high=1, shape=((self.obs_dist * 2 + 1) ** 2,), dtype=self.use_dtype)
            self.observation_space: spaces.Tuple[ObsType, ObsType] = spaces.Tuple((single_obs, single_obs))

        # We have 3 actions, corresponding to "forward", "up", "down"
        if self.allow_wait:
            self.action_space: spaces.Discrete = spaces.Discrete(4)
        else:
            self.action_space: spaces.Discrete = spaces.Discrete(3)

        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode must be one of {self.metadata['render_modes']}")
        self.render_mode = render_mode

        # If human-rendering is used, `self.window` will be a reference
        # to the window that we draw to. `self.clock` will be a clock that is used
        # to ensure that the environment is rendered at the correct framerate in
        # human-mode. They will remain `None` until human-mode is used for the
        # first time.
        self.window = None
        self.clock = None

    def _get_obs(self, new_mover: bool = True) -> np.ndarray:
        """
        Returns a new observation.
        Args:
            new_mover (bool, optional): Whether to select a new agent or not. Defaults to True.
        """
        # Select a random agent (a grid cell that is currently set to 1) and return
        # the observation of the grid around it
        if new_mover:
            agent_indices = np.argwhere(self.state != 0)
            self.current_mover = agent_indices[self.np_random.integers(len(agent_indices))]
        # The observation is a square of size `self.observation_distance * 2 + 1`
        # centered around the agent. NumPy's `take` function is used to select
        # the correct part of the grid, and the `mode="wrap"` argument ensures
        # periodic boundary conditions.
        obs = self.state.take(
            range(self.current_mover[0] - self.obs_dist,
                  self.current_mover[0] + self.obs_dist + 1),
            mode="wrap",
            axis=0,
        ).take(
            range(self.current_mover[1] - self.obs_dist,
                  self.current_mover[1] + self.obs_dist + 1),
            mode="wrap",
            axis=1,
        )
        if self.distinguishable_particles and not self.use_speeds:
            obs[obs != 0] = 1
        if self.use_speeds:
            obs[obs != 0] = obs[obs != 0] % 1
        return obs.flatten()

    @property
    def n(self) -> int:
        """
        Returns the number of particles in the system.
        """
        return np.count_nonzero(self.state)

    @property
    def rho(self) -> float:
        """
        Returns the density of particles in the system.
        """
        return self.n / (self.length * self.width)

    def _get_current(self) -> float:
        """
        Returns the current through the system averaged over the last `self.average_window` timesteps.
        """
        if self.average_window is None:
            return self.total_forward / self.total_timesteps if self.total_timesteps > 0 else 0
        return self._current

    def _get_info(self) -> dict[str, float]:
        """
        Returns a dictionary with information about the current state of the environment.

        Returns:
            dict: A dictionary containing the following keys:
                - "current": The current state of the environment.
        """
        return {
            # "state": self.state,
            # "N": self.n,
            # "rho": self.rho,
            "current": self._get_current(),
        }

    def render(self) -> Optional[np.ndarray]:
        """
        Renders the current state of the grid environment if render_mode is "rgb_array".

        Returns:
            If render_mode is "rgb_array", returns a numpy array representing the rendered frame.
        """
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self) -> Optional[np.ndarray]:
        """
        Renders the current state of the environment as an RGB array or on a PyGame window.

        Returns:
            If render_mode is "rgb_array", returns an RGB array of the current state of the environment.
            If render_mode is "human", renders the current state of the environment on a PyGame window.
        """
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Draw the state array. For mapping the state array to colors, we use the
        # HSL color space. The hue is determined by the value of the state array,
        # the saturation is set to 1, and the lightness is set to 0.5.
        if self.distinguishable_particles or self.use_speeds:
            if self.use_speeds:  # use only 1 to 120° of the hue spectrum (red to green) depending on speed
                h_array = (self.state.T % 1) / 3
            else:  # use the whole hue spectrum -> random color for each particle
                h_array = self.state.T / self.n
            s_array = np.full(self.state.T.shape, 1)
            s_array[self.state.T == 0] = 0
            l_array = np.full(self.state.T.shape, 0.5)
            l_array[self.state.T == 0] = 0.16
            hsl_array = np.stack([h_array, s_array, l_array], axis=2)
            rgb_array = hsl2rgb(hsl_array, self.use_dtype)
        else:  # binary observation space -> one fixed color for all particles on dark background
            rgb_array = np.stack([self.state.T * 245, self.state.T * 66, self.state.T * 69], axis=2)
            rgb_array[rgb_array == 0] = 255  # correct white background

        if self.render_mode == "human":
            # Create a PyGame surface from the array and scale it to the correct size
            state_surf = pygame.surfarray.make_surface(rgb_array)
            state_surf = pygame.transform.scale(state_surf,
                                                (self.window_width, self.window_height))
            # Add gridlines
            for i in range(self.width):
                pygame.draw.line(state_surf, (0, 0, 0), (0, i * self.pix_square_size),
                                 (self.window_width, i * self.pix_square_size))
            for i in range(self.length):
                pygame.draw.line(state_surf, (0, 0, 0), (i * self.pix_square_size, 0),
                                 (i * self.pix_square_size, self.window_height))

            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(state_surf, state_surf.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return rgb_array

    def reset(self, seed: int = None, options: dict = None, random_density: bool = False) -> tuple[
        WholeObsType, dict[str, Any]]:
        # We need the following line to seed self.np_random (for reproducibility)
        super().reset(seed=seed)

        self.total_forward = 0
        self.total_timesteps = 0
        self.avg_window_time = 0
        self.avg_window_forward = 0

        # Set the initial state of the environment
        # If no initial state is given, we set the initial state to be a checkerboard
        if self.initial_state is None:
            if self.initial_state_template == "checkerboard":
                self.state = np.indices((self.width, self.length)).sum(axis=0, dtype=self.use_dtype) % 2
            elif self.initial_state_template == "everyThird":
                self.state = np.zeros((self.width, self.length), dtype=self.use_dtype)
                self.state[::3, ::3] = 1
            else:
                self.density = self.density if not random_density else self.np_random.uniform(0.02, 0.7)
                self.state = self.np_random.random(size=(self.width, self.length), dtype=np.float32)
                self.state[self.state < self.density] = 1
                self.state[self.state != 1] = 0
                self.state = self.state.astype(self.use_dtype)
        else:
            self.state = self.initial_state
        if self.distinguishable_particles:
            random_integers = np.arange(1, self.n + 1, dtype=self.use_dtype)
            self.np_random.shuffle(random_integers)
            self.state[self.state == 1] = random_integers
            if self.use_speeds:
                random_speeds = truncated_normal(0.5, self.sigma, self.n)
                self.state[self.state != 0] += random_speeds
        elif not self.distinguishable_particles and self.use_speeds:
            random_speeds = truncated_normal(0.5, self.sigma, self.n)
            self.state[self.state == 1] = random_speeds
        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()
        if self.distinguishable_particles:
            return (observation, int(self.state[*self.current_mover])), info
        return (observation, observation), info

    def _move_if_possible(self, position: tuple) -> bool:
        """
        Moves the agent to the specified position if possible.
        Returns:
            bool: True if the agent moved, False otherwise.
        """
        if self.state[*position] == 0:  # if the next cell is empty, move
            self.state[*self.current_mover], self.state[*position] = self.state[*position], self.state[
                *self.current_mover]
            return True
        else:  # if the next cell is occupied, don't move
            return False

    def step(self, action: int) -> tuple[WholeObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # Move the agent in the specified direction if possible.
        # If the agent is at the boundary of the grid, it will wrap around
        reward = 0
        self.total_timesteps += 1
        self.avg_window_time += 1
        self._update_current()
        self._update_current_initial()
        if not self.use_speeds or self.np_random.random() < self.state[*self.current_mover] % 1:
            reward = self._perform_action(action)
        if not self.distinguishable_particles:
            react_observation = self._get_obs(new_mover=False)
        info = self._get_info()
        next_observation = self._get_obs(new_mover=True)
        self._render_if_human()
        if self.distinguishable_particles:
            return (next_observation, int(self.state[*self.current_mover])), reward, False, False, info
        return (react_observation, next_observation), reward, False, False, info

    def _update_current(self):
        """
        Updates the current if the average window time is greater than the average window.
        Note that time averaging only equals ensemble averaging if the steady state has been reached.
        """
        if self.avg_window_time >= self.average_window:
            self._current = self.avg_window_forward / self.average_window
            self.avg_window_time = 0
            self.avg_window_forward = 0

    def _update_current_initial(self):
        """
        Updates the current if the avg_window_time has not yet been reached. Note that this is only used for convenience.
        Time averaging only equals ensemble averaging if the steady state has been reached.
        """
        if self.average_window > self.total_timesteps > 50:
            self._current = self.total_forward / self.total_timesteps

    def _perform_action(self, action: int) -> int | float:
        """
        Performs the specified action and returns the reward.
        Args:
            action (int): The action to perform.

        Returns:
            int | float: The reward for the action.
        """
        if action == 0:  # forward
            reward = self._move_forward()
        elif action == 1:  # up
            above = self.width - 1 if self.current_mover[0] == 0 else self.current_mover[0] - 1
            reward = self._move_up_down(above)
            if self.social_reward:
                reward += self._calculate_social_reward(above)
        elif action == 2:  # down
            below = 0 if self.current_mover[0] == self.width - 1 else self.current_mover[0] + 1
            reward = self._move_up_down(below)
            if self.social_reward:
                reward += self._calculate_social_reward(below)
        else:  # wait
            reward = 0
        return reward

    def _calculate_social_reward(self, row: int) -> float:
        """
        Calculates the social reward for moving into the specified row/lane.
        Args:
            row: The row/lane to calculate the social reward for.
        Returns:
            float: The social reward for moving into the specified row/lane.
        """
        prev_x = self.length - 1 if self.current_mover[1] == 0 else self.current_mover[1] - 1
        if self.state[row, prev_x] != 0:
            return -self.social_reward if not self.use_speeds else -(self.state[row, prev_x] % 1) / 2
        return 0

    def _move_forward(self) -> int:
        """
        Moves the agent forward if possible and return the reward.
        Returns:
            int: The reward for moving forward. 1 if the agent moved, -1 otherwise.
        """
        next_x = 0 if self.current_mover[1] == self.length - 1 else self.current_mover[1] + 1
        has_moved = self._move_if_possible((self.current_mover[0], next_x))
        if not has_moved:
            return -1
        else:
            self.total_forward += 1
            self.avg_window_forward += 1
            return 1

    def _move_up_down(self, x_pos) -> int:
        """
        Moves the agent up or down if possible and return the reward.
        Args:
            x_pos: The row/lane to move to.
        Returns:
            int: The reward for moving up or down. 0 if the agent moved, -1 otherwise.
        """
        has_moved = self._move_if_possible((x_pos, self.current_mover[1]))
        if not has_moved:
            return -1
        return 0

    def _render_if_human(self):
        """
        Renders the environment if render_mode is "human" and the number of timesteps is a multiple of
        moves_per_timestep.
        """
        if self.render_mode == "human" and self.total_timesteps % self.moves_per_timestep == 0:
            self._render_frame()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
