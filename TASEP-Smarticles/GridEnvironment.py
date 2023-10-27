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


def truncated_normal_single(mean, std_dev):
    sample = -1
    while sample < 0 or sample > 1:
        sample = np.random.normal(mean, std_dev)
    return sample


def truncated_normal(mean, std_dev, size):
    samples = np.zeros(size, dtype=np.float32)
    for i in range(size):
        samples[i] = truncated_normal_single(mean, std_dev)
    return samples


class GridEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 144,
                "initial_state_templates": ["checkerboard", "everyThird"]}

    def __init__(self, render_mode=None, length=64, width=16, moves_per_timestep=5, window_height=256,
                 observation_distance=3, initial_state=None, initial_state_template=None,
                 distinguishable_particles=False, use_speeds=False, sigma=None):
        self.state: Optional[np.ndarray[np.uint8 | np.int32]] = None
        self.current_mover: Optional[np.ndarray] = None
        self.length = length  # The length of the grid
        self.width = width  # The number of "lanes"
        self.window_height = window_height  # The height of the PyGame window
        self.window_width = self.window_height * self.length / self.width  # The width of the PyGame window
        self.total_forward = 0
        self.timesteps = 0
        self.moves_per_timestep = moves_per_timestep
        self.distinguishable_particles = distinguishable_particles
        self.use_speeds = use_speeds
        if self.use_speeds:
            assert sigma is not None, "sigma must be specified if use_speeds is True"
            self.sigma = sigma
        assert initial_state_template is None or initial_state_template in self.metadata[
            "initial_state_templates"], "initial_state_template must be None, 'checkerboard', or 'everyThird'"
        self.initial_state_template = initial_state_template

        assert initial_state is None or initial_state.shape == (
            self.width, self.length), "initial_state must be None or have shape (width, length)"
        self.initial_state = initial_state

        # The agent perceives part of the surrounding grid, it has a square view
        # with viewing distance `self.observation_size`
        self.observation_distance = observation_distance

        # Each grid cell can be either empty or occupied by a particle
        # TODO: Specify the correct observation space for distinguishable particle
        if self.use_speeds and self.distinguishable_particles:
            self.internal_dtype = np.float32
            single_obs = spaces.Box(
                low=0, high=2 ** 20, shape=((self.observation_distance * 2 + 1) ** 2,), dtype=self.internal_dtype
            )
            self.observation_space: spaces.Tuple[ObsType,] = spaces.Tuple(
                (single_obs, spaces.Box(low=0, high=1, shape=(1,), dtype=self.internal_dtype)))
        elif distinguishable_particles and not self.use_speeds:
            self.internal_dtype = np.int32
            single_obs = spaces.Box(
                low=0, high=2 ** 20, shape=((self.observation_distance * 2 + 1) ** 2,), dtype=self.internal_dtype
            )
            self.observation_space: spaces.Tuple[ObsType,] = spaces.Tuple(
                (single_obs, spaces.Discrete(2 ** 20)))
        elif not distinguishable_particles and self.use_speeds:
            self.internal_dtype = np.float32
            single_obs = spaces.Box(
                low=0, high=1, shape=((self.observation_distance * 2 + 1) ** 2,), dtype=self.internal_dtype
            )
            self.observation_space: spaces.Tuple[ObsType,] = spaces.Tuple(
                (single_obs, spaces.Box(low=0, high=1, shape=(1,), dtype=self.internal_dtype)))
        else:
            self.internal_dtype = np.uint8
            single_obs = spaces.Box(
                low=0, high=1, shape=((self.observation_distance * 2 + 1) ** 2,), dtype=self.internal_dtype
            )
            self.observation_space: spaces.Tuple[ObsType, ObsType] = spaces.Tuple((single_obs, single_obs))

        # We have 4 actions, corresponding to "forward", "up", "down"
        self.action_space: spaces.Discrete = spaces.Discrete(3)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # If human-rendering is used, `self.window` will be a reference
        # to the window that we draw to. `self.clock` will be a clock that is used
        # to ensure that the environment is rendered at the correct framerate in
        # human-mode. They will remain `None` until human-mode is used for the
        # first time.
        self.window = None
        self.clock = None

    def _get_obs(self, new_mover=True):
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
            range(self.current_mover[0] - self.observation_distance,
                  self.current_mover[0] + self.observation_distance + 1),
            mode="wrap",
            axis=0,
        ).take(
            range(self.current_mover[1] - self.observation_distance,
                  self.current_mover[1] + self.observation_distance + 1),
            mode="wrap",
            axis=1,
        )
        # TODO: Allow agents to observe the speeds of the particles
        if self.distinguishable_particles or self.use_speeds:
            obs[obs != 0] = 1
        return obs.flatten()

    @property
    def n(self):
        # Return the number of particles in the system
        return np.count_nonzero(self.state)

    @property
    def rho(self):
        # Return the density of particles in the system
        return self.n / (self.length * self.width)

    def _get_current(self):
        # Return the current of the system
        # TODO: Implement averaging over n moves insead of the whole episode
        return self.total_forward / self.timesteps if self.timesteps > 0 else 0

    def _get_info(self):
        # Return dict with information about the current state of the environment
        return {
            # "state": self.state,
            # "N": self.n,
            # "rho": self.rho,
            "current": self._get_current(),
        }

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_width, self.window_height)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        pix_square_size = (
                self.window_height / self.width
        )

        # Draw the state array
        # First stack the three color channels
        if self.distinguishable_particles:
            if self.use_speeds:
                h_array = self.state.T % 1
            else:
                h_array = self.state.T / self.n
            s_array = np.full(self.state.T.shape, 0.82)
            s_array[self.state.T == 0] = 0
            l_array = np.full(self.state.T.shape, 0.56)
            l_array[self.state.T == 0] = 1
            hsl_array = np.stack([h_array, s_array, l_array], axis=2)
            rgb_array = hsl2rgb(hsl_array, self.internal_dtype)
        elif self.use_speeds and not self.distinguishable_particles:
            h_array = self.state.T % 1
            s_array = np.full(self.state.T.shape, 0.82)
            s_array[self.state.T == 0] = 0
            l_array = np.full(self.state.T.shape, 0.56)
            l_array[self.state.T == 0] = 1
            hsl_array = np.stack([h_array, s_array, l_array], axis=2)
            rgb_array = hsl2rgb(hsl_array, self.internal_dtype)
        else:
            rgb_array = np.stack([self.state.T * 245, self.state.T * 66, self.state.T * 69], axis=2)
            rgb_array[rgb_array == 0] = 255  # correct white background

        if self.render_mode == "human":
            # Create a PyGame surface from the array and scale it to the correct size
            state_surf = pygame.surfarray.make_surface(rgb_array)
            state_surf = pygame.transform.scale(state_surf,
                                                (self.window_width, self.window_height))
            # Add gridlines
            for i in range(self.width):
                pygame.draw.line(state_surf, (0, 0, 0), (0, i * pix_square_size),
                                 (self.window_width, i * pix_square_size))
            for i in range(self.length):
                pygame.draw.line(state_surf, (0, 0, 0), (i * pix_square_size, 0),
                                 (i * pix_square_size, self.window_height))

            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(state_surf, state_surf.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return rgb_array

    def reset(self, seed=None, options=None) -> tuple[WholeObsType, dict[str, Any]]:
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.total_forward = 0
        self.timesteps = 0

        # Set the initial state of the environment
        # If no initial state is given, we set the initial state to be a checkerboard
        if self.initial_state is None:
            if self.initial_state_template == "checkerboard":
                self.state = np.indices((self.width, self.length)).sum(axis=0, dtype=self.internal_dtype) % 2
            elif self.initial_state_template == "everyThird":
                self.state = np.zeros((self.width, self.length), dtype=self.internal_dtype)
                self.state[::3, ::3] = 1
            else:
                self.state = self.np_random.integers(2, size=(self.width, self.length), dtype=self.internal_dtype)
        else:
            self.state = self.initial_state
        if self.distinguishable_particles:
            random_integers = np.arange(1, self.n + 1, dtype=self.internal_dtype)
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
        if self.state[*position] == 0:  # if the next cell is empty, move
            self.state[*self.current_mover], self.state[*position] = self.state[*position], self.state[
                *self.current_mover]
            return True
        else:  # if the next cell is occupied, don't move
            return False

    def step(self, action) -> tuple[WholeObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # Move the agent in the specified direction if possible.
        # If the agent is at the boundary of the grid, it will wrap around
        reward = 0
        self.timesteps += 1
        if action == 0:  # forward
            # TODO: Clean this up, remove duplicate code
            if self.use_speeds:
                # check if random dice is smaller than speed
                dice = self.np_random.random()
                if self.distinguishable_particles:
                    if dice < self.state[*self.current_mover] % 1:
                        next_y = 0 if self.current_mover[1] == self.length - 1 else self.current_mover[1] + 1
                        has_moved = self._move_if_possible((self.current_mover[0], next_y))
                        if not has_moved:
                            reward = -1
                        else:
                            reward = 1
                            self.total_forward += 1
                else:
                    if dice < self.state[*self.current_mover]:
                        next_y = 0 if self.current_mover[1] == self.length - 1 else self.current_mover[1] + 1
                        has_moved = self._move_if_possible((self.current_mover[0], next_y))
                        if not has_moved:
                            reward = -1
                        else:
                            reward = 1
                            self.total_forward += 1
            else:
                next_y = 0 if self.current_mover[1] == self.length - 1 else self.current_mover[1] + 1
                has_moved = self._move_if_possible((self.current_mover[0], next_y))
                if not has_moved:
                    reward = -1
                else:
                    reward = 1
                    self.total_forward += 1
        else:  # up or down
            if action == 1:  # up
                above = self.width - 1 if self.current_mover[0] == 0 else self.current_mover[0] - 1
                has_moved = self._move_if_possible((above, self.current_mover[1]))
                if not has_moved:
                    reward = -1
            else:  # down
                below = 0 if self.current_mover[0] == self.width - 1 else self.current_mover[0] + 1
                has_moved = self._move_if_possible((below, self.current_mover[1]))
                if not has_moved:
                    reward = -1

        if not self.distinguishable_particles:
            react_observation = self._get_obs(new_mover=False)
        info = self._get_info()
        next_observation = self._get_obs(new_mover=True)

        if self.render_mode == "human" and self.timesteps % self.moves_per_timestep == 0:
            self._render_frame()
        if self.distinguishable_particles:
            return (next_observation, int(self.state[*self.current_mover])), reward, False, False, info
        return (react_observation, next_observation), reward, False, False, info

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
