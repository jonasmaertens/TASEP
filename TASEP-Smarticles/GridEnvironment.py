import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

from typing import SupportsFloat, TypeVar, Any

ObsType = TypeVar("ObsType")


class GridEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 144,
                "initial_state_templates": ["checkerboard", "everyThird"]}

    def __init__(self, render_mode=None, length=64, width=16, moves_per_timestep=5, window_height=256,
                 observation_distance=3, initial_state=None, initial_state_template=None):
        self.state: np.ndarray[np.uint8] = None
        self.current_mover = None
        self.length = length  # The length of the grid
        self.width = width  # The number of "lanes"
        self.window_height = window_height  # The height of the PyGame window
        self.window_width = self.window_height * self.length / self.width  # The width of the PyGame window
        self.total_forward = 0
        self.timesteps = 0
        self.moves_per_timestep = moves_per_timestep
        assert initial_state_template is None or initial_state_template in self.metadata[
            "initial_state_templates"], "initial_state_template must be None, 'checkerboard', or 'everyThird'"
        self.initial_state_template = initial_state_template

        assert initial_state is None or initial_state.shape == (self.width, self.length), "initial_state must be None or have shape (width, length)"
        self.initial_state = initial_state

        # The agent perceives part of the surrounding grid, it has a square view
        # with viewing distance `self.observation_size`
        self.observation_distance = observation_distance
        # Each grid cell can be either empty or occupied by a particle
        single_obs = spaces.Box(
            low=0, high=1, shape=((self.observation_distance * 2 + 1) ** 2,), dtype=np.uint8
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
            agent_indices = np.argwhere(self.state == 1)
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
        return obs.flatten()

    @property
    def n(self):
        # Return the number of particles in the system
        return np.sum(self.state)

    @property
    def rho(self):
        # Return the density of particles in the system
        return self.n / (self.length * self.width)

    def _get_info(self):
        # Return dict with information about the current state of the environment
        return {
            "state": self.state,
            "N": self.n,
            "rho": self.rho,
            "current": self.total_forward / self.timesteps if self.timesteps > 0 else 0,
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
        rgb_array = np.stack([self.state.T * 252, self.state.T * 172, self.state.T * 33], axis=2)
        # Then, replace all 0s with 255s ==> white background
        rgb_array[rgb_array == 0] = 255
        # Finally, create a PyGame surface from the array and scale it to the correct size
        state_surf = pygame.surfarray.make_surface(rgb_array)
        state_surf = pygame.transform.scale(state_surf,
                                            (self.window_width, self.window_height))
        # Finally, add some gridlines
        for i in range(self.width):
            pygame.draw.line(state_surf, (0, 0, 0), (0, i * pix_square_size), (self.window_width, i * pix_square_size))
        for i in range(self.length):
            pygame.draw.line(state_surf, (0, 0, 0), (i * pix_square_size, 0), (i * pix_square_size, self.window_height))

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(state_surf, state_surf.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return pygame.surfarray.array3d(state_surf)

    def reset(self, seed=None, options=None) -> tuple[tuple[ObsType, ObsType], dict[str, Any]]:
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.total_forward = 0
        self.timesteps = 0

        # Set the initial state of the environment
        # If no initial state is given, we set the initial state to be a checkerboard
        if self.initial_state is None:
            if self.initial_state_template == "checkerboard":
                self.state = np.indices((self.width, self.length)).sum(axis=0, dtype=np.uint8) % 2
            elif self.initial_state_template == "everyThird":
                self.state = np.zeros((self.width, self.length), dtype=np.uint8)
                self.state[::3, ::3] = 1
            else:
                self.state = self.np_random.integers(2, size=(self.width, self.length), dtype=np.uint8)
        else:
            self.state = self.initial_state

        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()
        return (observation, observation), info

    def _move_if_possible(self, position: tuple) -> bool:
        if self.state[*position] == 0:  # if the next cell is empty, move
            self.state[self.current_mover[0], self.current_mover[1]] = 0
            self.state[*position] = 1
            return True
        else:  # if the next cell is occupied, don't move
            return False

    def step(self, action) -> tuple[tuple[ObsType, ObsType], SupportsFloat, bool, bool, dict[str, Any]]:
        # Move the agent in the specified direction if possible.
        # If the agent is at the boundary of the grid, it will wrap around
        reward = 0
        self.timesteps += 1
        if action == 0:  # forward
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

        react_observation = self._get_obs(new_mover=False)
        info = self._get_info()
        next_observation = self._get_obs(new_mover=True)

        if self.render_mode == "human" and self.timesteps % self.moves_per_timestep == 0:
            self._render_frame()

        return (react_observation, next_observation), reward, False, False, info

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
