import numpy as np
import os
import pygame

import gymnasium as gym
from gymnasium import spaces
from .Hasel import hsl2rgb

from typing import Optional
from .GridEnvironmentInterface import ObsType, GridEnvInterface, EnvParams
from scipy.special import erf

default_env_params = {
    "render_mode": None,
    "length": 64,
    "width": 16,
    "moves_per_timestep": 5,
    "window_height": 256,
    "observation_distance": 3,
    "initial_state": None,
    "initial_state_template": None,
    "distinguishable_particles": False,
    "use_speeds": False,
    "sigma": None,
    "average_window": 1000,
    "allow_wait": False,
    "social_reward": None,
    "density": 0.5,
    "invert_speed_observation": False,
    "speed_observation_threshold": 0.35,
    "punish_inhomogeneities": False,
    "speed_gradient_reward": False,
    "speed_gradient_linearity": 0.1,
    "inh_rew_idx": -1,
    "binary_speeds": False,
    "choices": 2,
    "inflate_speeds": False
}


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


def binary_distribution(size, choices) -> np.ndarray:
    """
    Generates an array of random numbers from a binary distribution.

    Args:
        size (int): Number of samples to generate.
        choices (int): Number of different values to choose from.

    Returns:
        np.ndarray: Array of random numbers from the binary distribution.
    """
    if choices == 2:
        samples = np.random.choice([0.25, 0.95], size=size)
    elif choices == 4:
        samples = np.random.choice([0.18, 0.4, 0.65, 0.95], size=size)
    return samples


def invert_speed_obs(observation: np.ndarray, threshold: float) -> np.ndarray:
    """
    Inverts the speed observation. Higher speeds are represented by lower values in the observation.
    Args:
        observation: The observation to invert.
        threshold: The value that a particle with speed 1 should have in the observation.

    Returns:
        np.ndarray: The inverted observation.
    """
    observation[observation != 0] = 1 - observation[observation != 0] + threshold
    return observation


class GridEnv(gym.Env, GridEnvInterface):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 144,
                "initial_state_templates": ["checkerboard", "everyThird"]}

    def __init__(self, render_mode=None,
                 length=64,
                 width=16,
                 moves_per_timestep=5,
                 window_height=256,
                 observation_distance=3,
                 initial_state=None,
                 initial_state_template=None,
                 distinguishable_particles=False,
                 use_speeds=False,
                 sigma=None,
                 average_window=1000,
                 allow_wait=False,
                 social_reward=None,
                 density=0.5,
                 invert_speed_observation=False,
                 speed_observation_threshold=0.35,
                 punish_inhomogeneities=False,
                 speed_gradient_reward=False,
                 speed_gradient_linearity=0.1,
                 inh_rew_idx=-1,
                 forward_reward=1.5,
                 binary_speeds=False,
                 choices=2,
                 inflate_speeds=False):
        self.state: Optional[np.ndarray[np.uint8 | np.int32]] = None
        self.social_reward = social_reward
        self.forward_reward = forward_reward
        self.binary_speeds = binary_speeds
        self.choices = choices
        self.inflate_speeds = inflate_speeds
        self.punish_inhomogeneities = punish_inhomogeneities
        self.inh_rew_idx = inh_rew_idx
        self.density = density
        self.current_mover: Optional[np.ndarray] = None
        self.allow_wait = allow_wait
        self.length = length if initial_state is None else initial_state.shape[1]  # The length of the grid
        self.width = width if initial_state is None else initial_state.shape[0]  # The number of "lanes"
        self.window_height = window_height  # The height of the PyGame window
        self.window_width = self.window_height * self.length / self.width  # The width of the PyGame window
        # for pygame window
        os.environ['SDL_VIDEO_WINDOW_POS'] = f"0,{int(self.window_width / 2.5) + 70}"
        self.pix_square_size = (self.window_height / self.width)
        self.total_forward = 0
        self.total_timesteps = 0
        self.average_window = average_window
        self.avg_window_time = 0
        self.avg_window_forward = 0
        self.avg_window_reward = 0
        self._current = 0
        self._avg_reward = 0
        self.prepare_lane_gradient = None
        self.moves_per_timestep = moves_per_timestep
        self.distinguishable_particles = distinguishable_particles
        self.use_speeds = use_speeds
        self.speed_gradient_reward = speed_gradient_reward
        self.max_speed = 0.5 + sigma / 3 if self.use_speeds else None
        self.min_speed = 0.5 - sigma / 3 if self.use_speeds else None
        if self.use_speeds:
            if not self.distinguishable_particles:
                raise ValueError("use_speeds can only be True if distinguishable_particles is True")
            if sigma is None:
                raise ValueError("sigma must be specified if use_speeds is True")
        self.sigma = sigma
        self.invert_speed_observation = invert_speed_observation
        self.speed_observation_threshold = speed_observation_threshold
        if self.invert_speed_observation and not self.use_speeds:
            raise ValueError("invert_speed_observation can only be True if use_speeds is True")
        self.initial_state_template = initial_state_template
        self.initial_state = initial_state

        if self.speed_gradient_reward and not self.use_speeds:
            raise ValueError("speed_gradient_reward can only be True if use_speeds is True")

        self.speed_gradient_linearity = speed_gradient_linearity

        # The agent perceives part of the surrounding grid, it has a square view
        # with viewing distance `self.observation_size`
        self.obs_dist = observation_distance

        self._setup_spec()

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

    def _setup_spec(self):
        # With distinguishable particles, the observation is a tuple of the binary
        # observation space and the number of the agent
        if self.distinguishable_particles:
            self.use_dtype = np.float32 if self.use_speeds else np.int32
            obs_shape = ((self.obs_dist * 2 + 1) ** 2 + 1,) if self.speed_gradient_reward else (
                (self.obs_dist * 2 + 1) ** 2,)
            additional_high = self.width - 2 if self.speed_gradient_reward else 0
            if self.invert_speed_observation:
                single_obs = spaces.Box(low=0, high=1 + self.speed_observation_threshold + additional_high,
                                        shape=obs_shape, dtype=self.use_dtype)
                self.observation_space: spaces.Tuple[ObsType,] = spaces.Tuple(
                    (single_obs, spaces.Discrete(self.length * self.width)))
            else:
                single_obs = spaces.Box(low=0, high=1 + additional_high, shape=obs_shape,
                                        dtype=self.use_dtype)
                self.observation_space: spaces.Tuple[ObsType,] = spaces.Tuple(
                    (single_obs, spaces.Discrete(self.length * self.width)))
        else:
            self.use_dtype = np.uint8
            single_obs = spaces.Box(low=0, high=1, shape=((self.obs_dist * 2 + 1) ** 2,), dtype=self.use_dtype)
            self.observation_space: spaces.Tuple[ObsType, ObsType] = spaces.Tuple((single_obs, single_obs))
        # We have 3 actions, corresponding to "forward", "up", "down"
        if self.allow_wait:
            self.action_space: spaces.Discrete = spaces.Discrete(4)
        else:
            self.action_space: spaces.Discrete = spaces.Discrete(3)

    def _get_obs(self, new_mover=True):
        # simulate sometimes picking an empty cell by checking a random number against the density
        # in that case, increment the time and pick a new cell until a non-empty cell is found

        # Select a random agent (a grid cell that is currently set to 1) and return
        # the observation of the grid around it
        if new_mover:
            while np.random.random() > self.density:
                self.avg_window_time += 1
                self.total_timesteps += 1

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
            if self.inflate_speeds:
                obs[obs != 0] = 1 / 2 * (erf(0.5 / (np.sqrt(2) * self.sigma)) + erf(
                    (obs[obs != 0] - 0.5) / (np.sqrt(2) * self.sigma))) / erf(0.5 / (np.sqrt(2) * self.sigma))
            if self.invert_speed_observation:
                obs = invert_speed_obs(obs, self.speed_observation_threshold)
        return obs.flatten()

    @property
    def n(self):
        return np.count_nonzero(self.state)

    @property
    def rho(self):
        return self.n / (self.length * self.width)

    def _get_current(self):
        if self.average_window is None:
            return self.total_forward / self.total_timesteps if self.total_timesteps > 0 else 0
        return self._current

    def _get_info(self):
        return {
            # "state": self.state,
            # "N": self.n,
            # "rho": self.rho,
            "current": self._get_current(),
            "avg_reward": self._avg_reward
        }

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
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

    def reset(self, seed=None, options=None, random_density=False):
        # We need the following line to seed self.np_random (for reproducibility)
        super().reset(seed=seed)

        self.total_forward = 0
        self.total_timesteps = 0
        self.avg_window_time = 0
        self.avg_window_forward = 0
        self.avg_window_reward = 0

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
            self.state = self.initial_state.copy()
        if self.distinguishable_particles:
            random_integers = np.arange(1, self.n + 1, dtype=self.use_dtype)
            self.np_random.shuffle(random_integers)
            self.state[self.state == 1] = random_integers
            if self.use_speeds:
                random_speeds = truncated_normal(0.5, self.sigma,
                                                 self.n) if not self.binary_speeds else binary_distribution(self.n,
                                                                                                            self.choices)
                self.state[self.state != 0] += random_speeds
        elif not self.distinguishable_particles and self.use_speeds:
            random_speeds = truncated_normal(0.5, self.sigma,
                                             self.n) if not self.binary_speeds else binary_distribution(self.n,
                                                                                                        self.choices)
            self.state[self.state == 1] = random_speeds
        # update density to actual density
        self.density = self.n / (self.length * self.width)
        if self.prepare_lane_gradient:
            # prepare lane gradient by moving the slowest particles to the edges
            for i in range(self.length):
                # sort column by speed and shift indices by half of the width
                particles = self.state[:, i][self.state[:, i] != 0] % 1
                sorted_particles = np.sort(particles)[::-1]
                num = len(sorted_particles)
                order_indices = [num - 2 * i - 1 if i < num // 2 else 2 * i - num for i in
                                 range(num)] if num % 2 == 0 else [num - 2 * i - 1 if i <= num // 2 else 2 * i - num for i
                                                                   in range(num)]
                order_indices = np.array(order_indices)
                self.state[:, i][self.state[:, i] != 0] = sorted_particles[order_indices]
        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()
        if self.distinguishable_particles:
            if self.speed_gradient_reward:
                observation = np.append(observation, np.float32(self.current_mover[0]))
                return (observation, int(self.state[*self.current_mover])), info
            else:
                return (observation, int(self.state[*self.current_mover])), info
        return (observation, observation), info

    def _move_if_possible(self, position):
        if self.state[*position] == 0:  # if the next cell is empty, move
            self.state[*self.current_mover], self.state[*position] = self.state[*position], self.state[
                *self.current_mover]
            return True
        else:  # if the next cell is occupied, don't move
            return False

    def step(self, action):
        # Move the agent in the specified direction if possible.
        # If the agent is at the boundary of the grid, it will wrap around
        reward = 0
        self.avg_window_time += 1

        self.total_timesteps += 1
        self._update_current()
        self._update_current_initial()
        if not self.use_speeds or self.np_random.random() < self.state[*self.current_mover] % 1:
            reward = self._perform_action(action)
        react_observation = self._get_obs(new_mover=False)
        if self.punish_inhomogeneities:
            reward += self._calc_inhomo_reward(react_observation)
        if self.speed_gradient_reward:
            reward += self._calculate_speed_gradient_reward()
        self.avg_window_reward += reward
        info = self._get_info()
        next_observation = self._get_obs(new_mover=True)
        self._render_if_human()
        if self.distinguishable_particles:
            if self.speed_gradient_reward:
                next_observation = np.append(next_observation, np.float32(self.current_mover[0]))
                return (next_observation, int(self.state[*self.current_mover])), reward, False, False, info
            else:
                return (next_observation, int(self.state[*self.current_mover])), reward, False, False, info
        return (react_observation, next_observation), reward, False, False, info

    def _update_current(self):
        if self.avg_window_time >= self.average_window:
            self._current = self.avg_window_forward / self.average_window
            self._avg_reward = self.avg_window_reward / self.average_window
            self.avg_window_time = 0
            self.avg_window_forward = 0
            self.avg_window_reward = 0

    def _update_current_initial(self):
        if self.average_window > self.total_timesteps > 50:
            self._current = self.total_forward / self.total_timesteps
            self._avg_reward = self.avg_window_reward / self.total_timesteps

    def _perform_action(self, action):
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

    def _calculate_social_reward(self, row):
        prev_x = self.length - 1 if self.current_mover[1] == 0 else self.current_mover[1] - 1
        if self.state[row, prev_x] != 0:
            return -self.social_reward if not self.use_speeds else -(self.state[row, prev_x] % 1) * self.social_reward
        return 0

    def _calc_inhomo_reward(self, obs):
        obs = obs.reshape((2 * self.obs_dist + 1, 2 * self.obs_dist + 1))
        obs[obs != 0] = 1 - obs[obs != 0] + self.speed_observation_threshold
        center = obs[self.obs_dist, self.obs_dist]
        reward = 0
        particle_indices = np.argwhere(obs != 0)
        for particle in particle_indices:
            if particle[0] == self.obs_dist and particle[1] == self.obs_dist:
                continue
            # first working version
            if self.inh_rew_idx == 0:  # + changed forward to 1.5
                dv = abs(obs[*particle] - center)
                r = np.linalg.norm(particle - np.array([self.obs_dist, self.obs_dist]))
                if dv < 0.5:
                    if r <= 1:
                        pass
                    elif r > 5:
                        pass
                    else:
                        reward += -0.125 * r + 0.625
                else:
                    if r > 3.5:
                        pass
                    else:
                        reward -= 0.75 / (r ** 1.3) - 0.15
            elif self.inh_rew_idx == 1:
                dv = abs(obs[*particle] - center)
                r = np.linalg.norm(particle - np.array([self.obs_dist, self.obs_dist]))
                if dv < 0.5:
                    if r <= 1.5:
                        pass
                    elif r > 5:
                        pass
                    else:
                        reward += -0.125 * r + 0.625
                else:
                    if r > 3.5:
                        pass
                    else:
                        reward -= 0.75 / (r ** 1.3) - 0.15
        return reward

    def _speed_gradient(self, x):
        a = self.speed_gradient_linearity
        return a * ((1 + a) / a) ** x - a

    def _calculate_speed_gradient_reward(self):
        # the fastest particles should be in the middle lanes and slower particles should be farther to the edges
        speed = self.state[*self.current_mover] % 1
        gradient_factor = self._speed_gradient(speed)
        row = self.current_mover[0]
        desired_distance = self.width / 2 * (1 - gradient_factor)
        distance = abs(self.width / 2 - row)
        reward = -abs(desired_distance - distance) / self.width * 2
        return reward

    def _move_forward(self):
        next_x = 0 if self.current_mover[1] == self.length - 1 else self.current_mover[1] + 1
        has_moved = self._move_if_possible((self.current_mover[0], next_x))
        if not has_moved:
            return -1
        else:
            self.current_mover[1] = next_x
            self.total_forward += 1
            self.avg_window_forward += 1
            return self.forward_reward

    def _move_up_down(self, x_pos):
        has_moved = self._move_if_possible((x_pos, self.current_mover[1]))
        if not has_moved:
            return -1
        self.current_mover[0] = x_pos
        return 0

    def _render_if_human(self):
        if self.render_mode == "human" and self.total_timesteps % self.moves_per_timestep == 0:
            self._render_frame()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
