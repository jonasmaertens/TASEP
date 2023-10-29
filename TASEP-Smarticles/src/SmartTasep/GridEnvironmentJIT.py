import copy

import numpy as np
from numba import int64, float64
from numba.experimental import jitclass
import numba

spec = [
    ('length', int64),
    ('width', int64),
    ('moves_per_timestep', int64),
    ('window_height', int64),
    ('observation_distance', int64),
    ('initial_state_template', numba.types.string),
    ('distinguishable_particles', numba.types.boolean),
    ('use_speeds', numba.types.boolean),
    ('sigma', float64),
    ('average_window', int64),
    ('current_mover', int64[:]),
    ('state', float64[:, :]),
    ('total_timesteps', int64),
    ('total_forward', int64),
    ('avg_window_forward', int64),
    ('currents', float64[:]),
    ('window', numba.types.optional(numba.types.int64)),
    ('render_mode', numba.types.optional(numba.types.string)),
    ('_current', numba.types.optional(numba.types.float64)),
    ('use_dtype', numba.typeof(numba.types.float64)),
    ('avg_window_time', int64),
    ('action_space', int64),
]


@jitclass(spec)
class GridEnvJIT:
    def __init__(self, length, width, moves_per_timestep, window_height, observation_distance,
                 initial_state_template, distinguishable_particles, use_speeds, sigma, average_window):
        self.length = length
        self.width = width
        self.use_dtype = np.float64
        self.action_space = 3
        self.moves_per_timestep = moves_per_timestep
        self.window_height = window_height
        self.observation_distance = observation_distance
        self.initial_state_template = initial_state_template
        self.distinguishable_particles = distinguishable_particles
        self.use_speeds = use_speeds
        self.sigma = sigma
        self.average_window = average_window
        self.current_mover = np.array([0, 0], dtype=np.int64)
        self.state = np.zeros((width, length), dtype=np.float64)
        self.total_timesteps = 0
        self.total_forward = 0
        self.avg_window_forward = 0
        self.avg_window_time = 0
        self.currents = np.zeros(40, dtype=np.float64)
        self.window = None
        self.render_mode = None
        self._current = 0

    def truncated_normal_single(self, mean, std_dev):
        sample = -1
        while sample < 0 or sample > 1:
            sample = np.random.normal(mean, std_dev)
        return sample

    def truncated_normal(self, mean, std_dev, size):
        samples = np.zeros(size, dtype=np.float64)
        for i in range(size):
            samples[i] = self.truncated_normal_single(mean, std_dev)
        return samples

    def _get_obs(self, new_mover):
        if new_mover:
            agent_indices = np.argwhere(self.state != 0)
            self.current_mover = agent_indices[np.random.randint(len(agent_indices))]
        obs = np.zeros((self.observation_distance * 2 + 1, self.observation_distance * 2 + 1), dtype=self.use_dtype)
        for i in range(-self.observation_distance, self.observation_distance + 1):
            for j in range(-self.observation_distance, self.observation_distance + 1):
                pos = self.current_mover + np.array([i, j], dtype=np.int64)
                if pos[0] < 0:
                    pos[0] += self.width
                elif pos[0] >= self.width:
                    pos[0] -= self.width
                if pos[1] < 0:
                    pos[1] += self.length
                elif pos[1] >= self.length:
                    pos[1] -= self.length
                obs[i + self.observation_distance, j + self.observation_distance] = self.state[pos[0], pos[1]] % 1
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
        if self.average_window is None:
            return self.total_forward / self.total_timesteps if self.total_timesteps > 0 else 0
        return self._current

    def reset(self):
        self.total_forward = 0
        self.total_timesteps = 0
        self.avg_window_time = 0
        self.avg_window_forward = 0

        # Set the initial state of the environment
        # If no initial state is given, we set the initial state to be a checkerboard
        if self.initial_state_template == "checkerboard":
            self.state = np.zeros((self.width, self.length), dtype=self.use_dtype)
            self.state[::2, ::2] = 1
            self.state[1::2, 1::2] = 1
        else:
            self.state = np.zeros((self.width, self.length), dtype=self.use_dtype)
            self.state[0, 0] = 1
        if self.distinguishable_particles:
            random_integers = np.arange(1, self.n + 1, dtype=self.use_dtype)
            np.random.shuffle(random_integers)
            used = 0
            for i in range(self.width):
                for j in range(self.length):
                    if self.state[i, j] == 1:
                        self.state[i, j] = random_integers[used]
                        used += 1
            if self.use_speeds:
                random_speeds = self.truncated_normal(0.5, self.sigma, self.n)
                for i in range(self.width):
                    for j in range(self.length):
                        if self.state[i, j] != 0:
                            self.state[i, j] = self.state[i, j] + random_speeds[int(self.state[i, j]) - 1]
        elif not self.distinguishable_particles and self.use_speeds:
            random_speeds = self.truncated_normal(0.5, self.sigma, self.n)
            used = 0
            for i in range(self.width):
                for j in range(self.length):
                    if self.state[i, j] == 1:
                        self.state[i, j] = random_speeds[used]
                        used += 1
        observation = self._get_obs(True)
        info = numba.typed.Dict.empty(key_type=numba.types.string, value_type=numba.types.float64)
        info["current"] = self._get_current()
        mover = int(self.state[self.current_mover[0], self.current_mover[1]])
        return (observation, mover), info

    def _move_if_possible(self, position: tuple) -> bool:
        if self.state[position[0], position[1]] == 0:  # if the next cell is empty, move
            self.state[self.current_mover[0], self.current_mover[1]], self.state[position[0], position[1]] = self.state[
                position[0], position[1]], self.state[
                self.current_mover[0], self.current_mover[1]]
            return True
        else:  # if the next cell is occupied, don't move
            return False

    def step(self, action):
        # Move the agent in the specified direction if possible.
        # If the agent is at the boundary of the grid, it will wrap around
        reward = 0
        self.total_timesteps += 1
        self.avg_window_time += 1
        if self.avg_window_time >= self.average_window:
            self._current = self.avg_window_forward / self.average_window
            self.avg_window_time = 0
            self.avg_window_forward = 0
        if self.average_window > self.total_timesteps > 50:
            self._current = self.total_forward / self.total_timesteps
        if action == 0:  # forward
            # when using speeds, the probability to move forward is the speed of the particle
            if not self.use_speeds or np.random.random() < self.state[self.current_mover[0], self.current_mover[1]] % 1:
                next_y = 0 if self.current_mover[1] == self.length - 1 else self.current_mover[1] + 1
                has_moved = self._move_if_possible((self.current_mover[0], next_y))
                if not has_moved:
                    reward = -1
                else:
                    reward = 1
                    self.total_forward += 1
                    self.avg_window_forward += 1
        elif action == 1:  # up
            above = self.width - 1 if self.current_mover[0] == 0 else self.current_mover[0] - 1
            has_moved = self._move_if_possible((above, self.current_mover[1]))
            if not has_moved:
                reward = -1
        else:  # down
            below = 0 if self.current_mover[0] == self.width - 1 else self.current_mover[0] + 1
            has_moved = self._move_if_possible((below, self.current_mover[1]))
            if not has_moved:
                reward = -1

        info = numba.typed.Dict.empty(key_type=numba.types.string, value_type=numba.types.float64)
        info["current"] = self._get_current()
        next_observation = self._get_obs(new_mover=True)

        return (next_observation, int(
            self.state[self.current_mover[0], self.current_mover[1]])), reward, False, False, info


# Test the environment
if __name__ == '__main__':
    env = GridEnvJIT(length=8,
                          width=8,
                          moves_per_timestep=20,
                          window_height=256,
                          observation_distance=2,
                          initial_state_template="checkerboard",
                          distinguishable_particles=True,
                          use_speeds=True,
                          sigma=0.5,
                          average_window=2500)
    (obs, mover), info = env.reset()
    obs = obs.reshape((5, 5))
    state = copy.deepcopy(env.state)
    action = 0
    (next_obs, next_mover), reward, terminated, truncated, current = env.step(action)
    next_obs = next_obs.reshape((5, 5))
    next_state = copy.deepcopy(env.state)
    action = 0
    (next_next_obs, next_next_mover), next_reward, terminated, truncated, next_current = env.step(action)
    next_next_obs = next_next_obs.reshape((5, 5))
    next_next_state = env.state
    print("test")
