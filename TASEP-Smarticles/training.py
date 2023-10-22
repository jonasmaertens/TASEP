# %% Imports && Setup
import datetime

import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from DQN import DQN
from Memory import ReplayMemory

from GridEnvironment import GridEnv

gym.envs.registration.register(
    id='GridEnv',
    entry_point='GridEnvironment:GridEnv',
)
GridEnv.metadata["render_fps"] = 144
env: GridEnv | gym.Env = gym.make("GridEnv",
                                  render_mode=None,
                                  length=64,
                                  width=16,
                                  moves_per_timestep=10,
                                  window_height=256,
                                  observation_distance=3,
                                  initial_state_template="checkerboard")

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% Init
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 256
GAMMA = 0.6
EPS_START = 0.9
EPS_END = 0.03
EPS_DECAY = 40_000
TAU = 0.001
LR = 1e-2
MEMORY_SIZE = 30_000

# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
(state, _), info = env.reset()
n_observations = len(state)

# %% Init model, optimizer, memory
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(MEMORY_SIZE)

steps_done = 0

def get_current_eps():
    return EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = get_current_eps()
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            # max(1) gives max along axis 1 ==> max of every row
            # return tensor with first row = max values and second row = indices of max values
            # we want the indices of the max values, so we use [1]
            # we also need to reshape the tensor to be 1x1 instead of 1
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    batch = memory.sample(BATCH_SIZE)

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    # clears the .grad attribute of the weights tensor
    optimizer.zero_grad()
    # computes the gradient of loss w.r.t. all the weights
    # computation graph is embedded in the loss tensor because it is created
    # from the action values tensor which is computed by forward pass though
    # the network. Gradient values are stored in the .grad attribute of the
    # weights tensor
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    # uses the gradient values in the .grad attribute of the weights tensor
    # to update the weight values
    optimizer.step()


# %% Training loop
if torch.cuda.is_available():
    num_timesteps = int(420_000)
else:
    num_timesteps = int(1e6)

print(f"Training for {num_timesteps} timesteps")
currents = []
timesteps = []
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

while steps_done < num_timesteps:
    if steps_done > 400_000 and env.render_mode is None:
        env = gym.make("GridEnv",
                       render_mode="human",
                       length=64,
                       width=16,
                       moves_per_timestep=10,
                       window_height=256,
                       observation_distance=3,
                       initial_state_template="checkerboard")
        (state, _), info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    action = select_action(state)
    (react_observation, next_observation), reward, terminated, truncated, info = env.step(action.item())
    reward = torch.tensor([reward], device=device)
    done = terminated or truncated

    if terminated:
        next_state = None
    else:
        react_state = torch.tensor(react_observation, dtype=torch.float32, device=device).unsqueeze(0)
        next_state = torch.tensor(next_observation, dtype=torch.float32, device=device).unsqueeze(0)

    # Store the transition in memory
    memory.push(state, action, react_state, reward)

    # Move to the next state
    state = next_state

    # Perform one step of the optimization (on the policy network)
    optimize_model()

    # Soft update of the target network's weights
    # θ′ ← τ θ + (1 −τ )θ′
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
    target_net.load_state_dict(target_net_state_dict)

    if steps_done % 1000 == 0:
        print(
            f"{steps_done} steps finished: Current: {info['current']}, " +
            f"eps: {get_current_eps()}")
        currents.append(info['current'])
        timesteps.append(steps_done)
        plt.plot(timesteps, currents)
        plt.show()


torch.save(policy_net.state_dict(),
           f"policy_net_trained_{num_timesteps}_steps_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pt")
print('Complete')
