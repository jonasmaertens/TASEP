# %% Imports && Setup
import gymnasium as gym
from itertools import count
import torch

from DQN import DQN

env = gym.make("CartPole-v1", render_mode="human")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
policy_net.load_state_dict(torch.load("policy_net_trained.pt"))

steps_done = 0


def select_action(state):
    global steps_done
    steps_done += 1
    with torch.no_grad():
        return policy_net(state).max(1)[1].view(1, 1)


episode_durations = []
num_episodes = 600


for i_episode in range(num_episodes):
    # Initialize the environment and get it's state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        state = next_state

        if done:
            episode_durations.append(t + 1)
            print(f"Episode {i_episode} finished after {t + 1} timesteps")
            break


