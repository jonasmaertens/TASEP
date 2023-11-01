import numpy as np
import pygame
from DQN import DQN
import torch
import os


class Playground:
    def __init__(self, model, observation_distance=2):
        self.observation_distance = observation_distance
        self.state = np.zeros((self.observation_distance * 2 + 1, self.observation_distance * 2 + 1), dtype=int)
        self.state[self.observation_distance, self.observation_distance] = 2
        self.model = model
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN((self.observation_distance * 2 + 1) ** 2, 3).to(self.device)
        try:
            self.policy_net.load_state_dict(torch.load(self.model))
        except FileNotFoundError:
            self.model = os.path.join(os.getcwd(), self.model)
            self.policy_net.load_state_dict(torch.load(self.model))
        pygame.init()
        self.screen = pygame.display.set_mode((1200, 500))
        self.screen.fill((255, 255, 255))
        self.draw_grid()
        self.draw_matrix(self.state)
        pygame.display.flip()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONUP:
                    pos = pygame.mouse.get_pos()
                    if pos[0] > 500 or pos[1] > 500:
                        continue
                    self.state.T[pos[0] // 100, pos[1] // 100] = 1 - self.state.T[pos[0] // 100, pos[1] // 100]
                    self.screen.fill((255, 255, 255))
                    self.draw_grid()
                    self.draw_matrix(self.state.T)
                    action = self.select_action(
                        torch.tensor(self.state, dtype=torch.float32).flatten().unsqueeze(0).to(self.device))
                    self.next_state = self.state.copy()
                    if action == 0:  # right
                        self.next_state[observation_distance, observation_distance], self.next_state[
                            observation_distance, observation_distance + 1] = self.next_state[
                            observation_distance, observation_distance + 1], self.next_state[
                            observation_distance, observation_distance]
                    elif action == 1:  # up
                        self.next_state[observation_distance, observation_distance], self.next_state[
                            observation_distance - 1, observation_distance] = self.next_state[
                            observation_distance - 1, observation_distance], self.next_state[
                            observation_distance, observation_distance]
                    elif action == 2:  # down
                        self.next_state[observation_distance, observation_distance], self.next_state[
                            observation_distance + 1, observation_distance] = self.next_state[
                            observation_distance + 1, observation_distance], self.next_state[
                            observation_distance, observation_distance]
                    self.draw_matrix(self.next_state.T, right=True)
                    print(self.state)
                    print(self.next_state)
                    print(action)
                    pygame.display.flip()

    def draw_grid(self):
        for i in range(6):
            pygame.draw.line(self.screen, (0, 0, 0), (i * 100, 0), (i * 100, 500))
            pygame.draw.line(self.screen, (0, 0, 0), (0, i * 100), (500, i * 100))
            pygame.draw.line(self.screen, (0, 0, 0), (i * 100 + 700, 0), (i * 100 + 700, 500))
            pygame.draw.line(self.screen, (0, 0, 0), (700, i * 100), (1200, i * 100))

    def draw_matrix(self, a, right=False):
        if right:
            for i in range(a.shape[0]):
                for j in range(a.shape[1]):
                    if a[i, j] == 1:
                        pygame.draw.rect(self.screen, (0, 0, 0), (i * 100 + 700, j * 100, 100, 100))
                    elif a[i, j] == 2:
                        pygame.draw.rect(self.screen, (255, 0, 0), (i * 100 + 700, j * 100, 100, 100))
        else:
            for i in range(a.shape[0]):
                for j in range(a.shape[1]):
                    if a[i, j] == 1:
                        pygame.draw.rect(self.screen, (0, 0, 0), (i * 100, j * 100, 100, 100))
                    elif a[i, j] == 2:
                        pygame.draw.rect(self.screen, (255, 0, 0), (i * 100, j * 100, 100, 100))

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Selects an action using an epsilon-greedy policy
        :param state: Current state
        :param eps_greedy: Whether to use epsilon-greedy policy or greedy policy
        """
        with torch.no_grad():
            return self.policy_net(state).max(1)[1].view(1, 1)


if __name__ == '__main__':
    model = "models/different_speeds/individual_sigmas/model_100000_steps_sigma_1.00e+01_20231028170511.pt"
    Playground(model, observation_distance=2)
