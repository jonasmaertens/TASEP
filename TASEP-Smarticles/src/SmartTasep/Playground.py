import numpy as np
import pygame
from DQN import DQN
import torch
import os
from Hasel import hsl2rgb


class Playground:
    def __init__(self, model, observation_distance, actions):
        self.observation_distance = observation_distance
        self.length = self.observation_distance * 2 + 1
        self.state = np.zeros((self.length, self.length), dtype=np.float32)
        self.next_state = self.state.copy()
        self.state[self.observation_distance, self.observation_distance] = 2
        self.model = model
        self.mouse_down_pos = (0, 0)
        self.dragging = False
        self.dragging_distance = 0
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(self.length ** 2, actions).to(self.device)
        try:
            self.policy_net.load_state_dict(torch.load(self.model))
        except FileNotFoundError:
            self.model = os.path.join(os.getcwd(), self.model)
            self.policy_net.load_state_dict(torch.load(self.model))
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((self.length * 100 + 100, self.length * 50))
        self.font = pygame.font.SysFont("Arial", 20)
        pygame.display.set_caption("Smart TASEP")
        self.draw_background_and_arrow()
        self.draw_matrix(self.state.T)
        self.calc_next_state(0)
        self.draw_matrix(self.next_state.T, right=True)
        pygame.display.flip()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.mouse_down_pos = pygame.mouse.get_pos()
                    self.dragging = True
                elif event.type == pygame.MOUSEMOTION:
                    if self.dragging:
                        self.dragging_distance = min(
                            np.linalg.norm(np.array(pygame.mouse.get_pos()) - np.array(self.mouse_down_pos)) / 100, 1)
                        pos = self.mouse_down_pos
                        if not (pos[0] > 50 * self.length or pos[
                            1] > 50 * self.length) and self.dragging_distance > 0.05:
                            self.state.T[pos[0] // 50, pos[
                                1] // 50] = self.dragging_distance
                        text = self.font.render(f"{self.dragging_distance:.2f}", True, (0, 0, 0))
                        self.draw_background_and_arrow()
                        self.draw_matrix(self.state.T)
                        self.draw_matrix(self.next_state.T, right=True)
                        self.screen.blit(text,
                                         (self.length * 50 + 50 - text.get_width() / 2, self.length / 2 * 50 + 50))
                        pygame.display.flip()
                elif event.type == pygame.MOUSEBUTTONUP:
                    pos = self.mouse_down_pos
                    if self.dragging_distance < 0.05 and not (pos[0] > 50 * self.length or pos[1] > 50 * self.length):
                        self.state.T[pos[0] // 50, pos[1] // 50] = 1 if self.state.T[
                                                                            pos[0] // 50, pos[1] // 50] == 0 else 0
                    self.dragging = False
                    self.dragging_distance = 0
                    self.draw_background_and_arrow()
                    self.draw_matrix(self.state.T)
                    corrected_state = self.state.copy()
                    corrected_state[self.observation_distance, self.observation_distance] = 1
                    action = self.select_action(
                        torch.tensor(corrected_state, dtype=torch.float32).flatten().unsqueeze(0).to(self.device))
                    self.calc_next_state(action)
                    self.draw_matrix(self.next_state.T, right=True)
                    print(f"Current state:\n{self.state}")
                    pygame.display.flip()

    def draw_grid(self):
        for i in range(self.length + 1):
            pygame.draw.line(self.screen, (0, 0, 0), (i * 50, 0), (i * 50, self.length * 50))
            pygame.draw.line(self.screen, (0, 0, 0), (0, i * 50), (self.length * 50, i * 50))
            pygame.draw.line(self.screen, (0, 0, 0), (i * 50 + (self.length * 50 + 100), 0),
                             (i * 50 + (self.length * 50 + 100), self.length * 50))
            pygame.draw.line(self.screen, (0, 0, 0), ((self.length * 50 + 100), i * 50), (1200, i * 50))

    def calc_next_state(self, action):
        self.next_state = self.state.copy()
        if action == 0:  # right
            self.next_state[self.observation_distance, self.observation_distance], self.next_state[
                self.observation_distance, self.observation_distance + 1] = self.next_state[
                self.observation_distance, self.observation_distance + 1], self.next_state[
                self.observation_distance, self.observation_distance]
            print("Action: right")
        elif action == 1:  # up
            self.next_state[self.observation_distance, self.observation_distance], self.next_state[
                self.observation_distance - 1, self.observation_distance] = self.next_state[
                self.observation_distance - 1, self.observation_distance], self.next_state[
                self.observation_distance, self.observation_distance]
            print("Action: up")
        elif action == 2:  # down
            self.next_state[self.observation_distance, self.observation_distance], self.next_state[
                self.observation_distance + 1, self.observation_distance] = self.next_state[
                self.observation_distance + 1, self.observation_distance], self.next_state[
                self.observation_distance, self.observation_distance]
            print("Action: down")
        elif action == 3:  # wait
            print("Action: wait")

    def draw_matrix(self, a, right=False):
        if right:
            for i in range(a.shape[0]):
                for j in range(a.shape[1]):
                    if a[i, j] > 0:
                        hsl_color = np.array([a[i, j] / 3, 1, 0.5]).reshape((1, 1, 3))
                        rgb_color = hsl2rgb(hsl_color).flatten()
                        pygame.draw.rect(self.screen, rgb_color, (i * 50 + (self.length * 50 + 100), j * 50, 50, 50))
        else:
            for i in range(a.shape[0]):
                for j in range(a.shape[1]):
                    if a[i, j] > 0:
                        hsl_color = np.array([a[i, j] / 3, 1, 0.5]).reshape((1, 1, 3))
                        rgb_color = hsl2rgb(hsl_color).flatten()
                        pygame.draw.rect(self.screen, rgb_color, (i * 50, j * 50, 50, 50))

    def draw_background_and_arrow(self):
        self.screen.fill((255, 255, 255))
        self.draw_grid()
        # draw arrow between matrices
        pygame.draw.line(self.screen, (0, 0, 0), (self.length * 50 + 15, self.length * 25),
                         (self.length * 50 + 85, self.length * 25))
        pygame.draw.line(self.screen, (0, 0, 0), (self.length * 50 + 85, self.length * 25),
                         (self.length * 50 + 60, self.length * 21))
        pygame.draw.line(self.screen, (0, 0, 0), (self.length * 50 + 85, self.length * 25),
                         (self.length * 50 + 60, self.length * 29))
        pygame.draw.line(self.screen, (0, 0, 0), (self.length * 50 + 85, self.length * 25),
                         (self.length * 50 + 85, self.length * 25))

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Selects an action using an epsilon-greedy policy
        :param state: Current state
        :param eps_greedy: Whether to use epsilon-greedy policy or greedy policy
        """
        with torch.no_grad():
            return self.policy_net(state).max(1)[1].view(1, 1)


if __name__ == '__main__':
    # model = "models/same_speeds/allow_wait/model_500000_steps_sigma_1_render3.pt"
    # model = "models/same_speeds/allow_wait/model_200000_steps_sigma_1_no_gamma_2.pt"
    model = "models/same_speeds/allow_wait/model_1500000_steps_social_0.6_different_rho.pt"
    Playground(model, observation_distance=3, actions=4)
