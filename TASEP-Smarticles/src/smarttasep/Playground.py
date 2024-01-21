import numpy as np
import pygame
from .DQN import DQN
import torch
import os
from .Hasel import hsl2rgb
from .Trainer import Trainer
from .GridEnvironment import invert_speed_obs
import json
import datetime

choose_model = Trainer.choose_model


class Playground:
    def __init__(self, model_id: int = None):
        """
        The Playground class is used to test a trained policy network on a custom initial state. The user can draw
        particles on the left matrix and the policy network will choose an action. The resulting state immediately
        after the action is then displayed on the right matrix. If the matrix is clicked, the particle at the clicked
        position will be toggled with speed 1. If the mouse is dragged, the particle at the dragged position will
        have a speed between 0 and 1, depending on the distance of the mouse to the initial position.

        Args:
            model_id (int, optional): The id of the model to use. If None, the user will be prompted to choose a
                model from a table.
        """
        # init state
        if model_id is None:
            model_id = choose_model()
        with open("models/all_models.json", "r") as f:
            all_models = json.load(f)
        model = all_models[str(model_id)]
        self.model = model
        self.observation_distance = self.model["env_params"]["observation_distance"]
        self.actions = 4 if self.model["env_params"]["allow_wait"] else 3
        self.invert_speed_observation = self.model["env_params"]["invert_speed_observation"]
        self.speed_observation_threshold = self.model["env_params"]["speed_observation_threshold"]
        self.length = self.observation_distance * 2 + 1
        self.state = np.zeros((self.length, self.length), dtype=np.float32)
        self.next_state = self.state.copy()
        self.state[self.observation_distance, self.observation_distance] = 2
        self.mouse_down_pos = (0, 0)
        self.dragging = False
        self.dragging_distance = 0
        # init policy network
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(self.length ** 2, self.actions, self.model["new_model"]).to(self.device)
        self.policy_net.load_state_dict(torch.load(os.path.join(self.model["path"], "policy_net.pt")))
        # init pygame
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((self.length * 100 + 100 + 1, self.length * 50 + 1))
        self.font = pygame.font.SysFont("Arial", 20)
        pygame.display.set_caption("Smart TASEP")
        self.draw_background_and_arrow()
        self.draw_matrix(self.state.T)
        self.calc_next_state(0)
        self.draw_matrix(self.next_state.T, right=True)
        self.draw_grid()
        pygame.display.flip()

        # pygame main loop
        running = True
        while running:
            # event handling
            for event in pygame.event.get():
                # make sure the user can quit the program
                if event.type == pygame.QUIT:
                    running = False
                # when the mouse button is pushed down, a click or a drag can be started
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.mouse_down_pos = pygame.mouse.get_pos()
                    self.dragging = True
                # when the mouse is moved, the particles speed at the mouse position is set to the distance of the
                # mouse to the initial position
                elif event.type == pygame.MOUSEMOTION:
                    if self.dragging:
                        self.dragging_distance = min(
                            np.linalg.norm(np.array(pygame.mouse.get_pos()) - np.array(self.mouse_down_pos)) / 100, 1)
                        pos = self.mouse_down_pos
                        # only change the speed of the particle if the initial click is inside the left matrix
                        if (not (pos[0] > 50 * self.length or pos[1] > 50 * self.length or (
                                pos[0] // 50 == self.observation_distance
                                and pos[1] // 50 == self.observation_distance))
                                and self.dragging_distance > 0.05):
                            # prevent user from removing the particle at the center
                            self.state.T[pos[0] // 50, pos[1] // 50] = self.dragging_distance
                        text = self.font.render(f"{self.dragging_distance:.2f}", True, (0, 0, 0))
                        # redraw everything so the changes are visible
                        self.draw_background_and_arrow()
                        self.draw_matrix(self.state.T)
                        self.draw_matrix(self.next_state.T, right=True)
                        self.draw_grid()
                        self.screen.blit(text,
                                         (self.length * 50 + 50 - text.get_width() / 2, self.length / 2 * 50 + 50))
                        pygame.display.flip()
                # when the mouse button is released, the particle at the mouse position is toggled
                elif event.type == pygame.MOUSEBUTTONUP:
                    pos = self.mouse_down_pos
                    # if clicked on the right matrix, save an image of the current screen, with timestamp as filename
                    if pos[0] > 50 * self.length + 100:
                        os.makedirs("plots/playground", exist_ok=True)
                        pygame.image.save(self.screen, f"plots/playground/{datetime.datetime.now()}.png")
                    # only toggle the particle if the initial click is inside the left matrix and the mouse was not
                    # dragged. If the mouse was dragged, the particle speed is already set
                    if self.dragging_distance < 0.05 and not (
                            pos[0] > 50 * self.length or pos[1] > 50 * self.length or (
                            pos[0] // 50 == self.observation_distance and pos[1] // 50 == self.observation_distance)):
                        self.state.T[pos[0] // 50, pos[1] // 50] = 1 if self.state.T[
                                                                            pos[0] // 50, pos[1] // 50] == 0 else 0
                    # stop dragging measurement and redraw the left matrix
                    self.dragging = False
                    self.dragging_distance = 0
                    self.draw_background_and_arrow()
                    self.draw_matrix(self.state.T)
                    # map the state to the observation space of the policy network and select an action
                    corrected_state = self.state.copy()
                    corrected_state[self.observation_distance, self.observation_distance] = 1
                    if self.invert_speed_observation:
                        corrected_state = invert_speed_obs(corrected_state, self.speed_observation_threshold)
                    action = self.select_action(
                        torch.tensor(corrected_state, dtype=torch.float32).flatten().unsqueeze(0).to(self.device))
                    self.calc_next_state(action)
                    # redraw the right matrix
                    self.draw_matrix(self.next_state.T, right=True)
                    self.draw_grid()
                    # print(f"Current state:\n{self.state}")
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
        od = self.observation_distance
        if action == 0:  # right
            self.next_state[od, od], self.next_state[od, od + 1] = self.next_state[od, od + 1], self.next_state[od, od]
            print("Action: right")
        elif action == 1:  # up
            self.next_state[od, od], self.next_state[od - 1, od] = self.next_state[od - 1, od], self.next_state[od, od]
            print("Action: up")
        elif action == 2:  # down
            self.next_state[od, od], self.next_state[od + 1, od] = self.next_state[od + 1, od], self.next_state[od, od]
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
        with torch.no_grad():
            return self.policy_net(state).max(1)[1].view(1, 1)



















