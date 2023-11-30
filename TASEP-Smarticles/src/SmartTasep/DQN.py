import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, new_model):
        """
        The DQN class defines the policy network. It is a simple fully connected neural network with one hidden layer.
        Args:
            n_observations: The number of observations in the state. This is equal to the size of the first layer.
            n_actions: The number of actions. This is equal to the size of the output layer.
        """
        super(DQN, self).__init__()
        if not new_model:
            self.layer1 = nn.Linear(n_observations, 128)
            self.layer2 = nn.Linear(128, 128)
            self.layer3 = nn.Linear(128, n_actions)
        else:
            self.layer1 = nn.Linear(n_observations, 24)
            self.layer2 = nn.Linear(24, 12)
            self.layer3 = nn.Linear(12, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        if hasattr(self, "layer3"):
            x = F.relu(self.layer2(x))
            x = self.layer3(x)
        else:
            x = self.layer2(x)
        return x
