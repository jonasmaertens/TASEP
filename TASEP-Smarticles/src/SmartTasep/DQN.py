import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, new_model=None, hidden_sizes=None, activation_function=nn.ReLU()):
        """
        The DQN class defines the policy network. It is a simple fully connected neural network with one hidden layer.
        Args:
            n_observations: The number of observations in the state. This is equal to the size of the first layer.
            n_actions: The number of actions. This is equal to the size of the output layer.
        """
        super(DQN, self).__init__()
        self.layers = []
        if activation_function is None:
            activation_function = nn.ReLU()
        if new_model is False:
            self.layers.append(nn.Linear(n_observations, 128))
            self.layers.append(activation_function)
            self.layers.append(nn.Linear(128, 128))
            self.layers.append(activation_function)
            self.layers.append(nn.Linear(128, n_actions))
        elif new_model is True:
            self.layers.append(nn.Linear(n_observations, 24))
            self.layers.append(activation_function)
            self.layers.append(nn.Linear(24, 12))
            self.layers.append(activation_function)
            self.layers.append(nn.Linear(12, n_actions))
        else:
            if hidden_sizes is None:
                raise ValueError("hidden_sizes must be specified if new_model is not specified")
            # check if hidden_sizes is iterable
            if not hasattr(hidden_sizes, "__iter__"):
                raise ValueError("hidden_sizes must be iterable")
            # check if hidden_sizes is iterable of positive integers
            if not all(isinstance(x, int) and x > 0 for x in hidden_sizes):
                raise ValueError("hidden_sizes must be iterable of positive integers")
            # build layers
            if len(hidden_sizes) == 0:
                self.layers.append(nn.Linear(n_observations, n_actions))
            else:
                self.layers.append(nn.Linear(n_observations, hidden_sizes[0]))
                self.layers.append(activation_function)
                for i in range(len(hidden_sizes) - 1):
                    self.layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
                    self.layers.append(activation_function)
                self.layers.append(nn.Linear(hidden_sizes[-1], n_actions))

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        """
        The forward method defines the forward pass of the policy network.
        Args:
            x: The input tensor. This is the state or batch of states.

        Returns:
            The output tensor. This is the action or batch of actions.
        """
        return self.model(x)
