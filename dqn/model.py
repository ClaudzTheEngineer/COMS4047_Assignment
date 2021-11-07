from gym import spaces
import torch.nn as nn
import numpy as np
import torch

class DQN(nn.Module):
    """
    A basic implementation of a Deep Q-Network. The architecture is the same as that described in the
    Nature DQN paper.
    """

    def __init__(self, glyphs: spaces.Box, action_space: spaces.Discrete):
        """
        Initialise the DQN
        :param observation_space: the state space of the environment
        :param action_space: the action space of the environment
        """
        super().__init__()
        #create a CNN with 3 convolutional layers and 2 fully connected layers
        self.action_space = action_space.n
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(576, 512)
        self.fc2 = nn.Linear(512, action_space.n) #outputs tensor with an action value

    def forward(self, new_glyphs):
        x_glyphs = new_glyphs.unsqueeze(1).float() 

        # Implement the Deep Q-Network
        x = nn.functional.relu(self.conv1(x_glyphs))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))

        #Flatten the 4D tensor (bastch_size x color_channel x stack x dimensions) to 2D tensor
        x = nn.functional.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x)
