from gym import spaces
import torch.nn as nn


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

        self.glyphs_shape = self.glyphs_shape
        self.action_space = action_space.n
        self.h = self.glyphs_shape[0]
        self.w = self.glyphs_shape[1]
        self.conv1 = nn.Conv2d(self.h, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        #For the 84x84 input, the output from the convolution layer will have 3136
        self.fc1 = nn.Linear(81, 512)
        self.fc2 = nn.Linear(512, action_space.n)


    def forward(self, x):
        # TODO Implement forward pass [2]
        # Implement the Deep Q-Network
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        #Flatten the 4D tensor (bastch_size x color_channel x stack x dimensions) to 2D tensor
        x = nn.functional.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x)
