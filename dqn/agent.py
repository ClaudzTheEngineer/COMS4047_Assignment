from gym import spaces
import numpy as np
import torch

from dqn.model import DQN
from dqn.replay_buffer import ReplayBuffer

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNAgent:
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        replay_buffer: ReplayBuffer,
        use_double_dqn,
        lr,
        batch_size,
        gamma
    ):
        """
        Initialise the DQN algorithm using the RMSprop optimiser
        :param action_space: the action space of the environment
        :param observation_space: the state space of the environment
        :param replay_buffer: storage for experience replay
        :param lr: the learning rate for Adam
        :param batch_size: the batch size
        :param gamma: the discount factor
        """


        # TODO: Initialise agent's networks, optimiser and replay buffer [3]
        self.policy_network = DQN(observation_space, action_space).to(device)
        self.target_network = DQN(observation_space, action_space).to(device)
        self.update_target_network()
        self.target_network.eval()

        # Optimizer
        self.RMS = torch.optim.RMSprop(self.policy_network.parameters(), lr)

        # Replay Buffer
        self.replay_buffer = replay_buffer

        self.batch_size = batch_size
        self.gamma = gamma

    def optimise_td_loss(self):
        """
        Optimise the TD-error over a single minibatch of transitions
        :return: the loss
        """
        # TODO
        #   Optimise the TD-error over a single minibatch of transitions
        #   Sample the minibatch from the replay-memory
        #   using done (as a float) instead of if statement
        #   return loss


        # Sample the minibatch from the replay-memory
        mini_batch_sample = self.replay_buffer.sample(self.batch_size)
        states_batch,actions_batch,rewards_batch,next_states_batch,done_batch = mini_batch_sample

        # The pixel intensities are in the range [0 255] [3]
        # Scaled to 0 1
        states_batch = np.array(states_batch) /255.0
        next_states_batch = np.array(next_states_batch)/255.0

        # Wrap arrays with batch data in float tensors , and copy to GPU [2]
        states_batch = torch.tensor(states_batch,dtype=torch.float).to(device)
        actions_batch = torch.tensor(actions_batch,dtype=torch.long).to(device)
        rewards_batch = torch.tensor(rewards_batch, dtype=torch.float).to(device)
        next_states_batch = torch.tensor(next_states_batch, dtype=torch.float).to(device)
        done_batch = torch.tensor(done_batch, dtype=torch.float).to(device)

        # Choose the greedy action [3]
        none, greedy_action = self.policy_network(next_states_batch).max(1)
        # Pass next state to the target network and extract the specific Q-values [3]
        values_next_state = self.target_network(next_states_batch).gather(1, greedy_action.unsqueeze(1)).squeeze()
        # Prevent gradients from flowing into the target network [3]
        values_next_state = values_next_state.detach()

        # Bellman approximation
        approx_state_action_values = rewards_batch + (1 - done_batch) * self.gamma * values_next_state

        #Pass obseravtions to the policy network and extract the specific Q-values [2][3]
        state_action_values = self.policy_network(states_batch).gather(1, actions_batch.unsqueeze(1)).squeeze()

        # Compute the loss
        loss = torch.nn.functional.smooth_l1_loss(state_action_values, approx_state_action_values)

        # Update the main neural network using the SGD algorithm by minimizing the loss [2]
        self.RMS.zero_grad()
        loss.backward()
        self.RMS.step()

        del states_batch
        del next_states_batch

        return loss.item()

    def update_target_network(self):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """
        current_Q = self.policy_network.state_dict()
        self.target_network.load_state_dict(current_Q)

    def act(self, state: np.ndarray):
        """
        Select an action greedily from the Q-network given the state
        :param state: the current state
        :return: the action to take
        """
        # Wrap state data in float tensors , and copy to GPU [3]
        current_state = np.array(state).astype(np.float)/255.0
        current_state = torch.tensor(current_state , dtype=torch.float).unsqueeze(0).to(device)

        network_values = self.policy_network(current_state)
        # Choose the greedy action [3]
        none, greedy_action = network_values.max(1)
        # Prevent gradients from flowing into the target network
        network_values = network_values.detach()

        return greedy_action.item()
