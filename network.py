import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class FeedForwardNN(nn.Module):
    """
		A standard in_dim-64-64-out_dim Feed Forward Neural Network.
	"""
    def __init__(self, in_dim, out_dim):
        """
			Initialize the network and set up the layers.
			Parameters:
				in_dim - input dimensions as an int
				out_dim - output dimensions as an int
			Return:
				None
		"""
        super(FeedForwardNN, self).__init__()
        hidden_neurons = 256
        self.layer1 = nn.Linear(in_dim, hidden_neurons)
        self.layer2 = nn.Linear(hidden_neurons, hidden_neurons)
        self.layer3 = nn.Linear(hidden_neurons, hidden_neurons)
        self.layer4 = nn.Linear(hidden_neurons, out_dim)

    def forward(self, obs):
        """
			Runs a forward pass on the neural network.
			Parameters:
				obs - observation to pass as input
			Return:
				output - the output of our forward pass
		"""
        # Convert observation to tensor if it's a numpy array
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float).to(device)
        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        activation3 = F.relu(self.layer3(activation2))
        output = F.relu(self.layer4(activation3)) + 1e-6

        return output
