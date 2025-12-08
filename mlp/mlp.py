import logging

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.neuromodulator_out = nn.Linear(hidden_size, 2)
        self.alpha = nn.Parameter(.01 * (2.0 * torch.rand(hidden_size, hidden_size) - 1.0))
        self.value_out = nn.Linear(hidden_size, 1)

    def forward(self, items, plastic_weights):
        # Item shape is (batch_size, 2*item_size + 2) - includes prev_reward and prev_choice
        # Calculate the hidden state
        nonlinear_item_embeddings = torch.tanh(self.fc1(items))
        current_contribution = self.fc2(nonlinear_item_embeddings)

        # Plastic contribution: use item embeddings as presynaptic, apply plastic weights
        plastic_contribution = torch.einsum('bhi,bi->bh', self.alpha * plastic_weights, nonlinear_item_embeddings)

        pre_tanh_hidden = current_contribution + plastic_contribution
        hidden = torch.tanh(pre_tanh_hidden)

        # Log contribution magnitudes
        self.plastic_contribution_mag = plastic_contribution.abs().mean().item()
        self.current_contribution_mag = current_contribution.abs().mean().item()

        # Calculate the neuromodulator scalar and choice
        nm_out = torch.tanh(self.neuromodulator_out(hidden))
        neuromodulator = (nm_out[:, 0] - nm_out[:, 1]).unsqueeze(-1).unsqueeze(-1)
        choice = torch.sigmoid(self.fc3(hidden))

        # Calculate the value for RL
        value = self.value_out(hidden)

        # Compute Hebbian trace: outer product of presynaptic (item embeddings) and postsynaptic (hidden)
        # Shape: (batch, hidden, hidden) - but we want (batch, hidden_post, hidden_pre)
        hebbian_trace = torch.einsum('bh,bi->bhi', pre_tanh_hidden, nonlinear_item_embeddings)
        hebbian_trace = torch.tanh(hebbian_trace)*10

        # Update plastic weights with neuromodulator-gated Hebbian update
        plastic_weights = plastic_weights + neuromodulator * hebbian_trace
        torch.clip_(plastic_weights, min=-50.0, max=50.0)

        return choice, neuromodulator, value, plastic_weights, hidden
