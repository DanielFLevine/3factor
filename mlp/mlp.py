import logging

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, extra_layers=0):
        super(MLP, self).__init__()
        self.embedding_layer = nn.Linear(input_size, hidden_size)
        self.extra_hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(extra_layers)])
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.hidden_reward_combination_layer = nn.Linear(hidden_size, hidden_size)
        self.choice = nn.Linear(hidden_size, 1)
        self.reward_linear = nn.Linear(1, hidden_size)
        self.neuromodulator_out = nn.Linear(hidden_size, 2)
        self.alpha = nn.Parameter(.01 * (2.0 * torch.rand(hidden_size, hidden_size) - 1.0))
        self.value_out = nn.Linear(hidden_size, 1)

    def forward(self, items, plastic_weights, reward):
        # Item shape is (batch_size, 2*item_size + 2) - includes prev_reward and prev_choice
        # Calculate the hidden state
        nonlinear_item_embeddings = torch.tanh(self.embedding_layer(items))
        for layer in self.extra_hidden_layers:
            nonlinear_item_embeddings = torch.tanh(layer(nonlinear_item_embeddings))
        innate_contribution = self.fc2(nonlinear_item_embeddings)

        # Plastic contribution: use item embeddings as presynaptic, apply plastic weights
        plastic_contribution = torch.einsum('bhi,bi->bh', self.alpha * plastic_weights, nonlinear_item_embeddings)

        pre_tanh_hidden = innate_contribution + plastic_contribution
        hidden = torch.tanh(pre_tanh_hidden)

        # Log contribution magnitudes
        self.plastic_contribution_mag = plastic_contribution.abs().mean().item()
        self.current_contribution_mag = innate_contribution.abs().mean().item()

        # Calculate the neuromodulator scalar and choice
        # Ensure reward is shape (batch_size, 1) for the linear layer
        if reward.dim() == 1:
            reward = reward.unsqueeze(-1)
        reward_embedding = self.reward_linear(reward)
        hidden_reward_combination = torch.tanh(self.hidden_reward_combination_layer(hidden+reward_embedding))
        nm_out = torch.tanh(self.neuromodulator_out(hidden_reward_combination))
        neuromodulator = (nm_out[:, 0] - nm_out[:, 1]).unsqueeze(-1).unsqueeze(-1)
        choice = torch.sigmoid(self.choice(hidden))

        # Calculate the value for RL
        value = self.value_out(hidden)

        # Compute Hebbian trace: outer product of presynaptic (item embeddings) and postsynaptic (hidden)
        # Shape: (batch, hidden, hidden) - but we want (batch, hidden_post, hidden_pre)
        hebbian_trace = torch.einsum('bh,bi->bhi', pre_tanh_hidden, nonlinear_item_embeddings)
        hebbian_trace = torch.tanh(hebbian_trace)

        # Update plastic weights with neuromodulator-gated Hebbian update
        plastic_weights = plastic_weights + neuromodulator * hebbian_trace
        torch.clip_(plastic_weights, min=-50.0, max=50.0)

        return choice, neuromodulator, value, plastic_weights, hidden
