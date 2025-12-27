import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        extra_layers=0,
        plastic_weight_clip: Optional[float] = None,
        delay_steps=0
    ):
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
        self.hebbian_trace_multiplier = nn.Parameter(torch.tensor(1.0))
        self.plastic_weight_clip = plastic_weight_clip
        self.neuromodulator_multiplier = torch.nn.Parameter((1.0 * torch.ones(1)), requires_grad=True)
        self.hebbian_trace_multiplier = torch.nn.Parameter((1.0 * torch.ones(1)), requires_grad=True)
        self.delay_steps = delay_steps

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
        choice = torch.sigmoid(self.choice(hidden))

        # Calculate the neuromodulator scalar and choice
        # Ensure reward is shape (batch_size, 1) for the linear layer
        if reward.dim() == 1:
            reward = reward.unsqueeze(-1)
        reward_embedding = self.reward_linear(reward)
        pre_tanh_hidden_reward_combination = hidden + reward_embedding
        hidden_reward_combination = torch.tanh(self.hidden_reward_combination_layer(pre_tanh_hidden_reward_combination))
        nm_out = torch.tanh(self.neuromodulator_out(hidden_reward_combination))
        neuromodulator = (nm_out[:, 0] - nm_out[:, 1]).unsqueeze(-1).unsqueeze(-1)
        neuromodulator = self.neuromodulator_multiplier * neuromodulator

        # Calculate the value for RL
        value = self.value_out(hidden)

        # Compute Hebbian trace: outer product of presynaptic (item embeddings) and postsynaptic (hidden)
        # Shape: (batch, hidden, hidden) - but we want (batch, hidden_post, hidden_pre)
        hebbian_trace = torch.einsum('bh,bi->bhi', pre_tanh_hidden, nonlinear_item_embeddings)
        hebbian_trace = torch.tanh(hebbian_trace) * self.hebbian_trace_multiplier

        # Update plastic weights with neuromodulator-gated Hebbian update
        plastic_weights = plastic_weights + neuromodulator * hebbian_trace
        if self.plastic_weight_clip is not None:
            plastic_weights = torch.clamp(plastic_weights, min=-self.plastic_weight_clip, max=self.plastic_weight_clip)

        if self.delay_steps > 0:
            # Starting point
            preactivation_hidden = pre_tanh_hidden_reward_combination
            postactivation_hidden = hidden_reward_combination
            prev_postactivation_hidden = hidden
            for _ in range(self.delay_steps):
                preactivation_hidden, postactivation_hidden, prev_postactivation_hidden, plastic_weights = self.delay_step(preactivation_hidden, postactivation_hidden, prev_postactivation_hidden, plastic_weights)
            
        return choice, neuromodulator, value, plastic_weights, hidden

    def delay_step(self, preactivation_hidden, postactivation_hidden, prev_postactivation_hidden, plastic_weights):
        innate_delay_contribution = self.fc2(postactivation_hidden)
        plastic_delay_contribution = torch.einsum('bhi,bi->bh', self.alpha * plastic_weights, postactivation_hidden)
        pre_tanh_hidden_delay = innate_delay_contribution + plastic_delay_contribution
        hidden_delay = torch.tanh(pre_tanh_hidden_delay)

        nm_out_delay = torch.tanh(self.neuromodulator_out(hidden_delay))
        neuromodulator_delay = (nm_out_delay[:, 0] - nm_out_delay[:, 1]).unsqueeze(-1).unsqueeze(-1)
        neuromodulator_delay = self.neuromodulator_multiplier * neuromodulator_delay

        hebbian_trace_delay = torch.einsum('bh,bi->bhi', preactivation_hidden, prev_postactivation_hidden)
        hebbian_trace_delay = torch.tanh(hebbian_trace_delay) * self.hebbian_trace_multiplier

        plastic_weights = plastic_weights + neuromodulator_delay * hebbian_trace_delay
        if self.plastic_weight_clip is not None:
            plastic_weights = torch.clamp(plastic_weights, min=-self.plastic_weight_clip, max=self.plastic_weight_clip)
        
        return pre_tanh_hidden_delay, hidden_delay, postactivation_hidden, plastic_weights