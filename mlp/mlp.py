import logging
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class MLPOutput:
    choice: torch.Tensor
    sampled_choices: torch.Tensor
    neuromodulator: torch.Tensor
    value: torch.Tensor
    plastic_weights: torch.Tensor
    hidden: torch.Tensor
    extra_plastic_weights: List[torch.Tensor]
    embeddings: List[torch.Tensor]
    reward: torch.Tensor = None  # Actual reward: +1 if correct, -1 if incorrect

class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        extra_layers=0,
        plastic_weight_clip: Optional[float] = None,
        delay_steps=0,
        use_extra_neuromodulator=False,
        use_answer=False,
        use_sampled_choice_in_reward=False,
        add_additional_hidden_layer_pre_plastic=False,
        add_additional_hidden_layer_post_plastic=False,
        scalar_alpha_layers=None,
        simple_neuromodulator=False,
        simple_neuromodulator_bias=False,
        freeze_neuromodulator_multiplier=False,
        freeze_hebbian_trace_multiplier=False,
        multi_neuromodulator=1,
        multi_neuromodulator_shared_trace=True,
        direct_readout=False,
        use_sigmoid=False,
        use_capped_relu=False,
        single_nm_unit=False,
        linear_hebbian=False,
        no_alpha=False,
        no_embedding=False,
        linear_activation=False,
        ones_readout=False,
        antisymmetric_readout=False,
        antisymmetric_input_init=False,
        strong_antisymmetric_input_init=False,
        normal_init_std=None,
        no_bias_layers=None,
        simple_neuromodulator_init_weight=1.0,
        simple_neuromodulator_init_bias=0.0,
        direct_nm=False,
        direct_nm_pos_init=0.0,
        direct_nm_neg_init=-1.0,
        freeze_direct_nm_pos=False,
        freeze_direct_nm_neg=False,
    ):
        super(MLP, self).__init__()
        self.direct_readout = direct_readout
        self.ones_readout = ones_readout
        self.strong_antisymmetric_input_init = strong_antisymmetric_input_init
        self.no_embedding = no_embedding
        if linear_activation:
            self.activation = lambda x: x
        elif use_capped_relu:
            self.activation = lambda x: torch.clamp(F.relu(x), max=1.0)
        elif use_sigmoid:
            self.activation = torch.sigmoid
        else:
            self.activation = torch.tanh
        self.single_nm_unit = single_nm_unit
        self.linear_hebbian = linear_hebbian
        self.no_alpha = no_alpha
        self.scalar_alpha_layers = set(scalar_alpha_layers) if scalar_alpha_layers else set()
        self.num_extra_layers = extra_layers
        self.no_bias_layers = set(no_bias_layers) if no_bias_layers else set()

        # Compute key dimension variables
        # first_plastic_input_size: input dim of the first layer with plastic weights
        if not no_embedding or add_additional_hidden_layer_pre_plastic:
            self.first_plastic_input_size = hidden_size
        else:
            self.first_plastic_input_size = input_size  # raw item vectors flow directly

        # fc2_input_size: input dim of fc2 (final plastic layer)
        if extra_layers > 0:
            self.fc2_input_size = hidden_size  # last extra layer always outputs hidden_size
        else:
            self.fc2_input_size = self.first_plastic_input_size

        if not no_embedding:
            self.embedding_layer = nn.Linear(input_size, hidden_size, bias=0 not in self.no_bias_layers)
        self.add_additional_hidden_layer_pre_plastic = add_additional_hidden_layer_pre_plastic
        self.add_additional_hidden_layer_post_plastic = add_additional_hidden_layer_post_plastic
        if add_additional_hidden_layer_pre_plastic:
            self.pre_plastic_hidden_layer = nn.Linear(input_size if no_embedding else hidden_size, hidden_size, bias=0 not in self.no_bias_layers)
        if add_additional_hidden_layer_post_plastic:
            self.post_plastic_hidden_layer = nn.Linear(hidden_size, hidden_size)

        # Add all necessary components for extra layers and extra plastic weights
        # scalar_alpha_layers: 1..N for extra layers, N+1 for final layer
        self.extra_hidden_layers = nn.ModuleList([
            nn.Linear(self.first_plastic_input_size if i == 0 else hidden_size, hidden_size, bias=(i + 1) not in self.no_bias_layers)
            for i in range(extra_layers)
        ])
        if not no_alpha:
            self.alpha_extra = nn.ParameterList([
                nn.Parameter(torch.tensor(0.01)) if (i + 1) in self.scalar_alpha_layers
                else nn.Parameter(.01 * (2.0 * torch.rand(hidden_size, self.first_plastic_input_size if i == 0 else hidden_size) - 1.0))
                for i in range(extra_layers)
            ])
        self.neuromodulator_multiplier_extra = nn.ParameterList([torch.nn.Parameter((1.0 * torch.ones(1)), requires_grad=not freeze_neuromodulator_multiplier) for _ in range(extra_layers)])
        self.hebbian_trace_multiplier_extra = nn.ParameterList([torch.nn.Parameter((1.0 * torch.ones(1)), requires_grad=not freeze_hebbian_trace_multiplier) for _ in range(extra_layers)])
        self.hidden_to_reward_extra = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(extra_layers)])

        if direct_readout:
            self.fc2 = nn.Linear(self.fc2_input_size, 1, bias=not ones_readout and (extra_layers + 1) not in self.no_bias_layers)
            if ones_readout:
                nn.init.ones_(self.fc2.weight)
                if antisymmetric_readout:
                    self.fc2.weight.data[:, self.fc2_input_size // 2:] = -1
                self.fc2.weight.requires_grad = False
        else:
            self.fc2 = nn.Linear(self.fc2_input_size, hidden_size, bias=(extra_layers + 1) not in self.no_bias_layers)
            self.choice = nn.Linear(hidden_size, 1, bias=not ones_readout)
            if ones_readout:
                nn.init.ones_(self.choice.weight)
                if antisymmetric_readout:
                    self.choice.weight.data[:, hidden_size // 2:] = -1
                self.choice.weight.requires_grad = False
        nm_hidden_size = 1 if direct_readout else hidden_size
        self.hidden_to_reward = nn.Linear(nm_hidden_size, nm_hidden_size)
        self.use_sampled_choice_in_reward = use_sampled_choice_in_reward
        reward_input_size = 2 if use_sampled_choice_in_reward else 1
        self.reward_linear = nn.Linear(reward_input_size, nm_hidden_size)
        nm_out_size = 1 if single_nm_unit else 2
        self.neuromodulator_out = nn.Linear(nm_hidden_size, nm_out_size)
        self.use_extra_neuromodulator = use_extra_neuromodulator
        if not no_alpha:
            if (self.num_extra_layers + 1) in self.scalar_alpha_layers:
                self.alpha = nn.Parameter(torch.tensor(0.01))
            elif direct_readout:
                self.alpha = nn.Parameter(.01 * (2.0 * torch.rand(1, self.fc2_input_size) - 1.0))
            else:
                self.alpha = nn.Parameter(.01 * (2.0 * torch.rand(hidden_size, self.fc2_input_size) - 1.0))
        self.value_out = nn.Linear(self.fc2_input_size, 1)
        self.plastic_weight_clip = plastic_weight_clip
        self.neuromodulator_multiplier = torch.nn.Parameter((1.0 * torch.ones(1)), requires_grad=not freeze_neuromodulator_multiplier)
        self.hebbian_trace_multiplier = torch.nn.Parameter((1.0 * torch.ones(1)), requires_grad=not freeze_hebbian_trace_multiplier)
        self.delay_steps = delay_steps
        self.multi_neuromodulator = multi_neuromodulator
        self.multi_neuromodulator_shared_trace = multi_neuromodulator_shared_trace

        if use_extra_neuromodulator:
            self.neuromodulator_out_extra = nn.ModuleList([nn.Linear(hidden_size, nm_out_size) for _ in range(extra_layers)])

        self.simple_neuromodulator = simple_neuromodulator
        self.simple_neuromodulator_bias = simple_neuromodulator_bias
        if simple_neuromodulator:
            self.simple_nm_weight = nn.Parameter(torch.tensor(simple_neuromodulator_init_weight))
            if simple_neuromodulator_bias:
                self.simple_nm_bias = nn.Parameter(torch.tensor(simple_neuromodulator_init_bias))
            if use_extra_neuromodulator:
                self.simple_nm_weight_extra = nn.ParameterList([nn.Parameter(torch.tensor(simple_neuromodulator_init_weight)) for _ in range(extra_layers)])
                if simple_neuromodulator_bias:
                    self.simple_nm_bias_extra = nn.ParameterList([nn.Parameter(torch.tensor(simple_neuromodulator_init_bias)) for _ in range(extra_layers)])

        self.direct_nm = direct_nm
        if direct_nm:
            self.direct_nm_pos = nn.Parameter(torch.tensor(direct_nm_pos_init), requires_grad=not freeze_direct_nm_pos)
            self.direct_nm_neg = nn.Parameter(torch.tensor(direct_nm_neg_init), requires_grad=not freeze_direct_nm_neg)
            if use_extra_neuromodulator:
                self.direct_nm_pos_extra = nn.ParameterList([nn.Parameter(torch.tensor(direct_nm_pos_init), requires_grad=not freeze_direct_nm_pos) for _ in range(extra_layers)])
                self.direct_nm_neg_extra = nn.ParameterList([nn.Parameter(torch.tensor(direct_nm_neg_init), requires_grad=not freeze_direct_nm_neg) for _ in range(extra_layers)])

        # Multi-neuromodulator parameters (channels 1..N-1; channel 0 = existing params)
        if multi_neuromodulator > 1:
            N = multi_neuromodulator
            # Final layer: channels 1..N-1
            final_alpha_shape = (1, self.fc2_input_size) if direct_readout else (hidden_size, self.fc2_input_size)
            if not no_alpha:
                self.alpha_multi = nn.ParameterList([
                    nn.Parameter(torch.tensor(0.01)) if (self.num_extra_layers + 1) in self.scalar_alpha_layers
                    else nn.Parameter(.01 * (2.0 * torch.rand(*final_alpha_shape) - 1.0))
                    for _ in range(N - 1)
                ])
            self.neuromodulator_multiplier_multi = nn.ParameterList([
                torch.nn.Parameter((1.0 * torch.ones(1)), requires_grad=not freeze_neuromodulator_multiplier)
                for _ in range(N - 1)
            ])
            if not multi_neuromodulator_shared_trace:
                self.hebbian_trace_multiplier_multi = nn.ParameterList([
                    torch.nn.Parameter((1.0 * torch.ones(1)), requires_grad=not freeze_hebbian_trace_multiplier)
                    for _ in range(N - 1)
                ])

            if simple_neuromodulator:
                self.simple_nm_weight_multi = nn.ParameterList([nn.Parameter(torch.tensor(simple_neuromodulator_init_weight)) for _ in range(N - 1)])
                if simple_neuromodulator_bias:
                    self.simple_nm_bias_multi = nn.ParameterList([nn.Parameter(torch.tensor(simple_neuromodulator_init_bias)) for _ in range(N - 1)])
            else:
                self.hidden_to_reward_multi = nn.ModuleList([nn.Linear(nm_hidden_size, nm_hidden_size) for _ in range(N - 1)])
                self.neuromodulator_out_multi = nn.ModuleList([nn.Linear(nm_hidden_size, nm_out_size) for _ in range(N - 1)])

            # Extra layers: channels 1..N-1
            if not no_alpha:
                self.alpha_extra_multi = nn.ModuleList([
                    nn.ParameterList([
                        nn.Parameter(torch.tensor(0.01)) if (i + 1) in self.scalar_alpha_layers
                        else nn.Parameter(.01 * (2.0 * torch.rand(hidden_size, self.first_plastic_input_size if i == 0 else hidden_size) - 1.0))
                        for _ in range(N - 1)
                    ])
                    for i in range(extra_layers)
                ])
            self.neuromodulator_multiplier_extra_multi = nn.ModuleList([
                nn.ParameterList([
                    torch.nn.Parameter((1.0 * torch.ones(1)), requires_grad=not freeze_neuromodulator_multiplier)
                    for _ in range(N - 1)
                ])
                for i in range(extra_layers)
            ])
            if not multi_neuromodulator_shared_trace:
                self.hebbian_trace_multiplier_extra_multi = nn.ModuleList([
                    nn.ParameterList([
                        torch.nn.Parameter((1.0 * torch.ones(1)), requires_grad=not freeze_hebbian_trace_multiplier)
                        for _ in range(N - 1)
                    ])
                    for i in range(extra_layers)
                ])

            if use_extra_neuromodulator:
                if simple_neuromodulator:
                    self.simple_nm_weight_extra_multi = nn.ModuleList([
                        nn.ParameterList([nn.Parameter(torch.tensor(simple_neuromodulator_init_weight)) for _ in range(N - 1)])
                        for i in range(extra_layers)
                    ])
                    if simple_neuromodulator_bias:
                        self.simple_nm_bias_extra_multi = nn.ModuleList([
                            nn.ParameterList([nn.Parameter(torch.tensor(simple_neuromodulator_init_bias)) for _ in range(N - 1)])
                            for i in range(extra_layers)
                        ])
                else:
                    self.hidden_to_reward_extra_multi = nn.ModuleList([
                        nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(N - 1)])
                        for i in range(extra_layers)
                    ])
                    self.neuromodulator_out_extra_multi = nn.ModuleList([
                        nn.ModuleList([nn.Linear(hidden_size, nm_out_size) for _ in range(N - 1)])
                        for i in range(extra_layers)
                    ])

        # Normal initialization for all linear layers (skip frozen weights like ones_readout)
        if normal_init_std is not None:
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    if module.weight.requires_grad:
                        nn.init.normal_(module.weight, mean=0.0, std=normal_init_std)
                    if module.bias is not None and module.bias.requires_grad:
                        nn.init.zeros_(module.bias)

        # Antisymmetric input init: item 2 columns = -(item 1 columns)
        # Each neuron computes a projection of (X - Y), giving proper antisymmetry.
        if antisymmetric_input_init or strong_antisymmetric_input_init:
            if not no_embedding:
                target = self.embedding_layer
            elif add_additional_hidden_layer_pre_plastic:
                target = self.pre_plastic_hidden_layer
            elif extra_layers > 0:
                target = self.extra_hidden_layers[0]
            else:
                target = self.fc2
            w = target.weight.data
            half_rows = w.shape[0] // 2
            half_cols = w.shape[1] // 2
            if strong_antisymmetric_input_init:
                # [[A, -A], [-A, A]]: all blocks derived from upper-left A
                A = w[:half_rows, :half_cols].clone()
                w[:half_rows, half_cols:] = -A
                w[half_rows:, :half_cols] = -A
                w[half_rows:, half_cols:] = A
            else:
                w[:, half_cols:] = -w[:, :half_cols]

        self.use_answer = use_answer
        self.greedy_sampling = False

    def forward(self, items, plastic_weights, reward, extra_plastic_weights=None, store_embeddings=False):
        N = self.multi_neuromodulator

        # Normalize plastic weights to list format internally
        if N == 1:
            pw_list = [plastic_weights]
            epw_lists = [[epw] for epw in extra_plastic_weights] if extra_plastic_weights else []
        else:
            pw_list = plastic_weights  # already a list of N tensors
            epw_lists = extra_plastic_weights if extra_plastic_weights else []  # list of lists

        # Item shape is (batch_size, 2*item_size + 2) - includes prev_reward and prev_choice
        # Calculate the hidden state
        if self.no_embedding:
            hidden = items
            nonlinear_item_embeddings = items
        else:
            nonlinear_item_embeddings = self.activation(self.embedding_layer(items))
            hidden = nonlinear_item_embeddings
        if self.add_additional_hidden_layer_pre_plastic:
            hidden = self.activation(self.pre_plastic_hidden_layer(hidden))

        embeddings = []
        if store_embeddings:
            embeddings.append(nonlinear_item_embeddings)

        post_hiddens = []
        pre_hiddens = []

        for i, layer in enumerate(self.extra_hidden_layers):
            pre_hiddens.append(hidden)
            innate_contribution = layer(hidden)
            # Sum plastic contributions across all N channels
            if self.no_alpha:
                plastic_contribution = torch.einsum('bhi,bi->bh', epw_lists[i][0], hidden)
                for k in range(1, N):
                    plastic_contribution = plastic_contribution + torch.einsum('bhi,bi->bh', epw_lists[i][k], hidden)
            else:
                plastic_contribution = torch.einsum('bhi,bi->bh', self.alpha_extra[i] * epw_lists[i][0], hidden)
                for k in range(1, N):
                    plastic_contribution = plastic_contribution + torch.einsum('bhi,bi->bh', self.alpha_extra_multi[i][k-1] * epw_lists[i][k], hidden)

            pre_tanh_hidden_extra = innate_contribution + plastic_contribution
            hidden_extra = self.activation(pre_tanh_hidden_extra)
            post_hiddens.append(pre_tanh_hidden_extra)

            if store_embeddings:
                embeddings.append(hidden_extra)

            hidden = hidden_extra

        innate_contribution = self.fc2(hidden)

        # Plastic contribution: sum across all N channels
        if self.no_alpha:
            plastic_contribution = torch.einsum('bhi,bi->bh', pw_list[0], hidden)
            for k in range(1, N):
                plastic_contribution = plastic_contribution + torch.einsum('bhi,bi->bh', pw_list[k], hidden)
        else:
            plastic_contribution = torch.einsum('bhi,bi->bh', self.alpha * pw_list[0], hidden)
            for k in range(1, N):
                plastic_contribution = plastic_contribution + torch.einsum('bhi,bi->bh', self.alpha_multi[k-1] * pw_list[k], hidden)

        if self.direct_readout:
            # fc2 outputs (B, 1), plastic_contribution is (B, 1)
            # No tanh — this is the logit directly
            final_logit = innate_contribution + plastic_contribution  # (B, 1)
            final_hidden = final_logit  # NM uses the scalar fc2 output

            if store_embeddings:
                embeddings.append(final_logit)

            choice = torch.sigmoid(final_logit)
        else:
            final_pre_tanh_hidden = innate_contribution + plastic_contribution
            final_hidden = self.activation(final_pre_tanh_hidden)

            if self.add_additional_hidden_layer_post_plastic:
                final_hidden = self.activation(self.post_plastic_hidden_layer(final_hidden))

            if store_embeddings:
                embeddings.append(final_hidden)

            choice = torch.sigmoid(self.choice(final_hidden))

        # Sample choices (used for reward calculation and logging)
        if self.greedy_sampling:
            sampled_choices = (choice >= 0.5).float()
        else:
            sampled_choices = torch.bernoulli(choice)

        # Ensure reward is shape (batch_size, 1) for the linear layer
        if reward.dim() == 1:
            reward = reward.unsqueeze(-1)
        if not self.use_answer:
            reward = 2 * (reward == sampled_choices).float() - 1

        # Compute reward embedding (shared across all NM channels for complex NM)
        reward_embedding = None
        if not self.simple_neuromodulator:
            if self.use_sampled_choice_in_reward:
                sampled_choices_unsqueezed = sampled_choices.unsqueeze(-1) if sampled_choices.dim() == 1 else sampled_choices
                reward_input = torch.cat([reward, sampled_choices_unsqueezed], dim=-1)
            else:
                reward_input = reward
            reward_embedding = self.reward_linear(reward_input)

        # --- Compute N neuromodulators for the final layer ---
        neuromodulators_final = []
        # Channel 0
        if self.direct_nm:
            pos_mask = (reward > 0).float()  # (batch, 1)
            neg_mask = (reward < 0).float()
            nm_0 = (self.direct_nm_pos * pos_mask + self.direct_nm_neg * neg_mask).unsqueeze(-1)  # (batch, 1, 1)
        elif self.simple_neuromodulator:
            nm_0 = self.simple_nm_weight * reward  # (batch, 1)
            if self.simple_neuromodulator_bias:
                nm_0 = nm_0 + self.simple_nm_bias
            nm_0 = nm_0.unsqueeze(-1)  # (batch, 1, 1)
        else:
            hidden_reward_combination = torch.tanh(self.hidden_to_reward(final_hidden) + reward_embedding)
            nm_out = torch.tanh(self.neuromodulator_out(hidden_reward_combination))
            if self.single_nm_unit:
                nm_0 = nm_out[:, 0].unsqueeze(-1).unsqueeze(-1)
            else:
                nm_0 = (nm_out[:, 0] - nm_out[:, 1]).unsqueeze(-1).unsqueeze(-1)
            nm_0 = self.neuromodulator_multiplier * nm_0
        neuromodulators_final.append(nm_0)

        # Channels 1..N-1
        for k in range(1, N):
            if self.simple_neuromodulator:
                nm_k = self.simple_nm_weight_multi[k-1] * reward
                if self.simple_neuromodulator_bias:
                    nm_k = nm_k + self.simple_nm_bias_multi[k-1]
                nm_k = nm_k.unsqueeze(-1)
            else:
                hidden_reward_combination_k = torch.tanh(self.hidden_to_reward_multi[k-1](final_hidden) + reward_embedding)
                nm_out_k = torch.tanh(self.neuromodulator_out_multi[k-1](hidden_reward_combination_k))
                if self.single_nm_unit:
                    nm_k = nm_out_k[:, 0].unsqueeze(-1).unsqueeze(-1)
                else:
                    nm_k = (nm_out_k[:, 0] - nm_out_k[:, 1]).unsqueeze(-1).unsqueeze(-1)
                nm_k = self.neuromodulator_multiplier_multi[k-1] * nm_k
            neuromodulators_final.append(nm_k)

        # --- Extra layer Hebbian updates ---
        if self.use_extra_neuromodulator and len(self.extra_hidden_layers) > 0:
            tracking_neuromodulator_outputs = []

        for i, (pre_hidden, post_hidden) in enumerate(zip(pre_hiddens, post_hiddens)):
            if self.use_extra_neuromodulator:
                # Compute N neuromodulators for this extra layer
                neuromodulators_extra = []
                # Channel 0
                if self.direct_nm:
                    pos_mask = (reward > 0).float()
                    neg_mask = (reward < 0).float()
                    nm_extra_0 = (self.direct_nm_pos_extra[i] * pos_mask + self.direct_nm_neg_extra[i] * neg_mask).unsqueeze(-1)
                elif self.simple_neuromodulator:
                    nm_extra_0 = self.simple_nm_weight_extra[i] * reward
                    if self.simple_neuromodulator_bias:
                        nm_extra_0 = nm_extra_0 + self.simple_nm_bias_extra[i]
                    nm_extra_0 = nm_extra_0.unsqueeze(-1)
                else:
                    hidden_reward_combination_extra = torch.tanh(self.hidden_to_reward_extra[i](post_hidden) + reward_embedding)
                    nm_out_extra = torch.tanh(self.neuromodulator_out_extra[i](hidden_reward_combination_extra))
                    if self.single_nm_unit:
                        nm_extra_0 = nm_out_extra[:, 0].unsqueeze(-1).unsqueeze(-1)
                    else:
                        nm_extra_0 = (nm_out_extra[:, 0] - nm_out_extra[:, 1]).unsqueeze(-1).unsqueeze(-1)
                    nm_extra_0 = self.neuromodulator_multiplier_extra[i] * nm_extra_0
                neuromodulators_extra.append(nm_extra_0)
                tracking_neuromodulator_outputs.append(nm_extra_0)

                # Channels 1..N-1
                for k in range(1, N):
                    if self.simple_neuromodulator:
                        nm_extra_k = self.simple_nm_weight_extra_multi[i][k-1] * reward
                        if self.simple_neuromodulator_bias:
                            nm_extra_k = nm_extra_k + self.simple_nm_bias_extra_multi[i][k-1]
                        nm_extra_k = nm_extra_k.unsqueeze(-1)
                    else:
                        hidden_reward_combination_extra_k = torch.tanh(self.hidden_to_reward_extra_multi[i][k-1](post_hidden) + reward_embedding)
                        nm_out_extra_k = torch.tanh(self.neuromodulator_out_extra_multi[i][k-1](hidden_reward_combination_extra_k))
                        if self.single_nm_unit:
                            nm_extra_k = nm_out_extra_k[:, 0].unsqueeze(-1).unsqueeze(-1)
                        else:
                            nm_extra_k = (nm_out_extra_k[:, 0] - nm_out_extra_k[:, 1]).unsqueeze(-1).unsqueeze(-1)
                        nm_extra_k = self.neuromodulator_multiplier_extra_multi[i][k-1] * nm_extra_k
                    neuromodulators_extra.append(nm_extra_k)
            else:
                # Share final layer neuromodulators
                neuromodulators_extra = neuromodulators_final

            # Compute Hebbian trace once (outer product of pre/post)
            hebbian_trace_extra_raw = torch.einsum('bh,bi->bhi', post_hidden, pre_hidden)

            # Per-channel update
            for k in range(N):
                _trace_activated = hebbian_trace_extra_raw if self.linear_hebbian else torch.tanh(hebbian_trace_extra_raw)
                if self.multi_neuromodulator_shared_trace or k == 0:
                    trace_k = _trace_activated * self.hebbian_trace_multiplier_extra[i]
                else:
                    trace_k = _trace_activated * self.hebbian_trace_multiplier_extra_multi[i][k-1]
                epw_lists[i][k] = epw_lists[i][k] + neuromodulators_extra[k] * trace_k
                if self.plastic_weight_clip is not None:
                    epw_lists[i][k] = torch.clamp(epw_lists[i][k], min=-self.plastic_weight_clip, max=self.plastic_weight_clip)

        # Calculate the value for RL
        value = self.value_out(hidden)

        # Compute Hebbian trace for final layer
        if self.direct_readout:
            # Post-synaptic signal is the scalar logit (B, 1), pre-synaptic is hidden (B, hidden_size)
            # Trace shape: (B, 1, hidden_size) — scales the hidden vector by the logit
            trace_post = final_logit + 1 if (self.strong_antisymmetric_input_init and self.ones_readout) else final_logit
            hebbian_trace_raw = torch.einsum('bh,bi->bhi', trace_post, hidden)
        else:
            hebbian_trace_raw = torch.einsum('bh,bi->bhi', final_pre_tanh_hidden, hidden)

        # Per-channel update for final layer
        _trace_activated_final = hebbian_trace_raw if self.linear_hebbian else torch.tanh(hebbian_trace_raw)
        for k in range(N):
            if self.multi_neuromodulator_shared_trace or k == 0:
                trace_k = _trace_activated_final * self.hebbian_trace_multiplier
            else:
                trace_k = _trace_activated_final * self.hebbian_trace_multiplier_multi[k-1]
            pw_list[k] = pw_list[k] + neuromodulators_final[k] * trace_k
            if self.plastic_weight_clip is not None:
                pw_list[k] = torch.clamp(pw_list[k], min=-self.plastic_weight_clip, max=self.plastic_weight_clip)

        # --- Delay steps (final layer only) ---
        # Build tracking neuromodulator from channel 0 (legacy shape for delay compatibility)
        neuromodulator = neuromodulators_final[0]
        if self.delay_steps > 0:
            if self.simple_neuromodulator:
                # Simple NM doesn't depend on hidden state; append same value for tracking
                nm_val = neuromodulator
                for _ in range(self.delay_steps):
                    neuromodulator = torch.cat((neuromodulator, nm_val), dim=-1)
            else:
                # Starting point
                preactivation_hidden = pre_tanh_hidden_reward_combination
                postactivation_hidden = hidden_reward_combination
                prev_postactivation_hidden = final_hidden
                for _ in range(self.delay_steps):
                    preactivation_hidden, postactivation_hidden, prev_postactivation_hidden, pw_list, neuromodulator_delay = self.delay_step(preactivation_hidden, postactivation_hidden, prev_postactivation_hidden, pw_list)
                    neuromodulator = torch.cat((neuromodulator, neuromodulator_delay), dim=-1)

        # Concatenate all channel NMs for tracking output
        for k in range(1, N):
            neuromodulator = torch.cat((neuromodulator, neuromodulators_final[k]), dim=-1)

        if self.use_extra_neuromodulator and len(self.extra_hidden_layers) > 0:
            neuromodulator = torch.cat(tracking_neuromodulator_outputs + [neuromodulator], dim=-1)

        # Unwrap to legacy format when N=1
        if N == 1:
            return_pw = pw_list[0]
            return_epw = [epw_lists[i][0] for i in range(len(epw_lists))]
        else:
            return_pw = pw_list
            return_epw = epw_lists

        return MLPOutput(
            choice=choice,
            sampled_choices=sampled_choices,
            neuromodulator=neuromodulator,
            value=value,
            plastic_weights=return_pw,
            hidden=hidden,
            extra_plastic_weights=return_epw,
            embeddings=embeddings,
            reward=reward,
        )

    def delay_step(self, preactivation_hidden, postactivation_hidden, prev_postactivation_hidden, pw_list):
        N = self.multi_neuromodulator

        # Sum all channels for forward pass
        innate_delay_contribution = self.fc2(postactivation_hidden)
        if self.no_alpha:
            plastic_delay_contribution = torch.einsum('bhi,bi->bh', pw_list[0], postactivation_hidden)
            for k in range(1, N):
                plastic_delay_contribution = plastic_delay_contribution + torch.einsum('bhi,bi->bh', pw_list[k], postactivation_hidden)
        else:
            plastic_delay_contribution = torch.einsum('bhi,bi->bh', self.alpha * pw_list[0], postactivation_hidden)
            for k in range(1, N):
                plastic_delay_contribution = plastic_delay_contribution + torch.einsum('bhi,bi->bh', self.alpha_multi[k-1] * pw_list[k], postactivation_hidden)
        pre_tanh_hidden_delay = innate_delay_contribution + plastic_delay_contribution
        hidden_delay = self.activation(pre_tanh_hidden_delay)

        # Compute per-channel NMs
        neuromodulators_delay = []
        # Channel 0
        nm_out_delay_0 = torch.tanh(self.neuromodulator_out(hidden_delay))
        if self.single_nm_unit:
            nm_delay_0 = nm_out_delay_0[:, 0].unsqueeze(-1).unsqueeze(-1)
        else:
            nm_delay_0 = (nm_out_delay_0[:, 0] - nm_out_delay_0[:, 1]).unsqueeze(-1).unsqueeze(-1)
        nm_delay_0 = self.neuromodulator_multiplier * nm_delay_0
        neuromodulators_delay.append(nm_delay_0)
        # Channels 1..N-1
        for k in range(1, N):
            nm_out_k = torch.tanh(self.neuromodulator_out_multi[k-1](hidden_delay))
            if self.single_nm_unit:
                nm_k = nm_out_k[:, 0].unsqueeze(-1).unsqueeze(-1)
            else:
                nm_k = (nm_out_k[:, 0] - nm_out_k[:, 1]).unsqueeze(-1).unsqueeze(-1)
            nm_k = self.neuromodulator_multiplier_multi[k-1] * nm_k
            neuromodulators_delay.append(nm_k)

        # Compute Hebbian trace once
        hebbian_trace_delay_raw = torch.einsum('bh,bi->bhi', preactivation_hidden, prev_postactivation_hidden)

        # Per-channel update
        _trace_activated_delay = hebbian_trace_delay_raw if self.linear_hebbian else torch.tanh(hebbian_trace_delay_raw)
        for k in range(N):
            if self.multi_neuromodulator_shared_trace or k == 0:
                trace_k = _trace_activated_delay * self.hebbian_trace_multiplier
            else:
                trace_k = _trace_activated_delay * self.hebbian_trace_multiplier_multi[k-1]
            pw_list[k] = pw_list[k] + neuromodulators_delay[k] * trace_k
            if self.plastic_weight_clip is not None:
                pw_list[k] = torch.clamp(pw_list[k], min=-self.plastic_weight_clip, max=self.plastic_weight_clip)

        # Concatenate channel 0 NM for tracking (legacy shape)
        neuromodulator_delay = neuromodulators_delay[0]

        return pre_tanh_hidden_delay, hidden_delay, postactivation_hidden, pw_list, neuromodulator_delay


def create_plastic_weights(batch_size, hidden_size, extra_layers, multi_neuromodulator, device, direct_readout=False, first_plastic_input_size=None):
    """Create plastic weights for the model.

    When multi_neuromodulator=1 (default), returns legacy format:
        pw: (B, H, H) tensor (or (B, 1, H) if direct_readout)
        epw: list of (B, H, H) tensors
    When multi_neuromodulator>1, returns list format:
        pw: list of N (B, H, H) tensors (or (B, 1, H) if direct_readout)
        epw: list of N-element lists, each containing (B, H, H) tensors

    first_plastic_input_size: input dimension of the first plastic layer.
        Defaults to hidden_size if not specified.
    """
    if first_plastic_input_size is None:
        first_plastic_input_size = hidden_size
    N = multi_neuromodulator
    # fc2_input_size: if extra_layers > 0, last extra layer outputs hidden_size; else first_plastic_input_size
    fc2_input_size = hidden_size if extra_layers > 0 else first_plastic_input_size
    final_pw_shape = (batch_size, 1, fc2_input_size) if direct_readout else (batch_size, hidden_size, fc2_input_size)
    if N == 1:
        pw = torch.zeros(*final_pw_shape, dtype=torch.float32, requires_grad=False).to(device)
        epw = [torch.zeros(batch_size, hidden_size, first_plastic_input_size if i == 0 else hidden_size, dtype=torch.float32, requires_grad=False).to(device) for i in range(extra_layers)]
        return pw, epw
    else:
        pw = [torch.zeros(*final_pw_shape, dtype=torch.float32, requires_grad=False).to(device) for _ in range(N)]
        epw = [[torch.zeros(batch_size, hidden_size, first_plastic_input_size if i == 0 else hidden_size, dtype=torch.float32, requires_grad=False).to(device) for _ in range(N)] for i in range(extra_layers)]
        return pw, epw


def detach_plastic_weights(pw, epw, multi_neuromodulator):
    """Detach plastic weights between episodes."""
    N = multi_neuromodulator
    if N == 1:
        pw = pw.detach()
        epw = [epw[i].detach() for i in range(len(epw))]
    else:
        pw = [pw[k].detach() for k in range(N)]
        epw = [[epw[i][k].detach() for k in range(N)] for i in range(len(epw))]
    return pw, epw


def clone_plastic_weights(pw, epw):
    """Deep clone plastic weights. Auto-detects single-tensor vs list format."""
    if isinstance(pw, list):
        cloned_pw = [p.clone() for p in pw]
        cloned_epw = [[e.clone() for e in layer_epw] for layer_epw in epw]
    else:
        cloned_pw = pw.clone()
        cloned_epw = [e.clone() for e in epw]
    return cloned_pw, cloned_epw


def pw_batch_size(pw):
    """Get batch size from plastic weights. Auto-detects format."""
    if isinstance(pw, list):
        return pw[0].shape[0]
    return pw.shape[0]


def repeat_interleave_pw(pw, epw, repeats):
    """Repeat-interleave plastic weights along batch dim. Auto-detects format."""
    if isinstance(pw, list):
        ri_pw = [p.repeat_interleave(repeats, dim=0) for p in pw]
        ri_epw = [[e.repeat_interleave(repeats, dim=0) for e in layer_epw] for layer_epw in epw]
    else:
        ri_pw = pw.repeat_interleave(repeats, dim=0)
        ri_epw = [e.repeat_interleave(repeats, dim=0) for e in epw]
    return ri_pw, ri_epw


def zero_plastic_weights(pw, epw=None):
    """Zero out plastic weights in-place. Auto-detects format."""
    if isinstance(pw, list):
        for p in pw:
            p.zero_()
        if epw is not None:
            for layer_epw in epw:
                for e in layer_epw:
                    e.zero_()
    else:
        pw.zero_()
        if epw is not None:
            for e in epw:
                e.zero_()


def pw_mask_set(pw, mask, values):
    """Set pw[mask] = values[mask]. Auto-detects format. Modifies pw in-place."""
    if isinstance(pw, list):
        for k in range(len(pw)):
            pw[k][mask] = values[k][mask]
    else:
        pw[mask] = values[mask]


def pw_mask_set_scaled(pw, mask, base, ratio):
    """Set pw[mask] = base[mask] + ratio * (pw[mask] - base[mask]). Auto-detects format. Modifies pw in-place."""
    if isinstance(pw, list):
        for k in range(len(pw)):
            pw[k][mask] = base[k][mask] + ratio * (pw[k][mask] - base[k][mask])
    else:
        pw[mask] = base[mask] + ratio * (pw[mask] - base[mask])
