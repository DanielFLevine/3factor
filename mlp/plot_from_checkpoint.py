"""
Load a model from a checkpoint and plot TI readout correlation evolution to wandb.

For each of 10 networks, tracks how the correlation between each item's
representation (final layer) and the readout weights changes trial-by-trial.

Produces two plot types per network:
  1. Heatmap: items (y) x trials (x), pair label on each column
  2. Line plot: one line per item over trials, pair labels at each tick

Plus one pre-training baseline (shared, since plastic weights start at zero).
"""
import argparse
import itertools
import logging
import os
import warnings
from argparse import Namespace

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
from tqdm import tqdm
import wandb

from mlp import MLP, create_plastic_weights, clone_plastic_weights, pw_batch_size, repeat_interleave_pw, zero_plastic_weights, pw_mask_set, pw_mask_set_scaled
from generate_data import generate_batch_items, generate_batch_trials_ti, generate_batch_trials_ll
from plots import zero_shot_symbolic_distance_plot, delta_symbolic_distance_plot, plot_innate_weight_analysis, training_accuracy_by_trial_plot, pair_logit_by_trial_heatmap, item_dot_product_heatmaps

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", message=".*tight_layout.*")


def compute_item_readout_correlations(model, items, plastic_weights, extra_plastic_weights, readout_weights, final_layer_idx, device):
    """Compute correlation of each item's final-layer embedding with readout weights.

    Returns:
        pos1_corrs: array of shape (num_items,) - item in position 1 (left)
        pos2_corrs: array of shape (num_items,) - item in position 2 (right)
    """
    num_items = items.shape[0]
    item_size = items.shape[1]
    zeros = np.zeros(item_size)
    dummy_reward = torch.tensor([0.0], dtype=torch.float32).to(device)

    pos1_corrs = np.zeros(num_items)
    pos2_corrs = np.zeros(num_items)

    for item_idx in range(num_items):
        item = items[item_idx]

        # Position 1: [item, zeros]
        input_pos1 = np.concatenate([item, zeros])
        tensor_pos1 = torch.tensor(input_pos1, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.inference_mode():
            out_pos1 = model(tensor_pos1, plastic_weights, dummy_reward,
                             extra_plastic_weights=extra_plastic_weights, store_embeddings=True, embed_plastic_weights=embed_pw)
        emb_pos1 = out_pos1.embeddings[final_layer_idx][0].detach().cpu().numpy()

        if np.std(emb_pos1) > 1e-10:
            pos1_corrs[item_idx] = np.corrcoef(emb_pos1, readout_weights)[0, 1]

        # Position 2: [zeros, item]
        input_pos2 = np.concatenate([zeros, item])
        tensor_pos2 = torch.tensor(input_pos2, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.inference_mode():
            out_pos2 = model(tensor_pos2, plastic_weights, dummy_reward,
                             extra_plastic_weights=extra_plastic_weights, store_embeddings=True, embed_plastic_weights=embed_pw)
        emb_pos2 = out_pos2.embeddings[final_layer_idx][0].detach().cpu().numpy()

        if np.std(emb_pos2) > 1e-10:
            pos2_corrs[item_idx] = np.corrcoef(emb_pos2, readout_weights)[0, 1]

    return pos1_corrs, pos2_corrs


def compute_item_logits(model, items, plastic_weights, extra_plastic_weights, device):
    """Compute readout logit for each item presented alone in each position.

    Returns:
        pos1_logits: array of shape (num_items,) - item in position 1 (left), zeros in position 2
        pos2_logits: array of shape (num_items,) - item in position 2 (right), zeros in position 1
    """
    num_items = items.shape[0]
    item_size = items.shape[1]
    zeros = np.zeros(item_size)
    dummy_reward = torch.tensor([0.0], dtype=torch.float32).to(device)

    pos1_logits = np.zeros(num_items)
    pos2_logits = np.zeros(num_items)

    for item_idx in range(num_items):
        item = items[item_idx]

        for pos, logits_arr in [(1, pos1_logits), (2, pos2_logits)]:
            if pos == 1:
                inp = np.concatenate([item, zeros])
            else:
                inp = np.concatenate([zeros, item])
            tensor = torch.tensor(inp, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.inference_mode():
                out = model(tensor, plastic_weights, dummy_reward,
                            extra_plastic_weights=extra_plastic_weights, embed_plastic_weights=embed_pw)
            prob = out.choice.squeeze().item()
            prob = np.clip(prob, 1e-7, 1 - 1e-7)
            logits_arr[item_idx] = np.log(prob / (1 - prob))

    return pos1_logits, pos2_logits


def generate_single_pass_trials(batch_items, num_items):
    """Generate exactly num_items-1 training trials, one per adjacent pair, random positions, shuffled."""
    batch_size = batch_items.shape[0]
    item_size = batch_items.shape[2]
    num_pairs = num_items - 1

    trials = np.zeros((batch_size, num_pairs, 2 * item_size))
    correct_choices = np.zeros((batch_size, num_pairs))

    for b in range(batch_size):
        indices = list(range(num_pairs))
        np.random.shuffle(indices)  # random trial order
        for slot, pair_idx in enumerate(indices):
            high_idx, low_idx = pair_idx, pair_idx + 1
            high_item = batch_items[b, high_idx]
            low_item = batch_items[b, low_idx]
            if np.random.rand() < 0.5:
                # High on left
                trials[b, slot] = np.concatenate([high_item, low_item])
                correct_choices[b, slot] = 0.0
            else:
                # High on right
                trials[b, slot] = np.concatenate([low_item, high_item])
                correct_choices[b, slot] = 1.0

    return trials, correct_choices


def generate_high_left_ordered_trials(batch_items, num_items):
    """Generate num_items-1 trials in order AB BC CD ... FG GH (high item always on left)."""
    batch_size = batch_items.shape[0]
    item_size = batch_items.shape[2]
    num_pairs = num_items - 1

    trials = np.zeros((batch_size, num_pairs, 2 * item_size))
    correct_choices = np.zeros((batch_size, num_pairs))

    for b in range(batch_size):
        for pair_idx in range(num_pairs):
            high_item = batch_items[b, pair_idx]
            low_item = batch_items[b, pair_idx + 1]
            trials[b, pair_idx] = np.concatenate([high_item, low_item])
            correct_choices[b, pair_idx] = 0.0

    return trials, correct_choices


def generate_high_left_ordered_trials_skip_de(batch_items, num_items):
    """Generate num_items-1 trials in high-left order, with DE moved to last.
    Produces trials: AB BC CD EF FG GH DE (7 trials for 8 items)."""
    batch_size = batch_items.shape[0]
    item_size = batch_items.shape[2]
    skip_idx = 3  # DE pair — moved to end
    num_trials = num_items - 1

    trials = np.zeros((batch_size, num_trials, 2 * item_size))
    correct_choices = np.zeros((batch_size, num_trials))

    for b in range(batch_size):
        trial_slot = 0
        for pair_idx in range(num_items - 1):
            if pair_idx == skip_idx:
                continue
            high_item = batch_items[b, pair_idx]
            low_item = batch_items[b, pair_idx + 1]
            trials[b, trial_slot] = np.concatenate([high_item, low_item])
            correct_choices[b, trial_slot] = 0.0
            trial_slot += 1
        # Append DE as the final trial
        high_item = batch_items[b, skip_idx]
        low_item = batch_items[b, skip_idx + 1]
        trials[b, trial_slot] = np.concatenate([high_item, low_item])
        correct_choices[b, trial_slot] = 0.0

    return trials, correct_choices


def generate_high_right_ordered_trials(batch_items, num_items):
    """Generate num_items-1 trials in order BA CB DC ... GF HG (high item always on right)."""
    batch_size = batch_items.shape[0]
    item_size = batch_items.shape[2]
    num_pairs = num_items - 1

    trials = np.zeros((batch_size, num_pairs, 2 * item_size))
    correct_choices = np.zeros((batch_size, num_pairs))

    for b in range(batch_size):
        for pair_idx in range(num_pairs):
            high_item = batch_items[b, pair_idx]
            low_item = batch_items[b, pair_idx + 1]
            trials[b, pair_idx] = np.concatenate([low_item, high_item])
            correct_choices[b, pair_idx] = 1.0

    return trials, correct_choices


def generate_fixed_order_trials(batch_items, num_items):
    """Generate 2*(num_items-1) trials in canonical order: all high-left, then all high-right."""
    batch_size = batch_items.shape[0]
    item_size = batch_items.shape[2]
    num_pairs = num_items - 1
    num_trials = 2 * num_pairs

    trials = np.zeros((batch_size, num_trials, 2 * item_size))
    correct_choices = np.zeros((batch_size, num_trials))

    for b in range(batch_size):
        for pair_idx in range(num_pairs):
            high_idx, low_idx = pair_idx, pair_idx + 1
            high_item = batch_items[b, high_idx]
            low_item = batch_items[b, low_idx]
            # First half: high on left (AB, BC, CD, ...)
            trials[b, pair_idx] = np.concatenate([high_item, low_item])
            correct_choices[b, pair_idx] = 0.0
            # Second half: high on right (BA, CB, DC, ...)
            trials[b, num_pairs + pair_idx] = np.concatenate([low_item, high_item])
            correct_choices[b, num_pairs + pair_idx] = 1.0

    return trials, correct_choices


def compute_numpy_layer1_embeddings(inputs_np, epw_np, W_embed_np, W_extra_np_0, alpha_extra_np_0):
    """Compute layer 1 hidden representations via numpy forward pass (no bias).

    Matches plot_innate_weight_analysis: h = tanh(W @ input), then h = tanh(W_extra @ h + alpha * pw @ h).

    Args:
        inputs_np: (batch, input_size) numpy array
        epw_np: (batch, hidden, hidden) numpy array - extra plastic weights for layer 0
        W_embed_np: (hidden, input_size) embedding weight matrix
        W_extra_np_0: (hidden, hidden) first extra layer weight matrix
        alpha_extra_np_0: alpha for first extra layer (scalar or (hidden, hidden) matrix)

    Returns:
        h: (batch, hidden) layer 1 output (post-tanh)
    """
    # Embedding layer (no bias)
    h = np.tanh(inputs_np @ W_embed_np.T)  # (batch, hidden)
    # First extra hidden layer (no bias, with plastic)
    innate = h @ W_extra_np_0.T  # (batch, hidden)
    plastic = np.einsum('bhi,bi->bh', alpha_extra_np_0 * epw_np, h)  # (batch, hidden)
    h = np.tanh(innate + plastic)
    return h


def compute_batch_item_logits_fast(model, batch_items, plastic_weights, extra_plastic_weights,
                                    num_networks, num_items, item_size, device,
                                    return_embeddings_at_layers=None):
    """Compute item logits for all networks in a single batched forward pass.

    Args:
        return_embeddings_at_layers: if not None, list of layer indices to also return embeddings for.

    Returns:
        pos1_logits: (num_networks, num_items)
        pos2_logits: (num_networks, num_items)
        [if return_embeddings_at_layers]: embeddings_dict mapping layer_idx -> (pos1_emb, pos2_emb)
            where pos1_emb, pos2_emb have shape (num_networks, num_items, hidden_size)
    """
    zeros = np.zeros(item_size)
    items_per_net = num_items * 2
    total_batch = num_networks * items_per_net

    all_inputs = []
    for net_idx in range(num_networks):
        for item_idx in range(num_items):
            item = batch_items[net_idx][item_idx]
            all_inputs.append(np.concatenate([item, zeros]))   # pos1
            all_inputs.append(np.concatenate([zeros, item]))   # pos2

    batch_input = torch.tensor(np.stack(all_inputs), dtype=torch.float32).to(device)
    batch_pw, batch_epw, batch_embed_pw = repeat_interleave_pw(plastic_weights, extra_plastic_weights, items_per_net, embed_pw=embed_pw)
    batch_reward = torch.zeros(total_batch, dtype=torch.float32).to(device)

    store_embs = return_embeddings_at_layers is not None
    with torch.inference_mode():
        out = model(batch_input, batch_pw, batch_reward, extra_plastic_weights=batch_epw,
                    store_embeddings=store_embs, embed_plastic_weights=batch_embed_pw)

    probs = out.choice.squeeze(-1).detach().cpu().numpy()
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    all_logits = np.log(probs / (1 - probs))

    all_logits = all_logits.reshape(num_networks, num_items, 2)

    if store_embs:
        embeddings_dict = {}
        for layer_idx in return_embeddings_at_layers:
            layer_emb = out.embeddings[layer_idx].detach().cpu().numpy()
            layer_emb = layer_emb.reshape(num_networks, num_items, 2, -1)
            embeddings_dict[layer_idx] = (layer_emb[:, :, 0, :], layer_emb[:, :, 1, :])
        return all_logits[:, :, 0], all_logits[:, :, 1], embeddings_dict

    return all_logits[:, :, 0], all_logits[:, :, 1]


def compute_adjacent_pair_logits(model, items, plastic_weights, extra_plastic_weights, num_items, device):
    """Compute readout logits for all adjacent pairs (both orders).

    Returns:
        logits: array of shape (num_adjacent_pairs,) in order AB, BA, BC, CB, ...
    """
    item_size = items.shape[1]
    dummy_reward = torch.tensor([0.0], dtype=torch.float32).to(device)
    logits = []

    for i in range(num_items - 1):
        for left_idx, right_idx in [(i, i + 1), (i + 1, i)]:
            pair_input = np.concatenate([items[left_idx], items[right_idx]])
            pair_tensor = torch.tensor(pair_input, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.inference_mode():
                out = model(pair_tensor, plastic_weights, dummy_reward,
                            extra_plastic_weights=extra_plastic_weights, embed_plastic_weights=embed_pw)
            prob = out.choice.squeeze().item()
            prob = np.clip(prob, 1e-7, 1 - 1e-7)
            logits.append(np.log(prob / (1 - prob)))

    return np.array(logits)


def run_simulation(model, batch_items, saved_args, num_train_trials, num_networks, num_items, item_labels, final_layer_idx, readout_weights, device):
    """Run TI trial simulation and collect correlations, neuromodulator, logits, rewards.

    Returns:
        all_corrs: dict[net_idx] -> list of (pos1_corrs, pos2_corrs)
        trial_labels: dict[net_idx] -> list of pair label strings
        nm_values: dict[net_idx] -> list of floats
        logit_values: dict[net_idx] -> list of floats (readout logit per trial)
        reward_values: dict[net_idx] -> list of floats (+1 correct, -1 incorrect)
        adj_pair_logits: dict[net_idx] -> list of arrays, each (num_adj_pairs,)
        item_logits: dict[net_idx] -> list of (pos1_logits, pos2_logits)
        num_train_trials: actual number of train trials used
    """
    trials, correct_choices, pair_indices, num_train_trials = generate_batch_trials_ti(
        batch_items, num_train_trials, 0, arbitrary=saved_args.arbitrary
    )
    trials_t = torch.tensor(trials, dtype=torch.float32).to(device)
    correct_choices_t = torch.tensor(correct_choices, dtype=torch.float32).to(device)

    # Initialize plastic weights
    plastic_weights, extra_plastic_weights, embed_pw = create_plastic_weights(num_networks, saved_args.hidden_size, saved_args.extra_layers, getattr(saved_args, 'multi_neuromodulator', 1), device, direct_readout=getattr(saved_args, 'direct_readout', False), first_plastic_input_size=getattr(model, 'first_plastic_input_size', saved_args.hidden_size), plastic_embedding=getattr(saved_args, 'plastic_embedding', False), input_size=2*saved_args.item_size)

    all_corrs = {net_idx: [] for net_idx in range(num_networks)}
    trial_labels = {net_idx: [] for net_idx in range(num_networks)}
    nm_values = {net_idx: [] for net_idx in range(num_networks)}
    logit_values = {net_idx: [] for net_idx in range(num_networks)}
    reward_values = {net_idx: [] for net_idx in range(num_networks)}
    adj_pair_logits = {net_idx: [] for net_idx in range(num_networks)}
    item_logits = {net_idx: [] for net_idx in range(num_networks)}

    # Pre-training correlations, item logits, and adjacent pair logits
    logger.info("Computing pre-training correlations and pair logits...")
    for net_idx in range(num_networks):
        single_pw = plastic_weights[net_idx:net_idx+1]
        single_epw = [epw[net_idx:net_idx+1] for epw in extra_plastic_weights]
        pos1, pos2 = compute_item_readout_correlations(
            model, batch_items[net_idx], single_pw, single_epw,
            readout_weights, final_layer_idx, device
        )
        all_corrs[net_idx].append((pos1, pos2))
        il_pos1, il_pos2 = compute_item_logits(
            model, batch_items[net_idx], single_pw, single_epw, device
        )
        item_logits[net_idx].append((il_pos1, il_pos2))
        pair_logits = compute_adjacent_pair_logits(
            model, batch_items[net_idx], single_pw, single_epw, num_items, device
        )
        adj_pair_logits[net_idx].append(pair_logits)

    # Run through trials
    for trial_idx in range(num_train_trials):
        if trial_idx % 5 == 0:
            logger.info(f"Processing trial {trial_idx}/{num_train_trials}...")

        for net_idx in range(num_networks):
            high_item = int(pair_indices[net_idx, trial_idx, 0])
            low_item = int(pair_indices[net_idx, trial_idx, 1])
            if correct_choices[net_idx, trial_idx] == 0:
                pos1_item, pos2_item = high_item, low_item
            else:
                pos1_item, pos2_item = low_item, high_item
            trial_labels[net_idx].append(f"{item_labels[pos1_item]}{item_labels[pos2_item]}")

        batch_trial = trials_t[:, trial_idx, :]
        batch_correct = correct_choices_t[:, trial_idx]
        with torch.inference_mode():
            output = model(batch_trial, plastic_weights, batch_correct,
                          extra_plastic_weights=extra_plastic_weights, embed_plastic_weights=embed_pw)
        plastic_weights = output.plastic_weights
        extra_plastic_weights = output.extra_plastic_weights
        embed_pw = output.embed_plastic_weights

        nm = output.neuromodulator.detach().cpu().numpy()
        choice_prob = output.choice.squeeze(-1).detach().cpu().numpy()
        logits = np.log(np.clip(choice_prob, 1e-7, 1 - 1e-7) / np.clip(1 - choice_prob, 1e-7, 1))
        sampled = output.sampled_choices.squeeze(-1).detach().cpu().numpy()
        correct = correct_choices[:, trial_idx]
        rewards = 2.0 * (sampled == correct).astype(np.float64) - 1.0

        for net_idx in range(num_networks):
            nm_values[net_idx].append(float(nm[net_idx].flatten()[0]))
            logit_values[net_idx].append(float(logits[net_idx]))
            reward_values[net_idx].append(float(rewards[net_idx]))

        for net_idx in range(num_networks):
            single_pw = plastic_weights[net_idx:net_idx+1]
            single_epw = [epw[net_idx:net_idx+1] for epw in extra_plastic_weights]
            pos1, pos2 = compute_item_readout_correlations(
                model, batch_items[net_idx], single_pw, single_epw,
                readout_weights, final_layer_idx, device
            )
            all_corrs[net_idx].append((pos1, pos2))
            il_pos1, il_pos2 = compute_item_logits(
                model, batch_items[net_idx], single_pw, single_epw, device
            )
            item_logits[net_idx].append((il_pos1, il_pos2))
            pair_logits = compute_adjacent_pair_logits(
                model, batch_items[net_idx], single_pw, single_epw, num_items, device
            )
            adj_pair_logits[net_idx].append(pair_logits)

    return all_corrs, trial_labels, nm_values, logit_values, reward_values, adj_pair_logits, item_logits, num_train_trials


def run_aggregate_delta_analysis(model, saved_args, num_episodes, num_train_trials, num_items, item_labels, device, batch_size=50, external_pc1=None):
    """Run many episodes and collect per-trial item logit deltas grouped by trial type and reward.

    Also collects dot-product deltas between trial pair and individual item hidden representations
    at each extra hidden layer (when extra_layers > 0).

    Returns:
        delta_storage_pos1: [trial_num][pair_idx][reward_key] -> list of (num_items,) arrays
        delta_storage_pos2: [trial_num][pair_idx][reward_key] -> list of (num_items,) arrays
        nm_storage: [trial_num][pair_idx][reward_key] -> list of floats (NM scalars)
        adj_pair_labels: list of pair label strings (length 14 for 8 items)
        actual_num_trials: number of training trials used
        dot_delta_pos1: [layer][trial_num][pair_idx][reward_key] -> list of (num_items,) arrays (or empty if no extra layers)
        dot_delta_pos2: same structure as dot_delta_pos1
        dot_abs_pos1: [layer][trial_num][pair_idx][reward_key] -> list of (num_items,) arrays (post-trial dot products)
        dot_abs_pos2: same structure as dot_abs_pos1
        post_dot_pos1/pos2: [layer] -> list of (num_items, num_adj_pairs) arrays (extra layers + final layer)
        post_corr_pos1/pos2: same structure
        post_pair_dot: [layer] -> list of (num_adj_pairs, num_adj_pairs) arrays (pairwise trial-type dot products)
        post_pair_corr: [layer] -> list of (num_adj_pairs, num_adj_pairs) arrays (pairwise trial-type correlations)
        nonadj_pair_labels: list of nonadjacent pair label strings (length 42 for 8 items)
        post_nonadj_logits: list of (num_nonadj_pairs,) arrays
        post_cross_dot: [layer] -> list of (num_adj_pairs, num_nonadj_pairs) arrays (cross dot products)
        post_cross_corr: [layer] -> list of (num_adj_pairs, num_nonadj_pairs) arrays (cross correlations)
        post_dot_nonadj_pos1/pos2: [layer] -> list of (num_items, num_nonadj_pairs) arrays (item-vs-nonadj dot products)
        post_corr_nonadj_pos1/pos2: same structure for correlations
    """
    item_size = saved_args.item_size
    num_adj_pairs = 2 * (num_items - 1)
    num_extra_layers = saved_args.extra_layers
    # Layer indices for extra hidden layers in model embeddings: 1..extra_layers
    extra_layer_indices = list(range(1, num_extra_layers + 1)) if num_extra_layers > 0 else None

    adj_pair_labels = []
    for i in range(num_items - 1):
        adj_pair_labels.append(f"{item_labels[i]}{item_labels[i+1]}")
        adj_pair_labels.append(f"{item_labels[i+1]}{item_labels[i]}")
    pair_label_to_idx = {label: idx for idx, label in enumerate(adj_pair_labels)}

    # Storage: [trial_num][pair_idx][reward_key] -> list of delta arrays
    # reward_key: 0 = positive (+1), 1 = negative (-1)
    delta_storage_pos1 = [[{0: [], 1: []} for _ in range(num_adj_pairs)] for _ in range(num_train_trials)]
    delta_storage_pos2 = [[{0: [], 1: []} for _ in range(num_adj_pairs)] for _ in range(num_train_trials)]
    nm_storage = [[{0: [], 1: []} for _ in range(num_adj_pairs)] for _ in range(num_train_trials)]

    # Dot product delta storage: [layer_idx][trial_num][pair_idx][reward_key] -> list of (num_items,)
    dot_delta_pos1 = [[[{0: [], 1: []} for _ in range(num_adj_pairs)] for _ in range(num_train_trials)] for _ in range(num_extra_layers)]
    dot_delta_pos2 = [[[{0: [], 1: []} for _ in range(num_adj_pairs)] for _ in range(num_train_trials)] for _ in range(num_extra_layers)]
    # Absolute (post-trial) dot product storage: same structure
    dot_abs_pos1 = [[[{0: [], 1: []} for _ in range(num_adj_pairs)] for _ in range(num_train_trials)] for _ in range(num_extra_layers)]
    dot_abs_pos2 = [[[{0: [], 1: []} for _ in range(num_adj_pairs)] for _ in range(num_train_trials)] for _ in range(num_extra_layers)]
    # Pair-vs-pair dot product: trial emb dotted with all 14 pair embeddings
    # [layer_idx][trial_num][presented_pair_idx][reward_key] -> list of (num_adj_pairs,) arrays
    num_pair_dot_layers = num_extra_layers + 1  # extra layers + final layer
    pair_dot_delta = [[[{0: [], 1: []} for _ in range(num_adj_pairs)] for _ in range(num_train_trials)] for _ in range(num_pair_dot_layers)]
    pair_dot_abs = [[[{0: [], 1: []} for _ in range(num_adj_pairs)] for _ in range(num_train_trials)] for _ in range(num_pair_dot_layers)]
    # Pairwise pair-pair dot products: dot(pair_i_emb, pair_j_emb) for all i,j
    # [layer_idx][trial_num][presented_pair_idx][reward_key] -> list of (num_adj_pairs, num_adj_pairs) arrays
    pair_pair_dot_delta = [[[{0: [], 1: []} for _ in range(num_adj_pairs)] for _ in range(num_train_trials)] for _ in range(num_pair_dot_layers)]
    pair_pair_dot_abs = [[[{0: [], 1: []} for _ in range(num_adj_pairs)] for _ in range(num_train_trials)] for _ in range(num_pair_dot_layers)]
    # Pair logit changes: logits for all 14 pairs after presenting a trial
    # [trial_num][presented_pair_idx][reward_key] -> list of (num_adj_pairs,) arrays
    pair_logit_delta = [[{0: [], 1: []} for _ in range(num_adj_pairs)] for _ in range(num_train_trials)]
    pair_logit_abs = [[{0: [], 1: []} for _ in range(num_adj_pairs)] for _ in range(num_train_trials)]
    # Post-training dot product and correlation: [layer_idx] -> list of (num_items, num_adj_pairs) arrays
    # Includes extra layers (0..num_extra_layers-1) + final layer (index num_extra_layers)
    num_post_layers = num_extra_layers + 1  # extra layers + final layer
    post_dot_pos1 = [[] for _ in range(num_post_layers)]
    post_dot_pos2 = [[] for _ in range(num_post_layers)]
    post_corr_pos1 = [[] for _ in range(num_post_layers)]
    post_corr_pos2 = [[] for _ in range(num_post_layers)]
    # Pairwise trial-type dot product and correlation: [layer_idx] -> list of (num_adj_pairs, num_adj_pairs) arrays
    post_pair_dot = [[] for _ in range(num_post_layers)]
    post_pair_corr = [[] for _ in range(num_post_layers)]
    # Post-training logits per trial type: list of (num_adj_pairs,) arrays
    post_logits = []

    # Nonadjacent pair labels and storage
    nonadj_pair_labels = []
    for i in range(num_items - 1):
        for j in range(i + 2, num_items):
            nonadj_pair_labels.append(f"{item_labels[i]}{item_labels[j]}")
            nonadj_pair_labels.append(f"{item_labels[j]}{item_labels[i]}")
    num_nonadj_pairs = len(nonadj_pair_labels)
    post_nonadj_logits = []  # list of (num_nonadj_pairs,) arrays
    post_cross_dot = [[] for _ in range(num_post_layers)]   # [layer] -> list of (num_adj, num_nonadj) arrays
    post_cross_corr = [[] for _ in range(num_post_layers)]  # same structure
    # Item-vs-nonadjacent-pair dot product and correlation: [layer] -> list of (num_items, num_nonadj_pairs) arrays
    post_dot_nonadj_pos1 = [[] for _ in range(num_post_layers)]
    post_dot_nonadj_pos2 = [[] for _ in range(num_post_layers)]
    post_corr_nonadj_pos1 = [[] for _ in range(num_post_layers)]
    post_corr_nonadj_pos2 = [[] for _ in range(num_post_layers)]
    # Post-training individual item logits: list of (num_items,) arrays per position
    post_item_logits_pos1 = []
    post_item_logits_pos2 = []

    # Layer 1 PC1 analysis: fit PCA on post-training individual item embeddings only, then project pair embeddings onto PC1
    hidden_size = saved_args.hidden_size
    l1_emb_sum = np.zeros(hidden_size, dtype=np.float64)
    l1_emb_sq_sum = np.zeros((hidden_size, hidden_size), dtype=np.float64)
    l1_n = 0
    # Per-batch embedding sequences for computing PC1 projections after PCA is fit
    # Each entry: (init_embs, [(trial_idx, new_embs, labels, rewards, bs), ...])
    l1_batch_records = []
    # Post-training layer 1 pair embeddings for PC1 bar chart
    l1_post_train_embs = []  # list of (num_adj_pairs, hidden) arrays
    l1_post_train_nonadj_embs = []  # list of (num_nonadj_pairs, hidden) arrays
    l1_post_train_item_embs_pos1 = []  # list of (num_items, hidden) arrays
    l1_post_train_item_embs_pos2 = []  # list of (num_items, hidden) arrays

    # Extract innate weights and alphas for numpy forward pass (no bias, matching pca_frozen)
    # Used to compute individual item embeddings for PC1 analysis
    if hasattr(model, 'embedding_layer'):
        W_embed_np = model.embedding_layer.weight.detach().cpu().numpy()  # (hidden_size, input_size)
    else:
        W_embed_np = None
    W_extra_np = [model.extra_hidden_layers[i].weight.detach().cpu().numpy()
                  for i in range(saved_args.extra_layers)]
    alpha_extra_np = [model.alpha_extra[i].detach().cpu().numpy()
                      for i in range(saved_args.extra_layers)]

    num_batches = (num_episodes + batch_size - 1) // batch_size
    actual_num_trials = num_train_trials

    for batch_idx in range(num_batches):
        current_batch_size = min(batch_size, num_episodes - batch_idx * batch_size)
        logger.info(f"Aggregate batch {batch_idx+1}/{num_batches} ({current_batch_size} episodes)...")

        batch_items = generate_batch_items(num_items, item_size, current_batch_size, change_items_throughout_batch=True)
        trials, correct_choices, pair_indices, actual_num_trials = generate_batch_trials_ti(
            batch_items, num_train_trials, 0, arbitrary=saved_args.arbitrary
        )
        trials_t = torch.tensor(trials, dtype=torch.float32).to(device)
        correct_choices_t = torch.tensor(correct_choices, dtype=torch.float32).to(device)

        plastic_weights, extra_plastic_weights, embed_pw = create_plastic_weights(current_batch_size, saved_args.hidden_size, saved_args.extra_layers, getattr(saved_args, 'multi_neuromodulator', 1), device, direct_readout=getattr(saved_args, 'direct_readout', False), first_plastic_input_size=getattr(model, 'first_plastic_input_size', saved_args.hidden_size), plastic_embedding=getattr(saved_args, 'plastic_embedding', False), input_size=2*saved_args.item_size)

        # Pre-training item logits (and embeddings if extra layers exist)
        if extra_layer_indices:
            prev_pos1, prev_pos2, prev_item_embs = compute_batch_item_logits_fast(
                model, batch_items, plastic_weights, extra_plastic_weights,
                current_batch_size, num_items, item_size, device,
                return_embeddings_at_layers=extra_layer_indices
            )
        else:
            prev_pos1, prev_pos2 = compute_batch_item_logits_fast(
                model, batch_items, plastic_weights, extra_plastic_weights,
                current_batch_size, num_items, item_size, device
            )

        # Precompute pair input tensor (fixed across trials) and initial pair logits
        all_pair_inputs_train = []
        for net_idx in range(current_batch_size):
            for i in range(num_items - 1):
                all_pair_inputs_train.append(np.concatenate([batch_items[net_idx][i], batch_items[net_idx][i+1]]))
                all_pair_inputs_train.append(np.concatenate([batch_items[net_idx][i+1], batch_items[net_idx][i]]))
        pair_input_tensor = torch.tensor(np.stack(all_pair_inputs_train), dtype=torch.float32).to(device)

        pair_pw_init, pair_epw_init, pair_embed_pw_init = repeat_interleave_pw(plastic_weights, extra_plastic_weights, num_adj_pairs, embed_pw=embed_pw)
        pair_rew_init = torch.zeros(len(all_pair_inputs_train), dtype=torch.float32).to(device)
        with torch.inference_mode():
            pair_out_init = model(pair_input_tensor, pair_pw_init, pair_rew_init,
                                 extra_plastic_weights=pair_epw_init,
                                 store_embeddings=True, embed_plastic_weights=embed_pw)
        # Extract initial pair logits
        init_pair_probs = pair_out_init.choice.squeeze(-1).detach().cpu().numpy()
        init_pair_probs = np.clip(init_pair_probs, 1e-7, 1 - 1e-7)
        prev_pair_logits = np.log(init_pair_probs / (1 - init_pair_probs)).reshape(current_batch_size, num_adj_pairs)

        final_emb_idx = num_extra_layers + 1
        prev_pair_embs = {}
        if extra_layer_indices:
            for layer_idx in extra_layer_indices:
                emb = pair_out_init.embeddings[layer_idx].detach().cpu().numpy()
                prev_pair_embs[layer_idx] = emb.reshape(current_batch_size, num_adj_pairs, -1)

        # Compute layer 1 init pair embeddings via numpy (no bias) for PC1 projection
        l1_batch_trial_records = []
        if extra_layer_indices:
            pair_inputs_init_np = np.stack(all_pair_inputs_train)  # (batch*num_adj, input_size)
            epw0_init = np.zeros((current_batch_size * num_adj_pairs, hidden_size, hidden_size))
            l1_init_embs = compute_numpy_layer1_embeddings(
                pair_inputs_init_np, epw0_init,
                W_embed_np, W_extra_np[0], alpha_extra_np[0]
            ).reshape(current_batch_size, num_adj_pairs, -1)
        else:
            l1_init_embs = None

        # Always add final layer
        emb = pair_out_init.embeddings[final_emb_idx].detach().cpu().numpy()
        prev_pair_embs[final_emb_idx] = emb.reshape(current_batch_size, num_adj_pairs, -1)

        for trial_idx in range(actual_num_trials):
            # Determine trial type labels
            trial_type_labels = []
            for net_idx in range(current_batch_size):
                high_item = int(pair_indices[net_idx, trial_idx, 0])
                low_item = int(pair_indices[net_idx, trial_idx, 1])
                if correct_choices[net_idx, trial_idx] == 0:
                    p1, p2 = high_item, low_item
                else:
                    p1, p2 = low_item, high_item
                trial_type_labels.append(f"{item_labels[p1]}{item_labels[p2]}")

            # Run trial (with store_embeddings to get pre-update trial pair representations)
            batch_trial = trials_t[:, trial_idx, :]
            batch_correct = correct_choices_t[:, trial_idx]
            with torch.inference_mode():
                output = model(batch_trial, plastic_weights, batch_correct,
                              extra_plastic_weights=extra_plastic_weights,
                              store_embeddings=True, embed_plastic_weights=embed_pw)
            plastic_weights = output.plastic_weights
            extra_plastic_weights = output.extra_plastic_weights

            # Compute rewards and extract NM
            sampled = output.sampled_choices.squeeze(-1).detach().cpu().numpy()
            correct = correct_choices[:, trial_idx]
            rewards = 2.0 * (sampled == correct).astype(np.float64) - 1.0
            nm_vals = output.neuromodulator.detach().cpu().numpy().flatten()

            # Compute new item logits (and embeddings)
            if extra_layer_indices:
                new_pos1, new_pos2, new_item_embs = compute_batch_item_logits_fast(
                    model, batch_items, plastic_weights, extra_plastic_weights,
                    current_batch_size, num_items, item_size, device,
                    return_embeddings_at_layers=extra_layer_indices
                )
            else:
                new_pos1, new_pos2 = compute_batch_item_logits_fast(
                    model, batch_items, plastic_weights, extra_plastic_weights,
                    current_batch_size, num_items, item_size, device
                )

            # Compute logit deltas and store
            delta_pos1 = new_pos1 - prev_pos1  # (current_batch_size, num_items)
            delta_pos2 = new_pos2 - prev_pos2

            for net_idx in range(current_batch_size):
                pair_idx = pair_label_to_idx[trial_type_labels[net_idx]]
                reward_key = 0 if rewards[net_idx] > 0 else 1
                delta_storage_pos1[trial_idx][pair_idx][reward_key].append(delta_pos1[net_idx].copy())
                delta_storage_pos2[trial_idx][pair_idx][reward_key].append(delta_pos2[net_idx].copy())
                nm_storage[trial_idx][pair_idx][reward_key].append(float(nm_vals[net_idx]))

            # Post-update trial pair embeddings need a forward pass with updated weights
            dummy_reward = torch.zeros(current_batch_size, dtype=torch.float32).to(device)
            with torch.inference_mode():
                post_output = model(batch_trial, plastic_weights, dummy_reward,
                                   extra_plastic_weights=extra_plastic_weights,
                                   store_embeddings=True, embed_plastic_weights=embed_pw)

            # Compute dot product deltas at extra layers
            if extra_layer_indices:
                for li, layer_idx in enumerate(extra_layer_indices):
                    # Trial pair embeddings: (current_batch_size, hidden_size)
                    trial_emb_pre = output.embeddings[layer_idx].detach().cpu().numpy()
                    trial_emb_post = post_output.embeddings[layer_idx].detach().cpu().numpy()

                    for pos_idx in range(2):
                        # Item embeddings: (current_batch_size, num_items, hidden_size)
                        item_emb_pre = prev_item_embs[layer_idx][pos_idx]
                        item_emb_post = new_item_embs[layer_idx][pos_idx]

                        # Dot products: (current_batch_size, num_items)
                        dot_pre = np.einsum('bh,bih->bi', trial_emb_pre, item_emb_pre)
                        dot_post = np.einsum('bh,bih->bi', trial_emb_post, item_emb_post)
                        delta_dot = dot_post - dot_pre

                        for net_idx in range(current_batch_size):
                            pair_idx = pair_label_to_idx[trial_type_labels[net_idx]]
                            reward_key = 0 if rewards[net_idx] > 0 else 1
                            if pos_idx == 0:
                                dot_delta_pos1[li][trial_idx][pair_idx][reward_key].append(delta_dot[net_idx].copy())
                                dot_abs_pos1[li][trial_idx][pair_idx][reward_key].append(dot_post[net_idx].copy())
                            else:
                                dot_delta_pos2[li][trial_idx][pair_idx][reward_key].append(delta_dot[net_idx].copy())
                                dot_abs_pos2[li][trial_idx][pair_idx][reward_key].append(dot_post[net_idx].copy())

            # Compute new pair logits (and embeddings if extra layers) for all 14 pair types
            new_pair_pw, new_pair_epw, new_pair_embed_pw = repeat_interleave_pw(plastic_weights, extra_plastic_weights, num_adj_pairs, embed_pw=embed_pw)
            pair_rew = torch.zeros(current_batch_size * num_adj_pairs, dtype=torch.float32).to(device)
            with torch.inference_mode():
                new_pair_out = model(pair_input_tensor, new_pair_pw, pair_rew,
                                    extra_plastic_weights=new_pair_epw,
                                    store_embeddings=True, embed_plastic_weights=embed_pw)

            # Extract pair logits and compute delta/abs
            new_pair_probs = new_pair_out.choice.squeeze(-1).detach().cpu().numpy()
            new_pair_probs = np.clip(new_pair_probs, 1e-7, 1 - 1e-7)
            new_pair_logits = np.log(new_pair_probs / (1 - new_pair_probs)).reshape(current_batch_size, num_adj_pairs)
            delta_pair_logits = new_pair_logits - prev_pair_logits

            for net_idx in range(current_batch_size):
                pair_idx = pair_label_to_idx[trial_type_labels[net_idx]]
                reward_key = 0 if rewards[net_idx] > 0 else 1
                pair_logit_abs[trial_idx][pair_idx][reward_key].append(new_pair_logits[net_idx].copy())
                pair_logit_delta[trial_idx][pair_idx][reward_key].append(delta_pair_logits[net_idx].copy())

            prev_pair_logits = new_pair_logits

            new_pair_embs = {}
            if extra_layer_indices:
                for layer_idx in extra_layer_indices:
                    emb = new_pair_out.embeddings[layer_idx].detach().cpu().numpy()
                    new_pair_embs[layer_idx] = emb.reshape(current_batch_size, num_adj_pairs, -1)

                for li, layer_idx in enumerate(extra_layer_indices):
                    trial_emb_pre = output.embeddings[layer_idx].detach().cpu().numpy()
                    trial_emb_post = post_output.embeddings[layer_idx].detach().cpu().numpy()
                    p_emb_pre = prev_pair_embs[layer_idx]   # (batch, num_adj_pairs, hidden)
                    p_emb_post = new_pair_embs[layer_idx]    # (batch, num_adj_pairs, hidden)

                    p_dot_pre = np.einsum('bh,bih->bi', trial_emb_pre, p_emb_pre)
                    p_dot_post = np.einsum('bh,bih->bi', trial_emb_post, p_emb_post)
                    p_delta = p_dot_post - p_dot_pre

                    for net_idx in range(current_batch_size):
                        pair_idx = pair_label_to_idx[trial_type_labels[net_idx]]
                        reward_key = 0 if rewards[net_idx] > 0 else 1
                        pair_dot_delta[li][trial_idx][pair_idx][reward_key].append(p_delta[net_idx].copy())
                        pair_dot_abs[li][trial_idx][pair_idx][reward_key].append(p_dot_post[net_idx].copy())

                prev_item_embs = new_item_embs

            # Always add final layer pair embeddings and dot products
            emb = new_pair_out.embeddings[final_emb_idx].detach().cpu().numpy()
            new_pair_embs[final_emb_idx] = emb.reshape(current_batch_size, num_adj_pairs, -1)

            trial_emb_pre_final = output.embeddings[final_emb_idx].detach().cpu().numpy()
            trial_emb_post_final = post_output.embeddings[final_emb_idx].detach().cpu().numpy()
            p_emb_pre_final = prev_pair_embs[final_emb_idx]
            p_emb_post_final = new_pair_embs[final_emb_idx]

            p_dot_pre = np.einsum('bh,bih->bi', trial_emb_pre_final, p_emb_pre_final)
            p_dot_post = np.einsum('bh,bih->bi', trial_emb_post_final, p_emb_post_final)
            p_delta = p_dot_post - p_dot_pre

            final_li = num_extra_layers  # last index in pair_dot_delta
            for net_idx in range(current_batch_size):
                pair_idx = pair_label_to_idx[trial_type_labels[net_idx]]
                reward_key = 0 if rewards[net_idx] > 0 else 1
                pair_dot_delta[final_li][trial_idx][pair_idx][reward_key].append(p_delta[net_idx].copy())
                pair_dot_abs[final_li][trial_idx][pair_idx][reward_key].append(p_dot_post[net_idx].copy())

            # Compute pairwise pair-pair dot products for all layers
            for layer_idx_key, li_key in list(
                [(idx, i) for i, idx in enumerate(extra_layer_indices)] if extra_layer_indices else []
            ) + [(final_emb_idx, num_extra_layers)]:
                pp_pre = prev_pair_embs[layer_idx_key]   # (batch, num_adj_pairs, hidden)
                pp_post = new_pair_embs[layer_idx_key]    # (batch, num_adj_pairs, hidden)
                pp_dot_pre = np.einsum('bih,bjh->bij', pp_pre, pp_pre)
                pp_dot_post = np.einsum('bih,bjh->bij', pp_post, pp_post)
                pp_delta = pp_dot_post - pp_dot_pre

                for net_idx in range(current_batch_size):
                    pair_idx = pair_label_to_idx[trial_type_labels[net_idx]]
                    reward_key = 0 if rewards[net_idx] > 0 else 1
                    pair_pair_dot_delta[li_key][trial_idx][pair_idx][reward_key].append(pp_delta[net_idx].copy())
                    pair_pair_dot_abs[li_key][trial_idx][pair_idx][reward_key].append(pp_dot_post[net_idx].copy())

            # Compute layer 1 pair embeddings via numpy (no bias) for PC1 projection
            if extra_layer_indices:
                epw0_trial_np = extra_plastic_weights[0].detach().cpu().numpy()
                epw0_trial_repeated = np.repeat(epw0_trial_np, num_adj_pairs, axis=0)
                l1_new = compute_numpy_layer1_embeddings(
                    np.stack(all_pair_inputs_train), epw0_trial_repeated,
                    W_embed_np, W_extra_np[0], alpha_extra_np[0]
                ).reshape(current_batch_size, num_adj_pairs, -1)
                l1_batch_trial_records.append((
                    trial_idx, l1_new.copy(),
                    list(trial_type_labels), rewards.copy(), current_batch_size
                ))

            prev_pair_embs = new_pair_embs

            prev_pos1 = new_pos1
            prev_pos2 = new_pos2

        # Post-training evaluation: dot product and correlation between all adjacent
        # pair representations and individual item representations at each extra layer + final layer
        final_layer_idx = num_extra_layers + 1
        post_eval_layers = list(range(1, num_extra_layers + 1)) + [final_layer_idx]

        # Compute item logits and embeddings at all evaluation layers with final trained weights
        item_logits_p1, item_logits_p2, post_item_embs = compute_batch_item_logits_fast(
            model, batch_items, plastic_weights, extra_plastic_weights,
            current_batch_size, num_items, item_size, device,
            return_embeddings_at_layers=post_eval_layers
        )
        for net_idx in range(current_batch_size):
            post_item_logits_pos1.append(item_logits_p1[net_idx].copy())
            post_item_logits_pos2.append(item_logits_p2[net_idx].copy())

        # Compute layer 1 item embeddings via numpy forward pass (no bias) for PC1 analysis
        if extra_layer_indices:
            epw0_np = extra_plastic_weights[0].detach().cpu().numpy()  # (batch, hidden, hidden)
            zeros = np.zeros(item_size)
            # Build individual item inputs: pos1 = [item, zeros], pos2 = [zeros, item]
            all_item_inputs_p1 = []
            all_item_inputs_p2 = []
            for net_idx in range(current_batch_size):
                for item_idx in range(num_items):
                    item = batch_items[net_idx][item_idx]
                    all_item_inputs_p1.append(np.concatenate([item, zeros]))
                    all_item_inputs_p2.append(np.concatenate([zeros, item]))
            items_per_net = num_items
            epw0_repeated = np.repeat(epw0_np, items_per_net, axis=0)  # (batch*num_items, h, h)
            l1_item_p1 = compute_numpy_layer1_embeddings(
                np.stack(all_item_inputs_p1), epw0_repeated,
                W_embed_np, W_extra_np[0], alpha_extra_np[0]
            ).reshape(current_batch_size, num_items, -1)
            l1_item_p2 = compute_numpy_layer1_embeddings(
                np.stack(all_item_inputs_p2), epw0_repeated,
                W_embed_np, W_extra_np[0], alpha_extra_np[0]
            ).reshape(current_batch_size, num_items, -1)
            for net_idx in range(current_batch_size):
                l1_post_train_item_embs_pos1.append(l1_item_p1[net_idx].copy())
                l1_post_train_item_embs_pos2.append(l1_item_p2[net_idx].copy())
            # Accumulate covariance for PCA from post-training individual item embeddings only
            flat_p1 = l1_item_p1.reshape(-1, hidden_size)
            flat_p2 = l1_item_p2.reshape(-1, hidden_size)
            flat_items = np.concatenate([flat_p1, flat_p2], axis=0)
            l1_emb_sum += flat_items.sum(axis=0)
            l1_emb_sq_sum += flat_items.T @ flat_items
            l1_n += flat_items.shape[0]

        # Compute pair embeddings for all 14 adjacent pair types
        all_pair_inputs = []
        for net_idx in range(current_batch_size):
            for i in range(num_items - 1):
                all_pair_inputs.append(np.concatenate([batch_items[net_idx][i], batch_items[net_idx][i+1]]))
                all_pair_inputs.append(np.concatenate([batch_items[net_idx][i+1], batch_items[net_idx][i]]))

        pair_tensor = torch.tensor(np.stack(all_pair_inputs), dtype=torch.float32).to(device)
        pair_pw, pair_epw, pair_embed_pw = repeat_interleave_pw(plastic_weights, extra_plastic_weights, num_adj_pairs, embed_pw=embed_pw)
        pair_reward = torch.zeros(len(all_pair_inputs), dtype=torch.float32).to(device)

        with torch.inference_mode():
            pair_output = model(pair_tensor, pair_pw, pair_reward,
                               extra_plastic_weights=pair_epw, store_embeddings=True, embed_plastic_weights=pair_embed_pw)

        # Extract post-training logits per trial type
        pair_probs = pair_output.choice.squeeze(-1).detach().cpu().numpy()
        pair_probs = np.clip(pair_probs, 1e-7, 1 - 1e-7)
        pair_logits = np.log(pair_probs / (1 - pair_probs))
        pair_logits = pair_logits.reshape(current_batch_size, num_adj_pairs)
        for net_idx in range(current_batch_size):
            post_logits.append(pair_logits[net_idx].copy())

        # Compute post-training layer 1 pair embeddings via numpy (no bias) for PC1 projection
        if extra_layer_indices:
            pair_inputs_np = np.stack(all_pair_inputs)  # (batch*num_adj_pairs, input_size)
            epw0_pair = np.repeat(epw0_np, num_adj_pairs, axis=0)  # (batch*num_adj_pairs, h, h)
            l1_post_emb = compute_numpy_layer1_embeddings(
                pair_inputs_np, epw0_pair,
                W_embed_np, W_extra_np[0], alpha_extra_np[0]
            ).reshape(current_batch_size, num_adj_pairs, -1)
            for net_idx in range(current_batch_size):
                l1_post_train_embs.append(l1_post_emb[net_idx].copy())

        # Save batch records for PC1 projection computation
        if extra_layer_indices:
            l1_batch_records.append((l1_init_embs, l1_batch_trial_records))

        for li, layer_idx in enumerate(post_eval_layers):
            pair_emb = pair_output.embeddings[layer_idx].detach().cpu().numpy()
            pair_emb = pair_emb.reshape(current_batch_size, num_adj_pairs, -1)

            for pos_idx in range(2):
                item_emb = post_item_embs[layer_idx][pos_idx]  # (batch, num_items, hidden_size)

                # Dot product: (batch, num_items, num_adj_pairs)
                dot = np.einsum('bih,bjh->bij', item_emb, pair_emb)

                # Correlation: normalize then dot
                item_c = item_emb - item_emb.mean(axis=-1, keepdims=True)
                pair_c = pair_emb - pair_emb.mean(axis=-1, keepdims=True)
                item_norm = np.sqrt((item_c ** 2).sum(axis=-1, keepdims=True)) + 1e-10
                pair_norm = np.sqrt((pair_c ** 2).sum(axis=-1, keepdims=True)) + 1e-10
                corr = np.einsum('bih,bjh->bij', item_c / item_norm, pair_c / pair_norm)

                for net_idx in range(current_batch_size):
                    if pos_idx == 0:
                        post_dot_pos1[li].append(dot[net_idx].copy())
                        post_corr_pos1[li].append(corr[net_idx].copy())
                    else:
                        post_dot_pos2[li].append(dot[net_idx].copy())
                        post_corr_pos2[li].append(corr[net_idx].copy())

            # Pairwise trial-type dot product and correlation: (batch, 14, 14)
            pair_dot = np.einsum('bih,bjh->bij', pair_emb, pair_emb)
            pair_c = pair_emb - pair_emb.mean(axis=-1, keepdims=True)
            pair_norm = np.sqrt((pair_c ** 2).sum(axis=-1, keepdims=True)) + 1e-10
            pair_c_normed = pair_c / pair_norm
            pair_corr = np.einsum('bih,bjh->bij', pair_c_normed, pair_c_normed)

            for net_idx in range(current_batch_size):
                post_pair_dot[li].append(pair_dot[net_idx].copy())
                post_pair_corr[li].append(pair_corr[net_idx].copy())

        # Nonadjacent pair forward pass
        all_nonadj_inputs = []
        for net_idx in range(current_batch_size):
            for i in range(num_items - 1):
                for j in range(i + 2, num_items):
                    all_nonadj_inputs.append(np.concatenate([batch_items[net_idx][i], batch_items[net_idx][j]]))
                    all_nonadj_inputs.append(np.concatenate([batch_items[net_idx][j], batch_items[net_idx][i]]))

        nonadj_tensor = torch.tensor(np.stack(all_nonadj_inputs), dtype=torch.float32).to(device)
        nonadj_pw, nonadj_epw, nonadj_embed_pw = repeat_interleave_pw(plastic_weights, extra_plastic_weights, num_nonadj_pairs, embed_pw=embed_pw)
        nonadj_reward = torch.zeros(len(all_nonadj_inputs), dtype=torch.float32).to(device)

        with torch.inference_mode():
            nonadj_output = model(nonadj_tensor, nonadj_pw, nonadj_reward,
                                  extra_plastic_weights=nonadj_epw, store_embeddings=True, embed_plastic_weights=nonadj_embed_pw)

        # Extract nonadjacent logits
        nonadj_probs = nonadj_output.choice.squeeze(-1).detach().cpu().numpy()
        nonadj_probs = np.clip(nonadj_probs, 1e-7, 1 - 1e-7)
        nonadj_logits = np.log(nonadj_probs / (1 - nonadj_probs))
        nonadj_logits = nonadj_logits.reshape(current_batch_size, num_nonadj_pairs)
        for net_idx in range(current_batch_size):
            post_nonadj_logits.append(nonadj_logits[net_idx].copy())

        # Compute layer 1 nonadjacent pair embeddings via numpy (no bias) for PC1 analysis
        if extra_layer_indices:
            nonadj_inputs_np = np.stack(all_nonadj_inputs)  # (batch*num_nonadj_pairs, input_size)
            epw0_nonadj = np.repeat(epw0_np, num_nonadj_pairs, axis=0)
            l1_nonadj_emb = compute_numpy_layer1_embeddings(
                nonadj_inputs_np, epw0_nonadj,
                W_embed_np, W_extra_np[0], alpha_extra_np[0]
            ).reshape(current_batch_size, num_nonadj_pairs, -1)
            for net_idx in range(current_batch_size):
                l1_post_train_nonadj_embs.append(l1_nonadj_emb[net_idx].copy())

        # Cross dot product and correlation: adj (y-axis) x nonadj (x-axis)
        for li, layer_idx in enumerate(post_eval_layers):
            adj_pair_emb = pair_output.embeddings[layer_idx].detach().cpu().numpy()
            adj_pair_emb = adj_pair_emb.reshape(current_batch_size, num_adj_pairs, -1)
            nonadj_pair_emb = nonadj_output.embeddings[layer_idx].detach().cpu().numpy()
            nonadj_pair_emb = nonadj_pair_emb.reshape(current_batch_size, num_nonadj_pairs, -1)

            # Cross dot product: (batch, num_adj, num_nonadj)
            cross_dot = np.einsum('bih,bjh->bij', adj_pair_emb, nonadj_pair_emb)

            # Cross correlation: center + normalize, then einsum
            adj_c = adj_pair_emb - adj_pair_emb.mean(axis=-1, keepdims=True)
            nonadj_c = nonadj_pair_emb - nonadj_pair_emb.mean(axis=-1, keepdims=True)
            adj_norm = np.sqrt((adj_c ** 2).sum(axis=-1, keepdims=True)) + 1e-10
            nonadj_norm = np.sqrt((nonadj_c ** 2).sum(axis=-1, keepdims=True)) + 1e-10
            cross_corr = np.einsum('bih,bjh->bij', adj_c / adj_norm, nonadj_c / nonadj_norm)

            for net_idx in range(current_batch_size):
                post_cross_dot[li].append(cross_dot[net_idx].copy())
                post_cross_corr[li].append(cross_corr[net_idx].copy())

            # Item-vs-nonadjacent-pair dot product and correlation
            for pos_idx in range(2):
                item_emb = post_item_embs[layer_idx][pos_idx]  # (batch, num_items, hidden_size)

                # Dot product: (batch, num_items, num_nonadj_pairs)
                dot_nonadj = np.einsum('bih,bjh->bij', item_emb, nonadj_pair_emb)

                # Correlation: center + normalize, then einsum
                item_c = item_emb - item_emb.mean(axis=-1, keepdims=True)
                item_norm = np.sqrt((item_c ** 2).sum(axis=-1, keepdims=True)) + 1e-10
                corr_nonadj = np.einsum('bih,bjh->bij', item_c / item_norm, nonadj_c / nonadj_norm)

                for net_idx in range(current_batch_size):
                    if pos_idx == 0:
                        post_dot_nonadj_pos1[li].append(dot_nonadj[net_idx].copy())
                        post_corr_nonadj_pos1[li].append(corr_nonadj[net_idx].copy())
                    else:
                        post_dot_nonadj_pos2[li].append(dot_nonadj[net_idx].copy())
                        post_corr_nonadj_pos2[li].append(corr_nonadj[net_idx].copy())

    # Compute PC1 from layer 1 individual item embeddings and project to get PC1 "logits"
    pair_pc1_delta = None
    pair_pc1_abs = None
    post_pc1_logits = None
    post_pc1_nonadj_logits = None
    post_pc1_item_logits_pos1 = None
    post_pc1_item_logits_pos2 = None
    if extra_layer_indices and (external_pc1 is not None or l1_n > 0):
        if external_pc1 is not None:
            pc1 = external_pc1
            logger.info(f"Using external PC1 vector (from pca_frozen analysis)")
        else:
            l1_mean = l1_emb_sum / l1_n
            l1_cov = l1_emb_sq_sum / l1_n - np.outer(l1_mean, l1_mean)
            eigenvalues, eigenvectors = np.linalg.eigh(l1_cov)
            sort_idx = np.argsort(eigenvalues)[::-1]
            pc1 = eigenvectors[:, sort_idx[0]]
            logger.info(f"Layer 1 PC1: top 3 eigenvalues = {eigenvalues[sort_idx[:3]]}, "
                        f"explained variance ratio = {eigenvalues[sort_idx[0]] / eigenvalues.sum():.4f}")

        # Initialize storage (same structure as pair_logit_delta/abs)
        pair_pc1_delta = [[{0: [], 1: []} for _ in range(num_adj_pairs)] for _ in range(actual_num_trials)]
        pair_pc1_abs = [[{0: [], 1: []} for _ in range(num_adj_pairs)] for _ in range(actual_num_trials)]

        # Compute projections from stored batch records
        for l1_init, trial_records in l1_batch_records:
            prev_projs = l1_init @ pc1  # (batch, num_adj_pairs)
            for trial_idx, new_embs, labels, rewards, bs in trial_records:
                new_projs = new_embs @ pc1  # (batch, num_adj_pairs)
                delta_projs = new_projs - prev_projs
                for net_idx in range(bs):
                    pidx = pair_label_to_idx[labels[net_idx]]
                    rkey = 0 if rewards[net_idx] > 0 else 1
                    pair_pc1_abs[trial_idx][pidx][rkey].append(new_projs[net_idx].copy())
                    pair_pc1_delta[trial_idx][pidx][rkey].append(delta_projs[net_idx].copy())
                prev_projs = new_projs

        # Post-training PC1 logits (adjacent pairs)
        post_pc1_logits = []
        for embs in l1_post_train_embs:
            post_pc1_logits.append(embs @ pc1)  # (num_adj_pairs,)

        # Post-training PC1 logits (nonadjacent pairs)
        post_pc1_nonadj_logits = []
        for embs in l1_post_train_nonadj_embs:
            post_pc1_nonadj_logits.append(embs @ pc1)  # (num_nonadj_pairs,)

        # Post-training PC1 logits (individual items by position)
        post_pc1_item_logits_pos1 = []
        post_pc1_item_logits_pos2 = []
        for embs in l1_post_train_item_embs_pos1:
            post_pc1_item_logits_pos1.append(embs @ pc1)  # (num_items,)
        for embs in l1_post_train_item_embs_pos2:
            post_pc1_item_logits_pos2.append(embs @ pc1)  # (num_items,)

    return (delta_storage_pos1, delta_storage_pos2, nm_storage, adj_pair_labels, actual_num_trials,
            dot_delta_pos1, dot_delta_pos2, dot_abs_pos1, dot_abs_pos2,
            post_dot_pos1, post_dot_pos2, post_corr_pos1, post_corr_pos2,
            post_pair_dot, post_pair_corr, post_logits,
            nonadj_pair_labels, post_nonadj_logits, post_cross_dot, post_cross_corr,
            post_dot_nonadj_pos1, post_dot_nonadj_pos2, post_corr_nonadj_pos1, post_corr_nonadj_pos2,
            pair_dot_delta, pair_dot_abs,
            post_item_logits_pos1, post_item_logits_pos2,
            pair_logit_delta, pair_logit_abs,
            pair_pair_dot_delta, pair_pair_dot_abs,
            pair_pc1_delta, pair_pc1_abs, post_pc1_logits,
            post_pc1_nonadj_logits, post_pc1_item_logits_pos1, post_pc1_item_logits_pos2)


def run_aggregate_zero_shot(model, saved_args, num_episodes, num_train_trials, num_items, device,
                            batch_size=50, nm_override_positive=None, nm_override_negative=None,
                            freeze_extra_plastic_weights=False, freeze_final_plastic_weights=False):
    """Run many episodes with training, then test all pairs for symbolic distance accuracy.

    Args:
        nm_override_positive: if not None, override NM to this value on positive reward trials.
            0.0 means no Hebbian update. None means keep learned NM.
        nm_override_negative: if not None, override NM to this value on negative reward trials.
            None means keep learned NM.
        freeze_extra_plastic_weights: if True, revert extra (non-final) layer plastic weights
            to zero after each trial, so only the final hidden layer learns.
        freeze_final_plastic_weights: if True, revert final layer plastic weights
            to zero after each trial, so only extra hidden layers learn.

    Returns:
        zero_shot_trials: dict {(i, j): [0/1, ...]} for all pairs where i < j
    """
    item_size = saved_args.item_size
    zero_shot_trials = {(i, j): [] for i in range(num_items) for j in range(i + 1, num_items)}
    num_pairs = num_items * (num_items - 1) // 2
    needs_override = nm_override_positive is not None or nm_override_negative is not None

    num_batches = (num_episodes + batch_size - 1) // batch_size
    label_parts = []
    if nm_override_positive is not None:
        label_parts.append(f"pos_nm={nm_override_positive}")
    if nm_override_negative is not None:
        label_parts.append(f"neg_nm={nm_override_negative}")
    if freeze_extra_plastic_weights:
        label_parts.append("freeze_extra_pw")
    label = ", ".join(label_parts) if label_parts else "regular"

    for batch_idx in range(num_batches):
        current_batch_size = min(batch_size, num_episodes - batch_idx * batch_size)
        if batch_idx % 10 == 0:
            logger.info(f"Zero-shot ({label}) batch {batch_idx+1}/{num_batches} ({current_batch_size} episodes)...")

        batch_items = generate_batch_items(num_items, item_size, current_batch_size, change_items_throughout_batch=True)
        trials, correct_choices, pair_indices, actual_num_trials = generate_batch_trials_ti(
            batch_items, num_train_trials, 0, arbitrary=saved_args.arbitrary
        )
        trials_t = torch.tensor(trials, dtype=torch.float32).to(device)
        correct_choices_t = torch.tensor(correct_choices, dtype=torch.float32).to(device)

        plastic_weights, extra_plastic_weights, embed_pw = create_plastic_weights(current_batch_size, saved_args.hidden_size, saved_args.extra_layers, getattr(saved_args, 'multi_neuromodulator', 1), device, direct_readout=getattr(saved_args, 'direct_readout', False), first_plastic_input_size=getattr(model, 'first_plastic_input_size', saved_args.hidden_size), plastic_embedding=getattr(saved_args, 'plastic_embedding', False), input_size=2*saved_args.item_size)

        # Training phase
        for trial_idx in range(actual_num_trials):
            if needs_override:
                saved_pw, saved_epw, saved_embed_pw = clone_plastic_weights(plastic_weights, extra_plastic_weights, embed_pw=embed_pw)

            batch_trial = trials_t[:, trial_idx, :]
            batch_correct = correct_choices_t[:, trial_idx]
            with torch.inference_mode():
                output = model(batch_trial, plastic_weights, batch_correct,
                              extra_plastic_weights=extra_plastic_weights, embed_plastic_weights=embed_pw)
            plastic_weights, extra_plastic_weights, embed_pw = clone_plastic_weights(output.plastic_weights, output.extra_plastic_weights, embed_pw=output.embed_plastic_weights)

            if needs_override:
                sampled = output.sampled_choices.squeeze(-1).detach().cpu().numpy()
                correct = correct_choices[:, trial_idx]
                rewards = 2.0 * (sampled == correct).astype(np.float64) - 1.0
                actual_nm = output.neuromodulator[:, 0, 0].detach()  # (batch,) main NM

                for is_positive, nm_override in [(True, nm_override_positive), (False, nm_override_negative)]:
                    if nm_override is None:
                        continue
                    mask = torch.tensor(rewards > 0 if is_positive else rewards < 0, dtype=torch.bool).to(device)
                    if not mask.any():
                        continue
                    if nm_override == 0.0:
                        # Revert to saved weights (no Hebbian update)
                        pw_mask_set(plastic_weights, mask, saved_pw)
                        for layer_k in range(len(extra_plastic_weights)):
                            if isinstance(extra_plastic_weights[layer_k], list):
                                for ch in range(len(extra_plastic_weights[layer_k])):
                                    extra_plastic_weights[layer_k][ch][mask] = saved_epw[layer_k][ch][mask]
                            else:
                                extra_plastic_weights[layer_k][mask] = saved_epw[layer_k][mask]
                    else:
                        # Scale weight delta by (desired_nm / actual_nm)
                        nm_vals = actual_nm[mask]
                        ratio = torch.where(
                            nm_vals.abs() > 1e-8,
                            nm_override / nm_vals,
                            torch.zeros_like(nm_vals)
                        )
                        ratio = ratio.unsqueeze(-1).unsqueeze(-1)
                        pw_mask_set_scaled(plastic_weights, mask, saved_pw, ratio)
                        for layer_k in range(len(extra_plastic_weights)):
                            if isinstance(extra_plastic_weights[layer_k], list):
                                for ch in range(len(extra_plastic_weights[layer_k])):
                                    delta_epw = extra_plastic_weights[layer_k][ch][mask] - saved_epw[layer_k][ch][mask]
                                    extra_plastic_weights[layer_k][ch][mask] = saved_epw[layer_k][ch][mask] + ratio * delta_epw
                            else:
                                delta_epw = extra_plastic_weights[layer_k][mask] - saved_epw[layer_k][mask]
                                extra_plastic_weights[layer_k][mask] = saved_epw[layer_k][mask] + ratio * delta_epw

            if freeze_extra_plastic_weights:
                if isinstance(extra_plastic_weights[0], list):
                    for layer_epw in extra_plastic_weights:
                        for ch_epw in layer_epw:
                            ch_epw.zero_()
                else:
                    for k in range(len(extra_plastic_weights)):
                        extra_plastic_weights[k].zero_()
            if freeze_final_plastic_weights:
                zero_plastic_weights(plastic_weights)

        # Zero-shot test phase: test all pairs in both orderings (56 trials per network)
        num_test_per_net = num_pairs * 2
        all_test_inputs = []
        pair_info = []  # (net_idx, i, j, correct_choice)
        for net_idx in range(current_batch_size):
            for i in range(num_items):
                for j in range(i + 1, num_items):
                    # High item (i) on left
                    all_test_inputs.append(np.concatenate([batch_items[net_idx][i], batch_items[net_idx][j]]))
                    pair_info.append((net_idx, i, j, 0))
                    # High item (i) on right
                    all_test_inputs.append(np.concatenate([batch_items[net_idx][j], batch_items[net_idx][i]]))
                    pair_info.append((net_idx, i, j, 1))

        test_tensor = torch.tensor(np.stack(all_test_inputs), dtype=torch.float32).to(device)
        test_pw, test_epw, test_embed_pw = repeat_interleave_pw(plastic_weights, extra_plastic_weights, num_test_per_net, embed_pw=embed_pw)
        test_reward = torch.zeros(len(all_test_inputs), dtype=torch.float32).to(device)

        with torch.inference_mode():
            test_output = model(test_tensor, test_pw, test_reward, extra_plastic_weights=test_epw, embed_plastic_weights=test_embed_pw)

        sampled_choices = test_output.sampled_choices.squeeze(-1).detach().cpu().numpy()
        for k, (net_idx, i, j, correct_choice) in enumerate(pair_info):
            zero_shot_trials[(i, j)].append(int(sampled_choices[k] == correct_choice))

    return zero_shot_trials


def run_aggregate_zero_shot_ll(model, saved_args, num_episodes, num_items, device,
                               num_trials_list_1, num_trials_list_2, num_trials_linking_pair,
                               batch_size=50, put_linking_first=False, randomize_order=False):
    """Run many list-linking episodes with training, then test all pairs for symbolic distance accuracy.

    Training phase: list 1 trials + list 2 trials + linking pair trials.
    Test phase: all pairs in both orderings (same as TI zero-shot).

    Returns:
        zero_shot_trials: dict {(i, j): [0/1, ...]} for all pairs where i < j
    """
    item_size = saved_args.item_size
    zero_shot_trials = {(i, j): [] for i in range(num_items) for j in range(i + 1, num_items)}
    num_pairs = num_items * (num_items - 1) // 2
    num_train_trials = num_trials_list_1 + num_trials_list_2 + num_trials_linking_pair

    num_batches = (num_episodes + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        current_batch_size = min(batch_size, num_episodes - batch_idx * batch_size)
        if batch_idx % 10 == 0:
            logger.info(f"LL zero-shot batch {batch_idx+1}/{num_batches} ({current_batch_size} episodes)...")

        batch_items = generate_batch_items(num_items, item_size, current_batch_size, change_items_throughout_batch=True)
        trials, correct_choices, pair_indices = generate_batch_trials_ll(
            batch_items, num_trials_list_1, num_trials_list_2, num_trials_linking_pair, 0,
            put_linking_trials_first=put_linking_first, randomize_list_order=randomize_order
        )
        trials_t = torch.tensor(trials, dtype=torch.float32).to(device)
        correct_choices_t = torch.tensor(correct_choices, dtype=torch.float32).to(device)

        plastic_weights, extra_plastic_weights, embed_pw = create_plastic_weights(
            current_batch_size, saved_args.hidden_size, saved_args.extra_layers,
            getattr(saved_args, 'multi_neuromodulator', 1), device,
            direct_readout=getattr(saved_args, 'direct_readout', False),
            first_plastic_input_size=getattr(model, 'first_plastic_input_size', saved_args.hidden_size),
            plastic_embedding=getattr(saved_args, 'plastic_embedding', False),
            input_size=2*saved_args.item_size
        )

        # Training phase
        for trial_idx in range(num_train_trials):
            batch_trial = trials_t[:, trial_idx, :]
            batch_correct = correct_choices_t[:, trial_idx]
            with torch.inference_mode():
                output = model(batch_trial, plastic_weights, batch_correct,
                              extra_plastic_weights=extra_plastic_weights, embed_plastic_weights=embed_pw)
            plastic_weights, extra_plastic_weights, embed_pw = clone_plastic_weights(output.plastic_weights, output.extra_plastic_weights, embed_pw=output.embed_plastic_weights)

        # Zero-shot test phase: test all pairs in both orderings
        num_test_per_net = num_pairs * 2
        all_test_inputs = []
        pair_info = []
        for net_idx in range(current_batch_size):
            for i in range(num_items):
                for j in range(i + 1, num_items):
                    all_test_inputs.append(np.concatenate([batch_items[net_idx][i], batch_items[net_idx][j]]))
                    pair_info.append((net_idx, i, j, 0))
                    all_test_inputs.append(np.concatenate([batch_items[net_idx][j], batch_items[net_idx][i]]))
                    pair_info.append((net_idx, i, j, 1))

        test_tensor = torch.tensor(np.stack(all_test_inputs), dtype=torch.float32).to(device)
        test_pw, test_epw, test_embed_pw = repeat_interleave_pw(plastic_weights, extra_plastic_weights, num_test_per_net, embed_pw=embed_pw)
        test_reward = torch.zeros(len(all_test_inputs), dtype=torch.float32).to(device)

        with torch.inference_mode():
            test_output = model(test_tensor, test_pw, test_reward, extra_plastic_weights=test_epw, embed_plastic_weights=test_embed_pw)

        sampled_choices = test_output.sampled_choices.squeeze(-1).detach().cpu().numpy()
        for k, (net_idx, i, j, correct_choice) in enumerate(pair_info):
            zero_shot_trials[(i, j)].append(int(sampled_choices[k] == correct_choice))

    return zero_shot_trials


def run_zero_shot_with_activations(model, saved_args, num_episodes, num_train_trials,
                                    num_items, device, batch_size=50, num_bins=100):
    """Run zero-shot eval and collect per-layer activation histograms for adj vs nonadj pairs.

    Returns:
        zero_shot_trials: dict {(i, j): [0/1, ...]} (same as run_aggregate_zero_shot)
        activation_data: dict with keys 'adj', 'nonadj' (each layer_idx -> histogram counts),
                         'bins' (bin edges), 'layer_names' (list of str)
    """
    item_size = saved_args.item_size
    num_layers = saved_args.extra_layers + 2  # embedding + extra layers + final
    layer_indices = list(range(num_layers))
    layer_names = ['Embedding'] + [f'Hidden {i+1}' for i in range(saved_args.extra_layers)] + ['Final']

    zero_shot_trials = {(i, j): [] for i in range(num_items) for j in range(i + 1, num_items)}
    num_pairs = num_items * (num_items - 1) // 2

    bins = np.linspace(-1.0, 1.0, num_bins + 1)
    adj_counts = {L: np.zeros(num_bins) for L in layer_indices}
    nonadj_counts = {L: np.zeros(num_bins) for L in layer_indices}

    num_batches = (num_episodes + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        current_batch_size = min(batch_size, num_episodes - batch_idx * batch_size)
        if batch_idx % 50 == 0:
            logger.info(f"  Batch {batch_idx+1}/{num_batches} ({current_batch_size} episodes)")

        batch_items = generate_batch_items(num_items, item_size, current_batch_size, change_items_throughout_batch=True)
        trials, correct_choices, pair_indices, actual_num_trials = generate_batch_trials_ti(
            batch_items, num_train_trials, 0, arbitrary=saved_args.arbitrary
        )
        trials_t = torch.tensor(trials, dtype=torch.float32).to(device)
        correct_choices_t = torch.tensor(correct_choices, dtype=torch.float32).to(device)

        plastic_weights, extra_plastic_weights, embed_pw = create_plastic_weights(
            current_batch_size, saved_args.hidden_size, saved_args.extra_layers,
            getattr(saved_args, 'multi_neuromodulator', 1), device,
            direct_readout=getattr(saved_args, 'direct_readout', False),
            first_plastic_input_size=getattr(model, 'first_plastic_input_size', saved_args.hidden_size),
            plastic_embedding=getattr(saved_args, 'plastic_embedding', False),
            input_size=2*saved_args.item_size
        )

        # Training phase
        for trial_idx in range(actual_num_trials):
            batch_trial = trials_t[:, trial_idx, :]
            batch_correct = correct_choices_t[:, trial_idx]
            with torch.inference_mode():
                output = model(batch_trial, plastic_weights, batch_correct,
                              extra_plastic_weights=extra_plastic_weights, embed_plastic_weights=embed_pw)
            plastic_weights, extra_plastic_weights, embed_pw = clone_plastic_weights(
                output.plastic_weights, output.extra_plastic_weights, embed_pw=output.embed_plastic_weights
            )

        # Test phase: all pairs in both orderings, with embeddings
        num_test_per_net = num_pairs * 2
        all_test_inputs = []
        pair_info = []
        for net_idx in range(current_batch_size):
            for i in range(num_items):
                for j in range(i + 1, num_items):
                    all_test_inputs.append(np.concatenate([batch_items[net_idx][i], batch_items[net_idx][j]]))
                    pair_info.append((net_idx, i, j, 0))
                    all_test_inputs.append(np.concatenate([batch_items[net_idx][j], batch_items[net_idx][i]]))
                    pair_info.append((net_idx, i, j, 1))

        test_tensor = torch.tensor(np.stack(all_test_inputs), dtype=torch.float32).to(device)
        test_pw, test_epw, test_embed_pw = repeat_interleave_pw(plastic_weights, extra_plastic_weights, num_test_per_net, embed_pw=embed_pw)
        test_reward = torch.zeros(len(all_test_inputs), dtype=torch.float32).to(device)

        with torch.inference_mode():
            test_output = model(test_tensor, test_pw, test_reward,
                                extra_plastic_weights=test_epw, store_embeddings=True, embed_plastic_weights=test_embed_pw)

        # Record zero-shot accuracy
        sampled_choices = test_output.sampled_choices.squeeze(-1).detach().cpu().numpy()
        for k, (net_idx, i, j, correct_choice) in enumerate(pair_info):
            zero_shot_trials[(i, j)].append(int(sampled_choices[k] == correct_choice))

        # Accumulate activation histograms
        is_adj = np.array([abs(i - j) == 1 for (_, i, j, _) in pair_info])
        for L in layer_indices:
            emb = test_output.embeddings[L].detach().cpu().numpy()
            adj_vals = emb[is_adj].flatten()
            nonadj_vals = emb[~is_adj].flatten()
            adj_counts[L] += np.histogram(adj_vals, bins=bins)[0]
            nonadj_counts[L] += np.histogram(nonadj_vals, bins=bins)[0]

    activation_data = {
        'adj': adj_counts, 'nonadj': nonadj_counts,
        'bins': bins, 'layer_names': layer_names
    }
    return zero_shot_trials, activation_data


def run_controlled_zero_shot(model, saved_args, num_episodes, num_items, device,
                             trial_generator, batch_size=50, probe_pair_logits=False,
                             nm_override_positive=None, nm_override_negative=None,
                             probe_item_logits=False, probe_item_dot_layers=None,
                             probe_item_dots_no_pc1=False,
                             probe_item_dots_no_readout=False):
    """Run zero-shot evaluation with a custom trial generator.

    Args:
        trial_generator: callable(batch_items, num_items) -> (trials, correct_choices)
            trials: (batch, num_trials, 2*item_size)
            correct_choices: (batch, num_trials)
        probe_pair_logits: if True, measure logits for all 14 adjacent pairs AND all 56
            pairs (both orderings) after each training trial and return mean/SEM arrays
            in probe_results.
        nm_override_positive: if not None, override NM to this value on positive reward trials.
            0.0 means no Hebbian update. None means keep learned NM.
        nm_override_negative: if not None, override NM to this value on negative reward trials.
            None means keep learned NM.
        probe_item_logits: if True, measure individual item logits at both positions after
            each training trial and after training completes.
        probe_item_dot_layers: list of layer indices (e.g. [1] or [1, 2]) to compute pairwise
            dot products of individual item representations (both positions combined, 2N x 2N)
            before and after each trial. None or empty to disable.
        probe_item_dots_no_pc1: if True (requires probe_item_dot_layers), also compute dot
            products after projecting away PC1 from all item representations. PC1 is computed
            per slot from all item representations pooled across positions and episodes.
        probe_item_dots_no_readout: if True (requires probe_item_dot_layers), also compute
            dot products after projecting away the readout direction (get_readout_weight(),
            ignoring bias). Also computes a version with both readout and PC1 removed when
            probe_item_dots_no_pc1 is also True.

    Returns:
        (zero_shot_trials, per_trial_accuracy, probe_results):
            zero_shot_trials: dict {(i, j): [0/1, ...]} for all pairs where i < j
            per_trial_accuracy: array of shape (num_trials,) with mean accuracy per trial
            probe_results: dict with optional keys (present only if corresponding probe is True):
                'pair_logits_mean': array (num_trials, num_adj_pairs) — adjacent only
                'pair_logits_sem': array (num_trials, num_adj_pairs)
                'all_pair_logits_mean': array (num_trials, num_all_pairs) — all pairs
                'all_pair_logits_sem': array (num_trials, num_all_pairs)
                'all_pair_labels': list of str labels for all pairs
                'item_logits_pos1': list of (num_items,) arrays (post-training, per episode)
                'item_logits_pos2': list of (num_items,) arrays (post-training, per episode)
                'item_evo_pos1_mean': array (num_trials, num_items) — per-trial evolution
                'item_evo_pos1_sem': array (num_trials, num_items)
                'item_evo_pos2_mean': array (num_trials, num_items)
                'item_evo_pos2_sem': array (num_trials, num_items)
                For each layer L in probe_item_dot_layers:
                'item_dot_L{L}_mean': list of (num_trials+1) arrays, each (2N, 2N)
                'item_dot_L{L}_sem': same shape, SEM across episodes
                'item_dot_no_pc1_L{L}_mean': same but with PC1 projected away
                'item_dot_no_pc1_L{L}_sem': same shape
                'item_dot_no_readout_L{L}_mean': same but with readout direction removed
                'item_dot_no_readout_L{L}_sem': same shape
                'item_dot_no_pc1_no_readout_L{L}_mean': both PC1 and readout removed
                'item_dot_no_pc1_no_readout_L{L}_sem': same shape
    """
    item_size = saved_args.item_size
    zero_shot_trials = {(i, j): [] for i in range(num_items) for j in range(i + 1, num_items)}
    num_pairs = num_items * (num_items - 1) // 2
    num_adj_pairs = 2 * (num_items - 1)
    needs_override = nm_override_positive is not None or nm_override_negative is not None

    num_batches = (num_episodes + batch_size - 1) // batch_size

    trial_correct_counts = None  # shape (num_trials,), initialized on first batch
    trial_total_counts = None

    # Online accumulators for adjacent pair logits (initialized on first batch)
    pair_logit_sums = None
    pair_logit_sq_sums = None
    pair_logit_counts = None

    # Online accumulators for ALL pair logits (adjacent + nonadjacent)
    num_all_pairs = num_items * (num_items - 1)  # both orderings
    all_pair_logit_sums = None
    all_pair_logit_sq_sums = None

    # Per-episode item logit lists (post-training snapshots)
    post_item_logits_pos1 = [] if probe_item_logits else None
    post_item_logits_pos2 = [] if probe_item_logits else None

    # Online accumulators for per-trial item logit evolution
    item_evo_pos1_sums = None
    item_evo_pos1_sq_sums = None
    item_evo_pos2_sums = None
    item_evo_pos2_sq_sums = None
    item_evo_counts = None

    # Online accumulators for item dot products (2N x 2N matrices), per layer
    # Index 0 = before training, index t = after trial t-1
    dot_dim = 2 * num_items  # pos1 items + pos2 items
    dot_layers = probe_item_dot_layers if probe_item_dot_layers else []
    # item_dot_sums[layer][slot] = (dot_dim, dot_dim)
    item_dot_sums = None
    item_dot_sq_sums = None
    item_dot_counts = None
    # Per-slot raw embeddings for projection-removed dot products
    # Stored when probe_item_dots_no_pc1 or probe_item_dots_no_readout
    # item_dot_embeddings[layer][slot] = list of arrays, each (batch, 2N, hidden)
    store_dot_embeddings = probe_item_dots_no_pc1 or probe_item_dots_no_readout
    item_dot_embeddings = None

    for batch_idx in range(num_batches):
        current_batch_size = min(batch_size, num_episodes - batch_idx * batch_size)
        if batch_idx % 10 == 0:
            logger.info(f"Controlled zero-shot batch {batch_idx+1}/{num_batches} ({current_batch_size} episodes)...")

        batch_items = generate_batch_items(num_items, item_size, current_batch_size, change_items_throughout_batch=True)
        trials, correct_choices = trial_generator(batch_items, num_items)
        num_trials = trials.shape[1]
        trials_t = torch.tensor(trials, dtype=torch.float32).to(device)
        correct_choices_t = torch.tensor(correct_choices, dtype=torch.float32).to(device)

        if trial_correct_counts is None:
            trial_correct_counts = np.zeros(num_trials)
            trial_total_counts = np.zeros(num_trials)

        # Build pair probe inputs if needed
        if probe_pair_logits:
            # Adjacent pairs (14): AB, BA, BC, CB, ...
            all_pair_inputs = []
            for net_idx in range(current_batch_size):
                for i in range(num_items - 1):
                    all_pair_inputs.append(np.concatenate([batch_items[net_idx][i], batch_items[net_idx][i+1]]))
                    all_pair_inputs.append(np.concatenate([batch_items[net_idx][i+1], batch_items[net_idx][i]]))
            pair_input_tensor = torch.tensor(np.stack(all_pair_inputs), dtype=torch.float32).to(device)

            # All pairs (56): ordered by symbolic distance, both orderings
            all_all_pair_inputs = []
            for net_idx in range(current_batch_size):
                for sd in range(1, num_items):
                    for i in range(num_items - sd):
                        j = i + sd
                        all_all_pair_inputs.append(np.concatenate([batch_items[net_idx][i], batch_items[net_idx][j]]))
                        all_all_pair_inputs.append(np.concatenate([batch_items[net_idx][j], batch_items[net_idx][i]]))
            all_pair_input_tensor = torch.tensor(np.stack(all_all_pair_inputs), dtype=torch.float32).to(device)

            if pair_logit_sums is None:
                pair_logit_sums = np.zeros((num_trials, num_adj_pairs))
                pair_logit_sq_sums = np.zeros((num_trials, num_adj_pairs))
                pair_logit_counts = np.zeros(num_trials)
                all_pair_logit_sums = np.zeros((num_trials, num_all_pairs))
                all_pair_logit_sq_sums = np.zeros((num_trials, num_all_pairs))

        if probe_item_logits and item_evo_pos1_sums is None:
            item_evo_pos1_sums = np.zeros((num_trials, num_items))
            item_evo_pos1_sq_sums = np.zeros((num_trials, num_items))
            item_evo_pos2_sums = np.zeros((num_trials, num_items))
            item_evo_pos2_sq_sums = np.zeros((num_trials, num_items))
            item_evo_counts = np.zeros(num_trials)

        if dot_layers and item_dot_sums is None:
            item_dot_sums = {L: [np.zeros((dot_dim, dot_dim)) for _ in range(num_trials + 1)] for L in dot_layers}
            item_dot_sq_sums = {L: [np.zeros((dot_dim, dot_dim)) for _ in range(num_trials + 1)] for L in dot_layers}
            item_dot_counts = np.zeros(num_trials + 1)
            if store_dot_embeddings:
                item_dot_embeddings = {L: [[] for _ in range(num_trials + 1)] for L in dot_layers}

        plastic_weights, extra_plastic_weights, embed_pw = create_plastic_weights(current_batch_size, saved_args.hidden_size, saved_args.extra_layers, getattr(saved_args, 'multi_neuromodulator', 1), device, direct_readout=getattr(saved_args, 'direct_readout', False), first_plastic_input_size=getattr(model, 'first_plastic_input_size', saved_args.hidden_size), plastic_embedding=getattr(saved_args, 'plastic_embedding', False), input_size=2*saved_args.item_size)

        # Helper to probe item dots at requested layers and accumulate into slot_idx
        def _probe_item_dots(pw, epw, slot_idx):
            _, _, emb_dict = compute_batch_item_logits_fast(
                model, batch_items, pw, epw,
                current_batch_size, num_items, item_size, device,
                return_embeddings_at_layers=dot_layers
            )
            for L in dot_layers:
                p1_emb, p2_emb = emb_dict[L]  # each (batch, num_items, hidden)
                combined = np.concatenate([p1_emb, p2_emb], axis=1)  # (batch, 2N, hidden)
                dots = np.einsum('bih,bjh->bij', combined, combined)  # (batch, 2N, 2N)
                item_dot_sums[L][slot_idx] += dots.sum(axis=0)
                item_dot_sq_sums[L][slot_idx] += (dots ** 2).sum(axis=0)
                if store_dot_embeddings:
                    item_dot_embeddings[L][slot_idx].append(combined.astype(np.float32))
            item_dot_counts[slot_idx] += current_batch_size

        # Pre-training item dot products
        if dot_layers:
            _probe_item_dots(plastic_weights, extra_plastic_weights, 0)

        # Training phase
        for trial_idx in range(num_trials):
            if needs_override:
                saved_pw, saved_epw, saved_embed_pw = clone_plastic_weights(plastic_weights, extra_plastic_weights, embed_pw=embed_pw)

            batch_trial = trials_t[:, trial_idx, :]
            batch_correct = correct_choices_t[:, trial_idx]
            with torch.inference_mode():
                output = model(batch_trial, plastic_weights, batch_correct,
                              extra_plastic_weights=extra_plastic_weights, embed_plastic_weights=embed_pw)
            # Track per-trial accuracy
            sampled = output.sampled_choices.squeeze(-1).detach().cpu().numpy()
            correct = correct_choices[:, trial_idx]
            trial_correct_counts[trial_idx] += np.sum(sampled == correct)
            trial_total_counts[trial_idx] += current_batch_size

            plastic_weights, extra_plastic_weights, embed_pw = clone_plastic_weights(output.plastic_weights, output.extra_plastic_weights, embed_pw=output.embed_plastic_weights)

            if needs_override:
                rewards = 2.0 * (sampled == correct).astype(np.float64) - 1.0
                actual_nm = output.neuromodulator[:, 0, 0].detach()  # (batch,)

                for is_positive, nm_override in [(True, nm_override_positive), (False, nm_override_negative)]:
                    if nm_override is None:
                        continue
                    mask = torch.tensor(rewards > 0 if is_positive else rewards < 0, dtype=torch.bool).to(device)
                    if not mask.any():
                        continue
                    if nm_override == 0.0:
                        pw_mask_set(plastic_weights, mask, saved_pw)
                        for layer_k in range(len(extra_plastic_weights)):
                            if isinstance(extra_plastic_weights[layer_k], list):
                                for ch in range(len(extra_plastic_weights[layer_k])):
                                    extra_plastic_weights[layer_k][ch][mask] = saved_epw[layer_k][ch][mask]
                            else:
                                extra_plastic_weights[layer_k][mask] = saved_epw[layer_k][mask]
                    else:
                        nm_vals = actual_nm[mask]
                        ratio = torch.where(
                            nm_vals.abs() > 1e-8,
                            nm_override / nm_vals,
                            torch.zeros_like(nm_vals)
                        )
                        ratio = ratio.unsqueeze(-1).unsqueeze(-1)
                        pw_mask_set_scaled(plastic_weights, mask, saved_pw, ratio)
                        for layer_k in range(len(extra_plastic_weights)):
                            if isinstance(extra_plastic_weights[layer_k], list):
                                for ch in range(len(extra_plastic_weights[layer_k])):
                                    delta_epw = extra_plastic_weights[layer_k][ch][mask] - saved_epw[layer_k][ch][mask]
                                    extra_plastic_weights[layer_k][ch][mask] = saved_epw[layer_k][ch][mask] + ratio * delta_epw
                            else:
                                delta_epw = extra_plastic_weights[layer_k][mask] - saved_epw[layer_k][mask]
                                extra_plastic_weights[layer_k][mask] = saved_epw[layer_k][mask] + ratio * delta_epw

            # Probe all adjacent pair logits after this training trial
            if probe_pair_logits:
                probe_pw, probe_epw, probe_embed_pw = repeat_interleave_pw(plastic_weights, extra_plastic_weights, num_adj_pairs, embed_pw=embed_pw)
                probe_rew = torch.zeros(current_batch_size * num_adj_pairs, dtype=torch.float32).to(device)
                with torch.inference_mode():
                    probe_out = model(pair_input_tensor, probe_pw, probe_rew,
                                     extra_plastic_weights=probe_epw, embed_plastic_weights=probe_embed_pw)
                probe_probs = probe_out.choice.squeeze(-1).detach().cpu().numpy()
                probe_probs = np.clip(probe_probs, 1e-7, 1 - 1e-7)
                probe_logits = np.log(probe_probs / (1 - probe_probs)).reshape(current_batch_size, num_adj_pairs)
                pair_logit_sums[trial_idx] += probe_logits.sum(axis=0)
                pair_logit_sq_sums[trial_idx] += (probe_logits ** 2).sum(axis=0)
                pair_logit_counts[trial_idx] += current_batch_size

                # Probe ALL pairs (adjacent + nonadjacent)
                all_probe_pw, all_probe_epw, all_probe_embed_pw = repeat_interleave_pw(plastic_weights, extra_plastic_weights, num_all_pairs, embed_pw=embed_pw)
                all_probe_rew = torch.zeros(current_batch_size * num_all_pairs, dtype=torch.float32).to(device)
                with torch.inference_mode():
                    all_probe_out = model(all_pair_input_tensor, all_probe_pw, all_probe_rew,
                                         extra_plastic_weights=all_probe_epw, embed_plastic_weights=all_probe_embed_pw)
                all_probe_probs = all_probe_out.choice.squeeze(-1).detach().cpu().numpy()
                all_probe_probs = np.clip(all_probe_probs, 1e-7, 1 - 1e-7)
                all_probe_logits = np.log(all_probe_probs / (1 - all_probe_probs)).reshape(current_batch_size, num_all_pairs)
                all_pair_logit_sums[trial_idx] += all_probe_logits.sum(axis=0)
                all_pair_logit_sq_sums[trial_idx] += (all_probe_logits ** 2).sum(axis=0)

            # Probe individual item logits after this training trial
            if probe_item_logits:
                il_p1, il_p2 = compute_batch_item_logits_fast(
                    model, batch_items, plastic_weights, extra_plastic_weights,
                    current_batch_size, num_items, item_size, device
                )
                # il_p1, il_p2 shape: (current_batch_size, num_items)
                item_evo_pos1_sums[trial_idx] += il_p1.sum(axis=0)
                item_evo_pos1_sq_sums[trial_idx] += (il_p1 ** 2).sum(axis=0)
                item_evo_pos2_sums[trial_idx] += il_p2.sum(axis=0)
                item_evo_pos2_sq_sums[trial_idx] += (il_p2 ** 2).sum(axis=0)
                item_evo_counts[trial_idx] += current_batch_size

            # Probe item dot products after this training trial
            if dot_layers:
                _probe_item_dots(plastic_weights, extra_plastic_weights, trial_idx + 1)

        # Post-training individual item logits (per-episode for bar charts)
        if probe_item_logits:
            # Reuse last trial's il_p1, il_p2 which are still in scope
            for net_idx in range(current_batch_size):
                post_item_logits_pos1.append(il_p1[net_idx].copy())
                post_item_logits_pos2.append(il_p2[net_idx].copy())

        # Zero-shot test phase: test all pairs in both orderings (56 trials per network)
        num_test_per_net = num_pairs * 2
        all_test_inputs = []
        pair_info = []  # (net_idx, i, j, correct_choice)
        for net_idx in range(current_batch_size):
            for i in range(num_items):
                for j in range(i + 1, num_items):
                    # High item (i) on left
                    all_test_inputs.append(np.concatenate([batch_items[net_idx][i], batch_items[net_idx][j]]))
                    pair_info.append((net_idx, i, j, 0))
                    # High item (i) on right
                    all_test_inputs.append(np.concatenate([batch_items[net_idx][j], batch_items[net_idx][i]]))
                    pair_info.append((net_idx, i, j, 1))

        test_tensor = torch.tensor(np.stack(all_test_inputs), dtype=torch.float32).to(device)
        test_pw, test_epw, test_embed_pw = repeat_interleave_pw(plastic_weights, extra_plastic_weights, num_test_per_net, embed_pw=embed_pw)
        test_reward = torch.zeros(len(all_test_inputs), dtype=torch.float32).to(device)

        with torch.inference_mode():
            test_output = model(test_tensor, test_pw, test_reward, extra_plastic_weights=test_epw, embed_plastic_weights=test_embed_pw)

        sampled_choices = test_output.sampled_choices.squeeze(-1).detach().cpu().numpy()
        for k, (net_idx, i, j, correct_choice) in enumerate(pair_info):
            zero_shot_trials[(i, j)].append(int(sampled_choices[k] == correct_choice))

    per_trial_accuracy = trial_correct_counts / trial_total_counts

    probe_results = {}
    if probe_pair_logits:
        pl_mean = pair_logit_sums / pair_logit_counts[:, None]
        pl_var = pair_logit_sq_sums / pair_logit_counts[:, None] - pl_mean ** 2
        pl_var = np.maximum(pl_var, 0.0)
        pl_sem = np.sqrt(pl_var / pair_logit_counts[:, None])
        probe_results['pair_logits_mean'] = pl_mean
        probe_results['pair_logits_sem'] = pl_sem
        # All pairs (adjacent + nonadjacent)
        apl_mean = all_pair_logit_sums / pair_logit_counts[:, None]
        apl_var = np.maximum(all_pair_logit_sq_sums / pair_logit_counts[:, None] - apl_mean ** 2, 0.0)
        apl_sem = np.sqrt(apl_var / pair_logit_counts[:, None])
        probe_results['all_pair_logits_mean'] = apl_mean
        probe_results['all_pair_logits_sem'] = apl_sem
        # Generate labels for all pairs ordered by symbolic distance
        item_labels_local = [chr(ord('A') + i) for i in range(num_items)]
        all_pair_labels = []
        for sd in range(1, num_items):
            for i in range(num_items - sd):
                j = i + sd
                all_pair_labels.append(f"{item_labels_local[i]}{item_labels_local[j]}")
                all_pair_labels.append(f"{item_labels_local[j]}{item_labels_local[i]}")
        probe_results['all_pair_labels'] = all_pair_labels
    if probe_item_logits:
        probe_results['item_logits_pos1'] = post_item_logits_pos1
        probe_results['item_logits_pos2'] = post_item_logits_pos2
        for key, sums, sq_sums in [
            ('item_evo_pos1', item_evo_pos1_sums, item_evo_pos1_sq_sums),
            ('item_evo_pos2', item_evo_pos2_sums, item_evo_pos2_sq_sums),
        ]:
            m = sums / item_evo_counts[:, None]
            v = np.maximum(sq_sums / item_evo_counts[:, None] - m ** 2, 0.0)
            probe_results[f'{key}_mean'] = m
            probe_results[f'{key}_sem'] = np.sqrt(v / item_evo_counts[:, None])
    if dot_layers:
        for L in dot_layers:
            dot_mean_list = []
            dot_sem_list = []
            for s in range(len(item_dot_sums[L])):
                n = item_dot_counts[s]
                m = item_dot_sums[L][s] / n
                v = np.maximum(item_dot_sq_sums[L][s] / n - m ** 2, 0.0)
                dot_mean_list.append(m)
                dot_sem_list.append(np.sqrt(v / n))
            probe_results[f'item_dot_L{L}_mean'] = dot_mean_list
            probe_results[f'item_dot_L{L}_sem'] = dot_sem_list

            # Projection-removed dot product variants
            if store_dot_embeddings and item_dot_embeddings is not None:
                # Extract readout direction (get_readout_weight(), ignoring bias)
                readout_vec = get_readout_weight().data[0].cpu().numpy().astype(np.float64)
                readout_dir = readout_vec / np.linalg.norm(readout_vec)

                no_pc1_mean_list = [] if probe_item_dots_no_pc1 else None
                no_pc1_sem_list = [] if probe_item_dots_no_pc1 else None
                no_readout_mean_list = [] if probe_item_dots_no_readout else None
                no_readout_sem_list = [] if probe_item_dots_no_readout else None
                no_both_mean_list = [] if (probe_item_dots_no_pc1 and probe_item_dots_no_readout) else None
                no_both_sem_list = [] if (probe_item_dots_no_pc1 and probe_item_dots_no_readout) else None

                for s in range(len(item_dot_embeddings[L])):
                    all_emb = np.concatenate(item_dot_embeddings[L][s], axis=0).astype(np.float64)  # (n_ep, 2N, H)
                    n_ep_slot = all_emb.shape[0]
                    H = all_emb.shape[2]

                    # Compute PC1 from all items pooled across positions and episodes
                    pc1 = None
                    if probe_item_dots_no_pc1:
                        flat = all_emb.reshape(-1, H)  # (n_ep * 2N, H)
                        mean_vec = flat.mean(axis=0)
                        centered = flat - mean_vec
                        cov = (centered.T @ centered) / len(centered)
                        eigenvalues, eigenvectors = np.linalg.eigh(cov)
                        pc1 = eigenvectors[:, -1]  # top eigenvector (H,)

                    def _remove_and_dot(emb, directions):
                        """Remove list of (normalized) directions from embeddings and compute dots."""
                        rem = emb.copy()
                        for d in directions:
                            proj = np.einsum('bih,h->bi', rem, d)  # (n_ep, 2N)
                            rem = rem - proj[..., None] * d[None, None, :]
                        dots = np.einsum('bih,bjh->bij', rem, rem)  # (n_ep, 2N, 2N)
                        m = dots.mean(axis=0)
                        v = np.maximum(dots.var(axis=0, ddof=0), 0.0)
                        sem = np.sqrt(v / emb.shape[0])
                        return m, sem

                    if probe_item_dots_no_pc1:
                        m, sem = _remove_and_dot(all_emb, [pc1])
                        no_pc1_mean_list.append(m)
                        no_pc1_sem_list.append(sem)

                    if probe_item_dots_no_readout:
                        m, sem = _remove_and_dot(all_emb, [readout_dir])
                        no_readout_mean_list.append(m)
                        no_readout_sem_list.append(sem)

                    if probe_item_dots_no_pc1 and probe_item_dots_no_readout:
                        # First remove readout, then compute PC1 of the residual
                        ro_proj = np.einsum('bih,h->bi', all_emb, readout_dir)
                        emb_no_ro = all_emb - ro_proj[..., None] * readout_dir[None, None, :]
                        flat_no_ro = emb_no_ro.reshape(-1, H)
                        mean_no_ro = flat_no_ro.mean(axis=0)
                        centered_no_ro = flat_no_ro - mean_no_ro
                        cov_no_ro = (centered_no_ro.T @ centered_no_ro) / len(centered_no_ro)
                        _, eigvecs_no_ro = np.linalg.eigh(cov_no_ro)
                        pc1_after_ro = eigvecs_no_ro[:, -1]
                        # Remove that PC1 from the already readout-removed embeddings
                        m, sem = _remove_and_dot(emb_no_ro, [pc1_after_ro])
                        no_both_mean_list.append(m)
                        no_both_sem_list.append(sem)
                        del emb_no_ro

                    del all_emb  # free memory

                if no_pc1_mean_list is not None:
                    probe_results[f'item_dot_no_pc1_L{L}_mean'] = no_pc1_mean_list
                    probe_results[f'item_dot_no_pc1_L{L}_sem'] = no_pc1_sem_list
                if no_readout_mean_list is not None:
                    probe_results[f'item_dot_no_readout_L{L}_mean'] = no_readout_mean_list
                    probe_results[f'item_dot_no_readout_L{L}_sem'] = no_readout_sem_list
                if no_both_mean_list is not None:
                    probe_results[f'item_dot_no_pc1_no_readout_L{L}_mean'] = no_both_mean_list
                    probe_results[f'item_dot_no_pc1_no_readout_L{L}_sem'] = no_both_sem_list
        if store_dot_embeddings and item_dot_embeddings is not None:
            del item_dot_embeddings  # free all stored embeddings

    return zero_shot_trials, per_trial_accuracy, probe_results


def create_aggregate_figures(delta_storage_pos1, delta_storage_pos2, nm_storage,
                             adj_pair_labels, num_items, num_train_trials, item_labels, num_episodes,
                             skip_prefixes=frozenset()):
    """Create aggregate delta logit heatmaps (mean and SEM) per trial number."""
    figures = {}
    num_adj_pairs = len(adj_pair_labels)

    for trial_num in range(num_train_trials):
        # Pre-compute NM mean and SEM per pair for each reward (shared across positions)
        nm_mean = {0: np.full(num_adj_pairs, np.nan), 1: np.full(num_adj_pairs, np.nan)}
        nm_sem = {0: np.full(num_adj_pairs, np.nan), 1: np.full(num_adj_pairs, np.nan)}
        for reward_key in [0, 1]:
            for pair_idx in range(num_adj_pairs):
                nm_samples = nm_storage[trial_num][pair_idx][reward_key]
                if nm_samples:
                    nm_mean[reward_key][pair_idx] = np.mean(nm_samples)
                    if len(nm_samples) > 1:
                        nm_sem[reward_key][pair_idx] = np.std(nm_samples) / np.sqrt(len(nm_samples))
                    else:
                        nm_sem[reward_key][pair_idx] = 0.0

        for pos_name, delta_storage in [("pos1_left", delta_storage_pos1), ("pos2_right", delta_storage_pos2)]:
            for reward_label, reward_key, reward_str in [("pos_reward", 0, "+1"), ("neg_reward", 1, "-1")]:
                mean_data = np.full((num_items, num_adj_pairs), np.nan)
                sem_data = np.full((num_items, num_adj_pairs), np.nan)

                for pair_idx in range(num_adj_pairs):
                    samples = delta_storage[trial_num][pair_idx][reward_key]
                    if samples:
                        stacked = np.stack(samples)  # (n_samples, num_items)
                        mean_data[:, pair_idx] = stacked.mean(axis=0)
                        if len(samples) > 1:
                            sem_data[:, pair_idx] = stacked.std(axis=0) / np.sqrt(len(samples))
                        else:
                            sem_data[:, pair_idx] = 0.0

                pos_title = pos_name.replace("_", " ").title()

                # --- Mean heatmap with NM bars ---
                if 'ti_agg_delta_mean' not in skip_prefixes:
                    vmax = np.nanmax(np.abs(mean_data)) if not np.all(np.isnan(mean_data)) else 0.1
                    vmax = max(vmax, 0.01)

                    fig, (ax, ax_nm_mean, ax_nm_sem) = plt.subplots(
                        3, 1, figsize=(10, 7), dpi=150,
                        gridspec_kw={'height_ratios': [num_items, 1, 1], 'hspace': 0.05},
                        sharex=True
                    )

                    im = ax.imshow(mean_data, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
                    ax.set_yticks(np.arange(num_items))
                    ax.set_yticklabels(item_labels)
                    ax.set_ylabel('Item')
                    ax.set_title(f'Trial #{trial_num} - Mean Delta Logit ({pos_title}, Reward {reward_str})\nn={num_episodes} episodes')
                    plt.colorbar(im, ax=[ax, ax_nm_mean, ax_nm_sem], label='Mean Delta Logit')

                    for i in range(num_items):
                        for j in range(num_adj_pairs):
                            val = mean_data[i, j]
                            if not np.isnan(val):
                                text_color = 'white' if abs(val) > vmax * 0.5 else 'black'
                                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                                       color=text_color, fontsize=5)

                    # NM mean bar
                    nm_mean_row = np.ones((1, num_adj_pairs))
                    ax_nm_mean.imshow(nm_mean_row, cmap='Greys_r', aspect='auto', vmin=0, vmax=1)
                    ax_nm_mean.set_yticks([0])
                    ax_nm_mean.set_yticklabels(['NM\nmean'], fontsize=7)
                    ax_nm_mean.tick_params(axis='x', labelbottom=False)
                    for j in range(num_adj_pairs):
                        val = nm_mean[reward_key][j]
                        txt = f'{val:.3f}' if not np.isnan(val) else '–'
                        ax_nm_mean.text(j, 0, txt, ha='center', va='center', color='black', fontsize=5)

                    # NM SEM bar
                    nm_sem_row = np.ones((1, num_adj_pairs))
                    ax_nm_sem.imshow(nm_sem_row, cmap='Greys_r', aspect='auto', vmin=0, vmax=1)
                    ax_nm_sem.set_yticks([0])
                    ax_nm_sem.set_yticklabels(['NM\nSEM'], fontsize=7)
                    ax_nm_sem.set_xticks(np.arange(num_adj_pairs))
                    ax_nm_sem.set_xticklabels(adj_pair_labels, rotation=90, fontsize=7)
                    ax_nm_sem.set_xlabel('Trial Type')
                    for j in range(num_adj_pairs):
                        val = nm_sem[reward_key][j]
                        txt = f'{val:.4f}' if not np.isnan(val) else '–'
                        ax_nm_sem.text(j, 0, txt, ha='center', va='center', color='black', fontsize=5)

                    plt.tight_layout()
                    figures[f"ti_agg_delta_mean/trial{trial_num}_{pos_name}_{reward_label}"] = fig
                    plt.close(fig)

                # --- SEM heatmap with NM bars ---
                if 'ti_agg_delta_sem' not in skip_prefixes:
                    vmax_sem = np.nanmax(sem_data) if not np.all(np.isnan(sem_data)) else 0.01
                    vmax_sem = max(vmax_sem, 0.001)

                    fig_sem, (ax_sem, ax_nm_mean_s, ax_nm_sem_s) = plt.subplots(
                        3, 1, figsize=(10, 7), dpi=150,
                        gridspec_kw={'height_ratios': [num_items, 1, 1], 'hspace': 0.05},
                        sharex=True
                    )

                    im_sem = ax_sem.imshow(sem_data, cmap='Reds', aspect='auto', vmin=0, vmax=vmax_sem)
                    ax_sem.set_yticks(np.arange(num_items))
                    ax_sem.set_yticklabels(item_labels)
                    ax_sem.set_ylabel('Item')
                    ax_sem.set_title(f'Trial #{trial_num} - SEM of Delta Logit ({pos_title}, Reward {reward_str})\nn={num_episodes} episodes')
                    plt.colorbar(im_sem, ax=[ax_sem, ax_nm_mean_s, ax_nm_sem_s], label='SEM')

                    for i in range(num_items):
                        for j in range(num_adj_pairs):
                            val = sem_data[i, j]
                            if not np.isnan(val):
                                text_color = 'white' if val > vmax_sem * 0.5 else 'black'
                                ax_sem.text(j, i, f'{val:.4f}', ha='center', va='center',
                                           color=text_color, fontsize=5)

                    # NM mean bar
                    nm_mean_row_s = np.ones((1, num_adj_pairs))
                    ax_nm_mean_s.imshow(nm_mean_row_s, cmap='Greys_r', aspect='auto', vmin=0, vmax=1)
                    ax_nm_mean_s.set_yticks([0])
                    ax_nm_mean_s.set_yticklabels(['NM\nmean'], fontsize=7)
                    ax_nm_mean_s.tick_params(axis='x', labelbottom=False)
                    for j in range(num_adj_pairs):
                        val = nm_mean[reward_key][j]
                        txt = f'{val:.3f}' if not np.isnan(val) else '–'
                        ax_nm_mean_s.text(j, 0, txt, ha='center', va='center', color='black', fontsize=5)

                    # NM SEM bar
                    nm_sem_row_s = np.ones((1, num_adj_pairs))
                    ax_nm_sem_s.imshow(nm_sem_row_s, cmap='Greys_r', aspect='auto', vmin=0, vmax=1)
                    ax_nm_sem_s.set_yticks([0])
                    ax_nm_sem_s.set_yticklabels(['NM\nSEM'], fontsize=7)
                    ax_nm_sem_s.set_xticks(np.arange(num_adj_pairs))
                    ax_nm_sem_s.set_xticklabels(adj_pair_labels, rotation=90, fontsize=7)
                    ax_nm_sem_s.set_xlabel('Trial Type')
                    for j in range(num_adj_pairs):
                        val = nm_sem[reward_key][j]
                        txt = f'{val:.4f}' if not np.isnan(val) else '–'
                        ax_nm_sem_s.text(j, 0, txt, ha='center', va='center', color='black', fontsize=5)

                    plt.tight_layout()
                    figures[f"ti_agg_delta_sem/trial{trial_num}_{pos_name}_{reward_label}"] = fig_sem
                    plt.close(fig_sem)

    # === Bar charts: average NM by trial number, split by reward (aggregate) ===
    if 'ti_agg_nm_by_trial' not in skip_prefixes:
        logger.info("Creating aggregate NM by trial number bar charts...")

        nm_by_trial_agg = {0: {t: [] for t in range(num_train_trials)},
                           1: {t: [] for t in range(num_train_trials)}}
        for trial_num in range(num_train_trials):
            for pair_idx in range(num_adj_pairs):
                for reward_key in [0, 1]:
                    nm_by_trial_agg[reward_key][trial_num].extend(nm_storage[trial_num][pair_idx][reward_key])

        for reward_label, reward_key, color in [
            ("Positive Reward (+1)", 0, '#2ca02c'),
            ("Negative Reward (-1)", 1, '#d62728'),
        ]:
            trial_nums = np.arange(num_train_trials)
            means = np.array([np.mean(nm_by_trial_agg[reward_key][t]) if nm_by_trial_agg[reward_key][t] else np.nan for t in trial_nums])
            sems = np.array([np.std(nm_by_trial_agg[reward_key][t]) / np.sqrt(len(nm_by_trial_agg[reward_key][t])) if len(nm_by_trial_agg[reward_key][t]) > 1 else 0.0 for t in trial_nums])
            counts = np.array([len(nm_by_trial_agg[reward_key][t]) for t in trial_nums])

            valid = counts > 0
            if not np.any(valid):
                continue

            fig_bar, ax_bar = plt.subplots(figsize=(max(10, num_train_trials * 0.3), 5), dpi=150)
            ax_bar.bar(trial_nums[valid], means[valid], yerr=sems[valid],
                       color=color, alpha=0.7, edgecolor='black', linewidth=0.5,
                       capsize=2, error_kw={'linewidth': 0.8})
            ax_bar.set_xlabel('Trial Number')
            ax_bar.set_ylabel('Average Neuromodulator')
            ax_bar.set_title(f'{reward_label} - Average NM by Trial Number (n={num_episodes} episodes)')
            ax_bar.set_xticks(trial_nums)
            ax_bar.set_xticklabels([str(t) for t in trial_nums], fontsize=6, rotation=90)
            ax_bar.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

            plt.tight_layout()
            tag = "pos_reward" if reward_key == 0 else "neg_reward"
            figures[f"ti_agg_nm_by_trial/{tag}"] = fig_bar
            plt.close(fig_bar)

    return figures


def create_aggregate_dot_product_figures(dot_delta_pos1, dot_delta_pos2,
                                          dot_abs_pos1, dot_abs_pos2, nm_storage,
                                          adj_pair_labels, num_items, num_train_trials,
                                          item_labels, num_episodes, num_extra_layers):
    """Create aggregate dot-product heatmaps (absolute and delta, mean and SEM) per trial number per extra layer."""
    figures = {}
    num_adj_pairs = len(adj_pair_labels)

    for li in range(num_extra_layers):
        for trial_num in range(num_train_trials):
            # Pre-compute NM mean and SEM per pair for each reward (shared across positions)
            nm_mean = {0: np.full(num_adj_pairs, np.nan), 1: np.full(num_adj_pairs, np.nan)}
            nm_sem_vals = {0: np.full(num_adj_pairs, np.nan), 1: np.full(num_adj_pairs, np.nan)}
            for reward_key in [0, 1]:
                for pair_idx in range(num_adj_pairs):
                    nm_samples = nm_storage[trial_num][pair_idx][reward_key]
                    if nm_samples:
                        nm_mean[reward_key][pair_idx] = np.mean(nm_samples)
                        nm_sem_vals[reward_key][pair_idx] = (
                            np.std(nm_samples) / np.sqrt(len(nm_samples)) if len(nm_samples) > 1 else 0.0
                        )

            # Generate heatmaps for both absolute and delta dot products
            for plot_type, storage_pos1, storage_pos2, label_prefix, key_prefix in [
                ("abs", dot_abs_pos1, dot_abs_pos2, "Dot Product", "ti_agg_dot_abs"),
                ("delta", dot_delta_pos1, dot_delta_pos2, "Delta Dot Product", "ti_agg_dot_delta"),
            ]:
                for pos_name, dot_storage in [("pos1_left", storage_pos1), ("pos2_right", storage_pos2)]:
                    for reward_label, reward_key, reward_str in [("pos_reward", 0, "+1"), ("neg_reward", 1, "-1")]:
                        mean_data = np.full((num_items, num_adj_pairs), np.nan)
                        sem_data = np.full((num_items, num_adj_pairs), np.nan)

                        for pair_idx in range(num_adj_pairs):
                            samples = dot_storage[li][trial_num][pair_idx][reward_key]
                            if samples:
                                stacked = np.stack(samples)
                                mean_data[:, pair_idx] = stacked.mean(axis=0)
                                sem_data[:, pair_idx] = (
                                    stacked.std(axis=0) / np.sqrt(len(samples)) if len(samples) > 1 else 0.0
                                )

                        pos_title = pos_name.replace("_", " ").title()

                        # --- Mean heatmap with NM bars ---
                        vmax = np.nanmax(np.abs(mean_data)) if not np.all(np.isnan(mean_data)) else 0.1
                        vmax = max(vmax, 0.01)

                        fig, (ax, ax_nm_mean, ax_nm_sem) = plt.subplots(
                            3, 1, figsize=(10, 7), dpi=150,
                            gridspec_kw={'height_ratios': [num_items, 1, 1], 'hspace': 0.05},
                            sharex=True
                        )

                        im = ax.imshow(mean_data, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
                        ax.set_yticks(np.arange(num_items))
                        ax.set_yticklabels(item_labels)
                        ax.set_ylabel('Item')
                        ax.set_title(f'Trial #{trial_num} - Mean {label_prefix} (Layer {li+1}, {pos_title}, Reward {reward_str})\nn={num_episodes} episodes')
                        plt.colorbar(im, ax=[ax, ax_nm_mean, ax_nm_sem], label=f'Mean {label_prefix}')

                        for i in range(num_items):
                            for j in range(num_adj_pairs):
                                val = mean_data[i, j]
                                if not np.isnan(val):
                                    text_color = 'white' if abs(val) > vmax * 0.5 else 'black'
                                    ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                                           color=text_color, fontsize=5)

                        # NM mean bar
                        nm_mean_row = np.ones((1, num_adj_pairs))
                        ax_nm_mean.imshow(nm_mean_row, cmap='Greys_r', aspect='auto', vmin=0, vmax=1)
                        ax_nm_mean.set_yticks([0])
                        ax_nm_mean.set_yticklabels(['NM\nmean'], fontsize=7)
                        ax_nm_mean.tick_params(axis='x', labelbottom=False)
                        for j in range(num_adj_pairs):
                            val = nm_mean[reward_key][j]
                            txt = f'{val:.3f}' if not np.isnan(val) else '–'
                            ax_nm_mean.text(j, 0, txt, ha='center', va='center', color='black', fontsize=5)

                        # NM SEM bar
                        nm_sem_row = np.ones((1, num_adj_pairs))
                        ax_nm_sem.imshow(nm_sem_row, cmap='Greys_r', aspect='auto', vmin=0, vmax=1)
                        ax_nm_sem.set_yticks([0])
                        ax_nm_sem.set_yticklabels(['NM\nSEM'], fontsize=7)
                        ax_nm_sem.set_xticks(np.arange(num_adj_pairs))
                        ax_nm_sem.set_xticklabels(adj_pair_labels, rotation=90, fontsize=7)
                        ax_nm_sem.set_xlabel('Trial Type')
                        for j in range(num_adj_pairs):
                            val = nm_sem_vals[reward_key][j]
                            txt = f'{val:.4f}' if not np.isnan(val) else '–'
                            ax_nm_sem.text(j, 0, txt, ha='center', va='center', color='black', fontsize=5)

                        plt.tight_layout()
                        figures[f"{key_prefix}_mean/layer{li+1}_trial{trial_num}_{pos_name}_{reward_label}"] = fig
                        plt.close(fig)

                        # --- SEM heatmap with NM bars ---
                        vmax_sem = np.nanmax(sem_data) if not np.all(np.isnan(sem_data)) else 0.01
                        vmax_sem = max(vmax_sem, 0.001)

                        fig_sem, (ax_sem, ax_nm_mean_s, ax_nm_sem_s) = plt.subplots(
                            3, 1, figsize=(10, 7), dpi=150,
                            gridspec_kw={'height_ratios': [num_items, 1, 1], 'hspace': 0.05},
                            sharex=True
                        )

                        im_sem = ax_sem.imshow(sem_data, cmap='Reds', aspect='auto', vmin=0, vmax=vmax_sem)
                        ax_sem.set_yticks(np.arange(num_items))
                        ax_sem.set_yticklabels(item_labels)
                        ax_sem.set_ylabel('Item')
                        ax_sem.set_title(f'Trial #{trial_num} - SEM of {label_prefix} (Layer {li+1}, {pos_title}, Reward {reward_str})\nn={num_episodes} episodes')
                        plt.colorbar(im_sem, ax=[ax_sem, ax_nm_mean_s, ax_nm_sem_s], label='SEM')

                        for i in range(num_items):
                            for j in range(num_adj_pairs):
                                val = sem_data[i, j]
                                if not np.isnan(val):
                                    text_color = 'white' if val > vmax_sem * 0.5 else 'black'
                                    ax_sem.text(j, i, f'{val:.4f}', ha='center', va='center',
                                               color=text_color, fontsize=5)

                        # NM mean bar
                        nm_mean_row_s = np.ones((1, num_adj_pairs))
                        ax_nm_mean_s.imshow(nm_mean_row_s, cmap='Greys_r', aspect='auto', vmin=0, vmax=1)
                        ax_nm_mean_s.set_yticks([0])
                        ax_nm_mean_s.set_yticklabels(['NM\nmean'], fontsize=7)
                        ax_nm_mean_s.tick_params(axis='x', labelbottom=False)
                        for j in range(num_adj_pairs):
                            val = nm_mean[reward_key][j]
                            txt = f'{val:.3f}' if not np.isnan(val) else '–'
                            ax_nm_mean_s.text(j, 0, txt, ha='center', va='center', color='black', fontsize=5)

                        # NM SEM bar
                        nm_sem_row_s = np.ones((1, num_adj_pairs))
                        ax_nm_sem_s.imshow(nm_sem_row_s, cmap='Greys_r', aspect='auto', vmin=0, vmax=1)
                        ax_nm_sem_s.set_yticks([0])
                        ax_nm_sem_s.set_yticklabels(['NM\nSEM'], fontsize=7)
                        ax_nm_sem_s.set_xticks(np.arange(num_adj_pairs))
                        ax_nm_sem_s.set_xticklabels(adj_pair_labels, rotation=90, fontsize=7)
                        ax_nm_sem_s.set_xlabel('Trial Type')
                        for j in range(num_adj_pairs):
                            val = nm_sem_vals[reward_key][j]
                            txt = f'{val:.4f}' if not np.isnan(val) else '–'
                            ax_nm_sem_s.text(j, 0, txt, ha='center', va='center', color='black', fontsize=5)

                        plt.tight_layout()
                        figures[f"{key_prefix}_sem/layer{li+1}_trial{trial_num}_{pos_name}_{reward_label}"] = fig_sem
                        plt.close(fig_sem)

    return figures


def create_aggregate_pair_dot_figures(pair_dot_delta, pair_dot_abs, nm_storage,
                                       adj_pair_labels, num_train_trials,
                                       num_episodes, num_layers):
    """Create aggregate pair-vs-pair dot product heatmaps (y=trial types, x=presented trial type).

    Like create_aggregate_dot_product_figures but with adjacent trial types on the y-axis
    instead of individual items, split by reward only (no position split).
    num_layers includes extra layers (0..num_layers-2) + final layer (index num_layers-1).
    """
    figures = {}
    num_adj_pairs = len(adj_pair_labels)

    for li in range(num_layers):
        if li == num_layers - 1:
            layer_label = "Final Layer"
            layer_key = "final_layer"
        else:
            layer_label = f"Layer {li+1}"
            layer_key = f"layer{li+1}"

        for trial_num in range(num_train_trials):
            # Pre-compute NM mean and SEM per pair for each reward
            nm_mean = {0: np.full(num_adj_pairs, np.nan), 1: np.full(num_adj_pairs, np.nan)}
            nm_sem_vals = {0: np.full(num_adj_pairs, np.nan), 1: np.full(num_adj_pairs, np.nan)}
            for reward_key in [0, 1]:
                for pair_idx in range(num_adj_pairs):
                    nm_samples = nm_storage[trial_num][pair_idx][reward_key]
                    if nm_samples:
                        nm_mean[reward_key][pair_idx] = np.mean(nm_samples)
                        nm_sem_vals[reward_key][pair_idx] = (
                            np.std(nm_samples) / np.sqrt(len(nm_samples)) if len(nm_samples) > 1 else 0.0
                        )

            for plot_type, storage, label_prefix, key_prefix in [
                ("abs", pair_dot_abs, "Pair Dot Product", "ti_agg_pair_dot_abs"),
                ("delta", pair_dot_delta, "Delta Pair Dot Product", "ti_agg_pair_dot_delta"),
            ]:
                for reward_label, reward_key, reward_str in [("pos_reward", 0, "+1"), ("neg_reward", 1, "-1")]:
                    mean_data = np.full((num_adj_pairs, num_adj_pairs), np.nan)
                    sem_data = np.full((num_adj_pairs, num_adj_pairs), np.nan)

                    for pair_idx in range(num_adj_pairs):
                        samples = storage[li][trial_num][pair_idx][reward_key]
                        if samples:
                            stacked = np.stack(samples)  # (n_samples, num_adj_pairs)
                            mean_data[:, pair_idx] = stacked.mean(axis=0)
                            sem_data[:, pair_idx] = (
                                stacked.std(axis=0) / np.sqrt(len(samples)) if len(samples) > 1 else 0.0
                            )

                    # --- Mean heatmap with NM bars ---
                    vmax = np.nanmax(np.abs(mean_data)) if not np.all(np.isnan(mean_data)) else 0.1
                    vmax = max(vmax, 0.01)

                    fig, (ax, ax_nm_mean, ax_nm_sem) = plt.subplots(
                        3, 1, figsize=(10, 9), dpi=150,
                        gridspec_kw={'height_ratios': [num_adj_pairs, 1, 1], 'hspace': 0.05},
                        sharex=True
                    )

                    im = ax.imshow(mean_data, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
                    ax.set_yticks(np.arange(num_adj_pairs))
                    ax.set_yticklabels(adj_pair_labels, fontsize=7)
                    ax.set_ylabel('Trial Type')
                    ax.set_title(f'Trial #{trial_num} - Mean {label_prefix} ({layer_label}, Reward {reward_str})\nn={num_episodes} episodes')
                    plt.colorbar(im, ax=[ax, ax_nm_mean, ax_nm_sem], label=f'Mean {label_prefix}')

                    for i in range(num_adj_pairs):
                        for j in range(num_adj_pairs):
                            val = mean_data[i, j]
                            if not np.isnan(val):
                                text_color = 'white' if abs(val) > vmax * 0.5 else 'black'
                                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                                       color=text_color, fontsize=5)

                    # NM mean bar
                    nm_mean_row = np.ones((1, num_adj_pairs))
                    ax_nm_mean.imshow(nm_mean_row, cmap='Greys_r', aspect='auto', vmin=0, vmax=1)
                    ax_nm_mean.set_yticks([0])
                    ax_nm_mean.set_yticklabels(['NM\nmean'], fontsize=7)
                    ax_nm_mean.tick_params(axis='x', labelbottom=False)
                    for j in range(num_adj_pairs):
                        val = nm_mean[reward_key][j]
                        txt = f'{val:.3f}' if not np.isnan(val) else '–'
                        ax_nm_mean.text(j, 0, txt, ha='center', va='center', color='black', fontsize=5)

                    # NM SEM bar
                    nm_sem_row = np.ones((1, num_adj_pairs))
                    ax_nm_sem.imshow(nm_sem_row, cmap='Greys_r', aspect='auto', vmin=0, vmax=1)
                    ax_nm_sem.set_yticks([0])
                    ax_nm_sem.set_yticklabels(['NM\nSEM'], fontsize=7)
                    ax_nm_sem.set_xticks(np.arange(num_adj_pairs))
                    ax_nm_sem.set_xticklabels(adj_pair_labels, rotation=90, fontsize=7)
                    ax_nm_sem.set_xlabel('Presented Trial Type')
                    for j in range(num_adj_pairs):
                        val = nm_sem_vals[reward_key][j]
                        txt = f'{val:.4f}' if not np.isnan(val) else '–'
                        ax_nm_sem.text(j, 0, txt, ha='center', va='center', color='black', fontsize=5)

                    plt.tight_layout()
                    figures[f"{key_prefix}_mean/{layer_key}_trial{trial_num}_{reward_label}"] = fig
                    plt.close(fig)

                    # --- SEM heatmap with NM bars ---
                    vmax_sem = np.nanmax(sem_data) if not np.all(np.isnan(sem_data)) else 0.01
                    vmax_sem = max(vmax_sem, 0.001)

                    fig_sem, (ax_sem, ax_nm_mean_s, ax_nm_sem_s) = plt.subplots(
                        3, 1, figsize=(10, 9), dpi=150,
                        gridspec_kw={'height_ratios': [num_adj_pairs, 1, 1], 'hspace': 0.05},
                        sharex=True
                    )

                    im_sem = ax_sem.imshow(sem_data, cmap='Reds', aspect='auto', vmin=0, vmax=vmax_sem)
                    ax_sem.set_yticks(np.arange(num_adj_pairs))
                    ax_sem.set_yticklabels(adj_pair_labels, fontsize=7)
                    ax_sem.set_ylabel('Trial Type')
                    ax_sem.set_title(f'Trial #{trial_num} - SEM of {label_prefix} ({layer_label}, Reward {reward_str})\nn={num_episodes} episodes')
                    plt.colorbar(im_sem, ax=[ax_sem, ax_nm_mean_s, ax_nm_sem_s], label='SEM')

                    for i in range(num_adj_pairs):
                        for j in range(num_adj_pairs):
                            val = sem_data[i, j]
                            if not np.isnan(val):
                                text_color = 'white' if val > vmax_sem * 0.5 else 'black'
                                ax_sem.text(j, i, f'{val:.3f}', ha='center', va='center',
                                           color=text_color, fontsize=5)

                    # NM mean bar
                    nm_mean_row_s = np.ones((1, num_adj_pairs))
                    ax_nm_mean_s.imshow(nm_mean_row_s, cmap='Greys_r', aspect='auto', vmin=0, vmax=1)
                    ax_nm_mean_s.set_yticks([0])
                    ax_nm_mean_s.set_yticklabels(['NM\nmean'], fontsize=7)
                    ax_nm_mean_s.tick_params(axis='x', labelbottom=False)
                    for j in range(num_adj_pairs):
                        val = nm_mean[reward_key][j]
                        txt = f'{val:.3f}' if not np.isnan(val) else '–'
                        ax_nm_mean_s.text(j, 0, txt, ha='center', va='center', color='black', fontsize=5)

                    # NM SEM bar
                    nm_sem_row_s = np.ones((1, num_adj_pairs))
                    ax_nm_sem_s.imshow(nm_sem_row_s, cmap='Greys_r', aspect='auto', vmin=0, vmax=1)
                    ax_nm_sem_s.set_yticks([0])
                    ax_nm_sem_s.set_yticklabels(['NM\nSEM'], fontsize=7)
                    ax_nm_sem_s.set_xticks(np.arange(num_adj_pairs))
                    ax_nm_sem_s.set_xticklabels(adj_pair_labels, rotation=90, fontsize=7)
                    ax_nm_sem_s.set_xlabel('Presented Trial Type')
                    for j in range(num_adj_pairs):
                        val = nm_sem_vals[reward_key][j]
                        txt = f'{val:.4f}' if not np.isnan(val) else '–'
                        ax_nm_sem_s.text(j, 0, txt, ha='center', va='center', color='black', fontsize=5)

                    plt.tight_layout()
                    figures[f"{key_prefix}_sem/{layer_key}_trial{trial_num}_{reward_label}"] = fig_sem
                    plt.close(fig_sem)

    return figures


def create_aggregate_pair_pair_dot_figures(pair_pair_dot_delta, pair_pair_dot_abs, nm_storage,
                                            adj_pair_labels, num_train_trials,
                                            num_episodes, num_layers, focus_pair_labels):
    """Create pair-pair dot product heatmaps for specific presented pairs (e.g. DE/ED).

    Shows how presenting a specific pair affects the full pairwise dot product structure
    among all adjacent pair embeddings.  Both axes are adjacent pair labels.
    num_layers includes extra layers (0..num_layers-2) + final layer (index num_layers-1).
    focus_pair_labels: list of pair labels to create plots for (e.g. ["DE", "ED"]).
    """
    figures = {}
    num_adj_pairs = len(adj_pair_labels)
    pair_label_to_idx = {label: idx for idx, label in enumerate(adj_pair_labels)}

    # Filter to only focus pairs that exist in the label set
    valid_focus = [(label, pair_label_to_idx[label]) for label in focus_pair_labels
                   if label in pair_label_to_idx]
    if not valid_focus:
        return figures

    for li in range(num_layers):
        if li == num_layers - 1:
            layer_label = "Final Layer"
            layer_key = "final_layer"
        else:
            layer_label = f"Layer {li+1}"
            layer_key = f"layer{li+1}"

        for trial_num in range(num_train_trials):
            # Pre-compute NM mean and SEM for the focus pairs
            nm_mean_focus = {}
            nm_sem_focus = {}
            for focus_label, focus_idx in valid_focus:
                nm_mean_focus[focus_label] = {0: np.nan, 1: np.nan}
                nm_sem_focus[focus_label] = {0: np.nan, 1: np.nan}
                for reward_key in [0, 1]:
                    nm_samples = nm_storage[trial_num][focus_idx][reward_key]
                    if nm_samples:
                        nm_mean_focus[focus_label][reward_key] = np.mean(nm_samples)
                        nm_sem_focus[focus_label][reward_key] = (
                            np.std(nm_samples) / np.sqrt(len(nm_samples)) if len(nm_samples) > 1 else 0.0
                        )

            for plot_type, storage, label_prefix, key_prefix in [
                ("abs", pair_pair_dot_abs, "Pair-Pair Dot Product", "ti_agg_pair_pair_dot_abs"),
                ("delta", pair_pair_dot_delta, "Delta Pair-Pair Dot Product", "ti_agg_pair_pair_dot_delta"),
            ]:
                for focus_label, focus_idx in valid_focus:
                    for reward_label, reward_key, reward_str in [("pos_reward", 0, "+1"), ("neg_reward", 1, "-1")]:
                        samples = storage[li][trial_num][focus_idx][reward_key]
                        if not samples:
                            continue

                        stacked = np.stack(samples)  # (n_samples, num_adj_pairs, num_adj_pairs)
                        mean_data = stacked.mean(axis=0)
                        sem_data = (
                            stacked.std(axis=0) / np.sqrt(len(samples)) if len(samples) > 1
                            else np.zeros_like(mean_data)
                        )

                        nm_mean_val = nm_mean_focus[focus_label][reward_key]
                        nm_sem_val = nm_sem_focus[focus_label][reward_key]
                        nm_txt = f"NM mean={nm_mean_val:.3f}" if not np.isnan(nm_mean_val) else "NM mean=–"
                        if not np.isnan(nm_sem_val):
                            nm_txt += f" ± {nm_sem_val:.4f}"

                        # --- Mean heatmap ---
                        vmax = np.nanmax(np.abs(mean_data)) if not np.all(np.isnan(mean_data)) else 0.1
                        vmax = max(vmax, 0.01)

                        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
                        im = ax.imshow(mean_data, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
                        ax.set_yticks(np.arange(num_adj_pairs))
                        ax.set_yticklabels(adj_pair_labels, fontsize=7)
                        ax.set_ylabel('Pair i')
                        ax.set_xticks(np.arange(num_adj_pairs))
                        ax.set_xticklabels(adj_pair_labels, rotation=90, fontsize=7)
                        ax.set_xlabel('Pair j')
                        ax.set_title(
                            f'Trial #{trial_num} - Mean {label_prefix} ({layer_label}, Reward {reward_str})\n'
                            f'Presented: {focus_label}   {nm_txt}   n={num_episodes} episodes'
                        )
                        plt.colorbar(im, ax=ax, label=f'Mean {label_prefix}')

                        for i in range(num_adj_pairs):
                            for j in range(num_adj_pairs):
                                val = mean_data[i, j]
                                if not np.isnan(val):
                                    text_color = 'white' if abs(val) > vmax * 0.5 else 'black'
                                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                                           color=text_color, fontsize=5)

                        plt.tight_layout()
                        figures[f"{key_prefix}_mean/{layer_key}_trial{trial_num}_{focus_label}_{reward_label}"] = fig
                        plt.close(fig)

                        # --- SEM heatmap ---
                        vmax_sem = np.nanmax(sem_data) if not np.all(np.isnan(sem_data)) else 0.01
                        vmax_sem = max(vmax_sem, 0.001)

                        fig_sem, ax_sem = plt.subplots(figsize=(10, 8), dpi=150)
                        im_sem = ax_sem.imshow(sem_data, cmap='Reds', aspect='auto', vmin=0, vmax=vmax_sem)
                        ax_sem.set_yticks(np.arange(num_adj_pairs))
                        ax_sem.set_yticklabels(adj_pair_labels, fontsize=7)
                        ax_sem.set_ylabel('Pair i')
                        ax_sem.set_xticks(np.arange(num_adj_pairs))
                        ax_sem.set_xticklabels(adj_pair_labels, rotation=90, fontsize=7)
                        ax_sem.set_xlabel('Pair j')
                        ax_sem.set_title(
                            f'Trial #{trial_num} - SEM of {label_prefix} ({layer_label}, Reward {reward_str})\n'
                            f'Presented: {focus_label}   {nm_txt}   n={num_episodes} episodes'
                        )
                        plt.colorbar(im_sem, ax=ax_sem, label='SEM')

                        for i in range(num_adj_pairs):
                            for j in range(num_adj_pairs):
                                val = sem_data[i, j]
                                if not np.isnan(val):
                                    text_color = 'white' if val > vmax_sem * 0.5 else 'black'
                                    ax_sem.text(j, i, f'{val:.3f}', ha='center', va='center',
                                               color=text_color, fontsize=5)

                        plt.tight_layout()
                        figures[f"{key_prefix}_sem/{layer_key}_trial{trial_num}_{focus_label}_{reward_label}"] = fig_sem
                        plt.close(fig_sem)

    return figures


def create_pair_pair_dot_evolution_figures(pair_pair_dot_abs, adj_pair_labels,
                                           num_train_trials, num_episodes, num_layers):
    """Create pair-pair dot product evolution heatmaps across trials, pooled over rewards.

    For each layer and trial number, pools all samples across presented pairs and
    both reward types to show the overall pairwise dot product structure among all
    adjacent pair embeddings at that training stage.
    """
    figures = {}
    num_adj_pairs = len(adj_pair_labels)

    for li in range(num_layers):
        if li == num_layers - 1:
            layer_label = "Final Layer"
            layer_key = "final_layer"
        else:
            layer_label = f"Layer {li+1}"
            layer_key = f"layer{li+1}"

        for trial_num in range(num_train_trials):
            # Pool all samples across presented pairs and reward types
            all_samples = []
            for pair_idx in range(num_adj_pairs):
                for reward_key in [0, 1]:
                    all_samples.extend(pair_pair_dot_abs[li][trial_num][pair_idx][reward_key])

            if not all_samples:
                continue

            stacked = np.stack(all_samples)  # (n_total, num_adj_pairs, num_adj_pairs)
            mean_data = stacked.mean(axis=0)
            sem_data = (
                stacked.std(axis=0) / np.sqrt(len(all_samples)) if len(all_samples) > 1
                else np.zeros_like(mean_data)
            )

            # --- Mean heatmap ---
            vmax = np.nanmax(np.abs(mean_data)) if not np.all(np.isnan(mean_data)) else 0.1
            vmax = max(vmax, 0.01)

            fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
            im = ax.imshow(mean_data, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
            ax.set_yticks(np.arange(num_adj_pairs))
            ax.set_yticklabels(adj_pair_labels, fontsize=7)
            ax.set_ylabel('Pair i')
            ax.set_xticks(np.arange(num_adj_pairs))
            ax.set_xticklabels(adj_pair_labels, rotation=90, fontsize=7)
            ax.set_xlabel('Pair j')
            ax.set_title(
                f'Trial #{trial_num} - Mean Pair-Pair Dot Product ({layer_label})\n'
                f'Pooled over rewards and presented pairs   n={num_episodes} episodes'
            )
            plt.colorbar(im, ax=ax, label='Mean Dot Product')

            for i in range(num_adj_pairs):
                for j in range(num_adj_pairs):
                    val = mean_data[i, j]
                    if not np.isnan(val):
                        text_color = 'white' if abs(val) > vmax * 0.5 else 'black'
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                               color=text_color, fontsize=5)

            plt.tight_layout()
            figures[f"ti_agg_pair_pair_dot_evo_mean/{layer_key}_trial{trial_num}"] = fig
            plt.close(fig)

            # --- SEM heatmap ---
            vmax_sem = np.nanmax(sem_data) if not np.all(np.isnan(sem_data)) else 0.01
            vmax_sem = max(vmax_sem, 0.001)

            fig_sem, ax_sem = plt.subplots(figsize=(10, 8), dpi=150)
            im_sem = ax_sem.imshow(sem_data, cmap='Reds', aspect='auto', vmin=0, vmax=vmax_sem)
            ax_sem.set_yticks(np.arange(num_adj_pairs))
            ax_sem.set_yticklabels(adj_pair_labels, fontsize=7)
            ax_sem.set_ylabel('Pair i')
            ax_sem.set_xticks(np.arange(num_adj_pairs))
            ax_sem.set_xticklabels(adj_pair_labels, rotation=90, fontsize=7)
            ax_sem.set_xlabel('Pair j')
            ax_sem.set_title(
                f'Trial #{trial_num} - SEM Pair-Pair Dot Product ({layer_label})\n'
                f'Pooled over rewards and presented pairs   n={num_episodes} episodes'
            )
            plt.colorbar(im_sem, ax=ax_sem, label='SEM')

            for i in range(num_adj_pairs):
                for j in range(num_adj_pairs):
                    val = sem_data[i, j]
                    if not np.isnan(val):
                        text_color = 'white' if val > vmax_sem * 0.5 else 'black'
                        ax_sem.text(j, i, f'{val:.3f}', ha='center', va='center',
                                   color=text_color, fontsize=5)

            plt.tight_layout()
            figures[f"ti_agg_pair_pair_dot_evo_sem/{layer_key}_trial{trial_num}"] = fig_sem
            plt.close(fig_sem)

    return figures


def create_aggregate_pair_logit_figures(pair_logit_delta, pair_logit_abs, nm_storage,
                                         adj_pair_labels, num_train_trials,
                                         num_episodes,
                                         abs_label="Pair Logit", abs_key="ti_agg_pair_logit_abs",
                                         delta_label="Delta Pair Logit", delta_key="ti_agg_pair_logit_delta"):
    """Create aggregate pair logit heatmaps (y=trial types, x=presented trial type), split by reward.

    Like create_aggregate_pair_dot_figures but for logits instead of dot products.
    No layer dimension since logits come from model output.
    """
    figures = {}
    num_adj_pairs = len(adj_pair_labels)

    for trial_num in range(num_train_trials):
        # Pre-compute NM mean and SEM per pair for each reward
        nm_mean = {0: np.full(num_adj_pairs, np.nan), 1: np.full(num_adj_pairs, np.nan)}
        nm_sem_vals = {0: np.full(num_adj_pairs, np.nan), 1: np.full(num_adj_pairs, np.nan)}
        for reward_key in [0, 1]:
            for pair_idx in range(num_adj_pairs):
                nm_samples = nm_storage[trial_num][pair_idx][reward_key]
                if nm_samples:
                    nm_mean[reward_key][pair_idx] = np.mean(nm_samples)
                    nm_sem_vals[reward_key][pair_idx] = (
                        np.std(nm_samples) / np.sqrt(len(nm_samples)) if len(nm_samples) > 1 else 0.0
                    )

        for plot_type, storage, label_prefix, key_prefix in [
            ("abs", pair_logit_abs, abs_label, abs_key),
            ("delta", pair_logit_delta, delta_label, delta_key),
        ]:
            for reward_label, reward_key, reward_str in [("pos_reward", 0, "+1"), ("neg_reward", 1, "-1")]:
                mean_data = np.full((num_adj_pairs, num_adj_pairs), np.nan)
                sem_data = np.full((num_adj_pairs, num_adj_pairs), np.nan)

                for pair_idx in range(num_adj_pairs):
                    samples = storage[trial_num][pair_idx][reward_key]
                    if samples:
                        stacked = np.stack(samples)  # (n_samples, num_adj_pairs)
                        mean_data[:, pair_idx] = stacked.mean(axis=0)
                        sem_data[:, pair_idx] = (
                            stacked.std(axis=0) / np.sqrt(len(samples)) if len(samples) > 1 else 0.0
                        )

                # --- Mean heatmap with NM bars ---
                vmax = np.nanmax(np.abs(mean_data)) if not np.all(np.isnan(mean_data)) else 0.1
                vmax = max(vmax, 0.01)

                fig, (ax, ax_nm_mean, ax_nm_sem) = plt.subplots(
                    3, 1, figsize=(10, 9), dpi=150,
                    gridspec_kw={'height_ratios': [num_adj_pairs, 1, 1], 'hspace': 0.05},
                    sharex=True
                )

                im = ax.imshow(mean_data, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
                ax.set_yticks(np.arange(num_adj_pairs))
                ax.set_yticklabels(adj_pair_labels, fontsize=7)
                ax.set_ylabel('Trial Type')
                ax.set_title(f'Trial #{trial_num} - Mean {label_prefix} (Reward {reward_str})\nn={num_episodes} episodes')
                plt.colorbar(im, ax=[ax, ax_nm_mean, ax_nm_sem], label=f'Mean {label_prefix}')

                for i in range(num_adj_pairs):
                    for j in range(num_adj_pairs):
                        val = mean_data[i, j]
                        if not np.isnan(val):
                            text_color = 'white' if abs(val) > vmax * 0.5 else 'black'
                            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                                   color=text_color, fontsize=5)

                # NM mean bar
                nm_mean_row = np.ones((1, num_adj_pairs))
                ax_nm_mean.imshow(nm_mean_row, cmap='Greys_r', aspect='auto', vmin=0, vmax=1)
                ax_nm_mean.set_yticks([0])
                ax_nm_mean.set_yticklabels(['NM\nmean'], fontsize=7)
                ax_nm_mean.tick_params(axis='x', labelbottom=False)
                for j in range(num_adj_pairs):
                    val = nm_mean[reward_key][j]
                    txt = f'{val:.3f}' if not np.isnan(val) else '–'
                    ax_nm_mean.text(j, 0, txt, ha='center', va='center', color='black', fontsize=5)

                # NM SEM bar
                nm_sem_row = np.ones((1, num_adj_pairs))
                ax_nm_sem.imshow(nm_sem_row, cmap='Greys_r', aspect='auto', vmin=0, vmax=1)
                ax_nm_sem.set_yticks([0])
                ax_nm_sem.set_yticklabels(['NM\nSEM'], fontsize=7)
                ax_nm_sem.set_xticks(np.arange(num_adj_pairs))
                ax_nm_sem.set_xticklabels(adj_pair_labels, rotation=90, fontsize=7)
                ax_nm_sem.set_xlabel('Presented Trial Type')
                for j in range(num_adj_pairs):
                    val = nm_sem_vals[reward_key][j]
                    txt = f'{val:.4f}' if not np.isnan(val) else '–'
                    ax_nm_sem.text(j, 0, txt, ha='center', va='center', color='black', fontsize=5)

                plt.tight_layout()
                figures[f"{key_prefix}_mean/trial{trial_num}_{reward_label}"] = fig
                plt.close(fig)

                # --- SEM heatmap with NM bars ---
                vmax_sem = np.nanmax(sem_data) if not np.all(np.isnan(sem_data)) else 0.01
                vmax_sem = max(vmax_sem, 0.001)

                fig_sem, (ax_sem, ax_nm_mean_s, ax_nm_sem_s) = plt.subplots(
                    3, 1, figsize=(10, 9), dpi=150,
                    gridspec_kw={'height_ratios': [num_adj_pairs, 1, 1], 'hspace': 0.05},
                    sharex=True
                )

                im_sem = ax_sem.imshow(sem_data, cmap='Reds', aspect='auto', vmin=0, vmax=vmax_sem)
                ax_sem.set_yticks(np.arange(num_adj_pairs))
                ax_sem.set_yticklabels(adj_pair_labels, fontsize=7)
                ax_sem.set_ylabel('Trial Type')
                ax_sem.set_title(f'Trial #{trial_num} - SEM of {label_prefix} (Reward {reward_str})\nn={num_episodes} episodes')
                plt.colorbar(im_sem, ax=[ax_sem, ax_nm_mean_s, ax_nm_sem_s], label='SEM')

                for i in range(num_adj_pairs):
                    for j in range(num_adj_pairs):
                        val = sem_data[i, j]
                        if not np.isnan(val):
                            text_color = 'white' if val > vmax_sem * 0.5 else 'black'
                            ax_sem.text(j, i, f'{val:.3f}', ha='center', va='center',
                                       color=text_color, fontsize=5)

                # NM mean bar
                nm_mean_row_s = np.ones((1, num_adj_pairs))
                ax_nm_mean_s.imshow(nm_mean_row_s, cmap='Greys_r', aspect='auto', vmin=0, vmax=1)
                ax_nm_mean_s.set_yticks([0])
                ax_nm_mean_s.set_yticklabels(['NM\nmean'], fontsize=7)
                ax_nm_mean_s.tick_params(axis='x', labelbottom=False)
                for j in range(num_adj_pairs):
                    val = nm_mean[reward_key][j]
                    txt = f'{val:.3f}' if not np.isnan(val) else '–'
                    ax_nm_mean_s.text(j, 0, txt, ha='center', va='center', color='black', fontsize=5)

                # NM SEM bar
                nm_sem_row_s = np.ones((1, num_adj_pairs))
                ax_nm_sem_s.imshow(nm_sem_row_s, cmap='Greys_r', aspect='auto', vmin=0, vmax=1)
                ax_nm_sem_s.set_yticks([0])
                ax_nm_sem_s.set_yticklabels(['NM\nSEM'], fontsize=7)
                ax_nm_sem_s.set_xticks(np.arange(num_adj_pairs))
                ax_nm_sem_s.set_xticklabels(adj_pair_labels, rotation=90, fontsize=7)
                ax_nm_sem_s.set_xlabel('Presented Trial Type')
                for j in range(num_adj_pairs):
                    val = nm_sem_vals[reward_key][j]
                    txt = f'{val:.4f}' if not np.isnan(val) else '–'
                    ax_nm_sem_s.text(j, 0, txt, ha='center', va='center', color='black', fontsize=5)

                plt.tight_layout()
                figures[f"{key_prefix}_sem/trial{trial_num}_{reward_label}"] = fig_sem
                plt.close(fig_sem)

    return figures


def create_post_train_dot_corr_figures(post_dot_pos1, post_dot_pos2, post_corr_pos1, post_corr_pos2,
                                        adj_pair_labels, num_items, item_labels, num_episodes,
                                        num_extra_layers):
    """Create post-training dot product and correlation heatmaps between pair and item representations.

    Storage lists have num_extra_layers + 1 entries: indices 0..num_extra_layers-1 for extra layers,
    index num_extra_layers for the final layer.
    """
    figures = {}
    num_adj_pairs = len(adj_pair_labels)
    num_post_layers = num_extra_layers + 1  # extra layers + final layer

    for li in range(num_post_layers):
        # Label: "Layer 1", "Layer 2", ... for extra layers; "Final Layer" for the last
        if li < num_extra_layers:
            layer_label = f"Layer {li+1}"
            layer_key = f"layer{li+1}"
        else:
            layer_label = "Final Layer"
            layer_key = "final_layer"

        for measure_name, data_pos1, data_pos2, cmap, symmetric in [
            ("Dot Product", post_dot_pos1, post_dot_pos2, 'RdBu_r', True),
            ("Correlation", post_corr_pos1, post_corr_pos2, 'RdBu_r', True),
        ]:
            key_tag = "dot" if "Dot" in measure_name else "corr"

            for pos_name, data_list in [("pos1_left", data_pos1), ("pos2_right", data_pos2)]:
                if not data_list[li]:
                    continue
                stacked = np.stack(data_list[li])  # (num_episodes, num_items, num_adj_pairs)
                mean_data = stacked.mean(axis=0)
                sem_data = stacked.std(axis=0) / np.sqrt(stacked.shape[0]) if stacked.shape[0] > 1 else np.zeros_like(mean_data)

                pos_title = pos_name.replace("_", " ").title()

                # --- Mean heatmap ---
                if symmetric:
                    vmax = max(np.abs(mean_data).max(), 0.01)
                    vmin = -vmax
                else:
                    vmin, vmax = mean_data.min(), max(mean_data.max(), 0.01)

                fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
                im = ax.imshow(mean_data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
                ax.set_yticks(np.arange(num_items))
                ax.set_yticklabels(item_labels)
                ax.set_ylabel('Item')
                ax.set_xticks(np.arange(num_adj_pairs))
                ax.set_xticklabels(adj_pair_labels, rotation=90, fontsize=7)
                ax.set_xlabel('Trial Type')
                ax.set_title(f'Post-Training Mean {measure_name} ({layer_label}, {pos_title})\nn={num_episodes} episodes')
                plt.colorbar(im, ax=ax, label=f'Mean {measure_name}')

                for i in range(num_items):
                    for j in range(num_adj_pairs):
                        val = mean_data[i, j]
                        text_color = 'white' if abs(val) > vmax * 0.5 else 'black'
                        ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                               color=text_color, fontsize=5)

                plt.tight_layout()
                figures[f"ti_agg_post_train_{key_tag}_mean/{layer_key}_{pos_name}"] = fig
                plt.close(fig)

                # --- SEM heatmap ---
                vmax_sem = max(sem_data.max(), 0.001)

                fig_sem, ax_sem = plt.subplots(figsize=(10, 6), dpi=150)
                im_sem = ax_sem.imshow(sem_data, cmap='Reds', aspect='auto', vmin=0, vmax=vmax_sem)
                ax_sem.set_yticks(np.arange(num_items))
                ax_sem.set_yticklabels(item_labels)
                ax_sem.set_ylabel('Item')
                ax_sem.set_xticks(np.arange(num_adj_pairs))
                ax_sem.set_xticklabels(adj_pair_labels, rotation=90, fontsize=7)
                ax_sem.set_xlabel('Trial Type')
                ax_sem.set_title(f'Post-Training SEM {measure_name} ({layer_label}, {pos_title})\nn={num_episodes} episodes')
                plt.colorbar(im_sem, ax=ax_sem, label='SEM')

                for i in range(num_items):
                    for j in range(num_adj_pairs):
                        val = sem_data[i, j]
                        text_color = 'white' if val > vmax_sem * 0.5 else 'black'
                        ax_sem.text(j, i, f'{val:.4f}', ha='center', va='center',
                                   color=text_color, fontsize=5)

                plt.tight_layout()
                figures[f"ti_agg_post_train_{key_tag}_sem/{layer_key}_{pos_name}"] = fig_sem
                plt.close(fig_sem)

    return figures


def create_post_train_dot_corr_nonadj_figures(post_dot_nonadj_pos1, post_dot_nonadj_pos2,
                                                post_corr_nonadj_pos1, post_corr_nonadj_pos2,
                                                nonadj_pair_labels, num_items, item_labels,
                                                num_episodes, num_extra_layers):
    """Create post-training dot product and correlation heatmaps: items (y) x nonadjacent pairs (x)."""
    figures = {}
    num_nonadj_pairs = len(nonadj_pair_labels)
    num_post_layers = num_extra_layers + 1

    for li in range(num_post_layers):
        if li < num_extra_layers:
            layer_label = f"Layer {li+1}"
            layer_key = f"layer{li+1}"
        else:
            layer_label = "Final Layer"
            layer_key = "final_layer"

        for measure_name, data_pos1, data_pos2, cmap in [
            ("Dot Product", post_dot_nonadj_pos1, post_dot_nonadj_pos2, 'RdBu_r'),
            ("Correlation", post_corr_nonadj_pos1, post_corr_nonadj_pos2, 'RdBu_r'),
        ]:
            key_tag = "dot" if "Dot" in measure_name else "corr"

            for pos_name, data_list in [("pos1_left", data_pos1), ("pos2_right", data_pos2)]:
                if not data_list[li]:
                    continue
                stacked = np.stack(data_list[li])  # (num_episodes, num_items, num_nonadj_pairs)
                mean_data = stacked.mean(axis=0)
                sem_data = stacked.std(axis=0) / np.sqrt(stacked.shape[0]) if stacked.shape[0] > 1 else np.zeros_like(mean_data)

                pos_title = pos_name.replace("_", " ").title()

                # --- Mean heatmap ---
                vmax = max(np.abs(mean_data).max(), 0.01)
                vmin = -vmax

                fig, ax = plt.subplots(figsize=(max(14, num_nonadj_pairs * 0.35), 6), dpi=150)
                im = ax.imshow(mean_data, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
                ax.set_yticks(np.arange(num_items))
                ax.set_yticklabels(item_labels)
                ax.set_ylabel('Item')
                ax.set_xticks(np.arange(num_nonadj_pairs))
                ax.set_xticklabels(nonadj_pair_labels, rotation=90, fontsize=6)
                ax.set_xlabel('Nonadjacent Trial Type')
                ax.set_title(f'Post-Training Mean {measure_name} - Items x Nonadj ({layer_label}, {pos_title})\nn={num_episodes} episodes')
                plt.colorbar(im, ax=ax, label=f'Mean {measure_name}')

                for i in range(num_items):
                    for j in range(num_nonadj_pairs):
                        val = mean_data[i, j]
                        text_color = 'white' if abs(val) > vmax * 0.5 else 'black'
                        ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                               color=text_color, fontsize=4)

                plt.tight_layout()
                figures[f"ti_agg_post_train_nonadj_{key_tag}_mean/{layer_key}_{pos_name}"] = fig
                plt.close(fig)

                # --- SEM heatmap ---
                vmax_sem = max(sem_data.max(), 0.001)

                fig_sem, ax_sem = plt.subplots(figsize=(max(14, num_nonadj_pairs * 0.35), 6), dpi=150)
                im_sem = ax_sem.imshow(sem_data, cmap='Reds', aspect='auto', vmin=0, vmax=vmax_sem)
                ax_sem.set_yticks(np.arange(num_items))
                ax_sem.set_yticklabels(item_labels)
                ax_sem.set_ylabel('Item')
                ax_sem.set_xticks(np.arange(num_nonadj_pairs))
                ax_sem.set_xticklabels(nonadj_pair_labels, rotation=90, fontsize=6)
                ax_sem.set_xlabel('Nonadjacent Trial Type')
                ax_sem.set_title(f'Post-Training SEM {measure_name} - Items x Nonadj ({layer_label}, {pos_title})\nn={num_episodes} episodes')
                plt.colorbar(im_sem, ax=ax_sem, label='SEM')

                for i in range(num_items):
                    for j in range(num_nonadj_pairs):
                        val = sem_data[i, j]
                        text_color = 'white' if val > vmax_sem * 0.5 else 'black'
                        ax_sem.text(j, i, f'{val:.4f}', ha='center', va='center',
                                   color=text_color, fontsize=4)

                plt.tight_layout()
                figures[f"ti_agg_post_train_nonadj_{key_tag}_sem/{layer_key}_{pos_name}"] = fig_sem
                plt.close(fig_sem)

    return figures


def create_post_train_pairwise_figures(post_pair_dot, post_pair_corr,
                                        adj_pair_labels, num_episodes, num_extra_layers):
    """Create post-training pairwise trial-type dot product and correlation 14x14 heatmaps."""
    figures = {}
    num_adj_pairs = len(adj_pair_labels)
    num_post_layers = num_extra_layers + 1

    for li in range(num_post_layers):
        if li < num_extra_layers:
            layer_label = f"Layer {li+1}"
            layer_key = f"layer{li+1}"
        else:
            layer_label = "Final Layer"
            layer_key = "final_layer"

        for measure_name, data_list, cmap in [
            ("Dot Product", post_pair_dot, 'RdBu_r'),
            ("Correlation", post_pair_corr, 'RdBu_r'),
        ]:
            key_tag = "dot" if "Dot" in measure_name else "corr"

            if not data_list[li]:
                continue
            stacked = np.stack(data_list[li])  # (num_episodes, num_adj_pairs, num_adj_pairs)
            mean_data = stacked.mean(axis=0)
            sem_data = stacked.std(axis=0) / np.sqrt(stacked.shape[0]) if stacked.shape[0] > 1 else np.zeros_like(mean_data)

            # --- Mean heatmap ---
            vmax = max(np.abs(mean_data).max(), 0.01)

            fig, ax = plt.subplots(figsize=(10, 9), dpi=150)
            im = ax.imshow(mean_data, cmap=cmap, aspect='equal', vmin=-vmax, vmax=vmax)
            ax.set_xticks(np.arange(num_adj_pairs))
            ax.set_xticklabels(adj_pair_labels, rotation=90, fontsize=7)
            ax.set_yticks(np.arange(num_adj_pairs))
            ax.set_yticklabels(adj_pair_labels, fontsize=7)
            ax.set_xlabel('Trial Type')
            ax.set_ylabel('Trial Type')
            ax.set_title(f'Post-Training Pairwise Mean {measure_name} ({layer_label})\nn={num_episodes} episodes')
            plt.colorbar(im, ax=ax, label=f'Mean {measure_name}')

            for i in range(num_adj_pairs):
                for j in range(num_adj_pairs):
                    val = mean_data[i, j]
                    text_color = 'white' if abs(val) > vmax * 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                           color=text_color, fontsize=5)

            plt.tight_layout()
            figures[f"ti_agg_post_train_pairwise_{key_tag}_mean/{layer_key}"] = fig
            plt.close(fig)

            # --- SEM heatmap ---
            vmax_sem = max(sem_data.max(), 0.001)

            fig_sem, ax_sem = plt.subplots(figsize=(10, 9), dpi=150)
            im_sem = ax_sem.imshow(sem_data, cmap='Reds', aspect='equal', vmin=0, vmax=vmax_sem)
            ax_sem.set_xticks(np.arange(num_adj_pairs))
            ax_sem.set_xticklabels(adj_pair_labels, rotation=90, fontsize=7)
            ax_sem.set_yticks(np.arange(num_adj_pairs))
            ax_sem.set_yticklabels(adj_pair_labels, fontsize=7)
            ax_sem.set_xlabel('Trial Type')
            ax_sem.set_ylabel('Trial Type')
            ax_sem.set_title(f'Post-Training Pairwise SEM {measure_name} ({layer_label})\nn={num_episodes} episodes')
            plt.colorbar(im_sem, ax=ax_sem, label='SEM')

            for i in range(num_adj_pairs):
                for j in range(num_adj_pairs):
                    val = sem_data[i, j]
                    text_color = 'white' if val > vmax_sem * 0.5 else 'black'
                    ax_sem.text(j, i, f'{val:.3f}', ha='center', va='center',
                               color=text_color, fontsize=5)

            plt.tight_layout()
            figures[f"ti_agg_post_train_pairwise_{key_tag}_sem/{layer_key}"] = fig_sem
            plt.close(fig_sem)

    return figures


def create_alpha_weight_scatter_plots(model, saved_args):
    """Create scatter plots comparing alpha weight left half (item 1) vs right half (item 2) for each layer."""
    figures = {}
    hidden_size = saved_args.hidden_size
    half_size = hidden_size // 2

    # Extra hidden layer alpha weights
    for layer_idx in range(saved_args.extra_layers):
        alpha = model.alpha_extra[layer_idx].detach().cpu().numpy()  # (hidden_size, hidden_size)
        alpha_left = alpha[:, :half_size].flatten()
        alpha_right = alpha[:, half_size:].flatten()

        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
        ax.scatter(alpha_left, alpha_right, alpha=0.5, s=10)
        lim_min = min(alpha_left.min(), alpha_right.min())
        lim_max = max(alpha_left.max(), alpha_right.max())
        ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', alpha=0.5, label='y=x')
        slope, intercept = np.polyfit(alpha_left, alpha_right, 1)
        y_pred = slope * alpha_left + intercept
        ss_res = np.sum((alpha_right - y_pred) ** 2)
        ss_tot = np.sum((alpha_right - np.mean(alpha_right)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        x_line = np.array([lim_min, lim_max])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'g-', linewidth=2, label=f'Best fit: y={slope:.3f}x+{intercept:.3f}\n$R^2$={r_squared:.4f}')
        ax.set_xlabel('Left Half Alpha (Item 1)')
        ax.set_ylabel('Right Half Alpha (Item 2)')
        ax.set_title(f'Hidden Layer {layer_idx + 1}: Item 1 vs Item 2 Alpha Weights')
        ax.set_aspect('equal')
        ax.legend()
        plt.tight_layout()
        figures[f"alpha_weights_item1_vs_item2/hidden{layer_idx + 1}"] = fig
        plt.close(fig)

    # Final layer (main) alpha weights
    alpha_final = model.alpha.detach().cpu().numpy()  # (hidden_size, hidden_size)
    alpha_final_left = alpha_final[:, :half_size].flatten()
    alpha_final_right = alpha_final[:, half_size:].flatten()

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.scatter(alpha_final_left, alpha_final_right, alpha=0.5, s=10)
    lim_min = min(alpha_final_left.min(), alpha_final_right.min())
    lim_max = max(alpha_final_left.max(), alpha_final_right.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', alpha=0.5, label='y=x')
    slope, intercept = np.polyfit(alpha_final_left, alpha_final_right, 1)
    y_pred = slope * alpha_final_left + intercept
    ss_res = np.sum((alpha_final_right - y_pred) ** 2)
    ss_tot = np.sum((alpha_final_right - np.mean(alpha_final_right)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    x_line = np.array([lim_min, lim_max])
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'g-', linewidth=2, label=f'Best fit: y={slope:.3f}x+{intercept:.3f}\n$R^2$={r_squared:.4f}')
    ax.set_xlabel('Left Half Alpha (Item 1)')
    ax.set_ylabel('Right Half Alpha (Item 2)')
    ax.set_title('Final Layer: Item 1 vs Item 2 Alpha Weights')
    ax.set_aspect('equal')
    ax.legend()
    plt.tight_layout()
    figures["alpha_weights_item1_vs_item2/final"] = fig
    plt.close(fig)

    return figures


def create_innate_weight_scatter_plots(model, saved_args):
    """Create scatter plots comparing innate weight left half (item 1) vs right half (item 2) for each layer."""
    figures = {}
    hidden_size = saved_args.hidden_size
    half_size = hidden_size // 2
    item_size = saved_args.item_size

    def _scatter_with_fit(x_flat, y_flat, xlabel, ylabel, title, fig_key):
        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
        ax.scatter(x_flat, y_flat, alpha=0.5, s=10)
        lim_min = min(x_flat.min(), y_flat.min())
        lim_max = max(x_flat.max(), y_flat.max())
        ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', alpha=0.5, label='y=x')
        slope, intercept = np.polyfit(x_flat, y_flat, 1)
        y_pred = slope * x_flat + intercept
        ss_res = np.sum((y_flat - y_pred) ** 2)
        ss_tot = np.sum((y_flat - np.mean(y_flat)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        x_line = np.array([lim_min, lim_max])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'g-', linewidth=2, label=f'Best fit: y={slope:.3f}x+{intercept:.3f}\n$R^2$={r_squared:.4f}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.legend()
        plt.tight_layout()
        figures[fig_key] = fig
        plt.close(fig)

    def _scatter_cross_split(W, col_split, xlabel, ylabel, title, fig_key):
        """Scatter item1 vs item2 weights, colored by within-item vs cross-item rows.
        Top-half rows (item1 neurons): x=within, y=cross.
        Bottom-half rows (item2 neurons): x=cross, y=within."""
        W_left = W[:, :col_split]
        W_right = W[:, col_split:]
        # Top half rows = item1 output neurons
        x_top = W_left[:half_size].flatten()
        y_top = W_right[:half_size].flatten()
        # Bottom half rows = item2 output neurons
        x_bot = W_left[half_size:].flatten()
        y_bot = W_right[half_size:].flatten()

        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
        ax.scatter(x_top, y_top, alpha=0.4, s=10, color='tab:blue', label=f'Item1 neurons (n={len(x_top)}, x=within, y=cross)')
        ax.scatter(x_bot, y_bot, alpha=0.4, s=10, color='tab:orange', label=f'Item2 neurons (n={len(x_bot)}, x=cross, y=within)')
        all_vals = np.concatenate([x_top, y_top, x_bot, y_bot])
        lim_min, lim_max = all_vals.min(), all_vals.max()
        ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', alpha=0.5, label='y=x')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.legend(fontsize=8)
        plt.tight_layout()
        figures[fig_key] = fig
        plt.close(fig)

    # Embedding layer: item 1 columns vs item 2 columns
    if hasattr(model, 'embedding_layer'):
        W_embed = model.embedding_layer.weight.detach().cpu().numpy()  # (hidden_size, 2*item_size)
        W_embed_item1 = W_embed[:, :item_size].flatten()
        W_embed_item2 = W_embed[:, item_size:].flatten()
        _scatter_with_fit(W_embed_item1, W_embed_item2,
                          'Item 1 Weights', 'Item 2 Weights',
                          'Embedding Layer: Item 1 vs Item 2 Innate Weights',
                          'innate_weights_item1_vs_item2/embedding')
        _scatter_cross_split(W_embed, item_size,
                             'Item 1 Weights', 'Item 2 Weights',
                             'Embedding: Item 1 vs Item 2 (cross vs within)',
                             'innate_weights_item1_vs_item2/embedding_cross_split')

    # Compute first_plastic_input_size (mirrors mlp.py logic)
    no_embedding = getattr(saved_args, 'no_embedding', False)
    add_pre_plastic = getattr(saved_args, 'add_additional_hidden_layer_pre_plastic', False)
    if not no_embedding or add_pre_plastic:
        first_plastic_input_size = hidden_size
    else:
        first_plastic_input_size = 2 * item_size

    # Extra hidden layers: left half vs right half
    for layer_idx in range(saved_args.extra_layers):
        W_hidden = model.extra_hidden_layers[layer_idx].weight.detach().cpu().numpy()
        layer_input_size = first_plastic_input_size if layer_idx == 0 else hidden_size
        col_half = layer_input_size // 2
        W_left = W_hidden[:, :col_half].flatten()
        W_right = W_hidden[:, col_half:].flatten()
        _scatter_with_fit(W_left, W_right,
                          'Left Half Weights (Item 1)', 'Right Half Weights (Item 2)',
                          f'Hidden Layer {layer_idx + 1}: Item 1 vs Item 2 Innate Weights',
                          f'innate_weights_item1_vs_item2/hidden{layer_idx + 1}')
        _scatter_cross_split(W_hidden, col_half,
                             'Left Half Weights (Item 1)', 'Right Half Weights (Item 2)',
                             f'Hidden Layer {layer_idx + 1}: Item 1 vs Item 2 (cross vs within)',
                             f'innate_weights_item1_vs_item2/hidden{layer_idx + 1}_cross_split')

    # Final layer (fc2): left half vs right half
    W_final = model.fc2.weight.detach().cpu().numpy()
    fc2_input_size = hidden_size if saved_args.extra_layers > 0 else first_plastic_input_size
    fc2_col_half = fc2_input_size // 2
    W_final_left = W_final[:, :fc2_col_half].flatten()
    W_final_right = W_final[:, fc2_col_half:].flatten()
    _scatter_with_fit(W_final_left, W_final_right,
                      'Left Half Weights (Item 1)', 'Right Half Weights (Item 2)',
                      'Final Layer (fc2): Item 1 vs Item 2 Innate Weights',
                      'innate_weights_item1_vs_item2/final')
    _scatter_cross_split(W_final, fc2_col_half,
                         'Left Half Weights (Item 1)', 'Right Half Weights (Item 2)',
                         'Final Layer (fc2): Item 1 vs Item 2 (cross vs within)',
                         'innate_weights_item1_vs_item2/final_cross_split')

    # Alpha vs hidden weight scatter plots (same layer, element-wise)
    if not getattr(saved_args, 'no_alpha', False):
        for layer_idx in range(saved_args.extra_layers):
            alpha_extra = model.alpha_extra[layer_idx].detach().cpu().numpy()
            if alpha_extra.ndim < 2:
                continue
            W_hidden = model.extra_hidden_layers[layer_idx].weight.detach().cpu().numpy()
            _scatter_with_fit(W_hidden.flatten(), alpha_extra.flatten(),
                              'Hidden Weight', 'Alpha Weight',
                              f'Hidden Layer {layer_idx + 1}: Hidden vs Alpha Weights',
                              f'innate_weights_item1_vs_item2/alpha_vs_hidden{layer_idx + 1}')

        alpha_final = model.alpha.detach().cpu().numpy()
        if alpha_final.ndim >= 2:
            _scatter_with_fit(W_final.flatten(), alpha_final.flatten(),
                              'Hidden Weight (fc2)', 'Alpha Weight',
                              'Final Layer: Hidden vs Alpha Weights',
                              'innate_weights_item1_vs_item2/alpha_vs_hidden_final')

    # --- Top-half neurons vs bottom-half neurons (row split) ---
    # Embedding layer: top rows vs bottom rows
    if hasattr(model, 'embedding_layer'):
        W_embed = model.embedding_layer.weight.detach().cpu().numpy()
        row_half = W_embed.shape[0] // 2
        _scatter_with_fit(W_embed[:row_half].flatten(), W_embed[row_half:].flatten(),
                          'Top Half Neurons (Group 1)', 'Bottom Half Neurons (Group 2)',
                          'Embedding Layer: Top vs Bottom Neuron Weights',
                          'innate_weights_top_vs_bottom/embedding')

    # Extra hidden layers: top rows vs bottom rows
    for layer_idx in range(saved_args.extra_layers):
        W_hidden = model.extra_hidden_layers[layer_idx].weight.detach().cpu().numpy()
        row_half = W_hidden.shape[0] // 2
        _scatter_with_fit(W_hidden[:row_half].flatten(), W_hidden[row_half:].flatten(),
                          'Top Half Neurons (Group 1)', 'Bottom Half Neurons (Group 2)',
                          f'Hidden Layer {layer_idx + 1}: Top vs Bottom Neuron Weights',
                          f'innate_weights_top_vs_bottom/hidden{layer_idx + 1}')

    # Final layer (fc2): top rows vs bottom rows (skip if too few rows, e.g. direct_readout)
    W_final = model.fc2.weight.detach().cpu().numpy()
    if W_final.shape[0] >= 2:
        row_half = W_final.shape[0] // 2
        _scatter_with_fit(W_final[:row_half].flatten(), W_final[row_half:].flatten(),
                          'Top Half Neurons (Group 1)', 'Bottom Half Neurons (Group 2)',
                          'Final Layer (fc2): Top vs Bottom Neuron Weights',
                          'innate_weights_top_vs_bottom/final')

    # Choice/readout layer: top vs bottom (skip if too few rows)
    _is_direct_readout = getattr(saved_args, 'direct_readout', False)
    if not _is_direct_readout and hasattr(model, 'choice'):
        W_choice = model.choice.weight.detach().cpu().numpy()
        if W_choice.shape[1] >= 2:
            col_half = W_choice.shape[1] // 2
            _scatter_with_fit(W_choice[:, :col_half].flatten(), W_choice[:, col_half:].flatten(),
                              'Top Half Input (Group 1)', 'Bottom Half Input (Group 2)',
                              'Choice Layer: Top vs Bottom Neuron Input Weights',
                              'innate_weights_top_vs_bottom/choice')

    return figures


def create_weight_heatmaps(model, saved_args):
    """Create heatmap and histogram plots for all innate weight matrices in the model."""
    figures = {}

    def _heatmap(W, title, fig_key, figsize=None):
        if figsize is None:
            h = max(2, W.shape[0] / 50)
            w = max(4, W.shape[1] / 50)
            figsize = (w + 2, h + 1)
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        vmax = max(abs(W.min()), abs(W.max()))
        if vmax == 0:
            vmax = 1
        im = ax.imshow(W, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto', interpolation='nearest')
        ax.set_xlabel('Input Dimension')
        ax.set_ylabel('Output Dimension')
        ax.set_title(f'{title}\n(shape: {W.shape}, mean={W.mean():.4f}, std={W.std():.4f})')
        plt.colorbar(im, ax=ax, label='Weight')
        plt.tight_layout()
        figures[fig_key] = fig
        plt.close(fig)

    def _histogram(W, title, fig_key):
        flat = W.flatten()
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        ax.hist(flat, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=flat.mean(), color='green', linestyle='-', linewidth=2,
                   label=f'mean={flat.mean():.4f}')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Count')
        ax.set_title(f'{title}\n(n={len(flat)}, std={flat.std():.4f}, min={flat.min():.4f}, max={flat.max():.4f})')
        ax.legend()
        plt.tight_layout()
        figures[fig_key] = fig
        plt.close(fig)

    def _histogram_cross_split(W, row_split, col_split, title, fig_key):
        """Overlay histograms of within-item vs cross-item block weights."""
        if row_split >= W.shape[0] or col_split >= W.shape[1]:
            return  # Matrix too small for cross-split
        within = np.concatenate([W[:row_split, :col_split].flatten(),
                                 W[row_split:, col_split:].flatten()])
        cross = np.concatenate([W[:row_split, col_split:].flatten(),
                                W[row_split:, :col_split].flatten()])
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
        bins = np.linspace(min(within.min(), cross.min()), max(within.max(), cross.max()), 51)
        ax.hist(within, bins=bins, color='tab:blue', edgecolor='black', alpha=0.6,
                label=f'Within-item (n={len(within)}, mean={within.mean():.4f}, std={within.std():.4f})')
        ax.hist(cross, bins=bins, color='tab:orange', edgecolor='black', alpha=0.6,
                label=f'Cross-item (n={len(cross)}, mean={cross.mean():.4f}, std={cross.std():.4f})')
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Count')
        ax.set_title(f'{title}\n(within vs cross-item blocks)')
        ax.legend(fontsize=8)
        plt.tight_layout()
        figures[fig_key] = fig
        plt.close(fig)

    def _heatmap_2sigma(W, title, fig_key, figsize=None):
        """Heatmap with weights within 2σ of the mean zeroed out."""
        W_filtered = W.copy()
        mean, std = W.mean(), W.std()
        mask = np.abs(W_filtered - mean) <= 2 * std
        num_zeroed = mask.sum()
        W_filtered[mask] = 0.0
        if figsize is None:
            h = max(2, W.shape[0] / 50)
            w = max(4, W.shape[1] / 50)
            figsize = (w + 2, h + 1)
        fig, ax = plt.subplots(figsize=figsize, dpi=150)
        vmax = max(abs(W_filtered.min()), abs(W_filtered.max()))
        if vmax == 0:
            vmax = 1
        im = ax.imshow(W_filtered, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto', interpolation='nearest')
        ax.set_xlabel('Input Dimension')
        ax.set_ylabel('Output Dimension')
        ax.set_title(f'{title} (>2σ only)\n(zeroed {num_zeroed}/{W.size}, mean={mean:.4f}, std={std:.4f})')
        plt.colorbar(im, ax=ax, label='Weight')
        plt.tight_layout()
        figures[fig_key] = fig
        plt.close(fig)

    def _both(W, title, heatmap_key, hist_key):
        _heatmap(W, title, heatmap_key)
        _histogram(W, title, hist_key)

    def _all(W, title, heatmap_key, hist_key, cross_hist_key, row_split, col_split):
        _heatmap(W, title, heatmap_key)
        _histogram(W, title, hist_key)
        _histogram_cross_split(W, row_split, col_split, title, cross_hist_key)

    half_h = saved_args.hidden_size // 2
    _item_size = saved_args.item_size

    # --- Embedding layer ---
    if hasattr(model, 'embedding_layer'):
        W_embed = model.embedding_layer.weight.detach().cpu().numpy()
        _all(W_embed, 'Embedding Layer', 'weight_heatmaps/embedding', 'weight_histograms/embedding',
             'weight_histograms/embedding_cross_split', half_h, _item_size)

    # --- Extra hidden layers ---
    for i in range(saved_args.extra_layers):
        W = model.extra_hidden_layers[i].weight.detach().cpu().numpy()
        _all(W, f'Hidden Layer {i+1}', f'weight_heatmaps/hidden{i+1}', f'weight_histograms/hidden{i+1}',
             f'weight_histograms/hidden{i+1}_cross_split', half_h, half_h)

    # --- Final layer (fc2) ---
    W_fc2 = model.fc2.weight.detach().cpu().numpy()
    _all(W_fc2, 'Final Layer (fc2)', 'weight_heatmaps/final', 'weight_histograms/final',
         'weight_histograms/final_cross_split', half_h, half_h)

    # --- Choice (readout) layer ---
    _is_direct_readout = getattr(saved_args, 'direct_readout', False)
    readout_param = model.fc2.weight if _is_direct_readout else model.choice.weight
    W_choice = readout_param.detach().cpu().numpy()
    _both(W_choice, 'Choice (Readout)', 'weight_heatmaps/choice', 'weight_histograms/choice')

    # --- Alpha weights ---
    if not getattr(saved_args, 'no_alpha', False):
        alpha_final = model.alpha.detach().cpu().numpy()
        if alpha_final.ndim == 0:
            pass
        else:
            _all(alpha_final, 'Alpha (Final Layer)', 'weight_heatmaps/alpha_final', 'weight_histograms/alpha_final',
                 'weight_histograms/alpha_final_cross_split', half_h, half_h)
        for i in range(saved_args.extra_layers):
            alpha_extra = model.alpha_extra[i].detach().cpu().numpy()
            if alpha_extra.ndim == 0:
                continue
            _all(alpha_extra, f'Alpha (Hidden Layer {i+1})', f'weight_heatmaps/alpha_hidden{i+1}', f'weight_histograms/alpha_hidden{i+1}',
                 f'weight_histograms/alpha_hidden{i+1}_cross_split', half_h, half_h)

    # --- Reward linear ---
    W_reward = model.reward_linear.weight.detach().cpu().numpy()
    _both(W_reward, 'Reward Linear', 'weight_heatmaps/reward_linear', 'weight_histograms/reward_linear')

    # --- Hidden-to-reward (final layer NM pathway) ---
    W_h2r = model.hidden_to_reward.weight.detach().cpu().numpy()
    _all(W_h2r, 'Hidden-to-Reward (Final)', 'weight_heatmaps/hidden_to_reward', 'weight_histograms/hidden_to_reward',
         'weight_histograms/hidden_to_reward_cross_split', half_h, half_h)

    # --- Hidden-to-reward (extra layers) ---
    for i in range(saved_args.extra_layers):
        W_h2r_extra = model.hidden_to_reward_extra[i].weight.detach().cpu().numpy()
        _all(W_h2r_extra, f'Hidden-to-Reward (Hidden Layer {i+1})',
             f'weight_heatmaps/hidden_to_reward_hidden{i+1}', f'weight_histograms/hidden_to_reward_hidden{i+1}',
             f'weight_histograms/hidden_to_reward_hidden{i+1}_cross_split', half_h, half_h)

    # --- Neuromodulator output (final layer) ---
    W_nm = model.neuromodulator_out.weight.detach().cpu().numpy()
    _both(W_nm, 'Neuromodulator Output (Final)', 'weight_heatmaps/neuromodulator_out', 'weight_histograms/neuromodulator_out')

    # --- Neuromodulator output (extra layers) ---
    if saved_args.use_extra_neuromodulator:
        for i in range(saved_args.extra_layers):
            W_nm_extra = model.neuromodulator_out_extra[i].weight.detach().cpu().numpy()
            _both(W_nm_extra, f'Neuromodulator Output (Hidden Layer {i+1})',
                  f'weight_heatmaps/neuromodulator_out_hidden{i+1}', f'weight_histograms/neuromodulator_out_hidden{i+1}')

    # --- Value output ---
    W_value = model.value_out.weight.detach().cpu().numpy()
    _both(W_value, 'Value Output', 'weight_heatmaps/value_out', 'weight_histograms/value_out')

    # --- 2σ-filtered heatmaps (only outlier weights shown) ---
    if hasattr(model, 'embedding_layer'):
        _heatmap_2sigma(W_embed, 'Embedding Layer', 'weight_heatmaps_2sigma/embedding')
    for i in range(saved_args.extra_layers):
        W = model.extra_hidden_layers[i].weight.detach().cpu().numpy()
        _heatmap_2sigma(W, f'Hidden Layer {i+1}', f'weight_heatmaps_2sigma/hidden{i+1}')
    _heatmap_2sigma(W_fc2, 'Final Layer (fc2)', 'weight_heatmaps_2sigma/final')
    _heatmap_2sigma(W_choice, 'Choice (Readout)', 'weight_heatmaps_2sigma/choice')
    if not getattr(saved_args, 'no_alpha', False):
        if alpha_final.ndim >= 2:
            _heatmap_2sigma(alpha_final, 'Alpha (Final Layer)', 'weight_heatmaps_2sigma/alpha_final')
        for i in range(saved_args.extra_layers):
            alpha_extra = model.alpha_extra[i].detach().cpu().numpy()
            if alpha_extra.ndim >= 2:
                _heatmap_2sigma(alpha_extra, f'Alpha (Hidden Layer {i+1})', f'weight_heatmaps_2sigma/alpha_hidden{i+1}')
    _heatmap_2sigma(W_reward, 'Reward Linear', 'weight_heatmaps_2sigma/reward_linear')
    _heatmap_2sigma(W_h2r, 'Hidden-to-Reward (Final)', 'weight_heatmaps_2sigma/hidden_to_reward')
    for i in range(saved_args.extra_layers):
        W_h2r_extra = model.hidden_to_reward_extra[i].weight.detach().cpu().numpy()
        _heatmap_2sigma(W_h2r_extra, f'Hidden-to-Reward (Hidden Layer {i+1})', f'weight_heatmaps_2sigma/hidden_to_reward_hidden{i+1}')
    _heatmap_2sigma(W_nm, 'Neuromodulator Output (Final)', 'weight_heatmaps_2sigma/neuromodulator_out')
    if saved_args.use_extra_neuromodulator:
        for i in range(saved_args.extra_layers):
            W_nm_extra = model.neuromodulator_out_extra[i].weight.detach().cpu().numpy()
            _heatmap_2sigma(W_nm_extra, f'Neuromodulator Output (Hidden Layer {i+1})', f'weight_heatmaps_2sigma/neuromodulator_out_hidden{i+1}')
    _heatmap_2sigma(W_value, 'Value Output', 'weight_heatmaps_2sigma/value_out')

    return figures


def create_post_train_logit_bar_chart(post_logits, adj_pair_labels, num_episodes,
                                       title=None, fig_key="ti_agg_post_train_logits/bar_chart",
                                       ylabel="Logit"):
    """Create a bar chart of post-training logit values for each adjacent trial type."""
    figures = {}
    num_adj_pairs = len(adj_pair_labels)

    stacked = np.stack(post_logits)  # (num_episodes, num_adj_pairs)
    means = stacked.mean(axis=0)
    sems = stacked.std(axis=0) / np.sqrt(stacked.shape[0]) if stacked.shape[0] > 1 else np.zeros(num_adj_pairs)

    if title is None:
        title = f'Post-Training Readout Logit by Trial Type\nn={num_episodes} episodes'

    # Color bars by pair: green for correct ordering (high item in pos1), red for reversed
    colors = []
    for i in range(num_adj_pairs):
        if i % 2 == 0:
            colors.append('#2ca02c')  # correct ordering (e.g. AB)
        else:
            colors.append('#d62728')  # reversed ordering (e.g. BA)

    fig, ax = plt.subplots(figsize=(max(8, num_adj_pairs * 0.5), 5), dpi=150)
    x = np.arange(num_adj_pairs)
    ax.bar(x, means, yerr=sems, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5,
           capsize=3, error_kw={'linewidth': 0.8})
    ax.set_xticks(x)
    ax.set_xticklabels(adj_pair_labels, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Trial Type')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    # Add value labels on bars
    for i, (m, s) in enumerate(zip(means, sems)):
        ax.text(i, m + s + 0.02 * max(abs(means.max()), abs(means.min()), 0.1),
                f'{m:.3f}', ha='center', va='bottom', fontsize=6)

    legend_elements = [
        Line2D([0], [0], color='#2ca02c', lw=6, label='Correct order (high in pos1)'),
        Line2D([0], [0], color='#d62728', lw=6, label='Reversed order (low in pos1)'),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc='best')

    plt.tight_layout()
    figures[fig_key] = fig
    plt.close(fig)

    return figures


def create_post_train_nonadj_logit_bar_chart(post_nonadj_logits, nonadj_pair_labels, num_episodes,
                                              title=None, fig_key="ti_agg_post_train_logits/nonadj_bar_chart",
                                              ylabel="Logit"):
    """Create a bar chart of post-training logit values for each nonadjacent trial type."""
    figures = {}
    num_nonadj_pairs = len(nonadj_pair_labels)

    if title is None:
        title = f'Post-Training Readout Logit by Nonadjacent Trial Type\nn={num_episodes} episodes'

    stacked = np.stack(post_nonadj_logits)  # (num_episodes, num_nonadj_pairs)
    means = stacked.mean(axis=0)
    sems = stacked.std(axis=0) / np.sqrt(stacked.shape[0]) if stacked.shape[0] > 1 else np.zeros(num_nonadj_pairs)

    # Color bars: green for correct ordering (even index), red for reversed (odd index)
    colors = []
    for i in range(num_nonadj_pairs):
        if i % 2 == 0:
            colors.append('#2ca02c')  # correct ordering (e.g. AC)
        else:
            colors.append('#d62728')  # reversed ordering (e.g. CA)

    fig, ax = plt.subplots(figsize=(max(12, num_nonadj_pairs * 0.4), 5), dpi=150)
    x = np.arange(num_nonadj_pairs)
    ax.bar(x, means, yerr=sems, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5,
           capsize=2, error_kw={'linewidth': 0.8})
    ax.set_xticks(x)
    ax.set_xticklabels(nonadj_pair_labels, rotation=45, ha='right', fontsize=6)
    ax.set_xlabel('Trial Type')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    # Add value labels on bars
    for i, (m, s) in enumerate(zip(means, sems)):
        ax.text(i, m + s + 0.02 * max(abs(means.max()), abs(means.min()), 0.1),
                f'{m:.3f}', ha='center', va='bottom', fontsize=5)

    legend_elements = [
        Line2D([0], [0], color='#2ca02c', lw=6, label='Correct order (high in pos1)'),
        Line2D([0], [0], color='#d62728', lw=6, label='Reversed order (low in pos1)'),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc='best')

    plt.tight_layout()
    figures[fig_key] = fig
    plt.close(fig)

    return figures


def create_post_train_item_logit_bar_chart(post_item_logits_pos1, post_item_logits_pos2,
                                            item_labels, num_episodes,
                                            title_template=None,
                                            fig_key_prefix="ti_agg_post_train_logits/item",
                                            ylabel="Logit"):
    """Create bar charts of post-training individual item logits, split by position."""
    figures = {}
    num_items = len(item_labels)

    for pos_name, logit_list, color in [
        ("pos1_left", post_item_logits_pos1, '#1f77b4'),
        ("pos2_right", post_item_logits_pos2, '#ff7f0e'),
    ]:
        stacked = np.stack(logit_list)  # (num_episodes, num_items)
        means = stacked.mean(axis=0)
        sems = stacked.std(axis=0) / np.sqrt(stacked.shape[0]) if stacked.shape[0] > 1 else np.zeros(num_items)

        pos_title = pos_name.replace("_", " ").title()
        if title_template is not None:
            title = title_template.format(pos_title=pos_title, num_episodes=num_episodes)
        else:
            title = f'Post-Training Individual Item Logit ({pos_title})\nn={num_episodes} episodes'
        fig, ax = plt.subplots(figsize=(max(6, num_items * 0.6), 5), dpi=150)
        x = np.arange(num_items)
        ax.bar(x, means, yerr=sems, color=color, alpha=0.8, edgecolor='black', linewidth=0.5,
               capsize=3, error_kw={'linewidth': 0.8})
        ax.set_xticks(x)
        ax.set_xticklabels(item_labels, fontsize=9)
        ax.set_xlabel('Item')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

        for i, (m, s) in enumerate(zip(means, sems)):
            ax.text(i, m + s + 0.02 * max(abs(means.max()), abs(means.min()), 0.1),
                    f'{m:.3f}', ha='center', va='bottom', fontsize=7)

        plt.tight_layout()
        figures[f"{fig_key_prefix}_{pos_name}"] = fig
        plt.close(fig)

    return figures


def create_item_dot_weighted_logit_bar_chart(post_dot_pos1, post_dot_pos2, post_logits,
                                              item_labels, num_episodes, num_extra_layers,
                                              layer_li=None, title_template=None,
                                              fig_key_prefix="ti_agg_post_train_logits/item_dot_weighted",
                                              ylabel="Dot-Weighted Logit Sum"):
    """Create bar charts of dot-product-weighted logit sums for individual items, split by position.

    For each item i in position p, computes:
        weighted_logit[i] = sum_j(dot_pos_p[i, j] * adj_logit[j])
    where j indexes adjacent pair types. Computed per-episode, then averaged.
    Uses layer_li index (defaults to final layer = num_extra_layers).
    """
    figures = {}
    num_items = len(item_labels)
    if layer_li is None:
        layer_li = num_extra_layers  # last index

    logit_stacked = np.stack(post_logits)  # (num_episodes, num_adj_pairs)

    for pos_name, dot_list, color in [
        ("pos1_left", post_dot_pos1, '#1f77b4'),
        ("pos2_right", post_dot_pos2, '#ff7f0e'),
    ]:
        if not dot_list[layer_li]:
            continue
        dot_stacked = np.stack(dot_list[layer_li])  # (num_episodes, num_items, num_adj_pairs)

        # Per-episode weighted logit: dot_matrix @ adj_logit -> (num_episodes, num_items)
        weighted = np.einsum('bij,bj->bi', dot_stacked, logit_stacked)

        means = weighted.mean(axis=0)
        sems = weighted.std(axis=0) / np.sqrt(weighted.shape[0]) if weighted.shape[0] > 1 else np.zeros(num_items)

        pos_title = pos_name.replace("_", " ").title()
        if title_template is not None:
            title = title_template.format(pos_title=pos_title, num_episodes=num_episodes)
        else:
            title = f'Post-Training Item Dot-Weighted Logit Sum ({pos_title}, Final Layer)\nn={num_episodes} episodes'
        fig, ax = plt.subplots(figsize=(max(6, num_items * 0.6), 5), dpi=150)
        x = np.arange(num_items)
        ax.bar(x, means, yerr=sems, color=color, alpha=0.8, edgecolor='black', linewidth=0.5,
               capsize=3, error_kw={'linewidth': 0.8})
        ax.set_xticks(x)
        ax.set_xticklabels(item_labels, fontsize=9)
        ax.set_xlabel('Item')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

        for i, (m, s) in enumerate(zip(means, sems)):
            offset = s + 0.02 * max(abs(means.max()), abs(means.min()), 0.1)
            va = 'bottom' if m >= 0 else 'top'
            y = m + offset if m >= 0 else m - offset
            ax.text(i, y, f'{m:.2f}', ha='center', va=va, fontsize=7)

        plt.tight_layout()
        figures[f"{fig_key_prefix}_{pos_name}"] = fig
        plt.close(fig)

    return figures


def create_post_train_cross_figures(post_cross_dot, post_cross_corr,
                                     adj_pair_labels, nonadj_pair_labels,
                                     num_episodes, num_extra_layers):
    """Create cross dot-product and correlation heatmaps: adj pairs (y) x nonadj pairs (x)."""
    figures = {}
    num_adj_pairs = len(adj_pair_labels)
    num_nonadj_pairs = len(nonadj_pair_labels)
    num_post_layers = num_extra_layers + 1

    for li in range(num_post_layers):
        if li < num_extra_layers:
            layer_label = f"Layer {li+1}"
            layer_key = f"layer{li+1}"
        else:
            layer_label = "Final Layer"
            layer_key = "final_layer"

        for measure_name, data_list, cmap in [
            ("Dot Product", post_cross_dot, 'RdBu_r'),
            ("Correlation", post_cross_corr, 'RdBu_r'),
        ]:
            key_tag = "dot" if "Dot" in measure_name else "corr"

            if not data_list[li]:
                continue
            stacked = np.stack(data_list[li])  # (num_episodes, num_adj, num_nonadj)
            mean_data = stacked.mean(axis=0)
            sem_data = stacked.std(axis=0) / np.sqrt(stacked.shape[0]) if stacked.shape[0] > 1 else np.zeros_like(mean_data)

            # --- Mean heatmap ---
            vmax = max(np.abs(mean_data).max(), 0.01)

            fig, ax = plt.subplots(figsize=(max(14, num_nonadj_pairs * 0.35), max(6, num_adj_pairs * 0.4)), dpi=150)
            im = ax.imshow(mean_data, cmap=cmap, aspect='auto', vmin=-vmax, vmax=vmax)
            ax.set_yticks(np.arange(num_adj_pairs))
            ax.set_yticklabels(adj_pair_labels, fontsize=7)
            ax.set_ylabel('Adjacent Trial Type')
            ax.set_xticks(np.arange(num_nonadj_pairs))
            ax.set_xticklabels(nonadj_pair_labels, rotation=90, fontsize=6)
            ax.set_xlabel('Nonadjacent Trial Type')
            ax.set_title(f'Post-Training Cross Mean {measure_name} ({layer_label})\nn={num_episodes} episodes')
            plt.colorbar(im, ax=ax, label=f'Mean {measure_name}')

            for i in range(num_adj_pairs):
                for j in range(num_nonadj_pairs):
                    val = mean_data[i, j]
                    text_color = 'white' if abs(val) > vmax * 0.5 else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                           color=text_color, fontsize=4)

            plt.tight_layout()
            figures[f"ti_agg_post_train_cross_{key_tag}_mean/{layer_key}"] = fig
            plt.close(fig)

            # --- SEM heatmap ---
            vmax_sem = max(sem_data.max(), 0.001)

            fig_sem, ax_sem = plt.subplots(figsize=(max(14, num_nonadj_pairs * 0.35), max(6, num_adj_pairs * 0.4)), dpi=150)
            im_sem = ax_sem.imshow(sem_data, cmap='Reds', aspect='auto', vmin=0, vmax=vmax_sem)
            ax_sem.set_yticks(np.arange(num_adj_pairs))
            ax_sem.set_yticklabels(adj_pair_labels, fontsize=7)
            ax_sem.set_ylabel('Adjacent Trial Type')
            ax_sem.set_xticks(np.arange(num_nonadj_pairs))
            ax_sem.set_xticklabels(nonadj_pair_labels, rotation=90, fontsize=6)
            ax_sem.set_xlabel('Nonadjacent Trial Type')
            ax_sem.set_title(f'Post-Training Cross SEM {measure_name} ({layer_label})\nn={num_episodes} episodes')
            plt.colorbar(im_sem, ax=ax_sem, label='SEM')

            for i in range(num_adj_pairs):
                for j in range(num_nonadj_pairs):
                    val = sem_data[i, j]
                    text_color = 'white' if val > vmax_sem * 0.5 else 'black'
                    ax_sem.text(j, i, f'{val:.3f}', ha='center', va='center',
                               color=text_color, fontsize=4)

            plt.tight_layout()
            figures[f"ti_agg_post_train_cross_{key_tag}_sem/{layer_key}"] = fig_sem
            plt.close(fig_sem)

    return figures


def create_dot_weighted_logit_bar_chart(post_pair_dot, post_logits, adj_pair_labels,
                                         num_episodes, num_extra_layers,
                                         layer_li=None, title=None,
                                         fig_key="ti_agg_post_train_logits/dot_weighted_bar_chart",
                                         ylabel="Dot-Weighted Logit Sum"):
    """Create bar chart of dot-product-weighted logit sums per trial type.

    For each column j in the pairwise dot product matrix, computes:
        weighted_logit[j] = sum_i(dot[i, j] * logit[i])
    This is computed per-episode, then averaged.
    """
    figures = {}
    num_adj_pairs = len(adj_pair_labels)
    if layer_li is None:
        layer_li = num_extra_layers  # last index in post_pair_dot

    if not post_pair_dot[layer_li]:
        return figures

    dot_stacked = np.stack(post_pair_dot[layer_li])  # (num_episodes, 14, 14)
    logit_stacked = np.stack(post_logits)  # (num_episodes, 14)

    # Per-episode weighted logit: dot_matrix.T @ logit_vector -> (num_episodes, 14)
    weighted = np.einsum('bij,bi->bj', dot_stacked, logit_stacked)

    means = weighted.mean(axis=0)
    sems = weighted.std(axis=0) / np.sqrt(weighted.shape[0]) if weighted.shape[0] > 1 else np.zeros(num_adj_pairs)

    colors = []
    for i in range(num_adj_pairs):
        if i % 2 == 0:
            colors.append('#2ca02c')
        else:
            colors.append('#d62728')

    if title is None:
        title = f'Post-Training Dot-Weighted Logit Sum (Final Layer)\nn={num_episodes} episodes'

    fig, ax = plt.subplots(figsize=(max(8, num_adj_pairs * 0.5), 5), dpi=150)
    x = np.arange(num_adj_pairs)
    ax.bar(x, means, yerr=sems, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5,
           capsize=3, error_kw={'linewidth': 0.8})
    ax.set_xticks(x)
    ax.set_xticklabels(adj_pair_labels, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Trial Type')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    for i, (m, s) in enumerate(zip(means, sems)):
        offset = s + 0.02 * max(abs(means.max()), abs(means.min()), 0.1)
        va = 'bottom' if m >= 0 else 'top'
        y = m + offset if m >= 0 else m - offset
        ax.text(i, y, f'{m:.2f}', ha='center', va=va, fontsize=6)

    legend_elements = [
        Line2D([0], [0], color='#2ca02c', lw=6, label='Correct order (high in pos1)'),
        Line2D([0], [0], color='#d62728', lw=6, label='Reversed order (low in pos1)'),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc='best')

    plt.tight_layout()
    figures[fig_key] = fig
    plt.close(fig)

    return figures


def create_cross_dot_weighted_logit_bar_chart(post_cross_dot, post_logits, nonadj_pair_labels,
                                               num_episodes, num_extra_layers,
                                               layer_li=None, title=None,
                                               fig_key="ti_agg_post_train_logits/cross_dot_weighted_nonadj_bar_chart",
                                               ylabel="Cross-Dot-Weighted Logit Sum"):
    """Create bar chart of cross-dot-product-weighted logit sums for nonadjacent trial types.

    For each nonadjacent pair column j in the cross dot product matrix (adj x nonadj), computes:
        weighted_logit[j] = sum_i(cross_dot[i, j] * adj_logit[i])
    where i indexes adjacent pairs. Computed per-episode, then averaged.
    """
    figures = {}
    num_nonadj_pairs = len(nonadj_pair_labels)
    if layer_li is None:
        layer_li = num_extra_layers  # last index in post_cross_dot

    if not post_cross_dot[layer_li]:
        return figures

    cross_dot_stacked = np.stack(post_cross_dot[layer_li])  # (num_episodes, num_adj, num_nonadj)
    logit_stacked = np.stack(post_logits)  # (num_episodes, num_adj)

    # Per-episode weighted logit: cross_dot.T @ adj_logit -> (num_episodes, num_nonadj)
    weighted = np.einsum('bij,bi->bj', cross_dot_stacked, logit_stacked)

    means = weighted.mean(axis=0)
    sems = weighted.std(axis=0) / np.sqrt(weighted.shape[0]) if weighted.shape[0] > 1 else np.zeros(num_nonadj_pairs)

    colors = []
    for i in range(num_nonadj_pairs):
        if i % 2 == 0:
            colors.append('#2ca02c')
        else:
            colors.append('#d62728')

    if title is None:
        title = f'Post-Training Cross-Dot-Weighted Logit Sum for Nonadjacent Pairs (Final Layer)\nn={num_episodes} episodes'

    fig, ax = plt.subplots(figsize=(max(12, num_nonadj_pairs * 0.4), 5), dpi=150)
    x = np.arange(num_nonadj_pairs)
    ax.bar(x, means, yerr=sems, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5,
           capsize=2, error_kw={'linewidth': 0.8})
    ax.set_xticks(x)
    ax.set_xticklabels(nonadj_pair_labels, rotation=45, ha='right', fontsize=6)
    ax.set_xlabel('Nonadjacent Trial Type')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

    for i, (m, s) in enumerate(zip(means, sems)):
        offset = s + 0.02 * max(abs(means.max()), abs(means.min()), 0.1)
        va = 'bottom' if m >= 0 else 'top'
        y = m + offset if m >= 0 else m - offset
        ax.text(i, y, f'{m:.2f}', ha='center', va=va, fontsize=5)

    legend_elements = [
        Line2D([0], [0], color='#2ca02c', lw=6, label='Correct order (high in pos1)'),
        Line2D([0], [0], color='#d62728', lw=6, label='Reversed order (low in pos1)'),
    ]
    ax.legend(handles=legend_elements, fontsize=7, loc='best')

    plt.tight_layout()
    figures[fig_key] = fig
    plt.close(fig)

    return figures


def create_per_trial_type_figures(delta_storage_pos1, delta_storage_pos2, nm_storage,
                                   adj_pair_labels, num_items, num_train_trials, item_labels, num_episodes,
                                   skip_prefixes=frozenset()):
    """Create per-trial-type heatmaps: x-axis = trial number, y-axis = items, one figure per trial type."""
    figures = {}
    num_adj_pairs = len(adj_pair_labels)

    for pair_idx in range(num_adj_pairs):
        pair_label = adj_pair_labels[pair_idx]
        for pos_name, delta_storage in [("pos1_left", delta_storage_pos1), ("pos2_right", delta_storage_pos2)]:
            for reward_label, reward_key, reward_str in [("pos_reward", 0, "+1"), ("neg_reward", 1, "-1")]:
                mean_data = np.full((num_items, num_train_trials), np.nan)
                sem_data = np.full((num_items, num_train_trials), np.nan)
                nm_mean_arr = np.full(num_train_trials, np.nan)
                nm_sem_arr = np.full(num_train_trials, np.nan)

                for trial_num in range(num_train_trials):
                    samples = delta_storage[trial_num][pair_idx][reward_key]
                    if samples:
                        stacked = np.stack(samples)
                        mean_data[:, trial_num] = stacked.mean(axis=0)
                        if len(samples) > 1:
                            sem_data[:, trial_num] = stacked.std(axis=0) / np.sqrt(len(samples))
                        else:
                            sem_data[:, trial_num] = 0.0

                    nm_samples = nm_storage[trial_num][pair_idx][reward_key]
                    if nm_samples:
                        nm_mean_arr[trial_num] = np.mean(nm_samples)
                        nm_sem_arr[trial_num] = np.std(nm_samples) / np.sqrt(len(nm_samples)) if len(nm_samples) > 1 else 0.0

                pos_title = pos_name.replace("_", " ").title()

                # --- Mean heatmap with NM bars ---
                if 'ti_agg_delta_by_type_mean' not in skip_prefixes:
                    vmax = np.nanmax(np.abs(mean_data)) if not np.all(np.isnan(mean_data)) else 0.1
                    vmax = max(vmax, 0.01)

                    fig, (ax, ax_nm_mean, ax_nm_sem) = plt.subplots(
                        3, 1, figsize=(max(10, num_train_trials * 0.3), 7), dpi=150,
                        gridspec_kw={'height_ratios': [num_items, 1, 1], 'hspace': 0.05},
                        sharex=True
                    )

                    im = ax.imshow(mean_data, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
                    ax.set_yticks(np.arange(num_items))
                    ax.set_yticklabels(item_labels)
                    ax.set_ylabel('Item')
                    ax.set_title(f'Trial Type {pair_label} - Mean Delta Logit ({pos_title}, Reward {reward_str})\nn={num_episodes} episodes')
                    plt.colorbar(im, ax=[ax, ax_nm_mean, ax_nm_sem], label='Mean Delta Logit')

                    # NM mean bar
                    nm_mean_row = np.ones((1, num_train_trials))
                    ax_nm_mean.imshow(nm_mean_row, cmap='Greys_r', aspect='auto', vmin=0, vmax=1)
                    ax_nm_mean.set_yticks([0])
                    ax_nm_mean.set_yticklabels(['NM\nmean'], fontsize=7)
                    ax_nm_mean.tick_params(axis='x', labelbottom=False)
                    for j in range(num_train_trials):
                        val = nm_mean_arr[j]
                        txt = f'{val:.2f}' if not np.isnan(val) else ''
                        ax_nm_mean.text(j, 0, txt, ha='center', va='center', color='black', fontsize=4)

                    # NM SEM bar
                    nm_sem_row = np.ones((1, num_train_trials))
                    ax_nm_sem.imshow(nm_sem_row, cmap='Greys_r', aspect='auto', vmin=0, vmax=1)
                    ax_nm_sem.set_yticks([0])
                    ax_nm_sem.set_yticklabels(['NM\nSEM'], fontsize=7)
                    ax_nm_sem.set_xticks(np.arange(num_train_trials))
                    ax_nm_sem.set_xticklabels([str(t) for t in range(num_train_trials)], fontsize=5, rotation=90)
                    ax_nm_sem.set_xlabel('Trial Number')
                    for j in range(num_train_trials):
                        val = nm_sem_arr[j]
                        txt = f'{val:.3f}' if not np.isnan(val) else ''
                        ax_nm_sem.text(j, 0, txt, ha='center', va='center', color='black', fontsize=4)

                    plt.tight_layout()
                    figures[f"ti_agg_delta_by_type_mean/{pair_label}_{pos_name}_{reward_label}"] = fig
                    plt.close(fig)

                # --- SEM heatmap with NM bars ---
                if 'ti_agg_delta_by_type_sem' not in skip_prefixes:
                    vmax_sem = np.nanmax(sem_data) if not np.all(np.isnan(sem_data)) else 0.01
                    vmax_sem = max(vmax_sem, 0.001)

                    fig_sem, (ax_sem, ax_nm_mean_s, ax_nm_sem_s) = plt.subplots(
                        3, 1, figsize=(max(10, num_train_trials * 0.3), 7), dpi=150,
                        gridspec_kw={'height_ratios': [num_items, 1, 1], 'hspace': 0.05},
                        sharex=True
                    )

                    im_sem = ax_sem.imshow(sem_data, cmap='Reds', aspect='auto', vmin=0, vmax=vmax_sem)
                    ax_sem.set_yticks(np.arange(num_items))
                    ax_sem.set_yticklabels(item_labels)
                    ax_sem.set_ylabel('Item')
                    ax_sem.set_title(f'Trial Type {pair_label} - SEM of Delta Logit ({pos_title}, Reward {reward_str})\nn={num_episodes} episodes')
                    plt.colorbar(im_sem, ax=[ax_sem, ax_nm_mean_s, ax_nm_sem_s], label='SEM')

                    # NM mean bar
                    nm_mean_row_s = np.ones((1, num_train_trials))
                    ax_nm_mean_s.imshow(nm_mean_row_s, cmap='Greys_r', aspect='auto', vmin=0, vmax=1)
                    ax_nm_mean_s.set_yticks([0])
                    ax_nm_mean_s.set_yticklabels(['NM\nmean'], fontsize=7)
                    ax_nm_mean_s.tick_params(axis='x', labelbottom=False)
                    for j in range(num_train_trials):
                        val = nm_mean_arr[j]
                        txt = f'{val:.2f}' if not np.isnan(val) else ''
                        ax_nm_mean_s.text(j, 0, txt, ha='center', va='center', color='black', fontsize=4)

                    # NM SEM bar
                    nm_sem_row_s = np.ones((1, num_train_trials))
                    ax_nm_sem_s.imshow(nm_sem_row_s, cmap='Greys_r', aspect='auto', vmin=0, vmax=1)
                    ax_nm_sem_s.set_yticks([0])
                    ax_nm_sem_s.set_yticklabels(['NM\nSEM'], fontsize=7)
                    ax_nm_sem_s.set_xticks(np.arange(num_train_trials))
                    ax_nm_sem_s.set_xticklabels([str(t) for t in range(num_train_trials)], fontsize=5, rotation=90)
                    ax_nm_sem_s.set_xlabel('Trial Number')
                    for j in range(num_train_trials):
                        val = nm_sem_arr[j]
                        txt = f'{val:.3f}' if not np.isnan(val) else ''
                        ax_nm_sem_s.text(j, 0, txt, ha='center', va='center', color='black', fontsize=4)

                    plt.tight_layout()
                    figures[f"ti_agg_delta_by_type_sem/{pair_label}_{pos_name}_{reward_label}"] = fig_sem
                    plt.close(fig_sem)

    return figures


def create_per_trial_type_dot_figures(dot_delta_pos1, dot_delta_pos2, nm_storage,
                                       adj_pair_labels, num_items, num_train_trials,
                                       item_labels, num_episodes, num_extra_layers):
    """Create per-trial-type dot product delta heatmaps: x-axis = trial number, y-axis = items."""
    figures = {}
    num_adj_pairs = len(adj_pair_labels)

    for li in range(num_extra_layers):
        for pair_idx in range(num_adj_pairs):
            pair_label = adj_pair_labels[pair_idx]
            for pos_name, dot_storage in [("pos1_left", dot_delta_pos1), ("pos2_right", dot_delta_pos2)]:
                for reward_label, reward_key, reward_str in [("pos_reward", 0, "+1"), ("neg_reward", 1, "-1")]:
                    mean_data = np.full((num_items, num_train_trials), np.nan)
                    sem_data = np.full((num_items, num_train_trials), np.nan)
                    nm_mean_arr = np.full(num_train_trials, np.nan)
                    nm_sem_arr = np.full(num_train_trials, np.nan)

                    for trial_num in range(num_train_trials):
                        samples = dot_storage[li][trial_num][pair_idx][reward_key]
                        if samples:
                            stacked = np.stack(samples)
                            mean_data[:, trial_num] = stacked.mean(axis=0)
                            if len(samples) > 1:
                                sem_data[:, trial_num] = stacked.std(axis=0) / np.sqrt(len(samples))
                            else:
                                sem_data[:, trial_num] = 0.0

                        nm_samples = nm_storage[trial_num][pair_idx][reward_key]
                        if nm_samples:
                            nm_mean_arr[trial_num] = np.mean(nm_samples)
                            nm_sem_arr[trial_num] = np.std(nm_samples) / np.sqrt(len(nm_samples)) if len(nm_samples) > 1 else 0.0

                    pos_title = pos_name.replace("_", " ").title()

                    # --- Mean heatmap with NM bars ---
                    vmax = np.nanmax(np.abs(mean_data)) if not np.all(np.isnan(mean_data)) else 0.1
                    vmax = max(vmax, 0.01)

                    fig, (ax, ax_nm_mean, ax_nm_sem) = plt.subplots(
                        3, 1, figsize=(max(10, num_train_trials * 0.3), 7), dpi=150,
                        gridspec_kw={'height_ratios': [num_items, 1, 1], 'hspace': 0.05},
                        sharex=True
                    )

                    im = ax.imshow(mean_data, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
                    ax.set_yticks(np.arange(num_items))
                    ax.set_yticklabels(item_labels)
                    ax.set_ylabel('Item')
                    ax.set_title(f'Trial Type {pair_label} - Mean Delta Dot Product (Layer {li+1}, {pos_title}, Reward {reward_str})\nn={num_episodes} episodes')
                    plt.colorbar(im, ax=[ax, ax_nm_mean, ax_nm_sem], label='Mean Delta Dot Product')

                    # NM mean bar
                    nm_mean_row = np.ones((1, num_train_trials))
                    ax_nm_mean.imshow(nm_mean_row, cmap='Greys_r', aspect='auto', vmin=0, vmax=1)
                    ax_nm_mean.set_yticks([0])
                    ax_nm_mean.set_yticklabels(['NM\nmean'], fontsize=7)
                    ax_nm_mean.tick_params(axis='x', labelbottom=False)
                    for j in range(num_train_trials):
                        val = nm_mean_arr[j]
                        txt = f'{val:.2f}' if not np.isnan(val) else ''
                        ax_nm_mean.text(j, 0, txt, ha='center', va='center', color='black', fontsize=4)

                    # NM SEM bar
                    nm_sem_row = np.ones((1, num_train_trials))
                    ax_nm_sem.imshow(nm_sem_row, cmap='Greys_r', aspect='auto', vmin=0, vmax=1)
                    ax_nm_sem.set_yticks([0])
                    ax_nm_sem.set_yticklabels(['NM\nSEM'], fontsize=7)
                    ax_nm_sem.set_xticks(np.arange(num_train_trials))
                    ax_nm_sem.set_xticklabels([str(t) for t in range(num_train_trials)], fontsize=5, rotation=90)
                    ax_nm_sem.set_xlabel('Trial Number')
                    for j in range(num_train_trials):
                        val = nm_sem_arr[j]
                        txt = f'{val:.3f}' if not np.isnan(val) else ''
                        ax_nm_sem.text(j, 0, txt, ha='center', va='center', color='black', fontsize=4)

                    plt.tight_layout()
                    figures[f"ti_agg_dot_delta_by_type_mean/layer{li+1}_{pair_label}_{pos_name}_{reward_label}"] = fig
                    plt.close(fig)

                    # --- SEM heatmap with NM bars ---
                    vmax_sem = np.nanmax(sem_data) if not np.all(np.isnan(sem_data)) else 0.01
                    vmax_sem = max(vmax_sem, 0.001)

                    fig_sem, (ax_sem, ax_nm_mean_s, ax_nm_sem_s) = plt.subplots(
                        3, 1, figsize=(max(10, num_train_trials * 0.3), 7), dpi=150,
                        gridspec_kw={'height_ratios': [num_items, 1, 1], 'hspace': 0.05},
                        sharex=True
                    )

                    im_sem = ax_sem.imshow(sem_data, cmap='Reds', aspect='auto', vmin=0, vmax=vmax_sem)
                    ax_sem.set_yticks(np.arange(num_items))
                    ax_sem.set_yticklabels(item_labels)
                    ax_sem.set_ylabel('Item')
                    ax_sem.set_title(f'Trial Type {pair_label} - SEM of Delta Dot Product (Layer {li+1}, {pos_title}, Reward {reward_str})\nn={num_episodes} episodes')
                    plt.colorbar(im_sem, ax=[ax_sem, ax_nm_mean_s, ax_nm_sem_s], label='SEM')

                    # NM mean bar
                    nm_mean_row_s = np.ones((1, num_train_trials))
                    ax_nm_mean_s.imshow(nm_mean_row_s, cmap='Greys_r', aspect='auto', vmin=0, vmax=1)
                    ax_nm_mean_s.set_yticks([0])
                    ax_nm_mean_s.set_yticklabels(['NM\nmean'], fontsize=7)
                    ax_nm_mean_s.tick_params(axis='x', labelbottom=False)
                    for j in range(num_train_trials):
                        val = nm_mean_arr[j]
                        txt = f'{val:.2f}' if not np.isnan(val) else ''
                        ax_nm_mean_s.text(j, 0, txt, ha='center', va='center', color='black', fontsize=4)

                    # NM SEM bar
                    nm_sem_row_s = np.ones((1, num_train_trials))
                    ax_nm_sem_s.imshow(nm_sem_row_s, cmap='Greys_r', aspect='auto', vmin=0, vmax=1)
                    ax_nm_sem_s.set_yticks([0])
                    ax_nm_sem_s.set_yticklabels(['NM\nSEM'], fontsize=7)
                    ax_nm_sem_s.set_xticks(np.arange(num_train_trials))
                    ax_nm_sem_s.set_xticklabels([str(t) for t in range(num_train_trials)], fontsize=5, rotation=90)
                    ax_nm_sem_s.set_xlabel('Trial Number')
                    for j in range(num_train_trials):
                        val = nm_sem_arr[j]
                        txt = f'{val:.3f}' if not np.isnan(val) else ''
                        ax_nm_sem_s.text(j, 0, txt, ha='center', va='center', color='black', fontsize=4)

                    plt.tight_layout()
                    figures[f"ti_agg_dot_delta_by_type_sem/layer{li+1}_{pair_label}_{pos_name}_{reward_label}"] = fig_sem
                    plt.close(fig_sem)

    return figures


def create_figures(all_corrs, trial_labels, nm_values, logit_values, reward_values,
                   adj_pair_logits, item_logits, batch_items, model, num_networks, num_items,
                   num_train_trials, item_labels, readout_weights, saved_args, device):
    """Create heatmaps, line plots, scatter plots, and pairwise embedding correlation figures."""
    figures = {}
    colors = plt.cm.viridis(np.linspace(0, 1, num_items))
    num_timepoints = num_train_trials + 1

    for net_idx in range(num_networks):
        x_labels = ["(none)"] + trial_labels[net_idx]

        for pos_name, pos_idx in [("pos1_left", 0), ("pos2_right", 1)]:
            data = np.zeros((num_items, num_timepoints))
            for t in range(num_timepoints):
                data[:, t] = all_corrs[net_idx][t][pos_idx]

            # --- Heatmap with neuromodulator and reward bars ---
            fig_hm, (ax_hm, ax_nm, ax_rw) = plt.subplots(
                3, 1, figsize=(max(14, num_train_trials * 0.6), 7), dpi=150,
                gridspec_kw={'height_ratios': [num_items, 1, 1], 'hspace': 0.05},
                sharex=True
            )

            im = ax_hm.imshow(data, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            ax_hm.set_yticks(np.arange(num_items))
            ax_hm.set_yticklabels(item_labels)
            ax_hm.set_ylabel('Item')
            ax_hm.set_title(f'Network {net_idx + 1} - Item in {pos_name.replace("_", " ").title()}\nReadout Correlation Evolution (Final Layer)')
            plt.colorbar(im, ax=[ax_hm, ax_nm, ax_rw], label='Correlation')

            for i in range(num_items):
                for j in range(num_timepoints):
                    val = data[i, j]
                    text_color = 'white' if abs(val) > 0.5 else 'black'
                    ax_hm.text(j, i, f'{val:.2f}', ha='center', va='center',
                              color=text_color, fontsize=5)

            # Neuromodulator bar
            nm_row = np.ones((1, num_timepoints))
            ax_nm.imshow(nm_row, cmap='Greys_r', aspect='auto', vmin=0, vmax=1)
            ax_nm.set_yticks([0])
            ax_nm.set_yticklabels(['NM'], fontsize=8)
            ax_nm.tick_params(axis='x', labelbottom=False)

            ax_nm.text(0, 0, '–', ha='center', va='center', color='black', fontsize=6)
            for j, nm_val in enumerate(nm_values[net_idx]):
                ax_nm.text(j + 1, 0, f'{nm_val:.2f}', ha='center', va='center',
                          color='black', fontsize=6)

            # Reward bar: green for +1, red for -1
            reward_colors = np.zeros((1, num_timepoints, 3))
            reward_colors[0, 0] = [1, 1, 1]  # white for pre-training
            for j, rw_val in enumerate(reward_values[net_idx]):
                if rw_val > 0:
                    reward_colors[0, j + 1] = [0.2, 0.8, 0.2]  # green
                else:
                    reward_colors[0, j + 1] = [0.8, 0.2, 0.2]  # red
            ax_rw.imshow(reward_colors, aspect='auto')
            ax_rw.set_yticks([0])
            ax_rw.set_yticklabels(['Rew'], fontsize=8)
            ax_rw.set_xticks(np.arange(num_timepoints))
            ax_rw.set_xticklabels(x_labels, rotation=90, fontsize=7)
            ax_rw.set_xlabel('Trial Shown')

            plt.tight_layout()
            figures[f"ti_corr_evo_heatmap/net{net_idx+1}_{pos_name}"] = fig_hm
            plt.close(fig_hm)

            # --- Line plot ---
            fig_lp, ax_lp = plt.subplots(figsize=(max(14, num_train_trials * 0.6), 6), dpi=150)

            for item_idx in range(num_items):
                ax_lp.plot(np.arange(num_timepoints), data[item_idx, :],
                          color=colors[item_idx], label=item_labels[item_idx],
                          linewidth=1.5, alpha=0.8, marker='o', markersize=3)

            ax_lp.set_xticks(np.arange(num_timepoints))
            ax_lp.set_xticklabels(x_labels, rotation=90, fontsize=7)
            ax_lp.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
            ax_lp.set_xlabel('Trial Shown')
            ax_lp.set_ylabel('Correlation with Readout')
            ax_lp.set_title(f'Network {net_idx + 1} - Item in {pos_name.replace("_", " ").title()}\nReadout Correlation Evolution (Final Layer)')
            ax_lp.legend(loc='upper left', fontsize=8, ncol=2)
            ax_lp.set_ylim(-1.05, 1.05)

            plt.tight_layout()
            figures[f"ti_corr_evo_lines/net{net_idx+1}_{pos_name}"] = fig_lp
            plt.close(fig_lp)

    # === Item logit evolution heatmaps ===
    logger.info("Creating item logit evolution heatmaps...")

    for net_idx in range(num_networks):
        x_labels = ["(none)"] + trial_labels[net_idx]

        for pos_name, pos_idx in [("pos1_left", 0), ("pos2_right", 1)]:
            data = np.zeros((num_items, num_timepoints))
            for t in range(num_timepoints):
                data[:, t] = item_logits[net_idx][t][pos_idx]

            vmax_il = max(abs(data.min()), abs(data.max()), 0.1)

            fig_il, (ax_il, ax_nm_il, ax_rw_il) = plt.subplots(
                3, 1, figsize=(max(14, num_train_trials * 0.6), 7), dpi=150,
                gridspec_kw={'height_ratios': [num_items, 1, 1], 'hspace': 0.05},
                sharex=True
            )

            im_il = ax_il.imshow(data, cmap='RdBu_r', aspect='auto', vmin=-vmax_il, vmax=vmax_il)
            ax_il.set_yticks(np.arange(num_items))
            ax_il.set_yticklabels(item_labels)
            ax_il.set_ylabel('Item')
            ax_il.set_title(f'Network {net_idx + 1} - Item in {pos_name.replace("_", " ").title()}\nItem Logit Evolution (Final Layer)')
            plt.colorbar(im_il, ax=[ax_il, ax_nm_il, ax_rw_il], label='Logit')

            for i in range(num_items):
                for j in range(num_timepoints):
                    val = data[i, j]
                    text_color = 'white' if abs(val) > vmax_il * 0.5 else 'black'
                    ax_il.text(j, i, f'{val:.2f}', ha='center', va='center',
                              color=text_color, fontsize=5)

            # Neuromodulator bar
            nm_row = np.ones((1, num_timepoints))
            ax_nm_il.imshow(nm_row, cmap='Greys_r', aspect='auto', vmin=0, vmax=1)
            ax_nm_il.set_yticks([0])
            ax_nm_il.set_yticklabels(['NM'], fontsize=8)
            ax_nm_il.tick_params(axis='x', labelbottom=False)

            ax_nm_il.text(0, 0, '–', ha='center', va='center', color='black', fontsize=6)
            for j, nm_val in enumerate(nm_values[net_idx]):
                ax_nm_il.text(j + 1, 0, f'{nm_val:.2f}', ha='center', va='center',
                              color='black', fontsize=6)

            # Reward bar
            reward_colors = np.zeros((1, num_timepoints, 3))
            reward_colors[0, 0] = [1, 1, 1]  # white for pre-training
            for j, rw_val in enumerate(reward_values[net_idx]):
                if rw_val > 0:
                    reward_colors[0, j + 1] = [0.2, 0.8, 0.2]
                else:
                    reward_colors[0, j + 1] = [0.8, 0.2, 0.2]
            ax_rw_il.imshow(reward_colors, aspect='auto')
            ax_rw_il.set_yticks([0])
            ax_rw_il.set_yticklabels(['Rew'], fontsize=8)
            ax_rw_il.set_xticks(np.arange(num_timepoints))
            ax_rw_il.set_xticklabels(x_labels, rotation=90, fontsize=7)
            ax_rw_il.set_xlabel('Trial Shown')

            plt.tight_layout()
            figures[f"ti_item_logit_evo/net{net_idx+1}_{pos_name}"] = fig_il
            plt.close(fig_il)

            # --- Delta item logit heatmap ---
            delta_data = data[:, 1:] - data[:, :-1]
            delta_x_labels = trial_labels[net_idx]
            num_delta = delta_data.shape[1]
            vmax_delta = max(abs(delta_data.min()), abs(delta_data.max()), 0.01)

            fig_dil, (ax_dil, ax_nm_dil, ax_rw_dil) = plt.subplots(
                3, 1, figsize=(max(14, num_train_trials * 0.6), 7), dpi=150,
                gridspec_kw={'height_ratios': [num_items, 1, 1], 'hspace': 0.05},
                sharex=True
            )

            im_dil = ax_dil.imshow(delta_data, cmap='RdBu_r', aspect='auto', vmin=-vmax_delta, vmax=vmax_delta)
            ax_dil.set_yticks(np.arange(num_items))
            ax_dil.set_yticklabels(item_labels)
            ax_dil.set_ylabel('Item')
            ax_dil.set_title(f'Network {net_idx + 1} - Item in {pos_name.replace("_", " ").title()}\nDelta Item Logit per Trial (After - Before)')
            plt.colorbar(im_dil, ax=[ax_dil, ax_nm_dil, ax_rw_dil], label='Delta Logit')

            for i in range(num_items):
                for j in range(num_delta):
                    val = delta_data[i, j]
                    text_color = 'white' if abs(val) > vmax_delta * 0.5 else 'black'
                    ax_dil.text(j, i, f'{val:.2f}', ha='center', va='center',
                               color=text_color, fontsize=5)

            # NM bar
            nm_row_d = np.ones((1, num_delta))
            ax_nm_dil.imshow(nm_row_d, cmap='Greys_r', aspect='auto', vmin=0, vmax=1)
            ax_nm_dil.set_yticks([0])
            ax_nm_dil.set_yticklabels(['NM'], fontsize=8)
            ax_nm_dil.tick_params(axis='x', labelbottom=False)

            for j, nm_val in enumerate(nm_values[net_idx]):
                ax_nm_dil.text(j, 0, f'{nm_val:.2f}', ha='center', va='center',
                               color='black', fontsize=6)

            # Reward bar
            reward_colors_d = np.zeros((1, num_delta, 3))
            for j, rw_val in enumerate(reward_values[net_idx][:num_delta]):
                if rw_val > 0:
                    reward_colors_d[0, j] = [0.2, 0.8, 0.2]
                else:
                    reward_colors_d[0, j] = [0.8, 0.2, 0.2]
            ax_rw_dil.imshow(reward_colors_d, aspect='auto')
            ax_rw_dil.set_yticks([0])
            ax_rw_dil.set_yticklabels(['Rew'], fontsize=8)
            ax_rw_dil.set_xticks(np.arange(num_delta))
            ax_rw_dil.set_xticklabels(delta_x_labels, rotation=90, fontsize=7)
            ax_rw_dil.set_xlabel('Trial Shown')

            plt.tight_layout()
            figures[f"ti_item_logit_delta/net{net_idx+1}_{pos_name}"] = fig_dil
            plt.close(fig_dil)

    # === Adjacent pair logit evolution heatmaps ===
    logger.info("Creating adjacent pair logit evolution heatmaps...")

    # Build adjacent pair labels: AB, BA, BC, CB, ..., GH, HG
    adj_pair_labels = []
    for i in range(num_items - 1):
        adj_pair_labels.append(f"{item_labels[i]}{item_labels[i+1]}")
        adj_pair_labels.append(f"{item_labels[i+1]}{item_labels[i]}")
    num_adj_pairs = len(adj_pair_labels)

    for net_idx in range(num_networks):
        x_labels = ["(none)"] + trial_labels[net_idx]

        # Build data matrix: (num_adj_pairs, num_timepoints)
        logit_data = np.zeros((num_adj_pairs, num_timepoints))
        for t in range(num_timepoints):
            logit_data[:, t] = adj_pair_logits[net_idx][t]

        # Determine symmetric color range
        vmax_logit = max(abs(logit_data.min()), abs(logit_data.max()), 0.1)

        # --- Logit heatmap with NM and reward bars ---
        fig_al, (ax_al, ax_nm_al, ax_rw_al) = plt.subplots(
            3, 1, figsize=(max(14, num_train_trials * 0.6), 9), dpi=150,
            gridspec_kw={'height_ratios': [num_adj_pairs, 1, 1], 'hspace': 0.05},
            sharex=True
        )

        im_al = ax_al.imshow(logit_data, cmap='RdBu_r', aspect='auto', vmin=-vmax_logit, vmax=vmax_logit)
        ax_al.set_yticks(np.arange(num_adj_pairs))
        ax_al.set_yticklabels(adj_pair_labels, fontsize=7)
        ax_al.set_ylabel('Adjacent Pair')
        ax_al.set_title(f'Network {net_idx + 1} - Adjacent Pair Logit Evolution')
        plt.colorbar(im_al, ax=[ax_al, ax_nm_al, ax_rw_al], label='Logit')

        for i in range(num_adj_pairs):
            for j in range(num_timepoints):
                val = logit_data[i, j]
                text_color = 'white' if abs(val) > vmax_logit * 0.5 else 'black'
                ax_al.text(j, i, f'{val:.2f}', ha='center', va='center',
                          color=text_color, fontsize=4)

        # NM bar
        nm_row = np.ones((1, num_timepoints))
        ax_nm_al.imshow(nm_row, cmap='Greys_r', aspect='auto', vmin=0, vmax=1)
        ax_nm_al.set_yticks([0])
        ax_nm_al.set_yticklabels(['NM'], fontsize=8)
        ax_nm_al.tick_params(axis='x', labelbottom=False)

        ax_nm_al.text(0, 0, '–', ha='center', va='center', color='black', fontsize=6)
        for j, nm_val in enumerate(nm_values[net_idx]):
            ax_nm_al.text(j + 1, 0, f'{nm_val:.2f}', ha='center', va='center',
                         color='black', fontsize=6)

        # Reward bar
        reward_colors_al = np.zeros((1, num_timepoints, 3))
        reward_colors_al[0, 0] = [1, 1, 1]  # white for pre-training
        for j, rw_val in enumerate(reward_values[net_idx]):
            if rw_val > 0:
                reward_colors_al[0, j + 1] = [0.2, 0.8, 0.2]
            else:
                reward_colors_al[0, j + 1] = [0.8, 0.2, 0.2]
        ax_rw_al.imshow(reward_colors_al, aspect='auto')
        ax_rw_al.set_yticks([0])
        ax_rw_al.set_yticklabels(['Rew'], fontsize=8)
        ax_rw_al.set_xticks(np.arange(num_timepoints))
        ax_rw_al.set_xticklabels(x_labels, rotation=90, fontsize=7)
        ax_rw_al.set_xlabel('Trial Shown')

        plt.tight_layout()
        figures[f"ti_adj_logit_evo/net{net_idx+1}"] = fig_al
        plt.close(fig_al)

        # --- Delta logit heatmap with NM and reward bars ---
        # Delta = logit_after_trial - logit_before_trial
        # num_timepoints - 1 delta columns (one per trial)
        delta_data = logit_data[:, 1:] - logit_data[:, :-1]
        delta_x_labels = trial_labels[net_idx]
        num_delta = delta_data.shape[1]
        vmax_delta = max(abs(delta_data.min()), abs(delta_data.max()), 0.01)

        fig_dl, (ax_dl, ax_nm_dl, ax_rw_dl) = plt.subplots(
            3, 1, figsize=(max(14, num_train_trials * 0.6), 9), dpi=150,
            gridspec_kw={'height_ratios': [num_adj_pairs, 1, 1], 'hspace': 0.05},
            sharex=True
        )

        im_dl = ax_dl.imshow(delta_data, cmap='RdBu_r', aspect='auto', vmin=-vmax_delta, vmax=vmax_delta)
        ax_dl.set_yticks(np.arange(num_adj_pairs))
        ax_dl.set_yticklabels(adj_pair_labels, fontsize=7)
        ax_dl.set_ylabel('Adjacent Pair')
        ax_dl.set_title(f'Network {net_idx + 1} - Delta Logit per Trial (After - Before)')
        plt.colorbar(im_dl, ax=[ax_dl, ax_nm_dl, ax_rw_dl], label='Delta Logit')

        for i in range(num_adj_pairs):
            for j in range(num_delta):
                val = delta_data[i, j]
                text_color = 'white' if abs(val) > vmax_delta * 0.5 else 'black'
                ax_dl.text(j, i, f'{val:.2f}', ha='center', va='center',
                          color=text_color, fontsize=4)

        # NM bar
        nm_row_d = np.ones((1, num_delta))
        ax_nm_dl.imshow(nm_row_d, cmap='Greys_r', aspect='auto', vmin=0, vmax=1)
        ax_nm_dl.set_yticks([0])
        ax_nm_dl.set_yticklabels(['NM'], fontsize=8)
        ax_nm_dl.tick_params(axis='x', labelbottom=False)

        for j, nm_val in enumerate(nm_values[net_idx]):
            ax_nm_dl.text(j, 0, f'{nm_val:.2f}', ha='center', va='center',
                         color='black', fontsize=6)

        # Reward bar (delta has no pre-training column, so directly indexed)
        reward_colors_dl = np.zeros((1, num_delta, 3))
        for j, rw_val in enumerate(reward_values[net_idx][:num_delta]):
            if rw_val > 0:
                reward_colors_dl[0, j] = [0.2, 0.8, 0.2]
            else:
                reward_colors_dl[0, j] = [0.8, 0.2, 0.2]
        ax_rw_dl.imshow(reward_colors_dl, aspect='auto')
        ax_rw_dl.set_yticks([0])
        ax_rw_dl.set_yticklabels(['Rew'], fontsize=8)
        ax_rw_dl.set_xticks(np.arange(num_delta))
        ax_rw_dl.set_xticklabels(delta_x_labels, rotation=90, fontsize=7)
        ax_rw_dl.set_xlabel('Trial Shown')

        plt.tight_layout()
        figures[f"ti_adj_logit_delta/net{net_idx+1}"] = fig_dl
        plt.close(fig_dl)

    # === Scatter plots: readout logit vs neuromodulator, colored by reward ===
    logger.info("Creating logit vs neuromodulator scatter plots...")

    # Dot sizes: early trials get large dots, later trials get small
    max_dot_size = 120
    min_dot_size = 15

    # Per-network scatter plots
    for net_idx in range(num_networks):
        nm_arr = np.array(nm_values[net_idx])
        logit_arr = np.array(logit_values[net_idx])
        reward_arr = np.array(reward_values[net_idx])
        trial_nums = np.arange(len(nm_arr))
        dot_sizes = max_dot_size - (max_dot_size - min_dot_size) * trial_nums / max(len(trial_nums) - 1, 1)
        colors_scatter = ['green' if r > 0 else 'red' for r in reward_arr]

        fig_sc, ax_sc = plt.subplots(figsize=(8, 6), dpi=150)
        ax_sc.scatter(logit_arr, nm_arr, c=colors_scatter, s=dot_sizes, alpha=0.7, edgecolors='black', linewidths=0.3)
        ax_sc.set_xlabel('Readout Logit')
        ax_sc.set_ylabel('Neuromodulator')
        ax_sc.set_title(f'Network {net_idx + 1} - Readout Logit vs Neuromodulator')
        ax_sc.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax_sc.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)

        # Legend for reward colors
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Correct (+1)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Incorrect (-1)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Early trial (large)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=5, label='Late trial (small)'),
        ]
        ax_sc.legend(handles=legend_elements, loc='best', fontsize=8)

        plt.tight_layout()
        figures[f"ti_logit_vs_nm/net{net_idx+1}"] = fig_sc
        plt.close(fig_sc)

    # Aggregate scatter plot (all networks)
    fig_agg, ax_agg = plt.subplots(figsize=(8, 6), dpi=150)
    for net_idx in range(num_networks):
        nm_arr = np.array(nm_values[net_idx])
        logit_arr = np.array(logit_values[net_idx])
        reward_arr = np.array(reward_values[net_idx])
        trial_nums = np.arange(len(nm_arr))
        dot_sizes = max_dot_size - (max_dot_size - min_dot_size) * trial_nums / max(len(trial_nums) - 1, 1)
        colors_scatter = ['green' if r > 0 else 'red' for r in reward_arr]
        ax_agg.scatter(logit_arr, nm_arr, c=colors_scatter, s=dot_sizes, alpha=0.4, edgecolors='black', linewidths=0.2)

    ax_agg.set_xlabel('Readout Logit')
    ax_agg.set_ylabel('Neuromodulator')
    ax_agg.set_title('All Networks - Readout Logit vs Neuromodulator')
    ax_agg.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax_agg.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Correct (+1)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Incorrect (-1)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Early trial (large)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=5, label='Late trial (small)'),
    ]
    ax_agg.legend(handles=legend_elements, loc='best', fontsize=8)

    plt.tight_layout()
    figures["ti_logit_vs_nm/all_networks"] = fig_agg
    plt.close(fig_agg)

    # === Scatter plots colored by trial type (pair identity) ===
    logger.info("Creating logit vs neuromodulator scatter plots colored by trial type...")

    # Distinct colors for each adjacent pair (up to 7 pairs for 8 items)
    pair_type_colors = [
        '#1f77b4',  # blue
        '#d62728',  # red
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#e377c2',  # pink
        '#17becf',  # teal
        '#7f7f7f',  # gray
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#bcbd22',  # olive
    ]

    # Build canonical pair -> color mapping
    # Canonical pair = sorted letters, e.g. "AB" for both "AB" and "BA"
    canonical_pairs = []
    for i in range(num_items - 1):
        canonical_pairs.append(f"{item_labels[i]}{item_labels[i+1]}")
    pair_color_map = {pair: pair_type_colors[i % len(pair_type_colors)] for i, pair in enumerate(canonical_pairs)}

    def get_canonical_pair(label):
        return "".join(sorted(label))

    # Per-network scatter plots
    for net_idx in range(num_networks):
        nm_arr = np.array(nm_values[net_idx])
        logit_arr = np.array(logit_values[net_idx])
        trial_nums = np.arange(len(nm_arr))
        dot_sizes = max_dot_size - (max_dot_size - min_dot_size) * trial_nums / max(len(trial_nums) - 1, 1)
        pair_colors = [pair_color_map.get(get_canonical_pair(lbl), '#333333') for lbl in trial_labels[net_idx]]

        fig_pt, ax_pt = plt.subplots(figsize=(8, 6), dpi=150)
        ax_pt.scatter(logit_arr, nm_arr, c=pair_colors, s=dot_sizes, alpha=0.7, edgecolors='black', linewidths=0.3)
        ax_pt.set_xlabel('Readout Logit')
        ax_pt.set_ylabel('Neuromodulator')
        ax_pt.set_title(f'Network {net_idx + 1} - Readout Logit vs Neuromodulator (by Pair)')
        ax_pt.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax_pt.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)

        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=pair_color_map[p],
                   markersize=8, markeredgecolor='black', markeredgewidth=0.3, label=p)
            for p in canonical_pairs
        ]
        legend_elements += [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Early trial (large)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=5, label='Late trial (small)'),
        ]
        ax_pt.legend(handles=legend_elements, loc='best', fontsize=7, ncol=2)

        plt.tight_layout()
        figures[f"ti_logit_vs_nm_by_pair/net{net_idx+1}"] = fig_pt
        plt.close(fig_pt)

    # Aggregate scatter plot (all networks)
    fig_agg_pt, ax_agg_pt = plt.subplots(figsize=(8, 6), dpi=150)
    for net_idx in range(num_networks):
        nm_arr = np.array(nm_values[net_idx])
        logit_arr = np.array(logit_values[net_idx])
        trial_nums = np.arange(len(nm_arr))
        dot_sizes = max_dot_size - (max_dot_size - min_dot_size) * trial_nums / max(len(trial_nums) - 1, 1)
        pair_colors = [pair_color_map.get(get_canonical_pair(lbl), '#333333') for lbl in trial_labels[net_idx]]
        ax_agg_pt.scatter(logit_arr, nm_arr, c=pair_colors, s=dot_sizes, alpha=0.4, edgecolors='black', linewidths=0.2)

    ax_agg_pt.set_xlabel('Readout Logit')
    ax_agg_pt.set_ylabel('Neuromodulator')
    ax_agg_pt.set_title('All Networks - Readout Logit vs Neuromodulator (by Pair)')
    ax_agg_pt.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax_agg_pt.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=pair_color_map[p],
               markersize=8, markeredgecolor='black', markeredgewidth=0.3, label=p)
        for p in canonical_pairs
    ]
    legend_elements += [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Early trial (large)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=5, label='Late trial (small)'),
    ]
    ax_agg_pt.legend(handles=legend_elements, loc='best', fontsize=7, ncol=2)

    plt.tight_layout()
    figures["ti_logit_vs_nm_by_pair/all_networks"] = fig_agg_pt
    plt.close(fig_agg_pt)

    # === Pairwise embedding correlation heatmaps for adjacent pairs ===
    logger.info("Computing pairwise adjacent pair embedding correlations...")
    adj_pairs = []
    adj_labels = []
    for i in range(num_items - 1):
        adj_pairs.append((i, i + 1))
        adj_labels.append(f"{item_labels[i]}{item_labels[i+1]}")
        adj_pairs.append((i + 1, i))
        adj_labels.append(f"{item_labels[i+1]}{item_labels[i]}")
    num_adj = len(adj_pairs)

    zero_pw = torch.zeros(1, saved_args.hidden_size, saved_args.hidden_size,
                          dtype=torch.float32, requires_grad=False).to(device)
    zero_epw = [torch.zeros(1, saved_args.hidden_size, saved_args.hidden_size,
                            dtype=torch.float32, requires_grad=False).to(device)
                for _ in range(saved_args.extra_layers)]
    dummy_reward = torch.tensor([0.0], dtype=torch.float32).to(device)

    for net_idx in range(num_networks):
        items = batch_items[net_idx]
        embeddings = []

        for left_idx, right_idx in adj_pairs:
            pair_input = np.concatenate([items[left_idx], items[right_idx]])
            pair_tensor = torch.tensor(pair_input, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.inference_mode():
                out = model(pair_tensor, zero_pw, dummy_reward,
                            extra_plastic_weights=zero_epw, store_embeddings=True, embed_plastic_weights=None)
            embeddings.append(out.embeddings[0][0].detach().cpu().numpy())

        emb_matrix = np.stack(embeddings)
        corr_matrix = np.corrcoef(emb_matrix)
        np.fill_diagonal(corr_matrix, 1.0)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        fig_pw, ax_pw = plt.subplots(figsize=(10, 8), dpi=150)
        im = ax_pw.imshow(corr_matrix, cmap='RdBu_r', aspect='equal', vmin=-1, vmax=1)
        ax_pw.set_xticks(np.arange(num_adj))
        ax_pw.set_xticklabels(adj_labels, rotation=90, fontsize=8)
        ax_pw.set_yticks(np.arange(num_adj))
        ax_pw.set_yticklabels(adj_labels, fontsize=8)
        ax_pw.set_title(f'Network {net_idx + 1} - Pairwise Embedding Correlations\n(Adjacent Pairs, Embedding Layer)')
        plt.colorbar(im, ax=ax_pw, label='Correlation')

        for i in range(num_adj):
            for j in range(num_adj):
                val = corr_matrix[i, j]
                text_color = 'white' if abs(val) > 0.5 else 'black'
                ax_pw.text(j, i, f'{val:.2f}', ha='center', va='center',
                          color=text_color, fontsize=5)

        plt.tight_layout()
        figures[f"ti_pair_embedding_corr/net{net_idx+1}"] = fig_pw
        plt.close(fig_pw)

    # === Bar charts: average NM by trial number, split by reward ===
    logger.info("Creating NM by trial number bar charts (split by reward)...")

    # Collect NM and reward values per trial number across all networks
    nm_by_trial_pos = {t: [] for t in range(num_train_trials)}  # reward = +1
    nm_by_trial_neg = {t: [] for t in range(num_train_trials)}  # reward = -1

    for net_idx in range(num_networks):
        for t in range(num_train_trials):
            rw = reward_values[net_idx][t]
            nm = nm_values[net_idx][t]
            if rw > 0:
                nm_by_trial_pos[t].append(nm)
            else:
                nm_by_trial_neg[t].append(nm)

    for reward_label, nm_by_trial, color in [
        ("Positive Reward (+1)", nm_by_trial_pos, '#2ca02c'),
        ("Negative Reward (-1)", nm_by_trial_neg, '#d62728'),
    ]:
        trial_nums = np.arange(num_train_trials)
        means = np.array([np.mean(nm_by_trial[t]) if nm_by_trial[t] else np.nan for t in trial_nums])
        sems = np.array([np.std(nm_by_trial[t]) / np.sqrt(len(nm_by_trial[t])) if len(nm_by_trial[t]) > 1 else 0.0 for t in trial_nums])
        counts = np.array([len(nm_by_trial[t]) for t in trial_nums])

        # Only plot trials that have data
        valid = counts > 0
        if not np.any(valid):
            continue

        fig_bar, ax_bar = plt.subplots(figsize=(max(10, num_train_trials * 0.3), 5), dpi=150)
        ax_bar.bar(trial_nums[valid], means[valid], yerr=sems[valid],
                   color=color, alpha=0.7, edgecolor='black', linewidth=0.5,
                   capsize=2, error_kw={'linewidth': 0.8})
        ax_bar.set_xlabel('Trial Number')
        ax_bar.set_ylabel('Average Neuromodulator')
        ax_bar.set_title(f'{reward_label} - Average NM by Trial Number (n={num_networks} networks)')
        ax_bar.set_xticks(trial_nums)
        ax_bar.set_xticklabels([str(t) for t in trial_nums], fontsize=6, rotation=90)
        ax_bar.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)

        plt.tight_layout()
        tag = "pos_reward" if "Positive" in reward_label else "neg_reward"
        figures[f"ti_nm_by_trial/{tag}"] = fig_bar
        plt.close(fig_bar)

    return figures


def run_pca_frozen_pc1_analysis(model, saved_args, num_episodes, num_train_trials,
                                num_items, item_labels, device, batch_size=50):
    """Compute PC1 projections using the pca_frozen approach (manual forward pass, no bias).

    Ported directly from plots.py pca_frozen_analysis. Computes per-item embeddings
    via manual matrix multiplication (W @ h) without bias terms, fits PC1 on
    full layer outputs, and creates signed-projection bar charts.

    Returns:
        figures: dict of figure name -> matplotlib figure
    """
    figures = {}
    num_layers = saved_args.extra_layers + 2  # embedding + extra layers + final
    item_size = saved_args.item_size

    # Extract innate weights (weight only, no bias — matching pca_frozen)
    if hasattr(model, 'embedding_layer'):
        innate_weights = [model.embedding_layer.weight.detach().cpu().numpy()]
        layer_names = ['Embedding']
    else:
        innate_weights = []
        layer_names = []
    for i in range(saved_args.extra_layers):
        innate_weights.append(model.extra_hidden_layers[i].weight.detach().cpu().numpy())
        layer_names.append(f'Hidden {i+1}')
    innate_weights.append(model.fc2.weight.detach().cpu().numpy())
    layer_names.append('Final')

    alpha_extra_np = [model.alpha_extra[i].detach().cpu().numpy() for i in range(saved_args.extra_layers)]
    alpha_final_np = model.alpha.detach().cpu().numpy()

    # Storage for full layer outputs: position -> layer -> item -> list of vectors
    full_layer_outputs = {
        pos: {li: {ii: [] for ii in range(num_items)} for li in range(num_layers)}
        for pos in ['item1', 'item2']
    }

    # Process in batches
    num_batches = (num_episodes + batch_size - 1) // batch_size
    logger.info(f"=== Running pca_frozen PC1 analysis ({num_episodes} episodes) ===")

    for batch_idx in range(num_batches):
        current_batch = min(batch_size, num_episodes - batch_idx * batch_size)
        logger.info(f"pca_frozen batch {batch_idx+1}/{num_batches} ({current_batch} episodes)...")

        # Generate items and trials
        batch_items = generate_batch_items(num_items, item_size, current_batch,
                                           change_items_throughout_batch=True)
        trials, correct_choices, pair_indices, _ = generate_batch_trials_ti(
            batch_items, num_train_trials, 0, arbitrary=saved_args.arbitrary
        )
        trials_t = torch.tensor(trials, dtype=torch.float32).to(device)
        correct_choices_t = torch.tensor(correct_choices, dtype=torch.float32).to(device)

        # Initialize plastic weights
        pw = torch.zeros(current_batch, saved_args.hidden_size, saved_args.hidden_size,
                         dtype=torch.float32, requires_grad=False).to(device)
        epw = [torch.zeros(current_batch, saved_args.hidden_size, saved_args.hidden_size,
                           dtype=torch.float32, requires_grad=False).to(device)
               for _ in range(saved_args.extra_layers)]
        embed_pw_local = None
        if getattr(saved_args, 'plastic_embedding', False):
            embed_pw_local = torch.zeros(current_batch, saved_args.hidden_size, 2 * saved_args.item_size,
                                         dtype=torch.float32, requires_grad=False).to(device)

        # Run training trials to build plastic weights
        num_trials = trials_t.shape[1]
        with torch.no_grad():
            for trial_idx in range(num_trials):
                output = model(trials_t[:, trial_idx, :], pw, correct_choices_t[:, trial_idx],
                              extra_plastic_weights=epw, embed_plastic_weights=embed_pw_local)
                pw = output.plastic_weights
                epw = output.extra_plastic_weights
                embed_pw_local = output.embed_plastic_weights

        # Manual forward pass for each network (matching pca_frozen exactly)
        for net_idx in range(current_batch):
            single_items = batch_items[net_idx]
            single_pw = pw[net_idx].cpu().numpy()
            single_epw = [e[net_idx].cpu().numpy() for e in epw]

            for item_idx in range(num_items):
                item_emb = single_items[item_idx]
                zero_emb = np.zeros_like(item_emb)
                input_pos1 = np.concatenate([item_emb, zero_emb])
                input_pos2 = np.concatenate([zero_emb, item_emb])

                for position, input_vec in [('item1', input_pos1), ('item2', input_pos2)]:
                    # Embedding layer (no bias)
                    h = np.tanh(innate_weights[0] @ input_vec)
                    full_layer_outputs[position][0][item_idx].append(h.copy())

                    # Extra hidden layers (innate + plastic, no bias)
                    for layer_idx in range(1, num_layers - 1):
                        extra_layer_idx = layer_idx - 1
                        innate_contrib = innate_weights[layer_idx] @ h
                        plastic_contrib = (alpha_extra_np[extra_layer_idx] * single_epw[extra_layer_idx]) @ h
                        h = np.tanh(innate_contrib + plastic_contrib)
                        full_layer_outputs[position][layer_idx][item_idx].append(h.copy())

                    # Final layer (innate + plastic, no bias)
                    innate_contrib = innate_weights[-1] @ h
                    plastic_contrib = (alpha_final_np * single_pw) @ h
                    h = np.tanh(innate_contrib + plastic_contrib)
                    full_layer_outputs[position][num_layers - 1][item_idx].append(h.copy())

    # Compute PC1 from full layer outputs and create bar charts (matching pca_frozen exactly)
    for layer_idx in range(num_layers):
        layer_name = layer_names[layer_idx]

        # Collect all vectors for this layer
        all_vectors = []
        for position in ['item1', 'item2']:
            for item_idx in range(num_items):
                all_vectors.extend(full_layer_outputs[position][layer_idx][item_idx])

        if not all_vectors:
            continue

        all_vectors_matrix = np.array(all_vectors)
        mean_vec = np.mean(all_vectors_matrix, axis=0)
        centered = all_vectors_matrix - mean_vec
        cov_matrix = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        sort_idx = np.argsort(eigenvalues)[::-1]
        pc1 = eigenvectors[:, sort_idx[0]]
        pc1_norm = np.linalg.norm(pc1)

        evr = eigenvalues[sort_idx[0]] / eigenvalues.sum()
        logger.info(f"pca_frozen layer {layer_idx} ({layer_name}): "
                    f"top 3 eigenvalues = {eigenvalues[sort_idx[:3]]}, "
                    f"explained variance ratio = {evr:.4f}")

        # Compute signed projections per item per position
        width = 0.35
        x_pos = np.arange(num_items)

        mean_proj = {}
        se_proj = {}
        for position in ['item1', 'item2']:
            means = []
            ses = []
            for item_idx in range(num_items):
                vecs = full_layer_outputs[position][layer_idx][item_idx]
                projs = [np.dot(pc1, v) / pc1_norm if pc1_norm > 1e-10 else 0.0 for v in vecs]
                means.append(np.mean(projs) if projs else 0)
                ses.append(np.std(projs) / np.sqrt(len(projs)) if len(projs) > 1 else 0)
            mean_proj[position] = means
            se_proj[position] = ses

        layer_desc = "(innate only)" if layer_idx == 0 else "(innate + plastic)"
        fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
        ax.bar(x_pos - width/2, mean_proj['item1'], width, yerr=se_proj['item1'], capsize=3,
               label='Position 1', color='tab:blue', alpha=0.8)
        ax.bar(x_pos + width/2, mean_proj['item2'], width, yerr=se_proj['item2'], capsize=3,
               label='Position 2', color='tab:orange', alpha=0.8)
        ax.set_xlabel('Item')
        ax.set_ylabel('Signed Projection onto PC1')
        ax.set_title(f'TI: {layer_name} - Full Output Projection {layer_desc}\n'
                     f'(n={num_episodes} episodes, explained var = {evr:.1%})')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(item_labels[:num_items])
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.legend()
        plt.tight_layout()
        figures[f"pca_frozen/layer{layer_idx}_full_output_pc1_projection"] = fig
        plt.close(fig)

    return figures


def main():
    parser = argparse.ArgumentParser(description="Plot TI readout correlation evolution from a checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Custom wandb run name (default: auto from checkpoint)")
    parser.add_argument("--num_networks", type=int, default=10, help="Number of networks to plot")
    parser.add_argument("--num_train_trials", type=int, default=50, help="Number of training trials per episode")
    parser.add_argument("--num_aggregate_episodes", type=int, default=1000, help="Number of episodes for aggregate delta analysis")
    parser.add_argument("--neg_nm_override", type=float, default=-2.0, help="Constant NM value for negative reward ablations (default: -2.0)")
    parser.add_argument("--skip_plots", type=str, nargs='*', default=[],
        help="Plot group prefixes to skip. E.g.: --skip_plots ti_agg_delta_mean ti_agg_delta_sem")
    parser.add_argument("--single_neuron_top_k", type=int, default=0,
        help="Number of top neurons (by readout weight magnitude) to test individually. 0 = only >2σ neurons.")
    parser.add_argument("--ll_num_trials_list_1", type=int, default=7,
        help="Number of list 1 trials for list-linking zero-shot eval")
    parser.add_argument("--ll_num_trials_list_2", type=int, default=7,
        help="Number of list 2 trials for list-linking zero-shot eval")
    parser.add_argument("--ll_num_trials_linking_pair", type=int, default=1,
        help="Number of linking pair trials for list-linking zero-shot eval")
    cli_args = parser.parse_args()

    skip_set = set(cli_args.skip_plots or [])

    def should_skip(prefix):
        return prefix in skip_set

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load checkpoint
    checkpoint = torch.load(cli_args.checkpoint, map_location=device)
    saved_args = Namespace(**checkpoint['args'])
    episode = checkpoint['episode']
    logger.info(f"Loaded checkpoint from episode {episode}")

    # Migrate old scalar_alpha_layers convention: 0 meant final layer, now it's extra_layers+1
    sal = getattr(saved_args, 'scalar_alpha_layers', None)
    if sal and 0 in sal:
        saved_args.scalar_alpha_layers = [saved_args.extra_layers + 1 if x == 0 else x for x in sal]
        logger.info(f"Migrated scalar_alpha_layers from old convention: {sal} -> {saved_args.scalar_alpha_layers}")

    # Reconstruct model
    input_size = 2 * saved_args.item_size
    model = MLP(
        input_size=input_size,
        hidden_size=saved_args.hidden_size,
        batch_size=saved_args.batch_size,
        plastic_weight_clip=saved_args.plastic_weight_clip,
        delay_steps=saved_args.delay_steps,
        use_extra_neuromodulator=saved_args.use_extra_neuromodulator,
        extra_layers=saved_args.extra_layers,
        use_answer=saved_args.use_answer,
        use_sampled_choice_in_reward=saved_args.use_sampled_choice_in_reward,
        add_additional_hidden_layer_pre_plastic=saved_args.add_additional_hidden_layer_pre_plastic,
        add_additional_hidden_layer_post_plastic=saved_args.add_additional_hidden_layer_post_plastic,
        scalar_alpha_layers=getattr(saved_args, 'scalar_alpha_layers', None),
        simple_neuromodulator=getattr(saved_args, 'simple_neuromodulator', False),
        simple_neuromodulator_bias=getattr(saved_args, 'simple_neuromodulator_bias', False),
        freeze_neuromodulator_multiplier=getattr(saved_args, 'freeze_neuromodulator_multiplier', False),
        freeze_hebbian_trace_multiplier=getattr(saved_args, 'freeze_hebbian_trace_multiplier', False),
        direct_readout=getattr(saved_args, 'direct_readout', False),
        use_sigmoid=getattr(saved_args, 'use_sigmoid', False),
        use_capped_relu=getattr(saved_args, 'use_capped_relu', False),
        single_nm_unit=getattr(saved_args, 'single_nm_unit', False),
        linear_hebbian=getattr(saved_args, 'linear_hebbian', False),
        no_alpha=getattr(saved_args, 'no_alpha', False),
        no_embedding=getattr(saved_args, 'no_embedding', False),
        linear_activation=getattr(saved_args, 'linear_activation', False),
        ones_readout=getattr(saved_args, 'ones_readout', False),
        antisymmetric_readout=getattr(saved_args, 'antisymmetric_readout', False),
        antisymmetric_input_init=getattr(saved_args, 'antisymmetric_input_init', False),
        strong_antisymmetric_input_init=getattr(saved_args, 'strong_antisymmetric_input_init', False),
        no_bias_layers=getattr(saved_args, 'no_bias_layers', None),
        multi_neuromodulator=getattr(saved_args, 'multi_neuromodulator', 1),
        multi_neuromodulator_shared_trace=getattr(saved_args, 'multi_neuromodulator_shared_trace', True),
        simple_neuromodulator_init_weight=getattr(saved_args, 'simple_neuromodulator_init_weight', 1.0),
        simple_neuromodulator_init_bias=getattr(saved_args, 'simple_neuromodulator_init_bias', 0.0),
        direct_nm=getattr(saved_args, 'direct_nm', False),
        direct_nm_pos_init=getattr(saved_args, 'direct_nm_pos_init', 0.0),
        direct_nm_neg_init=getattr(saved_args, 'direct_nm_neg_init', -1.0),
        plastic_embedding=getattr(saved_args, 'plastic_embedding', False),
        disable_final_plastic=getattr(saved_args, 'disable_final_plastic', False),
        no_readout_bias=getattr(saved_args, 'no_readout_bias', False),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.greedy_sampling = saved_args.greedy_sampling
    model.eval()
    logger.info("Model loaded successfully")

    # Helper: readout weight accessor (model.choice.weight for standard, model.fc2.weight for direct_readout)
    _is_direct_readout = getattr(saved_args, 'direct_readout', False)
    def get_readout_weight():
        """Return the readout weight parameter (the one mapping hidden→scalar logit)."""
        return model.fc2.weight if _is_direct_readout else model.choice.weight

    num_networks = cli_args.num_networks
    num_items = saved_args.item_range[-1] - 1
    num_train_trials = cli_args.num_train_trials
    item_labels = [chr(ord('A') + i) for i in range(num_items)]
    final_layer_idx = saved_args.extra_layers + 1
    readout_weights = get_readout_weight().detach().cpu().numpy().squeeze()

    all_figures = {}

    # === Alpha weight scatter plots ===
    if not should_skip('alpha_weights_item1_vs_item2'):
        alpha_figs = create_alpha_weight_scatter_plots(model, saved_args)
        all_figures.update(alpha_figs)
        logger.info(f"Created {len(alpha_figs)} alpha weight scatter plots")
    else:
        logger.info("Skipping alpha weight scatter plots")

    # === Innate weight scatter plots ===
    if not should_skip('innate_weights_item1_vs_item2'):
        innate_figs = create_innate_weight_scatter_plots(model, saved_args)
        all_figures.update(innate_figs)
        logger.info(f"Created {len(innate_figs)} innate weight scatter plots")
    else:
        logger.info("Skipping innate weight scatter plots")

    # === Weight heatmaps ===
    if not should_skip('weight_heatmaps'):
        heatmap_figs = create_weight_heatmaps(model, saved_args)
        all_figures.update(heatmap_figs)
        logger.info(f"Created {len(heatmap_figs)} weight heatmaps")
    else:
        logger.info("Skipping weight heatmaps")

    # === Random items ===
    random_prefixes = {'ti_adj_logit_evo', 'ti_adj_logit_delta'}
    if not random_prefixes.issubset(skip_set):
        logger.info("=== Running with random items ===")
        batch_items_random = generate_batch_items(num_items, saved_args.item_size, num_networks, change_items_throughout_batch=True)
        all_corrs, trial_labels, nm_values, logit_values, reward_values, adj_pair_logits, item_logits, actual_num_train = run_simulation(
            model, batch_items_random, saved_args, num_train_trials, num_networks,
            num_items, item_labels, final_layer_idx, readout_weights, device
        )
        random_figs = create_figures(
            all_corrs, trial_labels, nm_values, logit_values, reward_values,
            adj_pair_logits, item_logits, batch_items_random, model, num_networks, num_items,
            actual_num_train, item_labels, readout_weights, saved_args, device
        )
        if skip_set:
            random_figs = {k: v for k, v in random_figs.items()
                           if k.split('/')[0] not in skip_set}
        all_figures.update(random_figs)
    else:
        logger.info("Skipping random items simulation (all prefixes skipped)")

    # === pca_frozen PC1 analysis (use original function from plots.py directly) ===
    # Run BEFORE aggregate analysis so we can extract and reuse the PC1 vector
    pca_frozen_pc1 = None
    if not should_skip('pca_frozen'):
        from copy import copy
        pca_args = copy(saved_args)
        # Override num_items to match what we use elsewhere (item_range derived)
        pca_args.num_items = num_items
        # Function uses num_networks = max(128, 4*batch_size)
        pca_args.batch_size = max(cli_args.num_aggregate_episodes, 128) // 4
        # Function uses args.num_train_trials // (num_items - 1), so multiply to undo division
        pca_args.num_train_trials = num_train_trials * (num_items - 1)
        pca_frozen_figs, pca_frozen_pc1s = plot_innate_weight_analysis(pca_args, model, task='ti')
        all_figures.update(pca_frozen_figs)
        logger.info(f"Created {len(pca_frozen_figs)} pca_frozen figures")
        # Extract layer 1 PC1 for use in aggregate analysis
        pca_frozen_pc1 = pca_frozen_pc1s.get(1)
        if pca_frozen_pc1 is not None:
            logger.info("Extracted layer 1 PC1 from pca_frozen for aggregate PC1 plots")
    else:
        logger.info("Skipping pca_frozen analysis")

    # === Aggregate delta analysis over many episodes ===
    # All sections that consume data from run_aggregate_delta_analysis:
    _agg_delta_sections = {
        'ti_agg_delta_mean', 'ti_agg_delta_sem', 'ti_agg_nm_by_trial',
        'ti_agg_delta_by_type_mean', 'ti_agg_delta_by_type_sem',
        'ti_agg_dot', 'ti_agg_pair_dot', 'ti_agg_pair_pair',
        'ti_agg_pair_logit', 'ti_agg_pair_pc1',
        'ti_agg_post_train_dot_mean', 'ti_agg_post_train_dot_sem',
        'ti_agg_post_train_corr_mean', 'ti_agg_post_train_corr_sem',
        'ti_agg_post_train_nonadj_corr_mean',
        'ti_agg_post_train_cross_dot_mean', 'ti_agg_post_train_cross_dot_sem',
        'ti_agg_post_train_cross_corr_mean', 'ti_agg_post_train_cross_corr_sem',
        'ti_agg_post_train_pairwise', 'ti_agg_post_train_logits',
    }
    _skip_agg_delta = _agg_delta_sections.issubset(skip_set)

    if not _skip_agg_delta:
        logger.info(f"=== Running aggregate delta analysis ({cli_args.num_aggregate_episodes} episodes) ===")
        (delta_pos1, delta_pos2, agg_nm_storage, agg_pair_labels, agg_num_trials,
         dot_delta_pos1, dot_delta_pos2, dot_abs_pos1, dot_abs_pos2,
         post_dot_pos1, post_dot_pos2, post_corr_pos1, post_corr_pos2,
         post_pair_dot, post_pair_corr, post_logits,
         nonadj_pair_labels, post_nonadj_logits, post_cross_dot, post_cross_corr,
         post_dot_nonadj_pos1, post_dot_nonadj_pos2, post_corr_nonadj_pos1, post_corr_nonadj_pos2,
         pair_dot_delta, pair_dot_abs,
         post_item_logits_pos1, post_item_logits_pos2,
         pair_logit_delta, pair_logit_abs,
         pair_pair_dot_delta, pair_pair_dot_abs,
         pair_pc1_delta, pair_pc1_abs, post_pc1_logits,
         post_pc1_nonadj_logits, post_pc1_item_logits_pos1, post_pc1_item_logits_pos2) = run_aggregate_delta_analysis(
            model, saved_args, cli_args.num_aggregate_episodes, num_train_trials,
            num_items, item_labels, device, external_pc1=pca_frozen_pc1
        )
        agg_figs = create_aggregate_figures(
            delta_pos1, delta_pos2, agg_nm_storage, agg_pair_labels,
            num_items, agg_num_trials, item_labels, cli_args.num_aggregate_episodes,
            skip_prefixes=skip_set
        )
        all_figures.update(agg_figs)
        logger.info(f"Created {len(agg_figs)} aggregate figures")

        # Per-trial-type heatmaps (x-axis = trial number) for readout logit deltas
        by_type_figs = create_per_trial_type_figures(
            delta_pos1, delta_pos2, agg_nm_storage, agg_pair_labels,
            num_items, agg_num_trials, item_labels, cli_args.num_aggregate_episodes,
            skip_prefixes=skip_set
        )
        all_figures.update(by_type_figs)
        logger.info(f"Created {len(by_type_figs)} per-trial-type delta logit figures")
    else:
        logger.info("Skipping aggregate delta analysis (all downstream sections skipped)")

    if saved_args.extra_layers > 0 and not should_skip('ti_agg_dot'):
        dot_figs = create_aggregate_dot_product_figures(
            dot_delta_pos1, dot_delta_pos2, dot_abs_pos1, dot_abs_pos2,
            agg_nm_storage, agg_pair_labels,
            num_items, agg_num_trials, item_labels, cli_args.num_aggregate_episodes,
            saved_args.extra_layers
        )
        all_figures.update(dot_figs)
        logger.info(f"Created {len(dot_figs)} dot product delta figures")

        # Per-trial-type heatmaps (x-axis = trial number) for dot product deltas
        dot_by_type_figs = create_per_trial_type_dot_figures(
            dot_delta_pos1, dot_delta_pos2, agg_nm_storage, agg_pair_labels,
            num_items, agg_num_trials, item_labels, cli_args.num_aggregate_episodes,
            saved_args.extra_layers
        )
        all_figures.update(dot_by_type_figs)
        logger.info(f"Created {len(dot_by_type_figs)} per-trial-type dot product delta figures")

    # Pair-vs-pair dot product heatmaps (y=trial types, x=presented trial type)
    if not should_skip('ti_agg_pair_dot'):
        pair_dot_figs = create_aggregate_pair_dot_figures(
            pair_dot_delta, pair_dot_abs, agg_nm_storage,
            agg_pair_labels, agg_num_trials,
            cli_args.num_aggregate_episodes, saved_args.extra_layers + 1
        )
        all_figures.update(pair_dot_figs)
        logger.info(f"Created {len(pair_dot_figs)} pair-vs-pair dot product figures")

    # Pair-pair dot product heatmaps filtered to boundary pairs (e.g. DE/ED)
    if not should_skip('ti_agg_pair_pair'):
        mid = num_items // 2
        focus_pair_labels = [
            item_labels[mid - 1] + item_labels[mid],
            item_labels[mid] + item_labels[mid - 1],
        ]
        pair_pair_dot_figs = create_aggregate_pair_pair_dot_figures(
            pair_pair_dot_delta, pair_pair_dot_abs, agg_nm_storage,
            agg_pair_labels, agg_num_trials,
            cli_args.num_aggregate_episodes, saved_args.extra_layers + 1,
            focus_pair_labels
        )
        all_figures.update(pair_pair_dot_figs)
        logger.info(f"Created {len(pair_pair_dot_figs)} pair-pair dot product figures (focus: {focus_pair_labels})")

        # Pair-pair dot product evolution across trials (pooled over rewards and presented pairs)
        pair_pair_evo_figs = create_pair_pair_dot_evolution_figures(
            pair_pair_dot_abs, agg_pair_labels, agg_num_trials,
            cli_args.num_aggregate_episodes, saved_args.extra_layers + 1
        )
        all_figures.update(pair_pair_evo_figs)
        logger.info(f"Created {len(pair_pair_evo_figs)} pair-pair dot product evolution figures")

    # Pair logit heatmaps (y=trial types, x=presented trial type, split by reward)
    if not should_skip('ti_agg_pair_logit'):
        pair_logit_figs = create_aggregate_pair_logit_figures(
            pair_logit_delta, pair_logit_abs, agg_nm_storage,
            agg_pair_labels, agg_num_trials,
            cli_args.num_aggregate_episodes
        )
        all_figures.update(pair_logit_figs)
        logger.info(f"Created {len(pair_logit_figs)} pair logit figures")

    # Layer 1 PC1 projection heatmaps (analogous to pair logit plots)
    if not should_skip('ti_agg_pair_pc1') and pair_pc1_delta is not None:
        pc1_figs = create_aggregate_pair_logit_figures(
            pair_pc1_delta, pair_pc1_abs, agg_nm_storage,
            agg_pair_labels, agg_num_trials,
            cli_args.num_aggregate_episodes,
            abs_label="Layer 1 PC1 Projection", abs_key="ti_agg_pair_pc1_abs",
            delta_label="Delta Layer 1 PC1 Projection", delta_key="ti_agg_pair_pc1_delta"
        )
        all_figures.update(pc1_figs)
        logger.info(f"Created {len(pc1_figs)} layer 1 PC1 projection figures")

    # Post-training dot product and correlation (extra layers + final layer)
    post_dot_corr_prefixes = {'ti_agg_post_train_dot_mean', 'ti_agg_post_train_dot_sem',
                               'ti_agg_post_train_corr_mean', 'ti_agg_post_train_corr_sem'}
    if not post_dot_corr_prefixes.issubset(skip_set):
        post_figs = create_post_train_dot_corr_figures(
            post_dot_pos1, post_dot_pos2, post_corr_pos1, post_corr_pos2,
            agg_pair_labels, num_items, item_labels, cli_args.num_aggregate_episodes,
            saved_args.extra_layers
        )
        if skip_set:
            post_figs = {k: v for k, v in post_figs.items()
                         if k.split('/')[0] not in skip_set}
        all_figures.update(post_figs)
        logger.info(f"Created {len(post_figs)} post-training dot/corr figures")
    else:
        logger.info("Skipping post-training dot/corr figures")

    # Post-training dot product and correlation: items x nonadjacent pairs
    if not should_skip('ti_agg_post_train_nonadj_corr_mean'):
        post_nonadj_figs = create_post_train_dot_corr_nonadj_figures(
            post_dot_nonadj_pos1, post_dot_nonadj_pos2, post_corr_nonadj_pos1, post_corr_nonadj_pos2,
            nonadj_pair_labels, num_items, item_labels, cli_args.num_aggregate_episodes,
            saved_args.extra_layers
        )
        all_figures.update(post_nonadj_figs)
        logger.info(f"Created {len(post_nonadj_figs)} post-training item x nonadj dot/corr figures")
    else:
        logger.info("Skipping post-training nonadj dot/corr figures")

    # Post-training pairwise trial-type dot product and correlation (14x14 matrices)
    if not should_skip('ti_agg_post_train_pairwise'):
        pairwise_figs = create_post_train_pairwise_figures(
            post_pair_dot, post_pair_corr,
            agg_pair_labels, cli_args.num_aggregate_episodes,
            saved_args.extra_layers
        )
        all_figures.update(pairwise_figs)
        logger.info(f"Created {len(pairwise_figs)} post-training pairwise figures")
    else:
        logger.info("Skipping post-training pairwise figures")

    # Post-training logit bar chart
    if not should_skip('ti_agg_post_train_logits'):
        logit_bar_figs = create_post_train_logit_bar_chart(
            post_logits, agg_pair_labels, cli_args.num_aggregate_episodes
        )
        all_figures.update(logit_bar_figs)
        logger.info(f"Created {len(logit_bar_figs)} post-training logit bar chart(s)")

        # Post-training PC1 logit bar chart (layer 1 embeddings projected onto PC1)
        if post_pc1_logits is not None:
            pc1_bar_figs = create_post_train_logit_bar_chart(
                post_pc1_logits, agg_pair_labels, cli_args.num_aggregate_episodes,
                title=f'Post-Training Layer 1 PC1 Projection by Trial Type\nn={cli_args.num_aggregate_episodes} episodes',
                fig_key="ti_agg_post_train_logits/pc1_bar_chart",
                ylabel="PC1 Projection"
            )
            all_figures.update(pc1_bar_figs)
            logger.info(f"Created {len(pc1_bar_figs)} post-training PC1 bar chart(s)")

        # Post-training PC1 nonadjacent bar chart
        if post_pc1_nonadj_logits is not None:
            pc1_nonadj_bar_figs = create_post_train_nonadj_logit_bar_chart(
                post_pc1_nonadj_logits, nonadj_pair_labels, cli_args.num_aggregate_episodes,
                title=f'Post-Training Layer 1 PC1 Projection by Nonadjacent Trial Type\nn={cli_args.num_aggregate_episodes} episodes',
                fig_key="ti_agg_post_train_logits/pc1_nonadj_bar_chart",
                ylabel="PC1 Projection"
            )
            all_figures.update(pc1_nonadj_bar_figs)
            logger.info(f"Created {len(pc1_nonadj_bar_figs)} post-training PC1 nonadj bar chart(s)")

        # Post-training PC1 individual item bar charts (split by position)
        if post_pc1_item_logits_pos1 is not None:
            pc1_item_figs = create_post_train_item_logit_bar_chart(
                post_pc1_item_logits_pos1, post_pc1_item_logits_pos2,
                item_labels, cli_args.num_aggregate_episodes,
                title_template='Post-Training Layer 1 PC1 Projection by Item ({pos_title})\nn={num_episodes} episodes',
                fig_key_prefix="ti_agg_post_train_logits/pc1_item",
                ylabel="PC1 Projection"
            )
            all_figures.update(pc1_item_figs)
            logger.info(f"Created {len(pc1_item_figs)} post-training PC1 item bar chart(s)")

        # Post-training nonadjacent logit bar chart
        nonadj_bar_figs = create_post_train_nonadj_logit_bar_chart(
            post_nonadj_logits, nonadj_pair_labels, cli_args.num_aggregate_episodes
        )
        all_figures.update(nonadj_bar_figs)
        logger.info(f"Created {len(nonadj_bar_figs)} post-training nonadjacent logit bar chart(s)")

        # Post-training individual item logit bar charts (split by position)
        item_logit_figs = create_post_train_item_logit_bar_chart(
            post_item_logits_pos1, post_item_logits_pos2,
            item_labels, cli_args.num_aggregate_episodes
        )
        all_figures.update(item_logit_figs)
        logger.info(f"Created {len(item_logit_figs)} post-training item logit bar chart(s)")

        # Item dot-weighted logit sum bar charts (split by position, final layer)
        item_dot_wt_figs = create_item_dot_weighted_logit_bar_chart(
            post_dot_pos1, post_dot_pos2, post_logits,
            item_labels, cli_args.num_aggregate_episodes, saved_args.extra_layers
        )
        all_figures.update(item_dot_wt_figs)
        logger.info(f"Created {len(item_dot_wt_figs)} item dot-weighted logit bar chart(s)")
    else:
        logger.info("Skipping post-training logit bar charts")

    # Post-training cross dot product and correlation heatmaps (adj x nonadj)
    cross_prefixes = {'ti_agg_post_train_cross_dot_mean', 'ti_agg_post_train_cross_dot_sem',
                       'ti_agg_post_train_cross_corr_mean', 'ti_agg_post_train_cross_corr_sem'}
    if not cross_prefixes.issubset(skip_set):
        cross_figs = create_post_train_cross_figures(
            post_cross_dot, post_cross_corr,
            agg_pair_labels, nonadj_pair_labels,
            cli_args.num_aggregate_episodes, saved_args.extra_layers
        )
        if skip_set:
            cross_figs = {k: v for k, v in cross_figs.items()
                          if k.split('/')[0] not in skip_set}
        all_figures.update(cross_figs)
        logger.info(f"Created {len(cross_figs)} post-training cross dot/corr figures")
    else:
        logger.info("Skipping post-training cross dot/corr figures")

    # Dot-weighted logit sum bar chart (final layer)
    if not should_skip('ti_agg_post_train_logits'):
        dot_wt_figs = create_dot_weighted_logit_bar_chart(
            post_pair_dot, post_logits, agg_pair_labels,
            cli_args.num_aggregate_episodes, saved_args.extra_layers
        )
        all_figures.update(dot_wt_figs)
        logger.info(f"Created {len(dot_wt_figs)} dot-weighted logit bar chart(s)")

        # Cross-dot-weighted logit sum bar chart for nonadjacent pairs (final layer)
        cross_dot_wt_figs = create_cross_dot_weighted_logit_bar_chart(
            post_cross_dot, post_logits, nonadj_pair_labels,
            cli_args.num_aggregate_episodes, saved_args.extra_layers
        )
        all_figures.update(cross_dot_wt_figs)
        logger.info(f"Created {len(cross_dot_wt_figs)} cross-dot-weighted nonadj logit bar chart(s)")

    # PC1 weighted sum analogs (use layer 1 dot products with PC1 projections)
    if not should_skip('ti_agg_post_train_logits') and post_pc1_logits is not None and saved_args.extra_layers > 0:
        n_ep = cli_args.num_aggregate_episodes

        # Item dot-weighted PC1 sum (layer 1 dot products, PC1 projections)
        pc1_item_dot_wt_figs = create_item_dot_weighted_logit_bar_chart(
            post_dot_pos1, post_dot_pos2, post_pc1_logits,
            item_labels, n_ep, saved_args.extra_layers,
            layer_li=0,
            title_template='Item Dot-Weighted PC1 Sum ({pos_title}, Layer 1)\nn={num_episodes} episodes',
            fig_key_prefix="ti_agg_post_train_logits/pc1_item_dot_weighted",
            ylabel="Dot-Weighted PC1 Sum"
        )
        all_figures.update(pc1_item_dot_wt_figs)
        logger.info(f"Created {len(pc1_item_dot_wt_figs)} PC1 item dot-weighted bar chart(s)")

        # Pair dot-weighted PC1 sum (layer 1 pairwise dot products, PC1 projections)
        pc1_dot_wt_figs = create_dot_weighted_logit_bar_chart(
            post_pair_dot, post_pc1_logits, agg_pair_labels,
            n_ep, saved_args.extra_layers,
            layer_li=0,
            title=f'Dot-Weighted PC1 Sum (Layer 1)\nn={n_ep} episodes',
            fig_key="ti_agg_post_train_logits/pc1_dot_weighted_bar_chart",
            ylabel="Dot-Weighted PC1 Sum"
        )
        all_figures.update(pc1_dot_wt_figs)
        logger.info(f"Created {len(pc1_dot_wt_figs)} PC1 dot-weighted bar chart(s)")

        # Cross-dot-weighted PC1 sum for nonadjacent pairs (layer 1 cross dots, PC1 projections)
        pc1_cross_dot_wt_figs = create_cross_dot_weighted_logit_bar_chart(
            post_cross_dot, post_pc1_logits, nonadj_pair_labels,
            n_ep, saved_args.extra_layers,
            layer_li=0,
            title=f'Cross-Dot-Weighted PC1 Sum for Nonadjacent Pairs (Layer 1)\nn={n_ep} episodes',
            fig_key="ti_agg_post_train_logits/pc1_cross_dot_weighted_nonadj_bar_chart",
            ylabel="Cross-Dot-Weighted PC1 Sum"
        )
        all_figures.update(pc1_cross_dot_wt_figs)
        logger.info(f"Created {len(pc1_cross_dot_wt_figs)} PC1 cross-dot-weighted nonadj bar chart(s)")

    # === Symbolic distance plot: regular only ===
    regular_zs = None
    if not should_skip('ti_agg_symbolic_distance_regular'):
        n_ep = cli_args.num_aggregate_episodes
        logger.info(f"=== Running regular zero-shot symbolic distance ({n_ep} episodes) ===")
        regular_zs = run_aggregate_zero_shot(
            model, saved_args, n_ep, num_train_trials, num_items, device
        )
        all_figures["ti_agg_symbolic_distance/regular"] = zero_shot_symbolic_distance_plot(
            regular_zs, num_items, title=f'Zero-Shot Symbolic Distance (n={n_ep})'
        )

    # === Symbolic distance plots: ablations ===
    if not should_skip('ti_agg_symbolic'):
        n_ep = cli_args.num_aggregate_episodes
        # Layer indices for dot products: all hidden layers + final layer
        # With extra_layers=1: [1, 2], with extra_layers=0: [1]
        dot_all_layers = list(range(1, saved_args.extra_layers + 2))

        # Run regular if not already done
        if regular_zs is None:
            logger.info(f"=== Running regular zero-shot symbolic distance ({n_ep} episodes) ===")
            regular_zs = run_aggregate_zero_shot(
                model, saved_args, n_ep, num_train_trials, num_items, device
            )
            all_figures["ti_agg_symbolic_distance/regular"] = zero_shot_symbolic_distance_plot(
                regular_zs, num_items, title=f'Zero-Shot Symbolic Distance (n={n_ep})'
            )

        logger.info(f"=== Ablation 1: NM=0 on positive reward ({n_ep} episodes) ===")
        abl1_zs = run_aggregate_zero_shot(
            model, saved_args, n_ep, num_train_trials, num_items, device,
            nm_override_positive=0.0
        )
        all_figures["ti_agg_symbolic_distance/abl1_nm0_pos"] = zero_shot_symbolic_distance_plot(
            abl1_zs, num_items, title=f'Zero-Shot SD - NM=0 on +Reward (n={n_ep})'
        )

        neg_nm = cli_args.neg_nm_override
        logger.info(f"=== Ablation 2: NM={neg_nm} on negative reward ({n_ep} episodes) ===")
        abl2_zs = run_aggregate_zero_shot(
            model, saved_args, n_ep, num_train_trials, num_items, device,
            nm_override_negative=neg_nm
        )
        all_figures["ti_agg_symbolic_distance/abl2_nm_neg"] = zero_shot_symbolic_distance_plot(
            abl2_zs, num_items, title=f'Zero-Shot SD - NM={neg_nm} on -Reward (n={n_ep})'
        )

        logger.info(f"=== Ablation 3: NM=0 on positive, NM={neg_nm} on negative ({n_ep} episodes) ===")
        abl3_zs = run_aggregate_zero_shot(
            model, saved_args, n_ep, num_train_trials, num_items, device,
            nm_override_positive=0.0, nm_override_negative=neg_nm
        )
        all_figures["ti_agg_symbolic_distance/abl3_nm0_pos_nm_neg"] = zero_shot_symbolic_distance_plot(
            abl3_zs, num_items, title=f'Zero-Shot SD - NM=0/+Reward, NM={neg_nm}/-Reward (n={n_ep})'
        )

        # Delta symbolic distance plots: all pairwise comparisons
        all_figures["ti_agg_symbolic_distance/delta_regular_vs_abl1"] = delta_symbolic_distance_plot(
            regular_zs, abl1_zs, num_items,
            title=f'Delta Accuracy: Regular - NM=0 on +Reward (n={n_ep})'
        )
        all_figures["ti_agg_symbolic_distance/delta_regular_vs_abl2"] = delta_symbolic_distance_plot(
            regular_zs, abl2_zs, num_items,
            title=f'Delta Accuracy: Regular - NM={neg_nm} on -Reward (n={n_ep})'
        )
        all_figures["ti_agg_symbolic_distance/delta_regular_vs_abl3"] = delta_symbolic_distance_plot(
            regular_zs, abl3_zs, num_items,
            title=f'Delta Accuracy: Regular - NM=0/+, NM={neg_nm}/- (n={n_ep})'
        )
        all_figures["ti_agg_symbolic_distance/delta_abl1_vs_abl2"] = delta_symbolic_distance_plot(
            abl1_zs, abl2_zs, num_items,
            title=f'Delta Accuracy: NM=0 on +Reward - NM={neg_nm} on -Reward (n={n_ep})'
        )
        all_figures["ti_agg_symbolic_distance/delta_abl3_vs_abl1"] = delta_symbolic_distance_plot(
            abl3_zs, abl1_zs, num_items,
            title=f'Delta Accuracy: NM=0/+,NM={neg_nm}/- vs NM=0 on +Reward (n={n_ep})'
        )
        all_figures["ti_agg_symbolic_distance/delta_abl3_vs_abl2"] = delta_symbolic_distance_plot(
            abl3_zs, abl2_zs, num_items,
            title=f'Delta Accuracy: NM=0/+,NM={neg_nm}/- vs NM={neg_nm} on -Reward (n={n_ep})'
        )

        if saved_args.extra_layers > 0:
            logger.info(f"=== Ablation 4: Freeze extra layer plastic weights ({n_ep} episodes) ===")
            abl4_zs = run_aggregate_zero_shot(
                model, saved_args, n_ep, num_train_trials, num_items, device,
                freeze_extra_plastic_weights=True
            )
            all_figures["ti_agg_symbolic_distance/abl4_freeze_extra_pw"] = zero_shot_symbolic_distance_plot(
                abl4_zs, num_items, title=f'Zero-Shot SD - Extra Layer PW Frozen (n={n_ep})'
            )
            all_figures["ti_agg_symbolic_distance/delta_regular_vs_abl4"] = delta_symbolic_distance_plot(
                regular_zs, abl4_zs, num_items,
                title=f'Delta Accuracy: Regular - Extra Layer PW Frozen (n={n_ep})'
            )

            logger.info(f"=== Ablation 5: Freeze final layer plastic weights ({n_ep} episodes) ===")
            abl5_zs = run_aggregate_zero_shot(
                model, saved_args, n_ep, num_train_trials, num_items, device,
                freeze_final_plastic_weights=True
            )
            all_figures["ti_agg_symbolic_distance/abl5_freeze_final_pw"] = zero_shot_symbolic_distance_plot(
                abl5_zs, num_items, title=f'Zero-Shot SD - Final Layer PW Frozen (n={n_ep})'
            )
            all_figures["ti_agg_symbolic_distance/delta_regular_vs_abl5"] = delta_symbolic_distance_plot(
                regular_zs, abl5_zs, num_items,
                title=f'Delta Accuracy: Regular - Final Layer PW Frozen (n={n_ep})'
            )

            logger.info(f"=== Ablation 6: Freeze all plastic weights ({n_ep} episodes) ===")
            abl6_zs = run_aggregate_zero_shot(
                model, saved_args, n_ep, num_train_trials, num_items, device,
                freeze_extra_plastic_weights=True, freeze_final_plastic_weights=True
            )
            all_figures["ti_agg_symbolic_distance/abl6_innate_only"] = zero_shot_symbolic_distance_plot(
                abl6_zs, num_items, title=f'Zero-Shot SD - Innate Only / All PW Frozen (n={n_ep})'
            )
            all_figures["ti_agg_symbolic_distance/delta_regular_vs_abl6"] = delta_symbolic_distance_plot(
                regular_zs, abl6_zs, num_items,
                title=f'Delta Accuracy: Regular - Innate Only (n={n_ep})'
            )
            all_figures["ti_agg_symbolic_distance/delta_abl4_vs_abl5"] = delta_symbolic_distance_plot(
                abl4_zs, abl5_zs, num_items,
                title=f'Delta Accuracy: Extra PW Frozen - Final PW Frozen (n={n_ep})'
            )

        # Ablation 7: Zero readout weights within 2 sigma of the mean
        logger.info(f"=== Ablation 7: Zero readout weights within 2σ ({n_ep} episodes) ===")
        with torch.no_grad():
            readout_w = get_readout_weight().data  # (1, hidden_size)
            readout_mean = readout_w.mean()
            readout_std = readout_w.std()
            readout_saved = readout_w.clone()
            mask_2sigma = (readout_w - readout_mean).abs() <= 2 * readout_std
            num_zeroed = mask_2sigma.sum().item()
            num_total = readout_w.numel()
            readout_w[mask_2sigma] = 0.0
            logger.info(f"  Zeroed {num_zeroed}/{num_total} readout weights (within 2σ of mean={readout_mean:.4f}, std={readout_std:.4f})")

        abl7_zs = run_aggregate_zero_shot(
            model, saved_args, n_ep, num_train_trials, num_items, device
        )

        with torch.no_grad():
            get_readout_weight().data.copy_(readout_saved)

        all_figures["ti_agg_symbolic_distance/abl7_readout_2sigma_zeroed"] = zero_shot_symbolic_distance_plot(
            abl7_zs, num_items, title=f'Zero-Shot SD - Readout ≤2σ Zeroed ({num_zeroed}/{num_total}, n={n_ep})'
        )
        all_figures["ti_agg_symbolic_distance/delta_regular_vs_abl7"] = delta_symbolic_distance_plot(
            regular_zs, abl7_zs, num_items,
            title=f'Delta: Regular - Readout ≤2σ Zeroed (n={n_ep})'
        )

        # Ablation 8: Zero readout + alpha weights within 2 sigma of the mean
        logger.info(f"=== Ablation 8: Zero readout + alpha weights within 2σ ({n_ep} episodes) ===")
        with torch.no_grad():
            # Readout
            readout_w = get_readout_weight().data
            readout_saved8 = readout_w.clone()
            r_mean, r_std = readout_w.mean(), readout_w.std()
            r_mask = (readout_w - r_mean).abs() <= 2 * r_std
            readout_w[r_mask] = 0.0
            r_zeroed = r_mask.sum().item()
            logger.info(f"  Readout: zeroed {r_zeroed}/{readout_w.numel()} (mean={r_mean:.4f}, std={r_std:.4f})")

            # Alpha (final)
            alpha_saved8 = model.alpha.data.clone()
            if model.alpha.ndim >= 2:
                a_mean, a_std = model.alpha.data.mean(), model.alpha.data.std()
                a_mask = (model.alpha.data - a_mean).abs() <= 2 * a_std
                model.alpha.data[a_mask] = 0.0
                a_zeroed = a_mask.sum().item()
                logger.info(f"  Alpha final: zeroed {a_zeroed}/{model.alpha.numel()} (mean={a_mean:.4f}, std={a_std:.4f})")

            # Alpha (extra layers)
            alpha_extra_saved8 = []
            for i in range(saved_args.extra_layers):
                alpha_extra_saved8.append(model.alpha_extra[i].data.clone())
                if model.alpha_extra[i].ndim >= 2:
                    ae_mean, ae_std = model.alpha_extra[i].data.mean(), model.alpha_extra[i].data.std()
                    ae_mask = (model.alpha_extra[i].data - ae_mean).abs() <= 2 * ae_std
                    model.alpha_extra[i].data[ae_mask] = 0.0
                    ae_zeroed = ae_mask.sum().item()
                    logger.info(f"  Alpha extra {i}: zeroed {ae_zeroed}/{model.alpha_extra[i].numel()} (mean={ae_mean:.4f}, std={ae_std:.4f})")

        abl8_zs = run_aggregate_zero_shot(
            model, saved_args, n_ep, num_train_trials, num_items, device
        )

        with torch.no_grad():
            get_readout_weight().data.copy_(readout_saved8)
            model.alpha.data.copy_(alpha_saved8)
            for i in range(saved_args.extra_layers):
                model.alpha_extra[i].data.copy_(alpha_extra_saved8[i])

        all_figures["ti_agg_symbolic_distance/abl8_readout_alpha_2sigma_zeroed"] = zero_shot_symbolic_distance_plot(
            abl8_zs, num_items, title=f'Zero-Shot SD - Readout+Alpha ≤2σ Zeroed (n={n_ep})'
        )
        all_figures["ti_agg_symbolic_distance/delta_regular_vs_abl8"] = delta_symbolic_distance_plot(
            regular_zs, abl8_zs, num_items,
            title=f'Delta: Regular - Readout+Alpha ≤2σ Zeroed (n={n_ep})'
        )

        # Ablation 9: Zero hidden + alpha + readout weights within 2 sigma of the mean
        logger.info(f"=== Ablation 9: Zero hidden + alpha + readout weights within 2σ ({n_ep} episodes) ===")
        with torch.no_grad():
            # Readout
            readout_w = get_readout_weight().data
            readout_saved9 = readout_w.clone()
            r_mean, r_std = readout_w.mean(), readout_w.std()
            r_mask = (readout_w - r_mean).abs() <= 2 * r_std
            readout_w[r_mask] = 0.0
            r_zeroed = r_mask.sum().item()
            logger.info(f"  Readout: zeroed {r_zeroed}/{readout_w.numel()} (mean={r_mean:.4f}, std={r_std:.4f})")

            # Alpha (final)
            alpha_saved9 = model.alpha.data.clone()
            if model.alpha.ndim >= 2:
                a_mean, a_std = model.alpha.data.mean(), model.alpha.data.std()
                a_mask = (model.alpha.data - a_mean).abs() <= 2 * a_std
                model.alpha.data[a_mask] = 0.0
                a_zeroed = a_mask.sum().item()
                logger.info(f"  Alpha final: zeroed {a_zeroed}/{model.alpha.numel()} (mean={a_mean:.4f}, std={a_std:.4f})")

            # Alpha (extra layers)
            alpha_extra_saved9 = []
            for i in range(saved_args.extra_layers):
                alpha_extra_saved9.append(model.alpha_extra[i].data.clone())
                if model.alpha_extra[i].ndim >= 2:
                    ae_mean, ae_std = model.alpha_extra[i].data.mean(), model.alpha_extra[i].data.std()
                    ae_mask = (model.alpha_extra[i].data - ae_mean).abs() <= 2 * ae_std
                    model.alpha_extra[i].data[ae_mask] = 0.0
                    ae_zeroed = ae_mask.sum().item()
                    logger.info(f"  Alpha extra {i}: zeroed {ae_zeroed}/{model.alpha_extra[i].numel()} (mean={ae_mean:.4f}, std={ae_std:.4f})")

            # Hidden (final layer - fc2)
            fc2_saved9 = model.fc2.weight.data.clone()
            h_mean, h_std = model.fc2.weight.data.mean(), model.fc2.weight.data.std()
            h_mask = (model.fc2.weight.data - h_mean).abs() <= 2 * h_std
            model.fc2.weight.data[h_mask] = 0.0
            h_zeroed = h_mask.sum().item()
            logger.info(f"  Hidden final (fc2): zeroed {h_zeroed}/{model.fc2.weight.numel()} (mean={h_mean:.4f}, std={h_std:.4f})")

            # Hidden (extra layers)
            extra_hidden_saved9 = []
            for i in range(saved_args.extra_layers):
                extra_hidden_saved9.append(model.extra_hidden_layers[i].weight.data.clone())
                eh_mean, eh_std = model.extra_hidden_layers[i].weight.data.mean(), model.extra_hidden_layers[i].weight.data.std()
                eh_mask = (model.extra_hidden_layers[i].weight.data - eh_mean).abs() <= 2 * eh_std
                model.extra_hidden_layers[i].weight.data[eh_mask] = 0.0
                eh_zeroed = eh_mask.sum().item()
                logger.info(f"  Hidden extra {i}: zeroed {eh_zeroed}/{model.extra_hidden_layers[i].weight.numel()} (mean={eh_mean:.4f}, std={eh_std:.4f})")

        abl9_zs = run_aggregate_zero_shot(
            model, saved_args, n_ep, num_train_trials, num_items, device
        )

        with torch.no_grad():
            get_readout_weight().data.copy_(readout_saved9)
            model.alpha.data.copy_(alpha_saved9)
            for i in range(saved_args.extra_layers):
                model.alpha_extra[i].data.copy_(alpha_extra_saved9[i])
            model.fc2.weight.data.copy_(fc2_saved9)
            for i in range(saved_args.extra_layers):
                model.extra_hidden_layers[i].weight.data.copy_(extra_hidden_saved9[i])

        all_figures["ti_agg_symbolic_distance/abl9_hidden_alpha_readout_2sigma_zeroed"] = zero_shot_symbolic_distance_plot(
            abl9_zs, num_items, title=f'Zero-Shot SD - Hidden+Alpha+Readout ≤2σ Zeroed (n={n_ep})'
        )
        all_figures["ti_agg_symbolic_distance/delta_regular_vs_abl9"] = delta_symbolic_distance_plot(
            regular_zs, abl9_zs, num_items,
            title=f'Delta: Regular - Hidden+Alpha+Readout ≤2σ Zeroed (n={n_ep})'
        )

        # Ablation 10: Keep only rows corresponding to >2σ readout dims in alpha
        logger.info(f"=== Ablation 10: Zero alpha rows for non->2σ readout dims ({n_ep} episodes) ===")
        with torch.no_grad():
            readout_w = get_readout_weight().data  # (1, hidden_size)
            r_mean, r_std = readout_w.mean(), readout_w.std()
            important_dims = ((readout_w - r_mean).abs() > 2 * r_std).squeeze(0)  # (hidden_size,)
            n_important = important_dims.sum().item()
            logger.info(f"  >2σ readout dims: {n_important}/{readout_w.numel()}")

            # Alpha (final) — zero rows not in important set
            alpha_saved10 = model.alpha.data.clone()
            if model.alpha.ndim >= 2:
                unimportant_rows = ~important_dims
                n_zeroed = unimportant_rows.sum().item() * model.alpha.shape[1]
                model.alpha.data[unimportant_rows, :] = 0.0
                logger.info(f"  Alpha final: zeroed {n_zeroed}/{model.alpha.numel()} (rows not in >2σ readout dims)")

            # Alpha (extra layers)
            alpha_extra_saved10 = []
            for i in range(saved_args.extra_layers):
                alpha_extra_saved10.append(model.alpha_extra[i].data.clone())
                if model.alpha_extra[i].ndim >= 2:
                    n_zeroed = unimportant_rows.sum().item() * model.alpha_extra[i].shape[1]
                    model.alpha_extra[i].data[unimportant_rows, :] = 0.0
                    logger.info(f"  Alpha extra {i}: zeroed {n_zeroed}/{model.alpha_extra[i].numel()} (rows not in >2σ readout dims)")

        # Heatmaps of ablated weights (Ablation 10)
        if model.alpha.ndim >= 2:
            W_abl10 = model.alpha.data.detach().cpu().numpy()
            fig_a10, ax_a10 = plt.subplots(figsize=(max(4, W_abl10.shape[1]/50)+2, max(2, W_abl10.shape[0]/50)+1), dpi=150)
            vmax = max(abs(W_abl10.min()), abs(W_abl10.max()), 1e-8)
            ax_a10.imshow(W_abl10, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto', interpolation='nearest')
            ax_a10.set_xlabel('Input Dimension'); ax_a10.set_ylabel('Output Dimension')
            ax_a10.set_title(f'Alpha Final - Abl10 (non->2σ rows zeroed)\n(shape: {W_abl10.shape}, nonzero rows: {int(important_dims.sum())})')
            plt.colorbar(ax_a10.images[0], ax=ax_a10, label='Weight')
            plt.tight_layout()
            all_figures["ti_agg_symbolic_distance/abl10_heatmap_alpha_final"] = fig_a10
        for i in range(saved_args.extra_layers):
            if model.alpha_extra[i].ndim >= 2:
                W_ae10 = model.alpha_extra[i].data.detach().cpu().numpy()
                fig_ae10, ax_ae10 = plt.subplots(figsize=(max(4, W_ae10.shape[1]/50)+2, max(2, W_ae10.shape[0]/50)+1), dpi=150)
                vmax = max(abs(W_ae10.min()), abs(W_ae10.max()), 1e-8)
                ax_ae10.imshow(W_ae10, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto', interpolation='nearest')
                ax_ae10.set_xlabel('Input Dimension'); ax_ae10.set_ylabel('Output Dimension')
                ax_ae10.set_title(f'Alpha Extra {i} - Abl10 (non->2σ rows zeroed)\n(shape: {W_ae10.shape}, nonzero rows: {int(important_dims.sum())})')
                plt.colorbar(ax_ae10.images[0], ax=ax_ae10, label='Weight')
                plt.tight_layout()
                all_figures[f"ti_agg_symbolic_distance/abl10_heatmap_alpha_extra{i}"] = fig_ae10

        abl10_zs = run_aggregate_zero_shot(
            model, saved_args, n_ep, num_train_trials, num_items, device
        )

        with torch.no_grad():
            model.alpha.data.copy_(alpha_saved10)
            for i in range(saved_args.extra_layers):
                model.alpha_extra[i].data.copy_(alpha_extra_saved10[i])

        all_figures["ti_agg_symbolic_distance/abl10_alpha_readout_dim_zeroed"] = zero_shot_symbolic_distance_plot(
            abl10_zs, num_items, title=f'Zero-Shot SD - Alpha Rows Zeroed (non->2σ readout dims, n={n_ep})'
        )
        all_figures["ti_agg_symbolic_distance/delta_regular_vs_abl10"] = delta_symbolic_distance_plot(
            regular_zs, abl10_zs, num_items,
            title=f'Delta: Regular - Alpha Rows Zeroed (non->2σ readout dims, n={n_ep})'
        )

        # Ablation 11: Keep only rows corresponding to >2σ readout dims in hidden + alpha
        logger.info(f"=== Ablation 11: Zero hidden + alpha rows for non->2σ readout dims ({n_ep} episodes) ===")
        with torch.no_grad():
            # Reuse important_dims from above

            # Alpha (final)
            alpha_saved11 = model.alpha.data.clone()
            if model.alpha.ndim >= 2:
                model.alpha.data[~important_dims, :] = 0.0
                logger.info(f"  Alpha final: zeroed rows not in >2σ readout dims")

            # Alpha (extra layers)
            alpha_extra_saved11 = []
            for i in range(saved_args.extra_layers):
                alpha_extra_saved11.append(model.alpha_extra[i].data.clone())
                if model.alpha_extra[i].ndim >= 2:
                    model.alpha_extra[i].data[~important_dims, :] = 0.0
                    logger.info(f"  Alpha extra {i}: zeroed rows not in >2σ readout dims")

            # Hidden (final layer - fc2)
            fc2_saved11 = model.fc2.weight.data.clone()
            if model.fc2.weight.data.shape[0] == 1:
                model.fc2.weight.data[:, ~important_dims] = 0.0
            else:
                model.fc2.weight.data[~important_dims, :] = 0.0
            logger.info(f"  Hidden final (fc2): zeroed rows not in >2σ readout dims")

            # Hidden (extra layers)
            extra_hidden_saved11 = []
            for i in range(saved_args.extra_layers):
                extra_hidden_saved11.append(model.extra_hidden_layers[i].weight.data.clone())
                model.extra_hidden_layers[i].weight.data[~important_dims, :] = 0.0
                logger.info(f"  Hidden extra {i}: zeroed rows not in >2σ readout dims")

        # Heatmaps of ablated weights (Ablation 11)
        def _abl_heatmap(W_np, title, fig_key):
            fig, ax = plt.subplots(figsize=(max(4, W_np.shape[1]/50)+2, max(2, W_np.shape[0]/50)+1), dpi=150)
            vmax = max(abs(W_np.min()), abs(W_np.max()), 1e-8)
            ax.imshow(W_np, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto', interpolation='nearest')
            ax.set_xlabel('Input Dimension'); ax.set_ylabel('Output Dimension')
            ax.set_title(f'{title}\n(shape: {W_np.shape}, nonzero rows: {int(important_dims.sum())})')
            plt.colorbar(ax.images[0], ax=ax, label='Weight')
            plt.tight_layout()
            all_figures[fig_key] = fig

        if model.alpha.ndim >= 2:
            _abl_heatmap(model.alpha.data.detach().cpu().numpy(), 'Alpha Final - Abl11 (non->2σ rows zeroed)',
                         'ti_agg_symbolic_distance/abl11_heatmap_alpha_final')
        for i in range(saved_args.extra_layers):
            if model.alpha_extra[i].ndim >= 2:
                _abl_heatmap(model.alpha_extra[i].data.detach().cpu().numpy(), f'Alpha Extra {i} - Abl11 (non->2σ rows zeroed)',
                             f'ti_agg_symbolic_distance/abl11_heatmap_alpha_extra{i}')
        _abl_heatmap(model.fc2.weight.data.detach().cpu().numpy(), 'Hidden Final (fc2) - Abl11 (non->2σ rows zeroed)',
                     'ti_agg_symbolic_distance/abl11_heatmap_fc2')
        for i in range(saved_args.extra_layers):
            _abl_heatmap(model.extra_hidden_layers[i].weight.data.detach().cpu().numpy(), f'Hidden Extra {i} - Abl11 (non->2σ rows zeroed)',
                         f'ti_agg_symbolic_distance/abl11_heatmap_hidden_extra{i}')

        abl11_zs = run_aggregate_zero_shot(
            model, saved_args, n_ep, num_train_trials, num_items, device
        )

        with torch.no_grad():
            model.alpha.data.copy_(alpha_saved11)
            for i in range(saved_args.extra_layers):
                model.alpha_extra[i].data.copy_(alpha_extra_saved11[i])
            model.fc2.weight.data.copy_(fc2_saved11)
            for i in range(saved_args.extra_layers):
                model.extra_hidden_layers[i].weight.data.copy_(extra_hidden_saved11[i])

        all_figures["ti_agg_symbolic_distance/abl11_hidden_alpha_readout_dim_zeroed"] = zero_shot_symbolic_distance_plot(
            abl11_zs, num_items, title=f'Zero-Shot SD - Hidden+Alpha Rows Zeroed (non->2σ readout dims, n={n_ep})'
        )
        all_figures["ti_agg_symbolic_distance/delta_regular_vs_abl11"] = delta_symbolic_distance_plot(
            regular_zs, abl11_zs, num_items,
            title=f'Delta: Regular - Hidden+Alpha Rows Zeroed (non->2σ readout dims, n={n_ep})'
        )

        # Greedy backward elimination of >2σ readout neurons
        # Performance measured by accuracy on critical pairs only (SD > 1, no end items)
        def critical_pair_accuracy(zs_trials, n_items):
            """Compute accuracy on critical pairs: SD > 1 and neither item is an end item."""
            correct = 0
            total = 0
            for (i, j), results in zs_trials.items():
                sd = j - i
                if sd <= 1:
                    continue
                if i == 0 or j == n_items - 1:
                    continue
                correct += sum(results)
                total += len(results)
            return correct / total if total > 0 else 0.0

        logger.info(f"=== Greedy backward elimination of >2σ readout neurons ({n_ep} episodes) ===")
        # Get the important dims and their readout weights
        with torch.no_grad():
            readout_w = get_readout_weight().data.squeeze(0)  # (hidden_size,)
            r_mean, r_std = readout_w.mean(), readout_w.std()
            important_mask = (readout_w - r_mean).abs() > 2 * r_std
            active_dims = sorted(important_mask.nonzero(as_tuple=True)[0].tolist())

        # Baseline: accuracy with all >2σ dims active (= regular model)
        baseline_crit_acc = critical_pair_accuracy(regular_zs, num_items)
        logger.info(f"  Baseline critical-pair accuracy (all dims): {baseline_crit_acc:.4f}")

        elimination_order = []  # (removed_dim, accuracy_after_removal)
        remaining_dims = list(active_dims)

        while len(remaining_dims) > 1:
            best_dim_to_remove = None
            best_acc_after_removal = -1.0

            for candidate_dim in remaining_dims:
                # Zero out the candidate dim's readout weight (effectively removing it)
                dims_after_removal = [d for d in remaining_dims if d != candidate_dim]
                dims_to_zero = [d for d in range(readout_w.shape[0]) if d not in dims_after_removal]

                with torch.no_grad():
                    readout_saved_ge = get_readout_weight().data.clone()
                    get_readout_weight().data[0, dims_to_zero] = 0.0

                zs = run_aggregate_zero_shot(
                    model, saved_args, n_ep, num_train_trials, num_items, device
                )

                with torch.no_grad():
                    get_readout_weight().data.copy_(readout_saved_ge)

                acc = critical_pair_accuracy(zs, num_items)
                logger.info(f"    Try removing dim {candidate_dim} (weight={readout_w[candidate_dim]:.4f}): crit acc={acc:.4f}")

                if acc > best_acc_after_removal:
                    best_acc_after_removal = acc
                    best_dim_to_remove = candidate_dim
                    best_zs = zs

            elimination_order.append((best_dim_to_remove, best_acc_after_removal))
            remaining_dims.remove(best_dim_to_remove)
            logger.info(f"  Step {len(elimination_order)}: removed dim {best_dim_to_remove} "
                        f"(weight={readout_w[best_dim_to_remove]:.4f}), "
                        f"crit acc={best_acc_after_removal:.4f}, "
                        f"remaining={remaining_dims}")

        # Log final single neuron
        last_dim = remaining_dims[0]
        with torch.no_grad():
            readout_saved_ge = get_readout_weight().data.clone()
            dims_to_zero = [d for d in range(readout_w.shape[0]) if d != last_dim]
            get_readout_weight().data[0, dims_to_zero] = 0.0
        last_zs = run_aggregate_zero_shot(
            model, saved_args, n_ep, num_train_trials, num_items, device
        )
        with torch.no_grad():
            get_readout_weight().data.copy_(readout_saved_ge)
        last_acc = critical_pair_accuracy(last_zs, num_items)
        logger.info(f"  Final single neuron: dim {last_dim} (weight={readout_w[last_dim]:.4f}), crit acc={last_acc:.4f}")

        # Plot: critical-pair accuracy vs number of active neurons
        fig_elim, ax_elim = plt.subplots(figsize=(8, 5))
        n_active = [len(active_dims)]  # start with all
        accs = [baseline_crit_acc]
        for removed_dim, acc in elimination_order:
            n_active.append(n_active[-1] - 1)
            accs.append(acc)
        # Add the final single-neuron point
        n_active.append(1)
        accs.append(last_acc)
        # Remove duplicate if last elimination step already got to 1
        if len(n_active) > len(set(n_active)):
            n_active = n_active[:-1]
            accs = accs[:-1]

        ax_elim.plot(n_active, accs, 'o-', color='steelblue', linewidth=2, markersize=8)
        ax_elim.set_xlabel('Number of Active Readout Neurons', fontsize=12)
        ax_elim.set_ylabel('Critical-Pair Accuracy', fontsize=12)
        ax_elim.set_title(f'Greedy Backward Elimination (n={n_ep})', fontsize=13)
        ax_elim.set_xticks(sorted(set(n_active)))
        ax_elim.axhline(y=baseline_crit_acc, color='gray', linestyle='--', alpha=0.5, label='Baseline (all >2σ)')
        ax_elim.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Chance')
        ax_elim.legend()
        ax_elim.set_ylim(0, 1.05)
        # Annotate each point with the dim removed
        for idx, (removed_dim, acc) in enumerate(elimination_order):
            ax_elim.annotate(f'-{removed_dim}', (n_active[idx + 1], accs[idx + 1]),
                           textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)
        fig_elim.tight_layout()
        all_figures["ti_agg_symbolic_distance/greedy_backward_elimination"] = fig_elim

        # Also log a summary table
        logger.info("  === Elimination summary ===")
        logger.info(f"  {'Step':>4} {'Removed':>8} {'Weight':>10} {'Crit Acc':>10} {'Remaining':>4}")
        logger.info(f"  {'0':>4} {'---':>8} {'---':>10} {baseline_crit_acc:>10.4f} {len(active_dims):>4}")
        for step, (removed_dim, acc) in enumerate(elimination_order, 1):
            logger.info(f"  {step:>4} {removed_dim:>8} {readout_w[removed_dim].item():>10.4f} {acc:>10.4f} {len(active_dims) - step:>4}")
        logger.info(f"  {'final':>4} {last_dim:>8} {readout_w[last_dim].item():>10.4f} {last_acc:>10.4f} {'1':>4}")

        # Single-pass experiment: 7 trials, one per adjacent pair
        logger.info(f"=== Single-pass trials ({n_ep} episodes) ===")
        single_pass_zs, single_pass_acc, _ = run_controlled_zero_shot(
            model, saved_args, n_ep, num_items, device, generate_single_pass_trials
        )
        all_figures["ti_agg_symbolic_distance/single_pass_7trials"] = zero_shot_symbolic_distance_plot(
            single_pass_zs, num_items, title=f'Zero-Shot SD - Single Pass (7 trials, n={n_ep})'
        )

        # High-left ordered experiment: 7 trials AB BC CD ... FG GH
        logger.info(f"=== High-left ordered trials ({n_ep} episodes) ===")
        high_left_zs, high_left_acc, hl_probes = run_controlled_zero_shot(
            model, saved_args, n_ep, num_items, device, generate_high_left_ordered_trials,
            probe_pair_logits=True, probe_item_logits=True, probe_item_dot_layers=dot_all_layers,
            probe_item_dots_no_pc1=True, probe_item_dots_no_readout=True
        )
        all_figures["ti_agg_symbolic_distance/high_left_7trials"] = zero_shot_symbolic_distance_plot(
            high_left_zs, num_items, title=f'Zero-Shot SD - High Left Ordered (7 trials, n={n_ep})'
        )

        # High-right ordered experiment: 7 trials BA CB DC ... GF HG
        logger.info(f"=== High-right ordered trials ({n_ep} episodes) ===")
        high_right_zs, high_right_acc, hr_probes = run_controlled_zero_shot(
            model, saved_args, n_ep, num_items, device, generate_high_right_ordered_trials,
            probe_pair_logits=True, probe_item_logits=True, probe_item_dot_layers=dot_all_layers[:1]
        )
        all_figures["ti_agg_symbolic_distance/high_right_7trials"] = zero_shot_symbolic_distance_plot(
            high_right_zs, num_items, title=f'Zero-Shot SD - High Right Ordered (7 trials, n={n_ep})'
        )

        # Fixed-order experiment: 14 trials in canonical order
        logger.info(f"=== Fixed-order trials ({n_ep} episodes) ===")
        fixed_order_zs, fixed_order_acc, _ = run_controlled_zero_shot(
            model, saved_args, n_ep, num_items, device, generate_fixed_order_trials
        )
        all_figures["ti_agg_symbolic_distance/fixed_order_14trials"] = zero_shot_symbolic_distance_plot(
            fixed_order_zs, num_items, title=f'Zero-Shot SD - Fixed Order (14 trials, n={n_ep})'
        )

        # Delta: regular vs single-pass
        all_figures["ti_agg_symbolic_distance/delta_regular_vs_single_pass"] = delta_symbolic_distance_plot(
            regular_zs, single_pass_zs, num_items,
            title=f'Delta Accuracy: Regular - Single Pass (n={n_ep})'
        )
        # Delta: regular vs high-left
        all_figures["ti_agg_symbolic_distance/delta_regular_vs_high_left"] = delta_symbolic_distance_plot(
            regular_zs, high_left_zs, num_items,
            title=f'Delta Accuracy: Regular - High Left Ordered (n={n_ep})'
        )
        # Delta: regular vs high-right
        all_figures["ti_agg_symbolic_distance/delta_regular_vs_high_right"] = delta_symbolic_distance_plot(
            regular_zs, high_right_zs, num_items,
            title=f'Delta Accuracy: Regular - High Right Ordered (n={n_ep})'
        )
        # Delta: regular vs fixed-order
        all_figures["ti_agg_symbolic_distance/delta_regular_vs_fixed_order"] = delta_symbolic_distance_plot(
            regular_zs, fixed_order_zs, num_items,
            title=f'Delta Accuracy: Regular - Fixed Order (n={n_ep})'
        )

        # Training accuracy by trial number
        all_figures["ti_agg_symbolic_distance/training_accuracy_by_trial"] = training_accuracy_by_trial_plot(
            {"High Left (AB BC...)": high_left_acc, "High Right (BA CB...)": high_right_acc},
            title=f'Training Accuracy by Trial (n={n_ep})'
        )

        # Pair logit heatmaps for high-left and high-right
        adj_pair_labels_hl = []
        for i in range(num_items - 1):
            adj_pair_labels_hl.append(f"{item_labels[i]}{item_labels[i+1]}")
            adj_pair_labels_hl.append(f"{item_labels[i+1]}{item_labels[i]}")
        hl_trial_labels = [f"{item_labels[i]}{item_labels[i+1]}" for i in range(num_items - 1)]
        hr_trial_labels = [f"{item_labels[i+1]}{item_labels[i]}" for i in range(num_items - 1)]

        all_figures["ti_agg_symbolic_distance/high_left_pair_logits"] = pair_logit_by_trial_heatmap(
            hl_probes['pair_logits_mean'], hl_probes['pair_logits_sem'], adj_pair_labels_hl, hl_trial_labels,
            title=f'Pair Logits During High-Left Training (n={n_ep})'
        )
        all_figures["ti_agg_symbolic_distance/high_right_pair_logits"] = pair_logit_by_trial_heatmap(
            hr_probes['pair_logits_mean'], hr_probes['pair_logits_sem'], adj_pair_labels_hl, hr_trial_labels,
            title=f'Pair Logits During High-Right Training (n={n_ep})'
        )

        # All-pairs logit heatmaps (adjacent + nonadjacent, ordered by symbolic distance)
        all_pl = hl_probes['all_pair_labels']
        all_figures["ti_agg_symbolic_distance/high_left_all_pair_logits"] = pair_logit_by_trial_heatmap(
            hl_probes['all_pair_logits_mean'], hl_probes['all_pair_logits_sem'], all_pl, hl_trial_labels,
            title=f'All Pair Logits During High-Left Training (n={n_ep})'
        )
        all_figures["ti_agg_symbolic_distance/high_right_all_pair_logits"] = pair_logit_by_trial_heatmap(
            hr_probes['all_pair_logits_mean'], hr_probes['all_pair_logits_sem'], all_pl, hr_trial_labels,
            title=f'All Pair Logits During High-Right Training (n={n_ep})'
        )

        # High-right with NM ablation: NM=0 on positive, NM=neg_nm_override on negative
        neg_nm = cli_args.neg_nm_override
        logger.info(f"=== High-right NM-ablated trials (pos_nm=0, neg_nm={neg_nm}, {n_ep} episodes) ===")
        hr_abl_zs, hr_abl_acc, hr_abl_probes = run_controlled_zero_shot(
            model, saved_args, n_ep, num_items, device, generate_high_right_ordered_trials,
            probe_pair_logits=True, nm_override_positive=0.0, nm_override_negative=neg_nm,
            probe_item_logits=True, probe_item_dot_layers=dot_all_layers[:1]
        )
        all_figures["ti_agg_symbolic_distance/high_right_nm_abl_7trials"] = zero_shot_symbolic_distance_plot(
            hr_abl_zs, num_items, title=f'Zero-Shot SD - High Right NM Ablated (pos=0, neg={neg_nm}, n={n_ep})'
        )
        all_figures["ti_agg_symbolic_distance/high_right_nm_abl_pair_logits"] = pair_logit_by_trial_heatmap(
            hr_abl_probes['pair_logits_mean'], hr_abl_probes['pair_logits_sem'], adj_pair_labels_hl, hr_trial_labels,
            title=f'Pair Logits During High-Right NM Ablated (pos=0, neg={neg_nm}, n={n_ep})'
        )
        all_figures["ti_agg_symbolic_distance/high_right_nm_abl_all_pair_logits"] = pair_logit_by_trial_heatmap(
            hr_abl_probes['all_pair_logits_mean'], hr_abl_probes['all_pair_logits_sem'], all_pl, hr_trial_labels,
            title=f'All Pair Logits During High-Right NM Ablated (pos=0, neg={neg_nm}, n={n_ep})'
        )
        all_figures["ti_agg_symbolic_distance/training_accuracy_by_trial_with_abl"] = training_accuracy_by_trial_plot(
            {"High Left (AB BC...)": high_left_acc,
             "High Right (BA CB...)": high_right_acc,
             f"High Right NM Abl (pos=0, neg={neg_nm})": hr_abl_acc},
            title=f'Training Accuracy by Trial (n={n_ep})'
        )
        all_figures["ti_agg_symbolic_distance/delta_high_right_vs_nm_abl"] = delta_symbolic_distance_plot(
            high_right_zs, hr_abl_zs, num_items,
            title=f'Delta Accuracy: High Right - High Right NM Ablated (n={n_ep})'
        )

        # Post-training individual item logit bar charts for high-left, high-right, and NM-ablated high-right
        hl_item_figs = create_post_train_item_logit_bar_chart(
            hl_probes['item_logits_pos1'], hl_probes['item_logits_pos2'], item_labels, n_ep,
            title_template='Post-Training Item Logit - High Left ({pos_title})\nn={num_episodes} episodes',
            fig_key_prefix="ti_agg_symbolic_distance/high_left_item"
        )
        all_figures.update(hl_item_figs)
        hr_item_figs = create_post_train_item_logit_bar_chart(
            hr_probes['item_logits_pos1'], hr_probes['item_logits_pos2'], item_labels, n_ep,
            title_template='Post-Training Item Logit - High Right ({pos_title})\nn={num_episodes} episodes',
            fig_key_prefix="ti_agg_symbolic_distance/high_right_item"
        )
        all_figures.update(hr_item_figs)
        hr_abl_item_figs = create_post_train_item_logit_bar_chart(
            hr_abl_probes['item_logits_pos1'], hr_abl_probes['item_logits_pos2'], item_labels, n_ep,
            title_template=f'Post-Training Item Logit - High Right NM Abl (pos=0, neg={neg_nm}) ({{pos_title}})\nn={{num_episodes}} episodes',
            fig_key_prefix="ti_agg_symbolic_distance/high_right_nm_abl_item"
        )
        all_figures.update(hr_abl_item_figs)

        # Item logit evolution heatmaps (per-trial, for each condition and position)
        for label, probes, trial_labs, prefix in [
            ("High Left", hl_probes, hl_trial_labels, "high_left"),
            ("High Right", hr_probes, hr_trial_labels, "high_right"),
            (f"High Right NM Abl (pos=0, neg={neg_nm})", hr_abl_probes, hr_trial_labels, "high_right_nm_abl"),
        ]:
            for pos_name, evo_key in [("pos1_left", "item_evo_pos1"), ("pos2_right", "item_evo_pos2")]:
                pos_title = pos_name.replace("_", " ").title()
                all_figures[f"ti_agg_symbolic_distance/{prefix}_item_evo_{pos_name}"] = pair_logit_by_trial_heatmap(
                    probes[f'{evo_key}_mean'], probes[f'{evo_key}_sem'],
                    item_labels, trial_labs,
                    title=f'Item Logit Evolution - {label} ({pos_title}, n={n_ep})',
                    ylabel='Item'
                )

            # Item dot product heatmaps at layer 1 (16x16: pos1+pos2)
            dot_figs = item_dot_product_heatmaps(
                probes['item_dot_L1_mean'], probes['item_dot_L1_sem'],
                num_items, trial_labs,
                title_prefix=f'Item Dot Products L1 - {label}',
                num_episodes=n_ep
            )
            for key_suffix, fig in dot_figs.items():
                all_figures[f"ti_agg_symbolic_distance/{prefix}_item_dot/{key_suffix}"] = fig

        # High-left: layer 1 + layer 2 dot products with projection-removed variants
        for L in dot_all_layers:
            # PC1-removed
            dot_figs_no_pc1 = item_dot_product_heatmaps(
                hl_probes[f'item_dot_no_pc1_L{L}_mean'], hl_probes[f'item_dot_no_pc1_L{L}_sem'],
                num_items, hl_trial_labels,
                title_prefix=f'Item Dot Products L{L} (PC1 removed) - High Left',
                num_episodes=n_ep
            )
            for key_suffix, fig in dot_figs_no_pc1.items():
                all_figures[f"ti_agg_symbolic_distance/high_left_item_dot_no_pc1_L{L}/{key_suffix}"] = fig
        # High-left final layer: regular + readout-removed + both-removed
        Lf = dot_all_layers[-1]
        dot_figs_lf = item_dot_product_heatmaps(
            hl_probes[f'item_dot_L{Lf}_mean'], hl_probes[f'item_dot_L{Lf}_sem'],
            num_items, hl_trial_labels,
            title_prefix=f'Item Dot Products L{Lf} - High Left',
            num_episodes=n_ep
        )
        for key_suffix, fig in dot_figs_lf.items():
            all_figures[f"ti_agg_symbolic_distance/high_left_item_dot_L{Lf}/{key_suffix}"] = fig
        dot_figs_no_ro = item_dot_product_heatmaps(
            hl_probes[f'item_dot_no_readout_L{Lf}_mean'], hl_probes[f'item_dot_no_readout_L{Lf}_sem'],
            num_items, hl_trial_labels,
            title_prefix=f'Item Dot Products L{Lf} (readout removed) - High Left',
            num_episodes=n_ep
        )
        for key_suffix, fig in dot_figs_no_ro.items():
            all_figures[f"ti_agg_symbolic_distance/high_left_item_dot_no_readout_L{Lf}/{key_suffix}"] = fig
        dot_figs_no_both = item_dot_product_heatmaps(
            hl_probes[f'item_dot_no_pc1_no_readout_L{Lf}_mean'], hl_probes[f'item_dot_no_pc1_no_readout_L{Lf}_sem'],
            num_items, hl_trial_labels,
            title_prefix=f'Item Dot Products L{Lf} (PC1+readout removed) - High Left',
            num_episodes=n_ep
        )
        for key_suffix, fig in dot_figs_no_both.items():
            all_figures[f"ti_agg_symbolic_distance/high_left_item_dot_no_pc1_no_readout_L{Lf}/{key_suffix}"] = fig

        # High-left skip-DE experiment: 6 trials AB BC CD EF FG GH (no DE trial)
        logger.info(f"=== High-left skip-DE trials ({n_ep} episodes) ===")
        hl_skipde_zs, hl_skipde_acc, hl_skipde_probes = run_controlled_zero_shot(
            model, saved_args, n_ep, num_items, device, generate_high_left_ordered_trials_skip_de,
            probe_item_dot_layers=dot_all_layers, probe_item_dots_no_pc1=True,
            probe_item_dots_no_readout=True
        )
        hl_skipde_trial_labels = [f"{item_labels[i]}{item_labels[i+1]}" for i in range(num_items - 1) if i != 3] + [f"{item_labels[3]}{item_labels[4]}"]
        for L in dot_all_layers:
            # Regular dot products
            dot_figs = item_dot_product_heatmaps(
                hl_skipde_probes[f'item_dot_L{L}_mean'], hl_skipde_probes[f'item_dot_L{L}_sem'],
                num_items, hl_skipde_trial_labels,
                title_prefix=f'Item Dot Products L{L} - High Left Skip DE',
                num_episodes=n_ep
            )
            for key_suffix, fig in dot_figs.items():
                all_figures[f"ti_agg_symbolic_distance/high_left_skip_de_item_dot_L{L}/{key_suffix}"] = fig
            # PC1-removed dot products
            dot_figs_no_pc1 = item_dot_product_heatmaps(
                hl_skipde_probes[f'item_dot_no_pc1_L{L}_mean'], hl_skipde_probes[f'item_dot_no_pc1_L{L}_sem'],
                num_items, hl_skipde_trial_labels,
                title_prefix=f'Item Dot Products L{L} (PC1 removed) - High Left Skip DE',
                num_episodes=n_ep
            )
            for key_suffix, fig in dot_figs_no_pc1.items():
                all_figures[f"ti_agg_symbolic_distance/high_left_skip_de_item_dot_no_pc1_L{L}/{key_suffix}"] = fig
        # Skip-DE final layer: readout-removed + both-removed
        dot_figs_no_ro = item_dot_product_heatmaps(
            hl_skipde_probes[f'item_dot_no_readout_L{Lf}_mean'], hl_skipde_probes[f'item_dot_no_readout_L{Lf}_sem'],
            num_items, hl_skipde_trial_labels,
            title_prefix=f'Item Dot Products L{Lf} (readout removed) - High Left Skip DE',
            num_episodes=n_ep
        )
        for key_suffix, fig in dot_figs_no_ro.items():
            all_figures[f"ti_agg_symbolic_distance/high_left_skip_de_item_dot_no_readout_L{Lf}/{key_suffix}"] = fig
        dot_figs_no_both = item_dot_product_heatmaps(
            hl_skipde_probes[f'item_dot_no_pc1_no_readout_L{Lf}_mean'], hl_skipde_probes[f'item_dot_no_pc1_no_readout_L{Lf}_sem'],
            num_items, hl_skipde_trial_labels,
            title_prefix=f'Item Dot Products L{Lf} (PC1+readout removed) - High Left Skip DE',
            num_episodes=n_ep
        )
        for key_suffix, fig in dot_figs_no_both.items():
            all_figures[f"ti_agg_symbolic_distance/high_left_skip_de_item_dot_no_pc1_no_readout_L{Lf}/{key_suffix}"] = fig

        logger.info("Created symbolic distance plots")
    else:
        logger.info("Skipping symbolic distance plots and ablations")

    # === Single-neuron and bias ablations ===
    if not should_skip('ti_agg_symbolic_distance_single_neuron_bias'):
        n_ep = cli_args.num_aggregate_episodes

        # Baseline: regular model
        logger.info(f"=== Single-neuron/bias section: baseline ({n_ep} episodes) ===")
        snb_baseline_zs = run_aggregate_zero_shot(
            model, saved_args, n_ep, num_train_trials, num_items, device
        )
        all_figures["ti_agg_symbolic_distance_single_neuron_bias/baseline"] = zero_shot_symbolic_distance_plot(
            snb_baseline_zs, num_items, title=f'Zero-Shot SD - Baseline (n={n_ep})'
        )

        # Identify >2σ readout dims and build neuron test list ranked by magnitude
        with torch.no_grad():
            readout_w = get_readout_weight().data.squeeze(0)  # (hidden_size,)
            r_mean, r_std = readout_w.mean(), readout_w.std()
            important_mask = (readout_w - r_mean).abs() > 2 * r_std
            important_dims = sorted(important_mask.nonzero(as_tuple=True)[0].tolist())
            logger.info(f"  >2σ readout dims ({len(important_dims)}): {important_dims}")

            # All neurons ranked by magnitude (descending)
            all_dims_by_magnitude = torch.argsort(readout_w.abs(), descending=True).tolist()

            # Build test list: >2σ neurons first (by magnitude), then remaining neurons by magnitude
            important_set = set(important_dims)
            important_by_mag = [d for d in all_dims_by_magnitude if d in important_set]
            rest_by_mag = [d for d in all_dims_by_magnitude if d not in important_set]

            if cli_args.single_neuron_top_k > 0:
                # Take top K overall by magnitude
                neurons_to_test = all_dims_by_magnitude[:cli_args.single_neuron_top_k]
            else:
                # Only >2σ neurons
                neurons_to_test = important_by_mag

            logger.info(f"  Testing {len(neurons_to_test)} neurons (top_k={cli_args.single_neuron_top_k})")

        # 1. Bias-only ablation: zero ALL readout weights, keep bias
        logger.info(f"=== Bias-only ablation ({n_ep} episodes) ===")
        with torch.no_grad():
            readout_saved_bias = get_readout_weight().data.clone()
            get_readout_weight().data.zero_()

        bias_zs = run_aggregate_zero_shot(
            model, saved_args, n_ep, num_train_trials, num_items, device
        )

        with torch.no_grad():
            get_readout_weight().data.copy_(readout_saved_bias)

        all_figures["ti_agg_symbolic_distance_single_neuron_bias/bias_only"] = zero_shot_symbolic_distance_plot(
            bias_zs, num_items, title=f'Zero-Shot SD - Bias Only (n={n_ep})'
        )
        all_figures["ti_agg_symbolic_distance_single_neuron_bias/delta_baseline_vs_bias_only"] = delta_symbolic_distance_plot(
            snb_baseline_zs, bias_zs, num_items,
            title=f'Delta: Baseline - Bias Only (n={n_ep})'
        )

        # Helper: critical pair accuracy (SD > 1, no end items)
        def critical_pair_accuracy(zs_trials, n_items):
            correct = 0
            total = 0
            for (i, j), results in zs_trials.items():
                sd = j - i
                if sd <= 1:
                    continue
                if i == 0 or j == n_items - 1:
                    continue
                correct += sum(results)
                total += len(results)
            return correct / total if total > 0 else 0.0

        baseline_crit_acc = critical_pair_accuracy(snb_baseline_zs, num_items)
        bias_crit_acc = critical_pair_accuracy(bias_zs, num_items)

        # 2. Each neuron individually (ranked by magnitude, >2σ first then rest)
        single_neuron_results = []  # (dim, weight, crit_acc, is_2sigma)
        for idx, dim in enumerate(neurons_to_test):
            is_2sigma = dim in important_set
            logger.info(f"=== Single neuron {dim} ({idx+1}/{len(neurons_to_test)}, "
                        f"weight={readout_w[dim]:.4f}, {'> 2σ' if is_2sigma else '≤ 2σ'}) ({n_ep} episodes) ===")
            with torch.no_grad():
                readout_saved_sn = get_readout_weight().data.clone()
                get_readout_weight().data.zero_()
                get_readout_weight().data[0, dim] = readout_saved_sn[0, dim]

            sn_zs = run_aggregate_zero_shot(
                model, saved_args, n_ep, num_train_trials, num_items, device
            )

            with torch.no_grad():
                get_readout_weight().data.copy_(readout_saved_sn)

            sn_crit_acc = critical_pair_accuracy(sn_zs, num_items)
            single_neuron_results.append((dim, readout_w[dim].item(), sn_crit_acc, is_2sigma))

            # Only generate SD plots for >2σ neurons to avoid plot bloat
            if is_2sigma:
                all_figures[f"ti_agg_symbolic_distance_single_neuron_bias/single_neuron_{dim}"] = zero_shot_symbolic_distance_plot(
                    sn_zs, num_items, title=f'Zero-Shot SD - Neuron {dim} Only (w={readout_w[dim]:.4f}, n={n_ep})'
                )
                all_figures[f"ti_agg_symbolic_distance_single_neuron_bias/delta_baseline_vs_single_neuron_{dim}"] = delta_symbolic_distance_plot(
                    snb_baseline_zs, sn_zs, num_items,
                    title=f'Delta: Baseline - Neuron {dim} Only (n={n_ep})'
                )

        # Table: single-neuron results sorted by decreasing critical-pair accuracy
        single_neuron_results.sort(key=lambda x: x[2], reverse=True)

        # Table figure
        n_rows = len(single_neuron_results) + 3  # header + baseline + bias + neurons
        fig_table, ax_table = plt.subplots(figsize=(7, 0.35 * n_rows + 1))
        ax_table.axis('off')
        col_labels = ['Rank', 'Neuron', 'Weight', 'Crit-Pair Acc', '>2σ']
        table_data = []
        table_data.append(['—', 'Baseline', '—', f'{baseline_crit_acc:.4f}', '—'])
        table_data.append(['—', 'Bias only', '—', f'{bias_crit_acc:.4f}', '—'])
        for rank, (dim, weight, acc, is_2sigma) in enumerate(single_neuron_results, 1):
            table_data.append([str(rank), str(dim), f'{weight:.4f}', f'{acc:.4f}', 'Y' if is_2sigma else ''])
        tbl = ax_table.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9 if len(single_neuron_results) > 30 else 10)
        tbl.scale(1, 1.3 if len(single_neuron_results) > 30 else 1.4)
        for j in range(len(col_labels)):
            tbl[0, j].set_text_props(fontweight='bold')
            tbl[1, j].set_facecolor('#d4edda')  # baseline green
            tbl[2, j].set_facecolor('#f8d7da')  # bias-only red
        # Highlight >2σ neuron rows
        for row_idx, (dim, weight, acc, is_2sigma) in enumerate(single_neuron_results, 3):
            if is_2sigma:
                for j in range(len(col_labels)):
                    tbl[row_idx, j].set_facecolor('#fff3cd')  # light yellow
        ax_table.set_title(f'Single-Neuron Critical-Pair Accuracy (n={n_ep}, {len(neurons_to_test)} neurons)', fontsize=12, pad=20)
        fig_table.tight_layout()
        all_figures["ti_agg_symbolic_distance_single_neuron_bias/single_neuron_accuracy_table"] = fig_table

        # Scatter plot: weight magnitude vs critical-pair accuracy
        fig_scatter, ax_scatter = plt.subplots(figsize=(8, 5))
        dims_arr = [r[0] for r in single_neuron_results]
        weights_arr = [abs(r[1]) for r in single_neuron_results]
        accs_arr = [r[2] for r in single_neuron_results]
        is_2sigma_arr = [r[3] for r in single_neuron_results]
        colors = ['#e8a838' if s else '#4a90d9' for s in is_2sigma_arr]
        ax_scatter.scatter(weights_arr, accs_arr, c=colors, s=40, alpha=0.7, edgecolors='k', linewidths=0.5)
        ax_scatter.set_xlabel('|Readout Weight|', fontsize=12)
        ax_scatter.set_ylabel('Critical-Pair Accuracy', fontsize=12)
        ax_scatter.set_title(f'Single-Neuron Accuracy vs Weight Magnitude (n={n_ep})', fontsize=13)
        ax_scatter.axhline(y=baseline_crit_acc, color='green', linestyle='--', alpha=0.5, label='Baseline')
        ax_scatter.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Chance')
        # Legend for colors
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#e8a838', markersize=8, label='>2σ'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#4a90d9', markersize=8, label='≤2σ'),
        ]
        ax_scatter.legend(handles=legend_elements + ax_scatter.get_legend_handles_labels()[0], loc='best')
        fig_scatter.tight_layout()
        all_figures["ti_agg_symbolic_distance_single_neuron_bias/accuracy_vs_weight_magnitude"] = fig_scatter

        # Log table
        logger.info("  === Single-neuron critical-pair accuracy (sorted by acc) ===")
        logger.info(f"  {'Rank':>4} {'Neuron':>8} {'Weight':>10} {'Crit Acc':>10} {'>2σ':>4}")
        logger.info(f"  {'—':>4} {'Baseline':>8} {'—':>10} {baseline_crit_acc:>10.4f} {'—':>4}")
        logger.info(f"  {'—':>4} {'Bias':>8} {'—':>10} {bias_crit_acc:>10.4f} {'—':>4}")
        for rank, (dim, weight, acc, is_2sigma) in enumerate(single_neuron_results, 1):
            logger.info(f"  {rank:>4} {dim:>8} {weight:>10.4f} {acc:>10.4f} {'Y' if is_2sigma else '':>4}")

        # 3. All >2σ except neuron 170 (only 170 zeroed)
        neuron_170_idx = 170
        logger.info(f"=== All >2σ except neuron {neuron_170_idx} ({n_ep} episodes) ===")
        with torch.no_grad():
            readout_saved_no170 = get_readout_weight().data.clone()
            # Zero everything, then restore only the >2σ dims except 170
            get_readout_weight().data.zero_()
            for dim in important_dims:
                if dim != neuron_170_idx:
                    get_readout_weight().data[0, dim] = readout_saved_no170[0, dim]

        no170_zs = run_aggregate_zero_shot(
            model, saved_args, n_ep, num_train_trials, num_items, device
        )

        with torch.no_grad():
            get_readout_weight().data.copy_(readout_saved_no170)

        all_figures["ti_agg_symbolic_distance_single_neuron_bias/all_2sigma_except_170"] = zero_shot_symbolic_distance_plot(
            no170_zs, num_items, title=f'Zero-Shot SD - All >2σ Except Neuron {neuron_170_idx} (n={n_ep})'
        )
        all_figures["ti_agg_symbolic_distance_single_neuron_bias/delta_baseline_vs_no170"] = delta_symbolic_distance_plot(
            snb_baseline_zs, no170_zs, num_items,
            title=f'Delta: Baseline - All >2σ Except Neuron {neuron_170_idx} (n={n_ep})'
        )

        logger.info("Created single-neuron/bias ablation plots")
    else:
        logger.info("Skipping single-neuron/bias ablation plots")

    # === List-linking symbolic distance ===
    if not should_skip('ll_agg_symbolic_distance'):
        n_ep = cli_args.num_aggregate_episodes
        ll_num_items = 8  # 2 lists of 4
        ll_n1 = cli_args.ll_num_trials_list_1
        ll_n2 = cli_args.ll_num_trials_list_2
        ll_nl = cli_args.ll_num_trials_linking_pair
        ll_put_first = getattr(saved_args, 'put_linking_trials_first', False)
        ll_rand_order = getattr(saved_args, 'randomize_list_order', False)

        logger.info(f"=== Running list-linking zero-shot symbolic distance ({n_ep} episodes) ===")
        logger.info(f"  List 1 trials: {ll_n1}, List 2 trials: {ll_n2}, Linking pair trials: {ll_nl}")
        ll_zs = run_aggregate_zero_shot_ll(
            model, saved_args, n_ep, ll_num_items, device,
            num_trials_list_1=ll_n1, num_trials_list_2=ll_n2, num_trials_linking_pair=ll_nl,
            put_linking_first=ll_put_first, randomize_order=ll_rand_order
        )
        all_figures["ll_agg_symbolic_distance/regular"] = zero_shot_symbolic_distance_plot(
            ll_zs, ll_num_items, title=f'LL Zero-Shot Symbolic Distance (n={n_ep})'
        )

        # --- LL Critical Pair Bar Chart ---
        # Critical LL pairs: cross-list, no end items, not linking pair,
        # and within-list position of list 1 item >= within-list position of list 2 item.
        half = ll_num_items // 2
        ll_critical_pairs = []
        for i in range(1, half):          # list 1, skip end item 0
            for j in range(half, ll_num_items - 1):  # list 2, skip end item N-1
                if i == half - 1 and j == half:       # skip linking pair
                    continue
                if i >= j - half:                     # within-list rank constraint
                    ll_critical_pairs.append((i, j))

        pair_labels_ll = []
        pair_accs_ll = []
        for (i, j) in sorted(ll_critical_pairs, key=lambda p: (p[1] - p[0], p[0])):
            label = chr(i + ord('A')) + chr(j + ord('A'))
            acc = np.mean(ll_zs[(i, j)]) if len(ll_zs[(i, j)]) > 0 else 0.0
            pair_labels_ll.append(label)
            pair_accs_ll.append(acc)

        overall_ll_crit = np.mean([v for (i, j) in ll_critical_pairs for v in ll_zs[(i, j)]]) if ll_critical_pairs else 0.0
        logger.info(f"  LL critical pair accuracy: {overall_ll_crit:.4f}")
        for lbl, acc in zip(pair_labels_ll, pair_accs_ll):
            logger.info(f"    {lbl}: {acc:.4f}")

        fig_crit, ax_crit = plt.subplots(figsize=(max(6, len(pair_labels_ll) * 0.8), 5), dpi=300)
        colors_crit = plt.cm.tab10(np.linspace(0, 0.7, len(pair_labels_ll)))
        bars = ax_crit.bar(range(len(pair_labels_ll)), pair_accs_ll, color=colors_crit, edgecolor='black', linewidth=0.5)
        ax_crit.axhline(y=0.5, color='lightgray', linestyle=':', linewidth=1, zorder=0)
        ax_crit.axhline(y=overall_ll_crit, color='red', linestyle='--', linewidth=1, label=f'Mean = {overall_ll_crit:.3f}')
        ax_crit.set_xticks(range(len(pair_labels_ll)))
        ax_crit.set_xticklabels(pair_labels_ll, fontsize=10)
        ax_crit.set_ylabel('Accuracy')
        ax_crit.set_title(f'LL Critical Pair Accuracy (n={n_ep})')
        ax_crit.set_ylim(0, 1.05)
        ax_crit.legend(fontsize=9)
        for idx, (bar, acc) in enumerate(zip(bars, pair_accs_ll)):
            ax_crit.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{acc:.2f}',
                        ha='center', va='bottom', fontsize=8)
        fig_crit.tight_layout()
        all_figures["ll_agg_symbolic_distance/critical_pairs"] = fig_crit

        logger.info("Created list-linking symbolic distance plots")
    else:
        logger.info("Skipping list-linking symbolic distance plots")

    # === Cross-term ablation: all combinations of removing cross-item blocks ===
    if not should_skip('ti_agg_symbolic_distance_nonlinear'):
        n_ep = cli_args.num_aggregate_episodes
        half_h = saved_args.hidden_size // 2
        _item_size = saved_args.item_size

        # Define the weight groups whose cross blocks can be removed
        # Each group is (label, list of (param, row_split, col_split) tuples)
        cross_groups = {}
        if hasattr(model, 'embedding_layer'):
            cross_groups['embed'] = [(model.embedding_layer.weight, half_h, _item_size)]
        hidden_params = []
        for i in range(saved_args.extra_layers):
            hidden_params.append((model.extra_hidden_layers[i].weight, half_h, half_h))
        hidden_params.append((model.fc2.weight, half_h, half_h))
        cross_groups['hidden'] = hidden_params
        alpha_params = []
        if model.alpha.ndim >= 2:
            alpha_params.append((model.alpha, half_h, half_h))
        for i in range(saved_args.extra_layers):
            if model.alpha_extra[i].ndim >= 2:
                alpha_params.append((model.alpha_extra[i], half_h, half_h))
        if alpha_params:
            cross_groups['alpha'] = alpha_params
        h2r_params = [(model.hidden_to_reward.weight, half_h, half_h)]
        for i in range(saved_args.extra_layers):
            h2r_params.append((model.hidden_to_reward_extra[i].weight, half_h, half_h))
        cross_groups['h2r'] = h2r_params

        group_names = sorted(cross_groups.keys())
        logger.info(f"Cross-term ablation groups: {group_names}")

        def _zero_cross_blocks(groups_to_zero):
            """Zero out cross blocks for the given groups, return saved values."""
            saved = {}
            with torch.no_grad():
                for gname in groups_to_zero:
                    for idx, (param, rs, cs) in enumerate(cross_groups[gname]):
                        key = f'{gname}_{idx}'
                        saved[key + '_tl'] = param.data[:rs, cs:].clone()
                        saved[key + '_br'] = param.data[rs:, :cs].clone()
                        param.data[:rs, cs:].zero_()
                        param.data[rs:, :cs].zero_()
            return saved

        def _restore_cross_blocks(groups_to_restore, saved):
            """Restore cross blocks from saved values."""
            with torch.no_grad():
                for gname in groups_to_restore:
                    for idx, (param, rs, cs) in enumerate(cross_groups[gname]):
                        key = f'{gname}_{idx}'
                        param.data[:rs, cs:].copy_(saved[key + '_tl'])
                        param.data[rs:, :cs].copy_(saved[key + '_br'])

        def _plot_activation_histograms(act_data, label, fig_prefix):
            """Create per-layer activation histograms (adj vs nonadj) from accumulated counts."""
            bin_edges = act_data['bins']
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            width = bin_centers[1] - bin_centers[0]
            for L, lname in enumerate(act_data['layer_names']):
                adj_c = act_data['adj'][L]
                nonadj_c = act_data['nonadj'][L]
                adj_total = adj_c.sum()
                nonadj_total = nonadj_c.sum()
                fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
                if adj_total > 0:
                    ax.bar(bin_centers, adj_c / adj_total, width=width, alpha=0.6,
                           color='tab:blue', label=f'Adjacent (n={int(adj_total):,})')
                if nonadj_total > 0:
                    ax.bar(bin_centers, nonadj_c / nonadj_total, width=width, alpha=0.6,
                           color='tab:orange', label=f'Non-adjacent (n={int(nonadj_total):,})')
                ax.set_xlabel('Activation Value')
                ax.set_ylabel('Density')
                ax.set_title(f'{lname} Activations - {label}\n(n={n_ep} episodes)')
                ax.legend()
                plt.tight_layout()
                lname_key = lname.lower().replace(' ', '_')
                all_figures[f"{fig_prefix}/L{L}_{lname_key}"] = fig
                plt.close(fig)

        # Run regular baseline for this section
        logger.info(f"=== Nonlinear ablation: regular baseline ({n_ep} episodes) ===")
        nl_regular_zs, nl_regular_act = run_zero_shot_with_activations(
            model, saved_args, n_ep, num_train_trials, num_items, device
        )
        all_figures["ti_agg_symbolic_distance_nonlinear/regular"] = zero_shot_symbolic_distance_plot(
            nl_regular_zs, num_items, title=f'Zero-Shot SD - Regular (n={n_ep})'
        )
        _plot_activation_histograms(nl_regular_act, 'Regular',
                                    'ti_agg_symbolic_distance_nonlinear/activations_regular')

        # Run every non-empty subset of groups
        all_combo_results = {}
        for r in range(1, len(group_names) + 1):
            for combo in itertools.combinations(group_names, r):
                combo_label = '+'.join(combo)
                logger.info(f"=== Nonlinear ablation: removing {combo_label} ({n_ep} episodes) ===")

                saved_vals = _zero_cross_blocks(combo)
                combo_zs, combo_act = run_zero_shot_with_activations(
                    model, saved_args, n_ep, num_train_trials, num_items, device
                )
                _restore_cross_blocks(combo, saved_vals)

                all_combo_results[combo_label] = combo_zs
                all_figures[f"ti_agg_symbolic_distance_nonlinear/no_{combo_label}"] = zero_shot_symbolic_distance_plot(
                    combo_zs, num_items, title=f'Zero-Shot SD - No Cross: {combo_label} (n={n_ep})'
                )
                all_figures[f"ti_agg_symbolic_distance_nonlinear/delta_regular_vs_no_{combo_label}"] = delta_symbolic_distance_plot(
                    nl_regular_zs, combo_zs, num_items,
                    title=f'Delta: Regular - No Cross {combo_label} (n={n_ep})'
                )
                _plot_activation_histograms(combo_act, f'No Cross: {combo_label}',
                                            f'ti_agg_symbolic_distance_nonlinear/activations_no_{combo_label}')

        logger.info(f"Created {len(all_combo_results)} cross-term ablation combinations")
    else:
        logger.info("Skipping nonlinear cross-term ablation")

    # Safety net: filter out any figures whose section name (before first /) is in skip set
    if skip_set:
        all_figures = {k: v for k, v in all_figures.items()
                       if k.split('/')[0] not in skip_set}
        logger.info(f"After skip_plots filter: {len(all_figures)} figures remain")

    # Init wandb and log everything at once
    checkpoint_basename = os.path.splitext(os.path.basename(cli_args.checkpoint))[0]
    run_name = cli_args.wandb_run_name or f"corr_evo_{checkpoint_basename}"
    wandb.init(project="3factor", name=run_name)

    wandb_log_dict = {"checkpoint_episode": episode}
    for fig_name, fig in tqdm(all_figures.items(), desc="Converting figures for wandb"):
        wandb_log_dict[fig_name] = wandb.Image(fig)

    wandb.log(wandb_log_dict)
    logger.info(f"Logged {len(all_figures)} figures to wandb")

    wandb.finish()
    logger.info("Done")


if __name__ == "__main__":
    main()
