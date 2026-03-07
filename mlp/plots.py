import logging
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import numpy as np
import torch

logger = logging.getLogger(__name__)

from generate_data import generate_batch_items, generate_batch_trials_ti
from generate_data import generate_batch_trials_ll

def plot_pca_inputs(batch_trials, model, episode):
    # batch_trials shape batch_size x num_trials x 2*item_size
    model.eval()
    if hasattr(model, 'embedding_layer'):
        batch_trial_embeddings = model.embedding_layer(batch_trials) # shape batch_size x num_trials x hidden_size
    else:
        batch_trial_embeddings = batch_trials
    model.train()
    batch_trial_embeddings = batch_trial_embeddings.detach().cpu().numpy()


    _, _, embed_dim = batch_trial_embeddings.shape
    item_dim = embed_dim // 2

    # Split into item1 and item2
    item1 = batch_trial_embeddings[:, :, :item_dim]  # (batch, trials, item_dim)
    item2 = batch_trial_embeddings[:, :, item_dim:]  # (batch, trials, item_dim)

    # Flatten batch and trials dimensions
    item1_flat = item1.reshape(-1, item_dim)  # (batch*trials, item_dim)
    item2_flat = item2.reshape(-1, item_dim)  # (batch*trials, item_dim)

    # Stack item1 on top of item2
    batch_trial_items_embeddings = np.concatenate([item1_flat, item2_flat], axis=0)
    pca = PCA(n_components=2)
    pca.fit(batch_trial_items_embeddings)
    split_batch_trial_pca = pca.transform(batch_trial_items_embeddings)

    half = len(split_batch_trial_pca) // 2
    plt.scatter(split_batch_trial_pca[:half, 0], split_batch_trial_pca[:half, 1], c='blue', label='Item 1')
    plt.scatter(split_batch_trial_pca[half:, 0], split_batch_trial_pca[half:, 1], c='red', label='Item 2')
    plt.title(f'PCA embeddings of item dimensions Episode {episode}')
    plt.xlabel('dim 1')
    plt.ylabel('dim 2')
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()
    fig = plt.gcf()
    plt.close()
    return fig


def plot_pca_frozen_by_symbolic_distance(args, model):
    """
    Create PCA plots with frozen plastic weights, sampling trials by symbolic distance.

    1. Run training phase to build up plastic weights
    2. Freeze plastic weights (no updates after training)
    3. For each symbolic distance, sample batch_size random trials
    4. Run inference with frozen weights, collecting embeddings
    5. Fit ONE PCA per layer on all collected embeddings
    6. Plot with signed symbolic distance coloring and lines connecting pairs

    Returns a dict of figures for wandb logging.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    batch_size = args.batch_size * 4
    num_items = args.item_range[-1] - 1
    num_symbolic_distances = num_items - 1  # SD 1 through num_items-1

    # Generate batch items for training phase (different items per network for robust averaging)
    batch_items = generate_batch_items(num_items, args.item_size, batch_size, change_items_throughout_batch=True)
    trials, correct_choices, pair_indices, num_train_trials = generate_batch_trials_ti(
        batch_items, args.num_train_trials, args.num_test_trials, arbitrary=args.arbitrary
    )

    trials = torch.tensor(trials, dtype=torch.float32).to(device)
    correct_choices = torch.tensor(correct_choices, dtype=torch.float32).to(device)

    # Initialize plastic weights
    plastic_weights = torch.zeros(batch_size, args.hidden_size, args.hidden_size,
                                   dtype=torch.float32, requires_grad=False).to(device)
    extra_plastic_weights = [torch.zeros(batch_size, args.hidden_size, args.hidden_size,
                                          dtype=torch.float32, requires_grad=False).to(device)
                             for _ in range(args.extra_layers)]

    # --- Neuromodulator tracking storage ---
    # Store neuromodulator values for each trial and network
    # Shape will be (num_train_trials, batch_size) for the main neuromodulator
    neuromodulator_history = []
    # Also track actual rewards (±1): +1 if model correct, -1 if incorrect
    reward_history = []
    # Also track model's readout (sigmoid output) for each trial
    choice_history = []
    # Also track which pair was presented: (high_item_idx, low_item_idx) for each trial/network
    # pair_indices has shape (batch_size, num_trials, 2)

    # Run training phase to build up plastic weights, saving checkpoints
    checkpoint_trials = [0, 5, 10, 15, num_train_trials - 1]  # Trial indices to save
    checkpoint_trials = [t for t in checkpoint_trials if t < num_train_trials]  # Filter valid
    checkpoint_trials = sorted(set(checkpoint_trials))  # Remove duplicates, sort

    pw_checkpoints = {}  # trial_idx -> (plastic_weights, extra_plastic_weights)

    for trial in range(num_train_trials):
        # Save checkpoint BEFORE this trial (state after previous trials)
        if trial in checkpoint_trials:
            pw_checkpoints[trial] = (
                plastic_weights.clone(),
                [epw.clone() for epw in extra_plastic_weights]
            )

        batch_trial = trials[:, trial, :]
        batch_correct_choice = correct_choices[:, trial]

        with torch.inference_mode():
            output = model(batch_trial, plastic_weights, batch_correct_choice,
                          extra_plastic_weights=extra_plastic_weights, store_embeddings=False)

        # Track neuromodulator values
        nm_values = output.neuromodulator.detach().cpu().numpy()  # (batch_size, num_neuromodulators)
        neuromodulator_history.append(nm_values)

        # Track actual rewards from model output
        reward_history.append(output.reward.squeeze().detach().cpu().numpy())

        # Track model's choice (sigmoid output)
        choice_history.append(output.choice.squeeze().detach().cpu().numpy())

        plastic_weights = output.plastic_weights
        extra_plastic_weights = output.extra_plastic_weights

    # Save final state (after all training trials)
    pw_checkpoints[num_train_trials] = (
        plastic_weights.clone(),
        [epw.clone() for epw in extra_plastic_weights]
    )

    # Now freeze plastic weights (just keep references, don't update)
    frozen_plastic_weights = plastic_weights.clone()
    frozen_extra_plastic_weights = [epw.clone() for epw in extra_plastic_weights]

    # --- Plastic weight heatmaps ---
    figures = {}

    # --- Neuromodulator Analysis Plots ---
    # Convert neuromodulator history to array: (num_train_trials, batch_size, num_neuromodulators)
    neuromodulator_history = np.array(neuromodulator_history)
    num_train_trials_actual = neuromodulator_history.shape[0]

    # For TI, use the first (or only) neuromodulator
    # Squeeze to remove any extra dimensions and get shape: (num_train_trials, batch_size)
    nm_main = np.squeeze(neuromodulator_history)
    if nm_main.ndim == 3:
        nm_main = nm_main[:, :, 0]  # First neuromodulator if multiple
    elif nm_main.ndim == 1:
        nm_main = nm_main[:, np.newaxis]  # Add batch dimension if missing

    # Get pair indices for training trials only
    # pair_indices shape: (batch_size, num_total_trials, 2) where [:, :, 0] is high_item, [:, :, 1] is low_item
    train_pair_indices = pair_indices[:, :args.num_train_trials, :]  # (batch_size, num_train_trials, 2)
    # Get correct choices for training trials to determine actual presentation order
    # correct_choice=0 means high-rank first (AB), correct_choice=1 means low-rank first (BA)
    train_correct_choices = correct_choices[:, :args.num_train_trials].cpu().numpy()  # (batch_size, num_train_trials)

    # Item labels for TI
    item_labels_nm = [chr(ord('A') + i) for i in range(num_items)]

    # --- Plot 1: Heatmap of individual networks (first 10) x trials ---
    # Now includes neuromodulator, readout (sigmoid), and actual reward (±1) with pair labels
    num_networks_to_show = min(10, batch_size)

    # Convert reward history to array
    reward_history_np = np.array(reward_history)  # (num_train_trials, batch_size) or (num_train_trials, batch_size, ...)
    # Ensure 2D shape (num_train_trials, batch_size)
    # First squeeze any dimensions of size 1
    reward_history_np = np.squeeze(reward_history_np)
    # If still more than 2D, take the first slice of extra dimensions
    while reward_history_np.ndim > 2:
        reward_history_np = reward_history_np[..., 0]
    if reward_history_np.ndim == 1:
        reward_history_np = reward_history_np[:, np.newaxis]

    # Convert choice history to array
    choice_history_np = np.array(choice_history)  # (num_train_trials, batch_size)
    choice_history_np = np.squeeze(choice_history_np)
    while choice_history_np.ndim > 2:
        choice_history_np = choice_history_np[..., 0]
    if choice_history_np.ndim == 1:
        choice_history_np = choice_history_np[:, np.newaxis]

    # Create figure with 3 subplots stacked vertically
    fig_nm_heatmap, axes_nm_heatmap = plt.subplots(3, 1, figsize=(14, 12), dpi=150)

    # Panel 1: Neuromodulator values
    nm_subset = nm_main[:, :num_networks_to_show].T  # (num_networks_to_show, num_train_trials)
    vmax_nm = max(abs(nm_subset.min()), abs(nm_subset.max()))
    if vmax_nm == 0:
        vmax_nm = 1
    im_nm = axes_nm_heatmap[0].imshow(nm_subset, cmap='RdBu_r', vmin=-vmax_nm, vmax=vmax_nm, aspect='auto')
    axes_nm_heatmap[0].set_ylabel('Network')
    axes_nm_heatmap[0].set_yticks(range(num_networks_to_show))
    axes_nm_heatmap[0].set_yticklabels([f'Net {i}' for i in range(num_networks_to_show)])
    axes_nm_heatmap[0].set_title('Neuromodulator Values')
    plt.colorbar(im_nm, ax=axes_nm_heatmap[0], label='Neuromodulator')

    # Add neuromodulator values as black text inside boxes
    for net_idx in range(num_networks_to_show):
        for trial_idx in range(args.num_train_trials):
            nm_val = nm_subset[net_idx, trial_idx]
            axes_nm_heatmap[0].text(trial_idx, net_idx, f'{nm_val:.2f}',
                                    ha='center', va='center', fontsize=5, color='black', fontweight='bold')

    # Panel 2: Readout values (sigmoid output) - model's confidence
    choice_subset = choice_history_np[:, :num_networks_to_show].T  # (num_networks_to_show, num_train_trials)
    im_choice = axes_nm_heatmap[1].imshow(choice_subset, cmap='RdBu_r', vmin=0, vmax=1, aspect='auto')
    axes_nm_heatmap[1].set_ylabel('Network')
    axes_nm_heatmap[1].set_yticks(range(num_networks_to_show))
    axes_nm_heatmap[1].set_yticklabels([f'Net {i}' for i in range(num_networks_to_show)])
    axes_nm_heatmap[1].set_title('Readout (Sigmoid) - P(choose position 1)')
    plt.colorbar(im_choice, ax=axes_nm_heatmap[1], label='P(pos 1)')

    # Add readout values as black text inside boxes
    for net_idx in range(num_networks_to_show):
        for trial_idx in range(args.num_train_trials):
            choice_val = choice_subset[net_idx, trial_idx]
            axes_nm_heatmap[1].text(trial_idx, net_idx, f'{choice_val:.2f}',
                                    ha='center', va='center', fontsize=5, color='black', fontweight='bold')

    # Panel 3: Actual reward (±1) with pair labels as white text
    reward_subset = reward_history_np[:, :num_networks_to_show].T  # (num_networks_to_show, num_train_trials)
    im_reward = axes_nm_heatmap[2].imshow(reward_subset, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    axes_nm_heatmap[2].set_xlabel('Trial')
    axes_nm_heatmap[2].set_ylabel('Network')
    axes_nm_heatmap[2].set_yticks(range(num_networks_to_show))
    axes_nm_heatmap[2].set_yticklabels([f'Net {i}' for i in range(num_networks_to_show)])
    axes_nm_heatmap[2].set_title('Reward & Pair Presented (+1=correct green, -1=incorrect red)')
    plt.colorbar(im_reward, ax=axes_nm_heatmap[2], label='Reward')

    # Add pair labels as white text on the reward heatmap
    for net_idx in range(num_networks_to_show):
        for trial_idx in range(args.num_train_trials):
            high_item = int(train_pair_indices[net_idx, trial_idx, 0])
            low_item = int(train_pair_indices[net_idx, trial_idx, 1])
            # Determine actual presentation order from correct_choice
            if train_correct_choices[net_idx, trial_idx] == 0:
                pair_label = f"{item_labels_nm[high_item]}{item_labels_nm[low_item]}"
            else:
                pair_label = f"{item_labels_nm[low_item]}{item_labels_nm[high_item]}"
            axes_nm_heatmap[2].text(trial_idx, net_idx, pair_label,
                                    ha='center', va='center', fontsize=5, color='white', fontweight='bold')

    plt.suptitle(f'TI: Neuromodulator, Readout, and Reward Analysis\n(first {num_networks_to_show} networks)', fontsize=14)
    plt.tight_layout()
    figures["pca_frozen/neuromodulator_heatmap"] = fig_nm_heatmap
    plt.close(fig_nm_heatmap)

    # --- Coefficient matrix A for individual networks (to compare with neuromodulator) ---
    # Compute A_coeffs for the same networks shown in neuromodulator_heatmap
    if args.extra_layers > 0:
        # Define adjacent pairs and presentations for TI
        adjacent_pairs_ti_nm = [(i, i+1) for i in range(num_items - 1)]
        all_presentations_ti_nm = []
        presentation_to_idx_ti_nm = {}
        for pair in adjacent_pairs_ti_nm:
            winner, loser = pair
            all_presentations_ti_nm.append((winner, loser))
            presentation_to_idx_ti_nm[(winner, loser)] = len(all_presentations_ti_nm) - 1
            all_presentations_ti_nm.append((loser, winner))
            presentation_to_idx_ti_nm[(loser, winner)] = len(all_presentations_ti_nm) - 1
        num_presentations_ti_nm = len(all_presentations_ti_nm)

        # Get model parameters for coefficient tracking
        alpha_first_ti_nm = model.alpha_extra[0].detach().cpu().numpy()
        m_hebb_ti_nm = model.hebbian_trace_multiplier_extra[0].item()

        # Handle scalar alpha: if 0-d, broadcast to a vector for mean computation
        alpha_is_scalar_ti_nm = alpha_first_ti_nm.ndim == 0

        # Compute A_coeffs for each of the first num_networks_to_show networks
        all_A_coeffs_ti = []
        all_A_col_sums_ti = []

        for net_idx in range(num_networks_to_show):
            single_items_ti_nm = batch_items[net_idx]

            # Compute embeddings for this network's items
            embeddings_u_ti_nm = []
            with torch.no_grad():
                for (item1_idx, item2_idx) in all_presentations_ti_nm:
                    item1_emb = single_items_ti_nm[item1_idx]
                    item2_emb = single_items_ti_nm[item2_idx]
                    input_vec = np.concatenate([item1_emb, item2_emb])
                    input_t = torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0).to(device)
                    if hasattr(model, 'embedding_layer'):
                        u = torch.tanh(model.embedding_layer(input_t))
                    else:
                        u = input_t
                    embeddings_u_ti_nm.append(u.squeeze(0).cpu().numpy())
            embeddings_u_ti_nm = np.array(embeddings_u_ti_nm)

            # Compute D_alpha for this network
            if alpha_is_scalar_ti_nm:
                alpha_mean_ti_nm = float(alpha_first_ti_nm)
            else:
                alpha_mean_ti_nm = alpha_first_ti_nm.mean(axis=0)
            D_alpha_ti_nm = (embeddings_u_ti_nm * alpha_mean_ti_nm) @ embeddings_u_ti_nm.T

            # Initialize and track coefficients
            A_coeffs_ti_nm = np.zeros((num_presentations_ti_nm, num_presentations_ti_nm))
            pw_track_ti_nm = torch.zeros(1, args.hidden_size, args.hidden_size, dtype=torch.float32).to(device)
            epw_track_ti_nm = [torch.zeros(1, args.hidden_size, args.hidden_size, dtype=torch.float32).to(device)
                               for _ in range(args.extra_layers)]

            single_trials_ti_nm = trials[net_idx:net_idx+1, :, :]
            single_correct_ti_nm = correct_choices[net_idx:net_idx+1, :]

            for trial_idx in range(args.num_train_trials):
                trial_input = single_trials_ti_nm[:, trial_idx, :]
                trial_correct = single_correct_ti_nm[:, trial_idx]

                item_size = args.item_size
                item1_emb_trial = trial_input[0, :item_size].cpu().numpy()
                item2_emb_trial = trial_input[0, item_size:2*item_size].cpu().numpy()

                item1_idx_found = None
                item2_idx_found = None
                for idx in range(num_items):
                    if np.allclose(single_items_ti_nm[idx], item1_emb_trial, atol=1e-5):
                        item1_idx_found = idx
                    if np.allclose(single_items_ti_nm[idx], item2_emb_trial, atol=1e-5):
                        item2_idx_found = idx

                if item1_idx_found is None or item2_idx_found is None:
                    with torch.no_grad():
                        output_track = model(trial_input, pw_track_ti_nm, trial_correct,
                                            extra_plastic_weights=epw_track_ti_nm, store_embeddings=False)
                    pw_track_ti_nm = output_track.plastic_weights
                    epw_track_ti_nm = output_track.extra_plastic_weights
                    continue

                presentation_key = (item1_idx_found, item2_idx_found)
                if presentation_key not in presentation_to_idx_ti_nm:
                    with torch.no_grad():
                        output_track = model(trial_input, pw_track_ti_nm, trial_correct,
                                            extra_plastic_weights=epw_track_ti_nm, store_embeddings=False)
                    pw_track_ti_nm = output_track.plastic_weights
                    epw_track_ti_nm = output_track.extra_plastic_weights
                    continue

                k = presentation_to_idx_ti_nm[presentation_key]

                with torch.no_grad():
                    output_track = model(trial_input, pw_track_ti_nm, trial_correct,
                                        extra_plastic_weights=epw_track_ti_nm, store_embeddings=False)

                nm_output = output_track.neuromodulator.squeeze()
                if args.use_extra_neuromodulator and args.extra_layers > 0:
                    eta_t = nm_output[0].item() if nm_output.dim() > 0 else nm_output.item()
                else:
                    eta_t = nm_output.item() if nm_output.dim() == 0 else nm_output[0].item()

                e_k = np.zeros(num_presentations_ti_nm)
                e_k[k] = 1.0
                tanh_scale = 0.9
                A_coeffs_ti_nm[:, k] += eta_t * m_hebb_ti_nm * tanh_scale * (e_k + A_coeffs_ti_nm @ D_alpha_ti_nm[:, k])

                pw_track_ti_nm = output_track.plastic_weights
                epw_track_ti_nm = output_track.extra_plastic_weights

            all_A_coeffs_ti.append(A_coeffs_ti_nm)
            all_A_col_sums_ti.append(A_coeffs_ti_nm.sum(axis=0))

        # Create figure showing A_col_sums for each network
        presentation_labels_ti_nm = []
        for (i1, i2) in all_presentations_ti_nm:
            presentation_labels_ti_nm.append(f"{item_labels_nm[i1]}{item_labels_nm[i2]}")

        fig_coeffs_networks_ti, axes_coeffs_ti = plt.subplots(2, 1, figsize=(14, 10), dpi=150)

        # Panel 1: A_col_sums heatmap (networks x presentations)
        A_col_sums_matrix_ti = np.array(all_A_col_sums_ti)
        vmax_col_ti = max(abs(A_col_sums_matrix_ti.min()), abs(A_col_sums_matrix_ti.max()))
        if vmax_col_ti == 0:
            vmax_col_ti = 1
        im_cols_ti = axes_coeffs_ti[0].imshow(A_col_sums_matrix_ti, cmap='RdBu_r', vmin=-vmax_col_ti, vmax=vmax_col_ti, aspect='auto')
        axes_coeffs_ti[0].set_xlabel('Adjacent Pair Presentation')
        axes_coeffs_ti[0].set_ylabel('Network')
        axes_coeffs_ti[0].set_xticks(range(num_presentations_ti_nm))
        axes_coeffs_ti[0].set_xticklabels(presentation_labels_ti_nm, rotation=45, ha='right', fontsize=8)
        axes_coeffs_ti[0].set_yticks(range(num_networks_to_show))
        axes_coeffs_ti[0].set_yticklabels([f'Net {i}' for i in range(num_networks_to_show)])
        axes_coeffs_ti[0].set_title('A_col_sums = Σ_i A[i,j] for each presentation j\n(Total coefficient weight per adjacent pair)')
        plt.colorbar(im_cols_ti, ax=axes_coeffs_ti[0], label='Column Sum')

        # Panel 2: Full A_coeffs matrices for first 3 networks side by side
        num_to_show_detail_ti = min(3, num_networks_to_show)
        axes_coeffs_ti[1].axis('off')
        for i in range(num_to_show_detail_ti):
            left = 0.05 + i * 0.32
            ax_inset = fig_coeffs_networks_ti.add_axes([left, 0.05, 0.28, 0.35])
            vmax_a_ti = max(abs(all_A_coeffs_ti[i].min()), abs(all_A_coeffs_ti[i].max()))
            if vmax_a_ti == 0:
                vmax_a_ti = 1
            im_a_ti = ax_inset.imshow(all_A_coeffs_ti[i], cmap='RdBu_r', vmin=-vmax_a_ti, vmax=vmax_a_ti, aspect='equal')
            ax_inset.set_xticks(range(0, num_presentations_ti_nm, 2))
            ax_inset.set_xticklabels([presentation_labels_ti_nm[j] for j in range(0, num_presentations_ti_nm, 2)], rotation=45, ha='right', fontsize=6)
            ax_inset.set_yticks(range(0, num_presentations_ti_nm, 2))
            ax_inset.set_yticklabels([presentation_labels_ti_nm[j] for j in range(0, num_presentations_ti_nm, 2)], fontsize=6)
            ax_inset.set_title(f'Net {i}: A[i,j]', fontsize=10)
            plt.colorbar(im_a_ti, ax=ax_inset, fraction=0.046)

        plt.suptitle('TI: Coefficient Matrices by Network\n(Compare with neuromodulator_heatmap to see correlation)', fontsize=12)
        figures["pca_frozen/pw_decomposition_coefficients_by_network"] = fig_coeffs_networks_ti
        plt.close(fig_coeffs_networks_ti)

    from matplotlib.colors import TwoSlopeNorm

    # Extract alpha parameters and innate weights from model
    alpha_final = model.alpha.detach().cpu().numpy()  # Shape: (hidden_size, hidden_size)
    W_innate_final = model.fc2.weight.detach().cpu().numpy()  # Shape: (hidden_size, hidden_size)

    alpha_extra_list = [model.alpha_extra[i].detach().cpu().numpy() for i in range(args.extra_layers)]
    W_innate_extra_list = [model.extra_hidden_layers[i].weight.detach().cpu().numpy() for i in range(args.extra_layers)]

    # Main plastic weights (mean across batch)
    pw_mean = frozen_plastic_weights.detach().cpu().numpy().mean(axis=0)  # Shape: (hidden_size, hidden_size)

    # Compute alpha-modulated plastic weights and effective weights for final layer
    alpha_pw_final = alpha_final * pw_mean  # Hadamard product
    W_effective_final = W_innate_final + alpha_pw_final

    # Helper function to create heatmap
    def create_heatmap(matrix, title, filename):
        fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
        vmax = max(abs(matrix.min()), abs(matrix.max()))
        if vmax == 0:
            vmax = 1
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(matrix, cmap='RdBu_r', norm=norm, aspect='equal')
        ax.set_xlabel('Pre-synaptic (input)')
        ax.set_ylabel('Post-synaptic (output)')
        ax.set_title(title)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Weight')
        plt.tight_layout()
        figures[filename] = fig
        plt.close(fig)

    # Final layer heatmaps
    create_heatmap(pw_mean, 'Plastic Weights (Mean) - Final Layer', 'pca_frozen/plastic_weights_final')
    if alpha_final.ndim >= 2:
        create_heatmap(alpha_final, 'Alpha (Meta-learned) - Final Layer', 'pca_frozen/alpha_final')
        create_heatmap(alpha_pw_final, 'Alpha × Plastic Weights - Final Layer', 'pca_frozen/alpha_pw_final')
    # else: alpha is scalar, skip matrix heatmaps (alpha_pw_final still used for effective weights)
    create_heatmap(W_innate_final, 'Innate Weights (fc2) - Final Layer', 'pca_frozen/innate_weights_final')
    create_heatmap(W_effective_final, 'Effective Weights (Innate + α×PW) - Final Layer', 'pca_frozen/effective_weights_final')

    # Extra plastic weights heatmaps for each layer
    alpha_pw_extra_list = []
    W_effective_extra_list = []

    for layer_idx, epw in enumerate(frozen_extra_plastic_weights):
        epw_mean = epw.detach().cpu().numpy().mean(axis=0)
        alpha_extra = alpha_extra_list[layer_idx]
        W_innate_extra = W_innate_extra_list[layer_idx]

        alpha_pw_extra = alpha_extra * epw_mean
        W_effective_extra = W_innate_extra + alpha_pw_extra

        alpha_pw_extra_list.append(alpha_pw_extra)
        W_effective_extra_list.append(W_effective_extra)

        create_heatmap(epw_mean, f'Plastic Weights (Mean) - Hidden Layer {layer_idx + 1}',
                       f'pca_frozen/plastic_weights_hidden{layer_idx + 1}')
        if alpha_extra.ndim >= 2:
            create_heatmap(alpha_extra, f'Alpha (Meta-learned) - Hidden Layer {layer_idx + 1}',
                           f'pca_frozen/alpha_hidden{layer_idx + 1}')
            create_heatmap(alpha_pw_extra, f'Alpha × Plastic Weights - Hidden Layer {layer_idx + 1}',
                           f'pca_frozen/alpha_pw_hidden{layer_idx + 1}')
        create_heatmap(W_innate_extra, f'Innate Weights - Hidden Layer {layer_idx + 1}',
                       f'pca_frozen/innate_weights_hidden{layer_idx + 1}')
        create_heatmap(W_effective_extra, f'Effective Weights (Innate + α×PW) - Hidden Layer {layer_idx + 1}',
                       f'pca_frozen/effective_weights_hidden{layer_idx + 1}')

    # --- Histograms for alpha and alpha*pw values ---
    # Helper function to create histogram
    def create_histogram(values, title, filename, bins=100):
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        ax.hist(values.flatten(), bins=bins, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='zero')
        ax.axvline(x=values.mean(), color='green', linestyle='-', alpha=0.7, label=f'mean={values.mean():.4f}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
        ax.set_title(f'{title}\n(n={values.size:,}, std={values.std():.4f})')
        ax.legend()
        plt.tight_layout()
        figures[filename] = fig
        plt.close(fig)

    # Helper function to create tail histogram (values beyond 3 STD)
    def create_tail_histogram(values, title, filename, num_std=3, bins=50):
        flat_values = values.flatten()
        mean_val = flat_values.mean()
        std_val = flat_values.std()
        threshold = num_std * std_val

        # Filter for outliers (beyond 3 STD from mean)
        outliers = flat_values[np.abs(flat_values - mean_val) > threshold]

        if len(outliers) == 0:
            return  # No outliers to plot

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

        # Left tail (values < mean - 3*std)
        left_tail = flat_values[flat_values < mean_val - threshold]
        ax_left = axes[0]
        if len(left_tail) > 0:
            ax_left.hist(left_tail, bins=bins, edgecolor='black', alpha=0.7, color='tab:blue')
            ax_left.axvline(x=mean_val - threshold, color='red', linestyle='--',
                           label=f'-{num_std}σ = {mean_val - threshold:.4f}')
            ax_left.set_xlabel('Value')
            ax_left.set_ylabel('Count')
            ax_left.set_title(f'Left Tail (< -{num_std}σ)\nn={len(left_tail):,} ({100*len(left_tail)/len(flat_values):.2f}%)')
            ax_left.legend()
        else:
            ax_left.text(0.5, 0.5, 'No values in left tail', ha='center', va='center', transform=ax_left.transAxes)
            ax_left.set_title(f'Left Tail (< -{num_std}σ)\nn=0')

        # Right tail (values > mean + 3*std)
        right_tail = flat_values[flat_values > mean_val + threshold]
        ax_right = axes[1]
        if len(right_tail) > 0:
            ax_right.hist(right_tail, bins=bins, edgecolor='black', alpha=0.7, color='tab:orange')
            ax_right.axvline(x=mean_val + threshold, color='red', linestyle='--',
                            label=f'+{num_std}σ = {mean_val + threshold:.4f}')
            ax_right.set_xlabel('Value')
            ax_right.set_ylabel('Count')
            ax_right.set_title(f'Right Tail (> +{num_std}σ)\nn={len(right_tail):,} ({100*len(right_tail)/len(flat_values):.2f}%)')
            ax_right.legend()
        else:
            ax_right.text(0.5, 0.5, 'No values in right tail', ha='center', va='center', transform=ax_right.transAxes)
            ax_right.set_title(f'Right Tail (> +{num_std}σ)\nn=0')

        fig.suptitle(f'{title} - Outliers (|x - μ| > {num_std}σ)\nμ={mean_val:.4f}, σ={std_val:.4f}', fontsize=12)
        plt.tight_layout()
        figures[filename] = fig
        plt.close(fig)

    # Alpha histograms (same for TI and LL, so only in pca_frozen)
    if alpha_final.ndim >= 2:
        create_histogram(alpha_final, 'Alpha - Final Layer', 'pca_frozen/alpha_final_histogram')
        create_tail_histogram(alpha_final, 'Alpha - Final Layer', 'pca_frozen/alpha_final_histogram_tails')
    for layer_idx in range(args.extra_layers):
        if alpha_extra_list[layer_idx].ndim >= 2:
            create_histogram(alpha_extra_list[layer_idx], f'Alpha - Hidden Layer {layer_idx + 1}',
                            f'pca_frozen/alpha_hidden{layer_idx + 1}_histogram')

    # Alpha × PW histograms for TI
    if alpha_final.ndim >= 2:
        create_histogram(alpha_pw_final, 'Alpha × PW (TI) - Final Layer', 'pca_frozen/alpha_pw_final_histogram')
        create_tail_histogram(alpha_pw_final, 'Alpha × PW (TI) - Final Layer', 'pca_frozen/alpha_pw_final_histogram_tails')
    for layer_idx in range(args.extra_layers):
        if alpha_extra_list[layer_idx].ndim >= 2:
            create_histogram(alpha_pw_extra_list[layer_idx], f'Alpha × PW (TI) - Hidden Layer {layer_idx + 1}',
                            f'pca_frozen/alpha_pw_hidden{layer_idx + 1}_histogram')

    # --- Innate Weights Item 1 vs Item 2 Scatter Plots ---
    # For embedding layer: compare weights for item 1 positions vs item 2 positions
    # For hidden layers: compare left half vs right half (paired by shifting by half the input dimension)

    # Embedding layer scatter plot
    if hasattr(model, 'embedding_layer'):
        W_embed = model.embedding_layer.weight.detach().cpu().numpy()  # (hidden_size, 2*item_size)
        item_size = args.item_size
        W_embed_item1 = W_embed[:, :item_size]  # (hidden_size, item_size)
        W_embed_item2 = W_embed[:, item_size:]  # (hidden_size, item_size)

        fig_embed_scatter, ax_embed_scatter = plt.subplots(figsize=(8, 8), dpi=150)
        x_flat = W_embed_item1.flatten()
        y_flat = W_embed_item2.flatten()
        ax_embed_scatter.scatter(x_flat, y_flat, alpha=0.5, s=10)
        # Add diagonal line for reference
        lim_min = min(x_flat.min(), y_flat.min())
        lim_max = max(x_flat.max(), y_flat.max())
        ax_embed_scatter.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', alpha=0.5, label='y=x')
        # Add line of best fit and R² value
        slope, intercept = np.polyfit(x_flat, y_flat, 1)
        y_pred = slope * x_flat + intercept
        ss_res = np.sum((y_flat - y_pred) ** 2)
        ss_tot = np.sum((y_flat - np.mean(y_flat)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        x_line = np.array([lim_min, lim_max])
        y_line = slope * x_line + intercept
        ax_embed_scatter.plot(x_line, y_line, 'g-', linewidth=2, label=f'Best fit: y={slope:.3f}x+{intercept:.3f}\n$R^2$={r_squared:.4f}')
        ax_embed_scatter.set_xlabel('Item 1 Weights')
        ax_embed_scatter.set_ylabel('Item 2 Weights')
        ax_embed_scatter.set_title('Embedding Layer: Item 1 vs Item 2 Innate Weights')
        ax_embed_scatter.set_aspect('equal')
        ax_embed_scatter.legend()
        plt.tight_layout()
        figures["pca_frozen/innate_weights_item1_vs_item2_embedding"] = fig_embed_scatter
        plt.close(fig_embed_scatter)

    # Extra hidden layers scatter plots
    for layer_idx in range(args.extra_layers):
        W_hidden = W_innate_extra_list[layer_idx]  # (hidden_size, hidden_size)
        half_size = args.hidden_size // 2
        W_hidden_left = W_hidden[:, :half_size]  # (hidden_size, hidden_size//2)
        W_hidden_right = W_hidden[:, half_size:]  # (hidden_size, hidden_size//2)

        fig_hidden_scatter, ax_hidden_scatter = plt.subplots(figsize=(8, 8), dpi=150)
        ax_hidden_scatter.scatter(W_hidden_left.flatten(), W_hidden_right.flatten(), alpha=0.5, s=10)
        lim_min = min(W_hidden_left.min(), W_hidden_right.min())
        lim_max = max(W_hidden_left.max(), W_hidden_right.max())
        ax_hidden_scatter.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', alpha=0.5, label='y=x')
        ax_hidden_scatter.set_xlabel('Left Half Weights (Item 1)')
        ax_hidden_scatter.set_ylabel('Right Half Weights (Item 2)')
        ax_hidden_scatter.set_title(f'Hidden Layer {layer_idx + 1}: Item 1 vs Item 2 Innate Weights')
        ax_hidden_scatter.set_aspect('equal')
        ax_hidden_scatter.legend()
        plt.tight_layout()
        figures[f"pca_frozen/innate_weights_item1_vs_item2_hidden{layer_idx + 1}"] = fig_hidden_scatter
        plt.close(fig_hidden_scatter)

    # Final layer (fc2) scatter plot
    half_size = args.hidden_size // 2
    W_final_left = W_innate_final[:, :half_size]  # (hidden_size, hidden_size//2)
    W_final_right = W_innate_final[:, half_size:]  # (hidden_size, hidden_size//2)

    fig_final_scatter, ax_final_scatter = plt.subplots(figsize=(8, 8), dpi=150)
    ax_final_scatter.scatter(W_final_left.flatten(), W_final_right.flatten(), alpha=0.5, s=10)
    lim_min = min(W_final_left.min(), W_final_right.min())
    lim_max = max(W_final_left.max(), W_final_right.max())
    ax_final_scatter.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', alpha=0.5, label='y=x')
    ax_final_scatter.set_xlabel('Left Half Weights (Item 1)')
    ax_final_scatter.set_ylabel('Right Half Weights (Item 2)')
    ax_final_scatter.set_title('Final Layer (fc2): Item 1 vs Item 2 Innate Weights')
    ax_final_scatter.set_aspect('equal')
    ax_final_scatter.legend()
    plt.tight_layout()
    figures["pca_frozen/innate_weights_item1_vs_item2_final"] = fig_final_scatter
    plt.close(fig_final_scatter)

    # --- Readout and Reward Layer Weight Heatmaps ---
    # Choice (readout) weights: shape (1, hidden_size)
    W_choice = model.choice.weight.detach().cpu().numpy()  # (1, hidden_size)
    fig_choice, ax_choice = plt.subplots(figsize=(12, 2), dpi=150)
    vmax_choice = max(abs(W_choice.min()), abs(W_choice.max()))
    if vmax_choice == 0:
        vmax_choice = 1
    im_choice = ax_choice.imshow(W_choice, cmap='RdBu_r', vmin=-vmax_choice, vmax=vmax_choice, aspect='auto')
    ax_choice.set_xlabel('Hidden Dimension')
    ax_choice.set_ylabel('Output')
    ax_choice.set_yticks([0])
    ax_choice.set_yticklabels(['Choice'])
    ax_choice.set_title(f'Choice (Readout) Weights\n(shape: {W_choice.shape}, mean={W_choice.mean():.4f}, std={W_choice.std():.4f})')
    plt.colorbar(im_choice, ax=ax_choice, label='Weight')
    plt.tight_layout()
    figures["pca_frozen/choice_weights_heatmap"] = fig_choice
    plt.close(fig_choice)

    # Sorted bar chart for readout weights
    W_choice_flat = W_choice.flatten()
    sorted_indices = np.argsort(W_choice_flat)
    sorted_weights = W_choice_flat[sorted_indices]

    fig_choice_sorted, ax_choice_sorted = plt.subplots(figsize=(12, 5), dpi=150)
    colors_sorted = ['tab:red' if w < 0 else 'tab:blue' for w in sorted_weights]
    ax_choice_sorted.bar(range(len(sorted_weights)), sorted_weights, color=colors_sorted, width=1.0, edgecolor='none')
    ax_choice_sorted.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax_choice_sorted.set_xlabel('Neuron (sorted by weight)')
    ax_choice_sorted.set_ylabel('Readout Weight')
    ax_choice_sorted.set_title(f'Readout Weights (Sorted)\n(mean={W_choice_flat.mean():.4f}, std={W_choice_flat.std():.4f})')
    plt.tight_layout()
    figures["pca_frozen/choice_weights_sorted_barchart"] = fig_choice_sorted
    plt.close(fig_choice_sorted)

    # Histogram of readout weights
    fig_choice_hist, ax_choice_hist = plt.subplots(figsize=(8, 6), dpi=150)
    ax_choice_hist.hist(W_choice_flat, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax_choice_hist.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax_choice_hist.axvline(x=W_choice_flat.mean(), color='green', linestyle='-', linewidth=2, label=f'mean={W_choice_flat.mean():.4f}')
    ax_choice_hist.set_xlabel('Readout Weight')
    ax_choice_hist.set_ylabel('Count')
    ax_choice_hist.set_title(f'Readout Weights Distribution\n(n={len(W_choice_flat)}, std={W_choice_flat.std():.4f})')
    ax_choice_hist.legend()
    plt.tight_layout()
    figures["pca_frozen/choice_weights_histogram"] = fig_choice_hist
    plt.close(fig_choice_hist)

    # Analysis: Readout weights vs Alpha*PW row sums (only if extra_layers > 0)
    if args.extra_layers > 0:
        # Use the first network's frozen plastic weights
        # alpha_pw for extra layer: alpha_extra[0] * extra_plastic_weights[0]
        alpha_extra_0 = model.alpha_extra[0].detach().cpu().numpy()  # (hidden_size, hidden_size)
        epw_0 = frozen_extra_plastic_weights[0][0].detach().cpu().numpy()  # (hidden_size, hidden_size) for network 0
        alpha_pw_extra = alpha_extra_0 * epw_0  # element-wise

        # Row sum of |alpha * pw| = total incoming weight magnitude per output neuron
        row_sums_alpha_pw = np.sum(np.abs(alpha_pw_extra), axis=1)  # (hidden_size,)

        # Correlation
        abs_readout = np.abs(W_choice_flat)
        correlation = np.corrcoef(abs_readout, row_sums_alpha_pw)[0, 1]

        # Scatter plot: |readout_weight| vs row_sum(|alpha * pw|)
        fig_scatter, ax_scatter = plt.subplots(figsize=(8, 8), dpi=150)
        ax_scatter.scatter(abs_readout, row_sums_alpha_pw, alpha=0.6, s=20)
        ax_scatter.set_xlabel('|Readout Weight|')
        ax_scatter.set_ylabel('Row Sum of |Alpha * PW| (Extra Layer 1)')
        ax_scatter.set_title(f'Readout Weight Magnitude vs Alpha*PW Incoming Weight\n(Correlation: r={correlation:.4f})')

        # Add line of best fit
        slope, intercept = np.polyfit(abs_readout, row_sums_alpha_pw, 1)
        x_line = np.array([abs_readout.min(), abs_readout.max()])
        y_line = slope * x_line + intercept
        ax_scatter.plot(x_line, y_line, 'r-', linewidth=2, label=f'Best fit: y={slope:.2f}x+{intercept:.2f}')
        ax_scatter.legend()
        plt.tight_layout()
        figures["pca_frozen/readout_vs_alpha_pw_scatter"] = fig_scatter
        plt.close(fig_scatter)

        # === PATHWAY ANALYSIS: Embedding → Extra Layer → Final Layer → Readout ===

        # --- Embedding layer output importance (row sums) ---
        if hasattr(model, 'embedding_layer'):
            W_embed = model.embedding_layer.weight.detach().cpu().numpy()  # (hidden_size, 2*item_size)
            embed_row_sums = np.sum(np.abs(W_embed), axis=1)  # (hidden_size,) - output importance

            # Sorted bar chart of embedding row sums
            sorted_idx_embed = np.argsort(embed_row_sums)
            fig_embed_rows, ax_embed_rows = plt.subplots(figsize=(12, 5), dpi=150)
            ax_embed_rows.bar(range(len(embed_row_sums)), embed_row_sums[sorted_idx_embed], color='steelblue', width=1.0)
            ax_embed_rows.set_xlabel('Neuron (sorted by row sum)')
            ax_embed_rows.set_ylabel('Row Sum of |W_embedding|')
            ax_embed_rows.set_title('Embedding Layer Output Importance\n(Row sums = total outgoing weight per output dimension)')
            plt.tight_layout()
            figures["pca_frozen/pathway_embed_row_sums"] = fig_embed_rows
            plt.close(fig_embed_rows)

            # --- Extra layer input importance (column sums) ---
            W_extra_innate = model.extra_hidden_layers[0].weight.detach().cpu().numpy()  # (hidden_size, hidden_size)
            extra_innate_col_sums = np.sum(np.abs(W_extra_innate), axis=0)  # (hidden_size,) - input importance
            if alpha_extra_0.ndim >= 2:
                alpha_extra_col_sums = np.sum(np.abs(alpha_extra_0), axis=0)  # (hidden_size,)
            alpha_pw_extra_col_sums = np.sum(np.abs(alpha_pw_extra), axis=0)  # (hidden_size,)

            # Scatter plots: Embedding output importance vs Extra layer input importance
            fig_pathway1, axes_pathway1 = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

            # Innate
            corr_innate = np.corrcoef(embed_row_sums, extra_innate_col_sums)[0, 1]
            axes_pathway1[0].scatter(embed_row_sums, extra_innate_col_sums, alpha=0.6, s=20)
            axes_pathway1[0].set_xlabel('Embedding Row Sum')
            axes_pathway1[0].set_ylabel('Extra Layer Innate Col Sum')
            axes_pathway1[0].set_title(f'Innate Weights\nr={corr_innate:.4f}')

            # Alpha (skip if scalar)
            if alpha_extra_0.ndim >= 2:
                corr_alpha = np.corrcoef(embed_row_sums, alpha_extra_col_sums)[0, 1]
                axes_pathway1[1].scatter(embed_row_sums, alpha_extra_col_sums, alpha=0.6, s=20, color='orange')
                axes_pathway1[1].set_xlabel('Embedding Row Sum')
                axes_pathway1[1].set_ylabel('Extra Layer Alpha Col Sum')
                axes_pathway1[1].set_title(f'Alpha Only\nr={corr_alpha:.4f}')
            else:
                axes_pathway1[1].set_title(f'Alpha Only\n(scalar alpha, skipped)')
                axes_pathway1[1].text(0.5, 0.5, f'Scalar alpha = {float(alpha_extra_0):.4f}', ha='center', va='center', transform=axes_pathway1[1].transAxes)

            # Alpha * PW
            corr_alpha_pw = np.corrcoef(embed_row_sums, alpha_pw_extra_col_sums)[0, 1]
            axes_pathway1[2].scatter(embed_row_sums, alpha_pw_extra_col_sums, alpha=0.6, s=20, color='green')
            axes_pathway1[2].set_xlabel('Embedding Row Sum')
            axes_pathway1[2].set_ylabel('Extra Layer Alpha*PW Col Sum')
            axes_pathway1[2].set_title(f'Alpha * PW\nr={corr_alpha_pw:.4f}')

            fig_pathway1.suptitle('Pathway: Embedding → Extra Layer\n(Output importance vs Input importance)', fontsize=14)
            plt.tight_layout()
            figures["pca_frozen/pathway_embed_to_extra"] = fig_pathway1
            plt.close(fig_pathway1)

        # --- Extra layer output importance (row sums) ---
        # We already have row_sums_alpha_pw for the extra layer output
        # But let's also compute for innate and alpha separately
        extra_innate_row_sums = np.sum(np.abs(W_extra_innate), axis=1)  # (hidden_size,)
        if alpha_extra_0.ndim >= 2:
            alpha_extra_row_sums = np.sum(np.abs(alpha_extra_0), axis=1)  # (hidden_size,)
        # row_sums_alpha_pw is already computed above

        # --- Final layer input importance (column sums) ---
        W_final_innate = model.fc2.weight.detach().cpu().numpy()  # (hidden_size, hidden_size)
        alpha_final_pw = model.alpha.detach().cpu().numpy()
        pw_final_0 = frozen_plastic_weights[0].detach().cpu().numpy()  # (hidden_size, hidden_size) for network 0
        alpha_pw_final = alpha_final_pw * pw_final_0  # element-wise

        final_innate_col_sums = np.sum(np.abs(W_final_innate), axis=0)  # (hidden_size,)
        if alpha_final_pw.ndim >= 2:
            alpha_final_col_sums = np.sum(np.abs(alpha_final_pw), axis=0)  # (hidden_size,)
        alpha_pw_final_col_sums = np.sum(np.abs(alpha_pw_final), axis=0)  # (hidden_size,)

        # Scatter plots: Extra layer output importance vs Final layer input importance
        fig_pathway2, axes_pathway2 = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

        # Using extra layer alpha*pw row sums as output importance
        # Innate
        corr_innate2 = np.corrcoef(row_sums_alpha_pw, final_innate_col_sums)[0, 1]
        axes_pathway2[0].scatter(row_sums_alpha_pw, final_innate_col_sums, alpha=0.6, s=20)
        axes_pathway2[0].set_xlabel('Extra Layer Alpha*PW Row Sum')
        axes_pathway2[0].set_ylabel('Final Layer Innate Col Sum')
        axes_pathway2[0].set_title(f'Innate Weights\nr={corr_innate2:.4f}')

        # Alpha (skip if scalar)
        if alpha_final_pw.ndim >= 2:
            corr_alpha2 = np.corrcoef(row_sums_alpha_pw, alpha_final_col_sums)[0, 1]
            axes_pathway2[1].scatter(row_sums_alpha_pw, alpha_final_col_sums, alpha=0.6, s=20, color='orange')
            axes_pathway2[1].set_xlabel('Extra Layer Alpha*PW Row Sum')
            axes_pathway2[1].set_ylabel('Final Layer Alpha Col Sum')
            axes_pathway2[1].set_title(f'Alpha Only\nr={corr_alpha2:.4f}')
        else:
            axes_pathway2[1].set_title(f'Alpha Only\n(scalar alpha, skipped)')
            axes_pathway2[1].text(0.5, 0.5, f'Scalar alpha = {float(alpha_final_pw):.4f}', ha='center', va='center', transform=axes_pathway2[1].transAxes)

        # Alpha * PW
        corr_alpha_pw2 = np.corrcoef(row_sums_alpha_pw, alpha_pw_final_col_sums)[0, 1]
        axes_pathway2[2].scatter(row_sums_alpha_pw, alpha_pw_final_col_sums, alpha=0.6, s=20, color='green')
        axes_pathway2[2].set_xlabel('Extra Layer Alpha*PW Row Sum')
        axes_pathway2[2].set_ylabel('Final Layer Alpha*PW Col Sum')
        axes_pathway2[2].set_title(f'Alpha * PW\nr={corr_alpha_pw2:.4f}')

        fig_pathway2.suptitle('Pathway: Extra Layer → Final Layer\n(Output importance vs Input importance)', fontsize=14)
        plt.tight_layout()
        figures["pca_frozen/pathway_extra_to_final"] = fig_pathway2
        plt.close(fig_pathway2)

        # --- Final layer output importance vs Readout ---
        # Row sums of final layer alpha*pw
        alpha_pw_final_row_sums = np.sum(np.abs(alpha_pw_final), axis=1)  # (hidden_size,)

        # We already have the readout vs extra layer alpha*pw scatter
        # Add one for final layer alpha*pw vs readout
        corr_final_readout = np.corrcoef(alpha_pw_final_row_sums, abs_readout)[0, 1]
        fig_pathway3, ax_pathway3 = plt.subplots(figsize=(8, 8), dpi=150)
        ax_pathway3.scatter(alpha_pw_final_row_sums, abs_readout, alpha=0.6, s=20)
        ax_pathway3.set_xlabel('Final Layer Alpha*PW Row Sum')
        ax_pathway3.set_ylabel('|Readout Weight|')
        ax_pathway3.set_title(f'Pathway: Final Layer → Readout\n(Correlation: r={corr_final_readout:.4f})')
        slope3, intercept3 = np.polyfit(alpha_pw_final_row_sums, abs_readout, 1)
        x_line3 = np.array([alpha_pw_final_row_sums.min(), alpha_pw_final_row_sums.max()])
        y_line3 = slope3 * x_line3 + intercept3
        ax_pathway3.plot(x_line3, y_line3, 'r-', linewidth=2, label=f'Best fit: y={slope3:.3f}x+{intercept3:.3f}')
        ax_pathway3.legend()
        plt.tight_layout()
        figures["pca_frozen/pathway_final_to_readout"] = fig_pathway3
        plt.close(fig_pathway3)

        # --- Summary: Full pathway correlation chain ---
        fig_summary, ax_summary = plt.subplots(figsize=(10, 6), dpi=150)
        pathway_labels = [
            'Embed→Extra\n(Innate)',
            'Embed→Extra\n(Alpha)',
            'Embed→Extra\n(Alpha*PW)',
            'Extra→Final\n(Innate)',
            'Extra→Final\n(Alpha)',
            'Extra→Final\n(Alpha*PW)',
            'Final→Readout'
        ]
        pathway_corrs = [corr_innate, corr_alpha, corr_alpha_pw, corr_innate2, corr_alpha2, corr_alpha_pw2, corr_final_readout]
        colors_summary = ['steelblue', 'orange', 'green', 'steelblue', 'orange', 'green', 'purple']
        bars = ax_summary.bar(range(len(pathway_corrs)), pathway_corrs, color=colors_summary)
        ax_summary.set_xticks(range(len(pathway_labels)))
        ax_summary.set_xticklabels(pathway_labels, rotation=45, ha='right')
        ax_summary.set_ylabel('Correlation (r)')
        ax_summary.set_title('Pathway Correlation Summary\n(Blue=Innate, Orange=Alpha, Green=Alpha*PW, Purple=Readout)')
        ax_summary.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax_summary.set_ylim(-1, 1)
        # Add correlation values on bars
        for bar, corr in zip(bars, pathway_corrs):
            ax_summary.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f'{corr:.2f}', ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        figures["pca_frozen/pathway_correlation_summary"] = fig_summary
        plt.close(fig_summary)

    # Reward linear weights: shape (hidden_size, reward_input_size) where reward_input_size is 1 or 2
    W_reward_linear = model.reward_linear.weight.detach().cpu().numpy()  # (hidden_size, reward_input_size)
    fig_reward_linear, ax_reward_linear = plt.subplots(figsize=(4, 10), dpi=150)
    vmax_reward_linear = max(abs(W_reward_linear.min()), abs(W_reward_linear.max()))
    if vmax_reward_linear == 0:
        vmax_reward_linear = 1
    im_reward_linear = ax_reward_linear.imshow(W_reward_linear, cmap='RdBu_r', vmin=-vmax_reward_linear, vmax=vmax_reward_linear, aspect='auto')
    ax_reward_linear.set_xlabel('Reward Input Dim')
    ax_reward_linear.set_ylabel('Hidden Dimension')
    if W_reward_linear.shape[1] == 1:
        ax_reward_linear.set_xticks([0])
        ax_reward_linear.set_xticklabels(['Reward'])
    else:
        ax_reward_linear.set_xticks([0, 1])
        ax_reward_linear.set_xticklabels(['Reward', 'Choice'])
    ax_reward_linear.set_title(f'Reward Linear Weights\n(shape: {W_reward_linear.shape}, mean={W_reward_linear.mean():.4f}, std={W_reward_linear.std():.4f})')
    plt.colorbar(im_reward_linear, ax=ax_reward_linear, label='Weight')
    plt.tight_layout()
    figures["pca_frozen/reward_linear_weights_heatmap"] = fig_reward_linear
    plt.close(fig_reward_linear)

    # Hidden to reward weights: shape (hidden_size, hidden_size)
    W_hidden_to_reward = model.hidden_to_reward.weight.detach().cpu().numpy()  # (hidden_size, hidden_size)
    fig_h2r, ax_h2r = plt.subplots(figsize=(10, 10), dpi=150)
    vmax_h2r = max(abs(W_hidden_to_reward.min()), abs(W_hidden_to_reward.max()))
    if vmax_h2r == 0:
        vmax_h2r = 1
    im_h2r = ax_h2r.imshow(W_hidden_to_reward, cmap='RdBu_r', vmin=-vmax_h2r, vmax=vmax_h2r, aspect='equal')
    ax_h2r.set_xlabel('Input Hidden Dimension')
    ax_h2r.set_ylabel('Output Hidden Dimension')
    ax_h2r.set_title(f'Hidden to Reward Weights\n(shape: {W_hidden_to_reward.shape}, mean={W_hidden_to_reward.mean():.4f}, std={W_hidden_to_reward.std():.4f})')
    plt.colorbar(im_h2r, ax=ax_h2r, label='Weight')
    plt.tight_layout()
    figures["pca_frozen/hidden_to_reward_weights_heatmap"] = fig_h2r
    plt.close(fig_h2r)

    # --- Plastic Weight Decomposition Analysis (for first hidden layer only) ---
    # Only do this if there are extra layers
    if args.extra_layers > 0:
        # For one network (network_idx = 0), track the coefficient matrix A
        # where P = Σ_ij A[i,j] * v_i * u_j^T with v_i = W @ u_i

        network_idx = 0
        single_items = batch_items[network_idx]  # (num_items, item_size)

        # Define adjacent pairs for TI (all adjacent item pairs)
        # Items: 0, 1, 2, ..., num_items-1
        # Adjacent pairs: (0,1), (1,2), (2,3), ..., (num_items-2, num_items-1)
        adjacent_pairs_ti = [(i, i+1) for i in range(num_items - 1)]

        # Create all presentations (both orderings: winner first, loser first)
        all_presentations = []
        presentation_to_idx = {}
        for pair in adjacent_pairs_ti:
            winner, loser = pair  # winner < loser in rank (lower index = higher rank)
            # Winner first (correct choice = 0)
            all_presentations.append((winner, loser))
            presentation_to_idx[(winner, loser)] = len(all_presentations) - 1
            # Loser first (correct choice = 1)
            all_presentations.append((loser, winner))
            presentation_to_idx[(loser, winner)] = len(all_presentations) - 1

        num_presentations = len(all_presentations)  # Should be 2 * (num_items - 1)

        # Compute embeddings u_i for each presentation (after embedding layer + tanh)
        embeddings_u = []
        with torch.no_grad():
            for (item1_idx, item2_idx) in all_presentations:
                item1_emb = single_items[item1_idx]
                item2_emb = single_items[item2_idx]
                input_vec = np.concatenate([item1_emb, item2_emb])
                input_t = torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0).to(device)
                # Run through embedding layer + tanh
                if hasattr(model, 'embedding_layer'):
                    u = torch.tanh(model.embedding_layer(input_t))  # (1, hidden_size)
                else:
                    u = input_t
                embeddings_u.append(u.squeeze(0).cpu().numpy())

        embeddings_u = np.array(embeddings_u)  # (num_presentations, hidden_size)

        # Get innate weights W for first hidden layer
        W_first_hidden = model.extra_hidden_layers[0].weight.detach().cpu().numpy()  # (hidden_size, hidden_size)

        # Get alpha matrix for first hidden layer
        alpha_first = model.alpha_extra[0].detach().cpu().numpy()  # (hidden_size, hidden_size)

        # Get hebbian trace multiplier for first hidden layer
        m_hebb = model.hebbian_trace_multiplier_extra[0].item()

        # Compute v_i = W @ u_i for each presentation
        embeddings_v = (W_first_hidden @ embeddings_u.T).T  # (num_presentations, hidden_size)

        # Compute D_alpha_full as 3D tensor: D_alpha_full[i,j,m] = sum_l alpha[m,l] * u_i[l] * u_j[l]
        # This captures the full alpha matrix without averaging
        if alpha_first.ndim >= 2:
            hidden_size_decomp = alpha_first.shape[0]
            D_alpha_full = np.zeros((num_presentations, num_presentations, hidden_size_decomp))
            for m in range(hidden_size_decomp):
                # For each output dimension m, compute the alpha[m,:]-weighted dot products
                D_alpha_full[:, :, m] = (embeddings_u * alpha_first[m, :]) @ embeddings_u.T

            # Compute mean and std across output dimensions
            D_alpha_mean = D_alpha_full.mean(axis=2)  # (num_presentations, num_presentations)
            D_alpha_std = D_alpha_full.std(axis=2)    # (num_presentations, num_presentations)
        else:
            # Scalar alpha: D_alpha = alpha * (U @ U^T)
            D_alpha_mean = float(alpha_first) * (embeddings_u @ embeddings_u.T)
            D_alpha_std = np.zeros_like(D_alpha_mean)

        # For coefficient tracking, we still use the mean approximation
        D_alpha = D_alpha_mean

        # Initialize coefficient matrix A = 0
        A_coeffs = np.zeros((num_presentations, num_presentations))

        # Re-run training to track coefficients
        # Reset plastic weights for tracking
        pw_track = torch.zeros(1, args.hidden_size, args.hidden_size, dtype=torch.float32).to(device)
        epw_track = [torch.zeros(1, args.hidden_size, args.hidden_size, dtype=torch.float32).to(device)
                     for _ in range(args.extra_layers)]

        # Get trials for this single network
        single_trials = trials[network_idx:network_idx+1, :, :]  # (1, num_trials, input_size)
        single_correct = correct_choices[network_idx:network_idx+1, :]  # (1, num_trials)

        for trial_idx in range(args.num_train_trials):
            trial_input = single_trials[:, trial_idx, :]  # (1, input_size)
            trial_correct = single_correct[:, trial_idx]  # (1,)

            # Identify which presentation this is
            # Extract the two item embeddings from the trial input
            item_size = args.item_size
            item1_emb_trial = trial_input[0, :item_size].cpu().numpy()
            item2_emb_trial = trial_input[0, item_size:2*item_size].cpu().numpy()

            # Find which items these correspond to
            item1_idx_found = None
            item2_idx_found = None
            for idx in range(num_items):
                if np.allclose(single_items[idx], item1_emb_trial, atol=1e-5):
                    item1_idx_found = idx
                if np.allclose(single_items[idx], item2_emb_trial, atol=1e-5):
                    item2_idx_found = idx

            if item1_idx_found is None or item2_idx_found is None:
                continue  # Skip if we can't identify the items

            # Check if this is an adjacent pair presentation
            presentation_key = (item1_idx_found, item2_idx_found)
            if presentation_key not in presentation_to_idx:
                continue  # Not an adjacent pair, skip

            k = presentation_to_idx[presentation_key]  # Embedding index

            # Run forward pass to get neuromodulator
            with torch.no_grad():
                output_track = model(trial_input, pw_track, trial_correct,
                                    extra_plastic_weights=epw_track, store_embeddings=False)

            # Get neuromodulator for first extra layer
            # The neuromodulator output has shape (batch, num_neuromodulators)
            # For use_extra_neuromodulator=True, first entries are for extra layers
            nm_output = output_track.neuromodulator.squeeze()
            if args.use_extra_neuromodulator and args.extra_layers > 0:
                # First neuromodulator is for first extra layer
                eta_t = nm_output[0].item() if nm_output.dim() > 0 else nm_output.item()
            else:
                eta_t = nm_output.item() if nm_output.dim() == 0 else nm_output[0].item()

            # Update coefficient matrix with full update rule:
            # A[:,k] += eta_t * m * (e_k + A @ D_alpha[:,k])
            # where m = hebbian_trace_multiplier, D_alpha accounts for alpha modulation
            e_k = np.zeros(num_presentations)
            e_k[k] = 1.0
            # Scale by ~0.9 to approximate tanh compression on outer products in [-1,1]
            tanh_scale = 0.9
            A_coeffs[:, k] += eta_t * m_hebb * tanh_scale * (e_k + A_coeffs @ D_alpha[:, k])

            # Update plastic weights for next iteration
            pw_track = output_track.plastic_weights
            epw_track = output_track.extra_plastic_weights

        # Create labels for presentations
        item_labels_decomp = [chr(ord('A') + i) for i in range(num_items)]
        presentation_labels = []
        for (item1_idx, item2_idx) in all_presentations:
            presentation_labels.append(f"{item_labels_decomp[item1_idx]}{item_labels_decomp[item2_idx]}")

        # Plot 1: Coefficient matrix A
        fig_coeffs, ax_coeffs = plt.subplots(figsize=(10, 8), dpi=150)
        vmax_coeffs = max(abs(A_coeffs.min()), abs(A_coeffs.max()))
        if vmax_coeffs == 0:
            vmax_coeffs = 1
        im_coeffs = ax_coeffs.imshow(A_coeffs, cmap='RdBu_r', vmin=-vmax_coeffs, vmax=vmax_coeffs, aspect='equal')
        ax_coeffs.set_xticks(range(num_presentations))
        ax_coeffs.set_xticklabels(presentation_labels, rotation=45, ha='right', fontsize=8)
        ax_coeffs.set_yticks(range(num_presentations))
        ax_coeffs.set_yticklabels(presentation_labels, fontsize=8)
        ax_coeffs.set_xlabel('j (u_j column)')
        ax_coeffs.set_ylabel('i (v_i row)')
        ax_coeffs.set_title('Coefficient Matrix A\n(P ≈ Σ_ij A[i,j] · v_i · u_j^T)')
        plt.colorbar(im_coeffs, ax=ax_coeffs, label='Coefficient value')
        plt.tight_layout()
        figures["pca_frozen/pw_decomposition_coefficients"] = fig_coeffs
        plt.close(fig_coeffs)

        # Plot 2: D_alpha matrix - Mean and Std side by side
        fig_dots, axes_dots = plt.subplots(1, 2, figsize=(16, 7), dpi=150)

        # Left: Mean D_alpha (with symmetric colormap around 0)
        vmax_dalpha = max(abs(D_alpha_mean.min()), abs(D_alpha_mean.max()))
        if vmax_dalpha == 0:
            vmax_dalpha = 1
        im_mean = axes_dots[0].imshow(D_alpha_mean, cmap='RdBu_r', vmin=-vmax_dalpha, vmax=vmax_dalpha, aspect='equal')
        axes_dots[0].set_xticks(range(num_presentations))
        axes_dots[0].set_xticklabels(presentation_labels, rotation=45, ha='right', fontsize=8)
        axes_dots[0].set_yticks(range(num_presentations))
        axes_dots[0].set_yticklabels(presentation_labels, fontsize=8)
        axes_dots[0].set_xlabel('j')
        axes_dots[0].set_ylabel('i')
        axes_dots[0].set_title('D_alpha Mean\n(averaged over output dimensions)')
        plt.colorbar(im_mean, ax=axes_dots[0], label='Mean alpha-weighted dot product')

        # Right: Std D_alpha
        im_std = axes_dots[1].imshow(D_alpha_std, cmap='plasma', aspect='equal')
        axes_dots[1].set_xticks(range(num_presentations))
        axes_dots[1].set_xticklabels(presentation_labels, rotation=45, ha='right', fontsize=8)
        axes_dots[1].set_yticks(range(num_presentations))
        axes_dots[1].set_yticklabels(presentation_labels, fontsize=8)
        axes_dots[1].set_xlabel('j')
        axes_dots[1].set_ylabel('i')
        axes_dots[1].set_title('D_alpha Std\n(variation across output dimensions)')
        plt.colorbar(im_std, ax=axes_dots[1], label='Std of alpha-weighted dot product')

        plt.suptitle('D_alpha Matrix: Mean vs Std across Output Dimensions', fontsize=12)
        plt.tight_layout()
        figures["pca_frozen/pw_decomposition_d_alpha"] = fig_dots
        plt.close(fig_dots)

        # Plot 3: Reconstruction error
        # Reconstruct P from coefficients and compare to actual P
        P_actual = frozen_extra_plastic_weights[0][network_idx].detach().cpu().numpy()
        P_reconstructed = np.zeros_like(P_actual)
        for i in range(num_presentations):
            for j in range(num_presentations):
                P_reconstructed += A_coeffs[i, j] * np.outer(embeddings_v[i], embeddings_u[j])

        reconstruction_error = np.linalg.norm(P_actual - P_reconstructed) / np.linalg.norm(P_actual)

        fig_recon, axes_recon = plt.subplots(1, 3, figsize=(18, 5), dpi=150)

        vmax_p = max(abs(P_actual.min()), abs(P_actual.max()), abs(P_reconstructed.min()), abs(P_reconstructed.max()))
        if vmax_p == 0:
            vmax_p = 1

        im0 = axes_recon[0].imshow(P_actual, cmap='RdBu_r', vmin=-vmax_p, vmax=vmax_p, aspect='equal')
        axes_recon[0].set_title('Actual P (first hidden layer)')
        plt.colorbar(im0, ax=axes_recon[0])

        im1 = axes_recon[1].imshow(P_reconstructed, cmap='RdBu_r', vmin=-vmax_p, vmax=vmax_p, aspect='equal')
        axes_recon[1].set_title('Reconstructed P from coefficients')
        plt.colorbar(im1, ax=axes_recon[1])

        P_diff = P_actual - P_reconstructed
        vmax_diff = max(abs(P_diff.min()), abs(P_diff.max()))
        if vmax_diff == 0:
            vmax_diff = 1
        im2 = axes_recon[2].imshow(P_diff, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff, aspect='equal')
        axes_recon[2].set_title(f'Difference (error = {reconstruction_error:.4f})')
        plt.colorbar(im2, ax=axes_recon[2])

        plt.suptitle('Plastic Weight Decomposition Verification (TI)', fontsize=12)
        plt.tight_layout()
        figures["pca_frozen/pw_decomposition_reconstruction"] = fig_recon
        plt.close(fig_recon)

        # Plot 4: Residual coefficients in the expanded basis {e_m ⊗ u_j}
        # Compute residual: R = P_actual - P_reconstructed
        # Then express R in the {e_m ⊗ u_j} basis: R[m,:] = sum_j R_coeff[m,j] * u_j^T
        # R_coeff[m,j] = R[m,:] projected onto u_j direction
        U_matrix = embeddings_u.T  # (hidden_size, num_presentations)
        U_pinv = np.linalg.pinv(U_matrix)  # (num_presentations, hidden_size)

        # R_coeff[m,j] = (R @ U @ (U^T U)^{-1})[m,j] but simpler with pinv of U^T
        # R[m,:] @ U_pinv.T gives the coefficients for row m
        R_coeff = P_diff @ U_pinv.T  # (hidden_size, num_presentations)

        # Compute reconstruction with residuals added
        P_with_residual = P_reconstructed + R_coeff @ U_matrix.T
        reconstruction_error_with_residual = np.linalg.norm(P_actual - P_with_residual) / np.linalg.norm(P_actual)

        # Plot residual coefficients heatmap
        fig_resid, ax_resid = plt.subplots(figsize=(12, 8), dpi=150)
        vmax_resid = max(abs(R_coeff.min()), abs(R_coeff.max()))
        if vmax_resid == 0:
            vmax_resid = 1
        im_resid = ax_resid.imshow(R_coeff, cmap='RdBu_r', vmin=-vmax_resid, vmax=vmax_resid, aspect='auto')
        ax_resid.set_xticks(range(num_presentations))
        ax_resid.set_xticklabels(presentation_labels, rotation=45, ha='right', fontsize=8)
        ax_resid.set_xlabel('j (u_j basis vector)')
        ax_resid.set_ylabel('m (output dimension)')
        ax_resid.set_title('Residual Coefficients R[m,j]\n(what the v_i basis cannot capture, in the e_m ⊗ u_j basis)')
        plt.colorbar(im_resid, ax=ax_resid, label='Residual coefficient')
        plt.tight_layout()
        figures["pca_frozen/pw_decomposition_residual"] = fig_resid
        plt.close(fig_resid)

        # Plot reconstruction comparison: without vs with residuals
        fig_recon_compare, axes_rc = plt.subplots(1, 3, figsize=(18, 5), dpi=150)

        im_rc0 = axes_rc[0].imshow(P_actual, cmap='RdBu_r', vmin=-vmax_p, vmax=vmax_p, aspect='equal')
        axes_rc[0].set_title('Actual P')
        plt.colorbar(im_rc0, ax=axes_rc[0])

        im_rc1 = axes_rc[1].imshow(P_reconstructed, cmap='RdBu_r', vmin=-vmax_p, vmax=vmax_p, aspect='equal')
        axes_rc[1].set_title(f'v_i ⊗ u_j basis only\n(error = {reconstruction_error:.4f})')
        plt.colorbar(im_rc1, ax=axes_rc[1])

        im_rc2 = axes_rc[2].imshow(P_with_residual, cmap='RdBu_r', vmin=-vmax_p, vmax=vmax_p, aspect='equal')
        axes_rc[2].set_title(f'With e_m ⊗ u_j residuals\n(error = {reconstruction_error_with_residual:.4f})')
        plt.colorbar(im_rc2, ax=axes_rc[2])

        plt.suptitle('Reconstruction Comparison: Scalar vs Expanded Basis (TI)', fontsize=12)
        plt.tight_layout()
        figures["pca_frozen/pw_decomposition_reconstruction_comparison"] = fig_recon_compare
        plt.close(fig_recon_compare)

        # Also plot the norm of residual per output dimension and per u_j
        fig_resid_summary, axes_resid_summary = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

        # Left: Residual norm per output dimension m
        resid_norm_per_m = np.linalg.norm(R_coeff, axis=1)  # (hidden_size,)
        axes_resid_summary[0].bar(range(len(resid_norm_per_m)), resid_norm_per_m, color='steelblue', alpha=0.7)
        axes_resid_summary[0].set_xlabel('Output dimension m')
        axes_resid_summary[0].set_ylabel('||R[m,:]||')
        axes_resid_summary[0].set_title('Residual norm per output dimension')

        # Right: Residual norm per u_j basis vector
        resid_norm_per_j = np.linalg.norm(R_coeff, axis=0)  # (num_presentations,)
        axes_resid_summary[1].bar(range(num_presentations), resid_norm_per_j, color='darkorange', alpha=0.7)
        axes_resid_summary[1].set_xticks(range(num_presentations))
        axes_resid_summary[1].set_xticklabels(presentation_labels, rotation=45, ha='right', fontsize=8)
        axes_resid_summary[1].set_xlabel('j (u_j basis vector)')
        axes_resid_summary[1].set_ylabel('||R[:,j]||')
        axes_resid_summary[1].set_title('Residual norm per u_j basis vector')

        plt.suptitle('Residual Analysis: Where does the v_i basis fail?', fontsize=12)
        plt.tight_layout()
        figures["pca_frozen/pw_decomposition_residual_summary"] = fig_resid_summary
        plt.close(fig_resid_summary)

        # ===================================================================================
        # LAYER 2 (Final Layer) Decomposition Analysis
        # P₂ = Σᵢⱼ B[i,j] · ṽᵢ ⊗ ũⱼ where ũⱼ = layer-1 activations, ṽᵢ = W₂ @ ũᵢ
        # ===================================================================================

        # Compute ũⱼ = layer-1 activations for each presentation
        # We need to run each presentation through the network with the final P₁
        embeddings_u_tilde = []
        with torch.no_grad():
            # Use the frozen plastic weights from layer 1
            pw_layer1_final = frozen_extra_plastic_weights[0][network_idx:network_idx+1]  # (1, H, H)

            for (item1_idx, item2_idx) in all_presentations:
                item1_emb = single_items[item1_idx]
                item2_emb = single_items[item2_idx]
                input_vec = np.concatenate([item1_emb, item2_emb])
                input_t = torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0).to(device)

                # Layer 0: embedding layer
                if hasattr(model, 'embedding_layer'):
                    u = torch.tanh(model.embedding_layer(input_t))  # (1, hidden_size)
                else:
                    u = input_t

                # Layer 1: first extra hidden layer with plastic weights
                alpha_layer1 = model.alpha_extra[0]  # (H, H)
                W_layer1 = model.extra_hidden_layers[0].weight  # (H, H)
                b_layer1 = model.extra_hidden_layers[0].bias
                # Include bias term: h1 = tanh(W @ u + bias + (α * P1) @ u)
                innate = W_layer1 @ u.T
                if b_layer1 is not None:
                    innate = innate + b_layer1.unsqueeze(1)
                h1 = torch.tanh(innate + (alpha_layer1 * pw_layer1_final.squeeze(0)) @ u.T)  # (H, 1)
                h1 = h1.T  # (1, H)

                embeddings_u_tilde.append(h1.squeeze(0).cpu().numpy())

        embeddings_u_tilde = np.array(embeddings_u_tilde)  # (num_presentations, hidden_size)

        # Get layer-2 (final layer) parameters
        W_layer2 = model.fc2.weight.detach().cpu().numpy()  # (H, H)
        alpha_layer2 = model.alpha.detach().cpu().numpy()  # (H, H)
        m_hebb_layer2 = model.hebbian_trace_multiplier.item()

        # Compute ṽᵢ = W₂ @ ũᵢ
        embeddings_v_tilde = (W_layer2 @ embeddings_u_tilde.T).T  # (num_presentations, hidden_size)

        # Compute D_alpha_full for layer 2: D_α²[i,j,m] = Σₗ α₂[m,l] · ũᵢ[l] · ũⱼ[l]
        if alpha_layer2.ndim >= 2:
            hidden_size_decomp_l2 = alpha_layer2.shape[0]
            D_alpha_full_layer2 = np.zeros((num_presentations, num_presentations, hidden_size_decomp_l2))
            for m in range(hidden_size_decomp_l2):
                D_alpha_full_layer2[:, :, m] = (embeddings_u_tilde * alpha_layer2[m, :]) @ embeddings_u_tilde.T

            D_alpha_mean_layer2 = D_alpha_full_layer2.mean(axis=2)
            D_alpha_std_layer2 = D_alpha_full_layer2.std(axis=2)
        else:
            D_alpha_mean_layer2 = float(alpha_layer2) * (embeddings_u_tilde @ embeddings_u_tilde.T)
            D_alpha_std_layer2 = np.zeros_like(D_alpha_mean_layer2)
        D_alpha_layer2 = D_alpha_mean_layer2  # For coefficient tracking

        # Initialize coefficient matrix B = 0 for layer 2
        B_coeffs = np.zeros((num_presentations, num_presentations))

        # Re-run training to track layer-2 coefficients with TIME-VARYING ũ basis
        pw_track_l2 = torch.zeros(1, args.hidden_size, args.hidden_size, dtype=torch.float32).to(device)
        epw_track_l2 = [torch.zeros(1, args.hidden_size, args.hidden_size, dtype=torch.float32).to(device)
                        for _ in range(args.extra_layers)]

        # Also accumulate the actual reconstruction using correct vectors at each step
        P_reconstructed_l2_timevarying = np.zeros((args.hidden_size, args.hidden_size))

        for trial_idx in range(args.num_train_trials):
            trial_input = single_trials[:, trial_idx, :]
            trial_correct = single_correct[:, trial_idx]

            item_size = args.item_size
            item1_emb_trial = trial_input[0, :item_size].cpu().numpy()
            item2_emb_trial = trial_input[0, item_size:2*item_size].cpu().numpy()

            item1_idx_found = None
            item2_idx_found = None
            for idx in range(num_items):
                if np.allclose(single_items[idx], item1_emb_trial, atol=1e-5):
                    item1_idx_found = idx
                if np.allclose(single_items[idx], item2_emb_trial, atol=1e-5):
                    item2_idx_found = idx

            if item1_idx_found is None or item2_idx_found is None:
                continue

            presentation_key = (item1_idx_found, item2_idx_found)
            if presentation_key not in presentation_to_idx:
                continue

            k = presentation_to_idx[presentation_key]

            # Compute time-varying ũⱼ⁽ᵗ⁾ for ALL presentations using current P₁
            with torch.no_grad():
                P1_current = epw_track_l2[0].squeeze(0)  # Current P₁ at this trial
                embeddings_u_tilde_t = []
                for (item1_idx, item2_idx) in all_presentations:
                    item1_emb = single_items[item1_idx]
                    item2_emb = single_items[item2_idx]
                    input_vec = np.concatenate([item1_emb, item2_emb])
                    input_t = torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0).to(device)
                    if hasattr(model, 'embedding_layer'):
                        u = torch.tanh(model.embedding_layer(input_t))
                    else:
                        u = input_t
                    alpha_layer1 = model.alpha_extra[0]
                    W_layer1 = model.extra_hidden_layers[0].weight
                    b_layer1 = model.extra_hidden_layers[0].bias
                    # Include bias term: h1 = tanh(W @ u + bias + (α * P1) @ u)
                    innate = W_layer1 @ u.T
                    if b_layer1 is not None:
                        innate = innate + b_layer1.unsqueeze(1)
                    h1 = torch.tanh(innate + (alpha_layer1 * P1_current) @ u.T)
                    embeddings_u_tilde_t.append(h1.T.squeeze(0).cpu().numpy())
                embeddings_u_tilde_t = np.array(embeddings_u_tilde_t)

            # Compute time-varying D_alpha using current ũ basis
            if alpha_layer2.ndim >= 2:
                D_alpha_t = (embeddings_u_tilde_t * alpha_layer2.mean(axis=0)) @ embeddings_u_tilde_t.T
            else:
                D_alpha_t = float(alpha_layer2) * (embeddings_u_tilde_t @ embeddings_u_tilde_t.T)

            with torch.no_grad():
                output_track_l2 = model(trial_input, pw_track_l2, trial_correct,
                                        extra_plastic_weights=epw_track_l2, store_embeddings=False)

            # Get neuromodulator for final layer (last value)
            nm_output = output_track_l2.neuromodulator.squeeze()
            if nm_output.dim() == 0:
                eta_t_layer2 = nm_output.item()
            else:
                # Last neuromodulator is for final layer
                eta_t_layer2 = nm_output[-1].item() if len(nm_output) > 1 else nm_output.item()

            # Update coefficient matrix B for layer 2 using TIME-VARYING D_alpha
            e_k = np.zeros(num_presentations)
            e_k[k] = 1.0
            tanh_scale = 0.9
            B_coeffs[:, k] += eta_t_layer2 * m_hebb_layer2 * tanh_scale * (e_k + B_coeffs @ D_alpha_t[:, k])

            # Accumulate actual reconstruction using the correct ũₖ⁽ᵗ⁾ and ṽₖ⁽ᵗ⁾ at this trial
            u_tilde_k_t = embeddings_u_tilde_t[k]  # ũₖ at current trial
            v_tilde_k_t = W_layer2 @ u_tilde_k_t   # ṽₖ = W₂ @ ũₖ
            P_reconstructed_l2_timevarying += eta_t_layer2 * m_hebb_layer2 * np.outer(v_tilde_k_t, u_tilde_k_t)

            pw_track_l2 = output_track_l2.plastic_weights
            epw_track_l2 = output_track_l2.extra_plastic_weights

        # --- Layer 2 Plots ---

        # Plot 1: Coefficient matrix B for layer 2
        fig_coeffs_l2, ax_coeffs_l2 = plt.subplots(figsize=(10, 8), dpi=150)
        vmax_coeffs_l2 = max(abs(B_coeffs.min()), abs(B_coeffs.max()))
        if vmax_coeffs_l2 == 0:
            vmax_coeffs_l2 = 1
        im_coeffs_l2 = ax_coeffs_l2.imshow(B_coeffs, cmap='RdBu_r', vmin=-vmax_coeffs_l2, vmax=vmax_coeffs_l2, aspect='equal')
        ax_coeffs_l2.set_xticks(range(num_presentations))
        ax_coeffs_l2.set_xticklabels(presentation_labels, rotation=45, ha='right', fontsize=8)
        ax_coeffs_l2.set_yticks(range(num_presentations))
        ax_coeffs_l2.set_yticklabels(presentation_labels, fontsize=8)
        ax_coeffs_l2.set_xlabel('j (ũ_j column)')
        ax_coeffs_l2.set_ylabel('i (ṽ_i row)')
        ax_coeffs_l2.set_title('Coefficient Matrix B - Layer 2 (Final)\n(P₂ ≈ Σᵢⱼ B[i,j] · ṽᵢ · ũⱼᵀ)')
        plt.colorbar(im_coeffs_l2, ax=ax_coeffs_l2, label='Coefficient value')
        plt.tight_layout()
        figures["pca_frozen/pw_decomposition_coefficients_layer2"] = fig_coeffs_l2
        plt.close(fig_coeffs_l2)

        # Plot 2: D_alpha mean and std for layer 2
        fig_dots_l2, axes_dots_l2 = plt.subplots(1, 2, figsize=(16, 7), dpi=150)

        vmax_dalpha_l2 = max(abs(D_alpha_mean_layer2.min()), abs(D_alpha_mean_layer2.max()))
        if vmax_dalpha_l2 == 0:
            vmax_dalpha_l2 = 1
        im_mean_l2 = axes_dots_l2[0].imshow(D_alpha_mean_layer2, cmap='RdBu_r', vmin=-vmax_dalpha_l2, vmax=vmax_dalpha_l2, aspect='equal')
        axes_dots_l2[0].set_xticks(range(num_presentations))
        axes_dots_l2[0].set_xticklabels(presentation_labels, rotation=45, ha='right', fontsize=8)
        axes_dots_l2[0].set_yticks(range(num_presentations))
        axes_dots_l2[0].set_yticklabels(presentation_labels, fontsize=8)
        axes_dots_l2[0].set_xlabel('j')
        axes_dots_l2[0].set_ylabel('i')
        axes_dots_l2[0].set_title('Mean D_α²[i,j] across output dimensions\n(D_α²[i,j,m] = Σₗ α₂[m,l] · ũᵢ[l] · ũⱼ[l])')
        plt.colorbar(im_mean_l2, ax=axes_dots_l2[0], label='Mean alpha-weighted dot product')

        im_std_l2 = axes_dots_l2[1].imshow(D_alpha_std_layer2, cmap='plasma', aspect='equal')
        axes_dots_l2[1].set_xticks(range(num_presentations))
        axes_dots_l2[1].set_xticklabels(presentation_labels, rotation=45, ha='right', fontsize=8)
        axes_dots_l2[1].set_yticks(range(num_presentations))
        axes_dots_l2[1].set_yticklabels(presentation_labels, fontsize=8)
        axes_dots_l2[1].set_xlabel('j')
        axes_dots_l2[1].set_ylabel('i')
        axes_dots_l2[1].set_title('Std D_α²[i,j] across output dimensions\n(variation in alpha weighting)')
        plt.colorbar(im_std_l2, ax=axes_dots_l2[1], label='Std of alpha-weighted dot product')

        plt.suptitle('D_alpha Matrix Analysis - Layer 2 (using ũ embeddings)', fontsize=12)
        plt.tight_layout()
        figures["pca_frozen/pw_decomposition_d_alpha_layer2"] = fig_dots_l2
        plt.close(fig_dots_l2)

        # Plot 3: Reconstruction error for layer 2 (comparing fixed basis vs time-varying)
        P_actual_l2 = frozen_plastic_weights[network_idx].detach().cpu().numpy()

        # Fixed basis reconstruction (using final ũ)
        P_reconstructed_l2_fixed = np.zeros_like(P_actual_l2)
        for i in range(num_presentations):
            for j in range(num_presentations):
                P_reconstructed_l2_fixed += B_coeffs[i, j] * np.outer(embeddings_v_tilde[i], embeddings_u_tilde[j])

        reconstruction_error_l2_fixed = np.linalg.norm(P_actual_l2 - P_reconstructed_l2_fixed) / np.linalg.norm(P_actual_l2) if np.linalg.norm(P_actual_l2) > 0 else 0.0

        # Time-varying reconstruction error
        reconstruction_error_l2_timevarying = np.linalg.norm(P_actual_l2 - P_reconstructed_l2_timevarying) / np.linalg.norm(P_actual_l2) if np.linalg.norm(P_actual_l2) > 0 else 0.0

        fig_recon_l2, axes_recon_l2 = plt.subplots(2, 2, figsize=(14, 12), dpi=150)

        vmax_p_l2 = max(abs(P_actual_l2.min()), abs(P_actual_l2.max()),
                        abs(P_reconstructed_l2_fixed.min()), abs(P_reconstructed_l2_fixed.max()),
                        abs(P_reconstructed_l2_timevarying.min()), abs(P_reconstructed_l2_timevarying.max()))
        if vmax_p_l2 == 0:
            vmax_p_l2 = 1

        im0_l2 = axes_recon_l2[0, 0].imshow(P_actual_l2, cmap='RdBu_r', vmin=-vmax_p_l2, vmax=vmax_p_l2, aspect='equal')
        axes_recon_l2[0, 0].set_title('Actual P₂ (final layer)')
        plt.colorbar(im0_l2, ax=axes_recon_l2[0, 0])

        im1_l2 = axes_recon_l2[0, 1].imshow(P_reconstructed_l2_fixed, cmap='RdBu_r', vmin=-vmax_p_l2, vmax=vmax_p_l2, aspect='equal')
        axes_recon_l2[0, 1].set_title(f'Fixed basis (final ũ)\n(error = {reconstruction_error_l2_fixed:.4f})')
        plt.colorbar(im1_l2, ax=axes_recon_l2[0, 1])

        im2_l2 = axes_recon_l2[1, 0].imshow(P_reconstructed_l2_timevarying, cmap='RdBu_r', vmin=-vmax_p_l2, vmax=vmax_p_l2, aspect='equal')
        axes_recon_l2[1, 0].set_title(f'Time-varying basis (ũ⁽ᵗ⁾)\n(error = {reconstruction_error_l2_timevarying:.4f})')
        plt.colorbar(im2_l2, ax=axes_recon_l2[1, 0])

        P_diff_l2 = P_actual_l2 - P_reconstructed_l2_timevarying
        vmax_diff_l2 = max(abs(P_diff_l2.min()), abs(P_diff_l2.max()))
        if vmax_diff_l2 == 0:
            vmax_diff_l2 = 1
        im3_l2 = axes_recon_l2[1, 1].imshow(P_diff_l2, cmap='RdBu_r', vmin=-vmax_diff_l2, vmax=vmax_diff_l2, aspect='equal')
        axes_recon_l2[1, 1].set_title(f'Time-varying residual\n(error = {reconstruction_error_l2_timevarying:.4f})')
        plt.colorbar(im3_l2, ax=axes_recon_l2[1, 1])

        plt.suptitle('Layer 2 Reconstruction: Fixed vs Time-Varying Basis (TI)', fontsize=12)
        plt.tight_layout()
        figures["pca_frozen/pw_decomposition_reconstruction_layer2"] = fig_recon_l2
        plt.close(fig_recon_l2)

        # Use time-varying reconstruction for subsequent analysis
        P_reconstructed_l2 = P_reconstructed_l2_timevarying
        reconstruction_error_l2 = reconstruction_error_l2_timevarying

        # Plot 4: Residual coefficients for layer 2
        U_matrix_l2 = embeddings_u_tilde.T  # (hidden_size, num_presentations)
        U_pinv_l2 = np.linalg.pinv(U_matrix_l2)
        R_coeff_l2 = P_diff_l2 @ U_pinv_l2.T

        P_with_residual_l2 = P_reconstructed_l2 + R_coeff_l2 @ U_matrix_l2.T
        reconstruction_error_with_residual_l2 = np.linalg.norm(P_actual_l2 - P_with_residual_l2) / np.linalg.norm(P_actual_l2) if np.linalg.norm(P_actual_l2) > 0 else 0.0

        fig_resid_l2, ax_resid_l2 = plt.subplots(figsize=(12, 8), dpi=150)
        vmax_resid_l2 = max(abs(R_coeff_l2.min()), abs(R_coeff_l2.max()))
        if vmax_resid_l2 == 0:
            vmax_resid_l2 = 1
        im_resid_l2 = ax_resid_l2.imshow(R_coeff_l2, cmap='RdBu_r', vmin=-vmax_resid_l2, vmax=vmax_resid_l2, aspect='auto')
        ax_resid_l2.set_xticks(range(num_presentations))
        ax_resid_l2.set_xticklabels(presentation_labels, rotation=45, ha='right', fontsize=8)
        ax_resid_l2.set_xlabel('j (ũ_j basis vector)')
        ax_resid_l2.set_ylabel('m (output dimension)')
        ax_resid_l2.set_title('Residual Coefficients R[m,j] - Layer 2\n(what the ṽ_i basis cannot capture)')
        plt.colorbar(im_resid_l2, ax=ax_resid_l2, label='Residual coefficient')
        plt.tight_layout()
        figures["pca_frozen/pw_decomposition_residual_layer2"] = fig_resid_l2
        plt.close(fig_resid_l2)

        # Plot 5: Reconstruction comparison for layer 2
        fig_recon_compare_l2, axes_rc_l2 = plt.subplots(1, 3, figsize=(18, 5), dpi=150)

        im_rc0_l2 = axes_rc_l2[0].imshow(P_actual_l2, cmap='RdBu_r', vmin=-vmax_p_l2, vmax=vmax_p_l2, aspect='equal')
        axes_rc_l2[0].set_title('Actual P₂')
        plt.colorbar(im_rc0_l2, ax=axes_rc_l2[0])

        im_rc1_l2 = axes_rc_l2[1].imshow(P_reconstructed_l2, cmap='RdBu_r', vmin=-vmax_p_l2, vmax=vmax_p_l2, aspect='equal')
        axes_rc_l2[1].set_title(f'ṽ_i ⊗ ũ_j basis only\n(error = {reconstruction_error_l2:.4f})')
        plt.colorbar(im_rc1_l2, ax=axes_rc_l2[1])

        im_rc2_l2 = axes_rc_l2[2].imshow(P_with_residual_l2, cmap='RdBu_r', vmin=-vmax_p_l2, vmax=vmax_p_l2, aspect='equal')
        axes_rc_l2[2].set_title(f'With e_m ⊗ ũ_j residuals\n(error = {reconstruction_error_with_residual_l2:.4f})')
        plt.colorbar(im_rc2_l2, ax=axes_rc_l2[2])

        plt.suptitle('Reconstruction Comparison: Scalar vs Expanded Basis - Layer 2 (TI)', fontsize=12)
        plt.tight_layout()
        figures["pca_frozen/pw_decomposition_reconstruction_comparison_layer2"] = fig_recon_compare_l2
        plt.close(fig_recon_compare_l2)

        # Plot 6: Residual norm summary for layer 2
        fig_resid_summary_l2, axes_resid_summary_l2 = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

        resid_norm_per_m_l2 = np.linalg.norm(R_coeff_l2, axis=1)
        axes_resid_summary_l2[0].bar(range(len(resid_norm_per_m_l2)), resid_norm_per_m_l2, color='steelblue', alpha=0.7)
        axes_resid_summary_l2[0].set_xlabel('Output dimension m')
        axes_resid_summary_l2[0].set_ylabel('||R[m,:]||')
        axes_resid_summary_l2[0].set_title('Residual norm per output dimension')

        resid_norm_per_j_l2 = np.linalg.norm(R_coeff_l2, axis=0)
        axes_resid_summary_l2[1].bar(range(num_presentations), resid_norm_per_j_l2, color='darkorange', alpha=0.7)
        axes_resid_summary_l2[1].set_xticks(range(num_presentations))
        axes_resid_summary_l2[1].set_xticklabels(presentation_labels, rotation=45, ha='right', fontsize=8)
        axes_resid_summary_l2[1].set_xlabel('j (ũ_j basis vector)')
        axes_resid_summary_l2[1].set_ylabel('||R[:,j]||')
        axes_resid_summary_l2[1].set_title('Residual norm per ũ_j basis vector')

        plt.suptitle('Residual Analysis - Layer 2: Where does the ṽ_i basis fail?', fontsize=12)
        plt.tight_layout()
        figures["pca_frozen/pw_decomposition_residual_summary_layer2"] = fig_resid_summary_l2
        plt.close(fig_resid_summary_l2)

        # =====================================================================
        # ATTENTION ANALYSIS: How non-adjacent pairs attend to adjacent pairs
        # =====================================================================
        # For each non-adjacent test pair, we analyze:
        # - Layer 1: attention via u_j^T @ u_test (embedding similarity)
        # - Layer 2: attention via ũ_j^T @ ũ_test (hidden activation similarity)

        # Generate all non-adjacent test pairs (symbolic distance >= 2)
        test_pairs = []
        for sd in range(2, num_items):
            for i in range(num_items - sd):
                # Both presentation orders
                test_pairs.append((i, i + sd, 0))  # winner first, correct=0
                test_pairs.append((i + sd, i, 1))  # loser first, correct=1

        # Compute test pair embeddings (u_test) and Layer-1 activations (ũ_test)
        test_embeddings_u = []
        test_embeddings_u_tilde = []
        test_labels = []
        test_signed_sds = []

        with torch.no_grad():
            # Use frozen P1 for computing ũ_test
            pw_layer1_final = frozen_extra_plastic_weights[0][network_idx:network_idx+1].squeeze(0)  # (H, H)

            for (item1_idx, item2_idx, correct) in test_pairs:
                item1_emb = single_items[item1_idx]
                item2_emb = single_items[item2_idx]
                input_vec = np.concatenate([item1_emb, item2_emb])
                input_t = torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0).to(device)

                # Layer 0: embedding layer -> u_test
                if hasattr(model, 'embedding_layer'):
                    u_test = torch.tanh(model.embedding_layer(input_t))  # (1, H)
                else:
                    u_test = input_t
                test_embeddings_u.append(u_test.squeeze(0).cpu().numpy())

                # Layer 1: compute ũ_test = tanh(W @ u + bias + (α * P1) @ u)
                alpha_layer1 = model.alpha_extra[0]
                W_layer1 = model.extra_hidden_layers[0].weight
                b_layer1 = model.extra_hidden_layers[0].bias
                innate = W_layer1 @ u_test.T
                if b_layer1 is not None:
                    innate = innate + b_layer1.unsqueeze(1)
                u_tilde_test = torch.tanh(innate + (alpha_layer1 * pw_layer1_final) @ u_test.T)
                test_embeddings_u_tilde.append(u_tilde_test.T.squeeze(0).cpu().numpy())

                # Labels and metadata
                test_labels.append(f"{item_labels_decomp[item1_idx]}{item_labels_decomp[item2_idx]}")
                signed_sd = item2_idx - item1_idx  # positive when winner (lower index) is first
                test_signed_sds.append(signed_sd)

        test_embeddings_u = np.array(test_embeddings_u)  # (num_test_pairs, hidden_size)
        test_embeddings_u_tilde = np.array(test_embeddings_u_tilde)  # (num_test_pairs, hidden_size)

        # --- Layer 1 Attention Analysis ---
        # attention_L1[adj_idx, test_idx] = u_adj^T @ u_test
        attention_matrix_L1 = embeddings_u @ test_embeddings_u.T  # (num_adj, num_test)

        # --- Layer 2 Attention Analysis ---
        # attention_L2[adj_idx, test_idx] = ũ_adj^T @ ũ_test
        attention_matrix_L2 = embeddings_u_tilde @ test_embeddings_u_tilde.T  # (num_adj, num_test)

        # =====================================================================
        # Plot 1: Layer 1 Attention Heatmap (u-space)
        # =====================================================================
        fig_attn_L1, ax_attn_L1 = plt.subplots(figsize=(14, 8), dpi=150)
        vmax_attn_L1 = max(abs(attention_matrix_L1.min()), abs(attention_matrix_L1.max()))
        if vmax_attn_L1 == 0:
            vmax_attn_L1 = 1
        im_attn_L1 = ax_attn_L1.imshow(attention_matrix_L1, cmap='RdBu_r', vmin=-vmax_attn_L1, vmax=vmax_attn_L1, aspect='auto')
        ax_attn_L1.set_yticks(range(num_presentations))
        ax_attn_L1.set_yticklabels(presentation_labels, fontsize=7)
        ax_attn_L1.set_xticks(range(len(test_labels)))
        ax_attn_L1.set_xticklabels(test_labels, fontsize=7, rotation=45, ha='right')
        ax_attn_L1.set_ylabel('Adjacent Pair Embedding (u_j)')
        ax_attn_L1.set_xlabel('Test Pair Embedding (u_test)')
        ax_attn_L1.set_title('Layer 1 Attention: u_j^T @ u_test\n(Embedding-space similarity)')
        plt.colorbar(im_attn_L1, ax=ax_attn_L1, label='Dot product')
        plt.tight_layout()
        figures["pca_frozen/pw_attention_layer1"] = fig_attn_L1
        plt.close(fig_attn_L1)

        # =====================================================================
        # Plot 2: Layer 2 Attention Heatmap (ũ-space)
        # =====================================================================
        fig_attn_L2, ax_attn_L2 = plt.subplots(figsize=(14, 8), dpi=150)
        vmax_attn_L2 = max(abs(attention_matrix_L2.min()), abs(attention_matrix_L2.max()))
        if vmax_attn_L2 == 0:
            vmax_attn_L2 = 1
        im_attn_L2 = ax_attn_L2.imshow(attention_matrix_L2, cmap='RdBu_r', vmin=-vmax_attn_L2, vmax=vmax_attn_L2, aspect='auto')
        ax_attn_L2.set_yticks(range(num_presentations))
        ax_attn_L2.set_yticklabels(presentation_labels, fontsize=7)
        ax_attn_L2.set_xticks(range(len(test_labels)))
        ax_attn_L2.set_xticklabels(test_labels, fontsize=7, rotation=45, ha='right')
        ax_attn_L2.set_ylabel('Adjacent Pair Activation (ũ_j)')
        ax_attn_L2.set_xlabel('Test Pair Activation (ũ_test)')
        ax_attn_L2.set_title('Layer 2 Attention: ũ_j^T @ ũ_test\n(Hidden activation similarity - after plastic layer 1)')
        plt.colorbar(im_attn_L2, ax=ax_attn_L2, label='Dot product')
        plt.tight_layout()
        figures["pca_frozen/pw_attention_layer2"] = fig_attn_L2
        plt.close(fig_attn_L2)

        # =====================================================================
        # Plot 3: Layer 1 vs Layer 2 Attention Comparison (side by side)
        # =====================================================================
        fig_attn_compare, axes_attn_compare = plt.subplots(1, 2, figsize=(20, 8), dpi=150)

        vmax_compare = max(vmax_attn_L1, vmax_attn_L2)

        im_c1 = axes_attn_compare[0].imshow(attention_matrix_L1, cmap='RdBu_r', vmin=-vmax_compare, vmax=vmax_compare, aspect='auto')
        axes_attn_compare[0].set_yticks(range(num_presentations))
        axes_attn_compare[0].set_yticklabels(presentation_labels, fontsize=7)
        axes_attn_compare[0].set_xticks(range(len(test_labels)))
        axes_attn_compare[0].set_xticklabels(test_labels, fontsize=7, rotation=45, ha='right')
        axes_attn_compare[0].set_ylabel('Adjacent Pair')
        axes_attn_compare[0].set_xlabel('Test Pair')
        axes_attn_compare[0].set_title('Layer 1: u_j^T @ u_test\n(Before plastic layer)')
        plt.colorbar(im_c1, ax=axes_attn_compare[0], label='Dot product')

        im_c2 = axes_attn_compare[1].imshow(attention_matrix_L2, cmap='RdBu_r', vmin=-vmax_compare, vmax=vmax_compare, aspect='auto')
        axes_attn_compare[1].set_yticks(range(num_presentations))
        axes_attn_compare[1].set_yticklabels(presentation_labels, fontsize=7)
        axes_attn_compare[1].set_xticks(range(len(test_labels)))
        axes_attn_compare[1].set_xticklabels(test_labels, fontsize=7, rotation=45, ha='right')
        axes_attn_compare[1].set_ylabel('Adjacent Pair')
        axes_attn_compare[1].set_xlabel('Test Pair')
        axes_attn_compare[1].set_title('Layer 2: ũ_j^T @ ũ_test\n(After plastic layer 1)')
        plt.colorbar(im_c2, ax=axes_attn_compare[1], label='Dot product')

        plt.suptitle('Attention Comparison: How does Layer 1 plasticity transform attention patterns?', fontsize=12)
        plt.tight_layout()
        figures["pca_frozen/pw_attention_comparison"] = fig_attn_compare
        plt.close(fig_attn_compare)

        # =====================================================================
        # Plot 4: Attention by Item Overlap (Layer 1)
        # =====================================================================
        overlap_categories = ['shares_item1', 'shares_item2', 'shares_both', 'shares_none']
        colors_overlap = {'shares_item1': 'blue', 'shares_item2': 'green', 'shares_both': 'purple', 'shares_none': 'gray'}

        # Pick representative test pairs across different symbolic distances
        representative_test_indices = [0, 1, len(test_pairs)//2, len(test_pairs)//2 + 1, -2, -1]
        representative_test_indices = [i % len(test_pairs) for i in representative_test_indices]

        fig_overlap_L1, axes_overlap_L1 = plt.subplots(2, 3, figsize=(15, 10), dpi=150)

        for plot_idx, test_idx in enumerate(representative_test_indices[:6]):
            ax = axes_overlap_L1.flatten()[plot_idx]
            item1_test, item2_test, _ = test_pairs[test_idx]

            overlap_data = {cat: [] for cat in overlap_categories}
            overlap_labels_data = {cat: [] for cat in overlap_categories}

            for adj_idx, (item1_adj, item2_adj) in enumerate(all_presentations):
                shares_1 = (item1_adj == item1_test or item2_adj == item1_test)
                shares_2 = (item1_adj == item2_test or item2_adj == item2_test)

                if shares_1 and shares_2:
                    cat = 'shares_both'
                elif shares_1:
                    cat = 'shares_item1'
                elif shares_2:
                    cat = 'shares_item2'
                else:
                    cat = 'shares_none'

                overlap_data[cat].append(attention_matrix_L1[adj_idx, test_idx])
                overlap_labels_data[cat].append(f"{item_labels_decomp[item1_adj]}{item_labels_decomp[item2_adj]}")

            x_pos = 0
            for cat in overlap_categories:
                if overlap_data[cat]:
                    ax.bar(range(x_pos, x_pos + len(overlap_data[cat])),
                           overlap_data[cat], color=colors_overlap[cat],
                           alpha=0.7, label=cat if plot_idx == 0 else None)
                    x_pos += len(overlap_data[cat]) + 1

            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.set_title(f'Test: {item_labels_decomp[item1_test]}{item_labels_decomp[item2_test]}, SD={test_signed_sds[test_idx]}')
            ax.set_ylabel('Attention (u_j^T @ u_test)')

        axes_overlap_L1[0, 0].legend(fontsize=8)
        plt.suptitle('Layer 1 Attention Grouped by Item Overlap', fontsize=12)
        plt.tight_layout()
        figures["pca_frozen/pw_attention_by_overlap_layer1"] = fig_overlap_L1
        plt.close(fig_overlap_L1)

        # =====================================================================
        # Plot 5: Attention by Item Overlap (Layer 2)
        # =====================================================================
        fig_overlap_L2, axes_overlap_L2 = plt.subplots(2, 3, figsize=(15, 10), dpi=150)

        for plot_idx, test_idx in enumerate(representative_test_indices[:6]):
            ax = axes_overlap_L2.flatten()[plot_idx]
            item1_test, item2_test, _ = test_pairs[test_idx]

            overlap_data = {cat: [] for cat in overlap_categories}

            for adj_idx, (item1_adj, item2_adj) in enumerate(all_presentations):
                shares_1 = (item1_adj == item1_test or item2_adj == item1_test)
                shares_2 = (item1_adj == item2_test or item2_adj == item2_test)

                if shares_1 and shares_2:
                    cat = 'shares_both'
                elif shares_1:
                    cat = 'shares_item1'
                elif shares_2:
                    cat = 'shares_item2'
                else:
                    cat = 'shares_none'

                overlap_data[cat].append(attention_matrix_L2[adj_idx, test_idx])

            x_pos = 0
            for cat in overlap_categories:
                if overlap_data[cat]:
                    ax.bar(range(x_pos, x_pos + len(overlap_data[cat])),
                           overlap_data[cat], color=colors_overlap[cat],
                           alpha=0.7, label=cat if plot_idx == 0 else None)
                    x_pos += len(overlap_data[cat]) + 1

            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax.set_title(f'Test: {item_labels_decomp[item1_test]}{item_labels_decomp[item2_test]}, SD={test_signed_sds[test_idx]}')
            ax.set_ylabel('Attention (ũ_j^T @ ũ_test)')

        axes_overlap_L2[0, 0].legend(fontsize=8)
        plt.suptitle('Layer 2 Attention Grouped by Item Overlap (After Plastic Layer 1)', fontsize=12)
        plt.tight_layout()
        figures["pca_frozen/pw_attention_by_overlap_layer2"] = fig_overlap_L2
        plt.close(fig_overlap_L2)

        # =====================================================================
        # Plot 6: Weighted Contribution from Adjacent Pairs to Test Pairs
        # For each test pair, compute the contribution:
        #   - Layer 1: Σ_j A[i,j] * (u_j^T @ u_test) summed over i
        #   - Layer 2: Σ_j B[i,j] * (ũ_j^T @ ũ_test) summed over i
        # =====================================================================

        # A_col_sums[j] = Σ_i A[i,j] (total contribution weight for embedding j)
        A_col_sums = A_coeffs.sum(axis=0)  # (num_presentations,)
        B_col_sums = B_coeffs.sum(axis=0)  # (num_presentations,)

        # Weighted contribution for each test pair
        weighted_contrib_L1 = A_col_sums @ attention_matrix_L1  # (num_test,)
        weighted_contrib_L2 = B_col_sums @ attention_matrix_L2  # (num_test,)

        fig_contrib, axes_contrib = plt.subplots(2, 2, figsize=(16, 12), dpi=150)

        # Top left: Layer 1 weighted contribution vs symbolic distance
        scatter_L1 = axes_contrib[0, 0].scatter(range(len(test_signed_sds)), weighted_contrib_L1,
                                                 c=test_signed_sds, cmap='coolwarm', s=50, edgecolors='black')
        for i, label in enumerate(test_labels):
            axes_contrib[0, 0].annotate(label, (i, weighted_contrib_L1[i]), fontsize=6, ha='center', va='bottom')
        axes_contrib[0, 0].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        axes_contrib[0, 0].set_xlabel('Test pair index')
        axes_contrib[0, 0].set_ylabel('Weighted contribution Σ_j A_col[j] · (u_j^T @ u_test)')
        axes_contrib[0, 0].set_title('Layer 1: Coefficient-weighted attention')
        plt.colorbar(scatter_L1, ax=axes_contrib[0, 0], label='Signed SD')

        # Top right: Layer 2 weighted contribution vs symbolic distance
        scatter_L2 = axes_contrib[0, 1].scatter(range(len(test_signed_sds)), weighted_contrib_L2,
                                                 c=test_signed_sds, cmap='coolwarm', s=50, edgecolors='black')
        for i, label in enumerate(test_labels):
            axes_contrib[0, 1].annotate(label, (i, weighted_contrib_L2[i]), fontsize=6, ha='center', va='bottom')
        axes_contrib[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        axes_contrib[0, 1].set_xlabel('Test pair index')
        axes_contrib[0, 1].set_ylabel('Weighted contribution Σ_j B_col[j] · (ũ_j^T @ ũ_test)')
        axes_contrib[0, 1].set_title('Layer 2: Coefficient-weighted attention')
        plt.colorbar(scatter_L2, ax=axes_contrib[0, 1], label='Signed SD')

        # Bottom left: Both layers combined
        axes_contrib[1, 0].scatter(weighted_contrib_L1, weighted_contrib_L2,
                                    c=test_signed_sds, cmap='coolwarm', s=50, edgecolors='black')
        for i, label in enumerate(test_labels):
            axes_contrib[1, 0].annotate(label, (weighted_contrib_L1[i], weighted_contrib_L2[i]), fontsize=6)
        axes_contrib[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        axes_contrib[1, 0].axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        axes_contrib[1, 0].set_xlabel('Layer 1 weighted contribution')
        axes_contrib[1, 0].set_ylabel('Layer 2 weighted contribution')
        axes_contrib[1, 0].set_title('Layer 1 vs Layer 2 contributions')

        # Bottom right: Contribution by absolute symbolic distance
        abs_sds = [abs(sd) for sd in test_signed_sds]
        unique_abs_sds = sorted(set(abs_sds))

        L1_by_sd = {sd: [] for sd in unique_abs_sds}
        L2_by_sd = {sd: [] for sd in unique_abs_sds}
        for i, sd in enumerate(abs_sds):
            L1_by_sd[sd].append(weighted_contrib_L1[i])
            L2_by_sd[sd].append(weighted_contrib_L2[i])

        x_positions = np.arange(len(unique_abs_sds))
        width = 0.35

        L1_means = [np.mean(L1_by_sd[sd]) for sd in unique_abs_sds]
        L1_stds = [np.std(L1_by_sd[sd]) for sd in unique_abs_sds]
        L2_means = [np.mean(L2_by_sd[sd]) for sd in unique_abs_sds]
        L2_stds = [np.std(L2_by_sd[sd]) for sd in unique_abs_sds]

        axes_contrib[1, 1].bar(x_positions - width/2, L1_means, width, yerr=L1_stds, label='Layer 1', alpha=0.7, capsize=3)
        axes_contrib[1, 1].bar(x_positions + width/2, L2_means, width, yerr=L2_stds, label='Layer 2', alpha=0.7, capsize=3)
        axes_contrib[1, 1].set_xticks(x_positions)
        axes_contrib[1, 1].set_xticklabels([f'SD={sd}' for sd in unique_abs_sds])
        axes_contrib[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        axes_contrib[1, 1].set_xlabel('Absolute Symbolic Distance')
        axes_contrib[1, 1].set_ylabel('Mean weighted contribution')
        axes_contrib[1, 1].set_title('Contribution by Symbolic Distance')
        axes_contrib[1, 1].legend()

        plt.suptitle('Weighted Contribution Analysis: How Adjacent Pairs Contribute to Test Pairs', fontsize=12)
        plt.tight_layout()
        figures["pca_frozen/pw_attention_weighted_contribution"] = fig_contrib
        plt.close(fig_contrib)

        # =====================================================================
        # Plot 7: Per-adjacent-pair contribution breakdown for each test pair
        # Shows which adjacent pairs contribute most to each test outcome
        # =====================================================================

        # Compute individual contributions: contrib[adj, test] = A_col[adj] * attention[adj, test]
        individual_contrib_L1 = A_col_sums[:, np.newaxis] * attention_matrix_L1  # (num_adj, num_test)
        individual_contrib_L2 = B_col_sums[:, np.newaxis] * attention_matrix_L2  # (num_adj, num_test)

        fig_breakdown, axes_breakdown = plt.subplots(1, 2, figsize=(18, 8), dpi=150)

        vmax_breakdown = max(abs(individual_contrib_L1.min()), abs(individual_contrib_L1.max()),
                             abs(individual_contrib_L2.min()), abs(individual_contrib_L2.max()))
        if vmax_breakdown == 0:
            vmax_breakdown = 1

        im_bd1 = axes_breakdown[0].imshow(individual_contrib_L1, cmap='RdBu_r', vmin=-vmax_breakdown, vmax=vmax_breakdown, aspect='auto')
        axes_breakdown[0].set_yticks(range(num_presentations))
        axes_breakdown[0].set_yticklabels(presentation_labels, fontsize=7)
        axes_breakdown[0].set_xticks(range(len(test_labels)))
        axes_breakdown[0].set_xticklabels(test_labels, fontsize=7, rotation=45, ha='right')
        axes_breakdown[0].set_ylabel('Adjacent Pair')
        axes_breakdown[0].set_xlabel('Test Pair')
        axes_breakdown[0].set_title('Layer 1: A_col[j] * (u_j^T @ u_test)\n(Per-pair contribution to output)')
        plt.colorbar(im_bd1, ax=axes_breakdown[0], label='Contribution')

        im_bd2 = axes_breakdown[1].imshow(individual_contrib_L2, cmap='RdBu_r', vmin=-vmax_breakdown, vmax=vmax_breakdown, aspect='auto')
        axes_breakdown[1].set_yticks(range(num_presentations))
        axes_breakdown[1].set_yticklabels(presentation_labels, fontsize=7)
        axes_breakdown[1].set_xticks(range(len(test_labels)))
        axes_breakdown[1].set_xticklabels(test_labels, fontsize=7, rotation=45, ha='right')
        axes_breakdown[1].set_ylabel('Adjacent Pair')
        axes_breakdown[1].set_xlabel('Test Pair')
        axes_breakdown[1].set_title('Layer 2: B_col[j] * (ũ_j^T @ ũ_test)\n(Per-pair contribution to output)')
        plt.colorbar(im_bd2, ax=axes_breakdown[1], label='Contribution')

        plt.suptitle('Per-Adjacent-Pair Contribution Breakdown', fontsize=12)
        plt.tight_layout()
        figures["pca_frozen/pw_attention_contribution_breakdown"] = fig_breakdown
        plt.close(fig_breakdown)

        # =====================================================================
        # Plot 8: Attention transformation by Layer 1 plasticity
        # Shows how the attention pattern changes from L1 to L2
        # =====================================================================
        attention_diff = attention_matrix_L2 - attention_matrix_L1

        fig_attn_diff, axes_attn_diff = plt.subplots(1, 3, figsize=(20, 6), dpi=150)

        vmax_attn_all = max(vmax_attn_L1, vmax_attn_L2)
        vmax_diff = max(abs(attention_diff.min()), abs(attention_diff.max()))
        if vmax_diff == 0:
            vmax_diff = 1

        im_diff1 = axes_attn_diff[0].imshow(attention_matrix_L1, cmap='RdBu_r', vmin=-vmax_attn_all, vmax=vmax_attn_all, aspect='auto')
        axes_attn_diff[0].set_yticks(range(num_presentations))
        axes_attn_diff[0].set_yticklabels(presentation_labels, fontsize=6)
        axes_attn_diff[0].set_xticks(range(len(test_labels)))
        axes_attn_diff[0].set_xticklabels(test_labels, fontsize=6, rotation=45, ha='right')
        axes_attn_diff[0].set_title('Before: u_j^T @ u_test')
        plt.colorbar(im_diff1, ax=axes_attn_diff[0])

        im_diff2 = axes_attn_diff[1].imshow(attention_matrix_L2, cmap='RdBu_r', vmin=-vmax_attn_all, vmax=vmax_attn_all, aspect='auto')
        axes_attn_diff[1].set_yticks(range(num_presentations))
        axes_attn_diff[1].set_yticklabels(presentation_labels, fontsize=6)
        axes_attn_diff[1].set_xticks(range(len(test_labels)))
        axes_attn_diff[1].set_xticklabels(test_labels, fontsize=6, rotation=45, ha='right')
        axes_attn_diff[1].set_title('After: ũ_j^T @ ũ_test')
        plt.colorbar(im_diff2, ax=axes_attn_diff[1])

        im_diff3 = axes_attn_diff[2].imshow(attention_diff, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff, aspect='auto')
        axes_attn_diff[2].set_yticks(range(num_presentations))
        axes_attn_diff[2].set_yticklabels(presentation_labels, fontsize=6)
        axes_attn_diff[2].set_xticks(range(len(test_labels)))
        axes_attn_diff[2].set_xticklabels(test_labels, fontsize=6, rotation=45, ha='right')
        axes_attn_diff[2].set_title('Difference: (ũ_j^T @ ũ_test) - (u_j^T @ u_test)')
        plt.colorbar(im_diff3, ax=axes_attn_diff[2])

        plt.suptitle('How Layer 1 Plasticity Transforms Attention Patterns', fontsize=12)
        plt.tight_layout()
        figures["pca_frozen/pw_attention_transformation"] = fig_attn_diff
        plt.close(fig_attn_diff)

        # =====================================================================
        # BATCH-AVERAGED ATTENTION ANALYSIS (TI)
        # Compute attention matrices averaged across all networks in the batch
        # =====================================================================
        num_networks = batch_items.shape[0]

        # Accumulators for attention matrices
        all_attention_L1_ti = []
        all_attention_L2_ti = []

        for net_idx in range(num_networks):
            net_items = batch_items[net_idx]
            net_pw_layer1 = frozen_extra_plastic_weights[0][net_idx]  # (H, H)

            # Compute embeddings for adjacent pairs
            net_embeddings_u = []
            net_embeddings_u_tilde = []
            with torch.no_grad():
                for (item1_idx, item2_idx) in all_presentations:
                    item1_emb = net_items[item1_idx]
                    item2_emb = net_items[item2_idx]
                    input_vec = np.concatenate([item1_emb, item2_emb])
                    input_t = torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0).to(device)

                    # u = embedding layer output
                    if hasattr(model, 'embedding_layer'):
                        u = torch.tanh(model.embedding_layer(input_t))
                    else:
                        u = input_t
                    net_embeddings_u.append(u.squeeze(0).cpu().numpy())

                    # ũ = layer 1 output with plastic weights
                    alpha_layer1 = model.alpha_extra[0]
                    W_layer1 = model.extra_hidden_layers[0].weight
                    b_layer1 = model.extra_hidden_layers[0].bias
                    innate = W_layer1 @ u.T
                    if b_layer1 is not None:
                        innate = innate + b_layer1.unsqueeze(1)
                    u_tilde = torch.tanh(innate + (alpha_layer1 * net_pw_layer1) @ u.T)
                    net_embeddings_u_tilde.append(u_tilde.T.squeeze(0).cpu().numpy())

            net_embeddings_u = np.array(net_embeddings_u)
            net_embeddings_u_tilde = np.array(net_embeddings_u_tilde)

            # Compute embeddings for test pairs
            net_test_embeddings_u = []
            net_test_embeddings_u_tilde = []
            with torch.no_grad():
                for (item1_idx, item2_idx, correct) in test_pairs:
                    item1_emb = net_items[item1_idx]
                    item2_emb = net_items[item2_idx]
                    input_vec = np.concatenate([item1_emb, item2_emb])
                    input_t = torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0).to(device)

                    if hasattr(model, 'embedding_layer'):
                        u_test = torch.tanh(model.embedding_layer(input_t))
                    else:
                        u_test = input_t
                    net_test_embeddings_u.append(u_test.squeeze(0).cpu().numpy())

                    alpha_layer1 = model.alpha_extra[0]
                    W_layer1 = model.extra_hidden_layers[0].weight
                    b_layer1 = model.extra_hidden_layers[0].bias
                    innate = W_layer1 @ u_test.T
                    if b_layer1 is not None:
                        innate = innate + b_layer1.unsqueeze(1)
                    u_tilde_test = torch.tanh(innate + (alpha_layer1 * net_pw_layer1) @ u_test.T)
                    net_test_embeddings_u_tilde.append(u_tilde_test.T.squeeze(0).cpu().numpy())

            net_test_embeddings_u = np.array(net_test_embeddings_u)
            net_test_embeddings_u_tilde = np.array(net_test_embeddings_u_tilde)

            # Compute attention matrices for this network
            net_attention_L1 = net_embeddings_u @ net_test_embeddings_u.T
            net_attention_L2 = net_embeddings_u_tilde @ net_test_embeddings_u_tilde.T

            all_attention_L1_ti.append(net_attention_L1)
            all_attention_L2_ti.append(net_attention_L2)

        # Convert to arrays and compute statistics
        all_attention_L1_ti = np.array(all_attention_L1_ti)  # (num_networks, num_adj, num_test)
        all_attention_L2_ti = np.array(all_attention_L2_ti)

        mean_attention_L1_ti = all_attention_L1_ti.mean(axis=0)
        std_attention_L1_ti = all_attention_L1_ti.std(axis=0)
        mean_attention_L2_ti = all_attention_L2_ti.mean(axis=0)
        std_attention_L2_ti = all_attention_L2_ti.std(axis=0)

        # =====================================================================
        # Plot: Batch-Averaged Attention Comparison (L1 vs L2) for TI
        # =====================================================================
        fig_avg_compare_ti, axes_avg_compare_ti = plt.subplots(2, 2, figsize=(20, 14), dpi=150)

        vmax_avg_ti = max(abs(mean_attention_L1_ti.min()), abs(mean_attention_L1_ti.max()),
                          abs(mean_attention_L2_ti.min()), abs(mean_attention_L2_ti.max()))
        if vmax_avg_ti == 0:
            vmax_avg_ti = 1

        # Top row: Mean attention
        im_avg1_ti = axes_avg_compare_ti[0, 0].imshow(mean_attention_L1_ti, cmap='RdBu_r', vmin=-vmax_avg_ti, vmax=vmax_avg_ti, aspect='auto')
        axes_avg_compare_ti[0, 0].set_yticks(range(num_presentations))
        axes_avg_compare_ti[0, 0].set_yticklabels(presentation_labels, fontsize=7)
        axes_avg_compare_ti[0, 0].set_xticks(range(len(test_labels)))
        axes_avg_compare_ti[0, 0].set_xticklabels(test_labels, fontsize=6, rotation=45, ha='right')
        axes_avg_compare_ti[0, 0].set_ylabel('Adjacent Pair')
        axes_avg_compare_ti[0, 0].set_xlabel('Test Pair')
        axes_avg_compare_ti[0, 0].set_title(f'Layer 1 MEAN Attention (n={num_networks} networks)')
        plt.colorbar(im_avg1_ti, ax=axes_avg_compare_ti[0, 0], label='Mean dot product')

        im_avg2_ti = axes_avg_compare_ti[0, 1].imshow(mean_attention_L2_ti, cmap='RdBu_r', vmin=-vmax_avg_ti, vmax=vmax_avg_ti, aspect='auto')
        axes_avg_compare_ti[0, 1].set_yticks(range(num_presentations))
        axes_avg_compare_ti[0, 1].set_yticklabels(presentation_labels, fontsize=7)
        axes_avg_compare_ti[0, 1].set_xticks(range(len(test_labels)))
        axes_avg_compare_ti[0, 1].set_xticklabels(test_labels, fontsize=6, rotation=45, ha='right')
        axes_avg_compare_ti[0, 1].set_ylabel('Adjacent Pair')
        axes_avg_compare_ti[0, 1].set_xlabel('Test Pair')
        axes_avg_compare_ti[0, 1].set_title(f'Layer 2 MEAN Attention (n={num_networks} networks)')
        plt.colorbar(im_avg2_ti, ax=axes_avg_compare_ti[0, 1], label='Mean dot product')

        # Bottom row: Std attention (to see consistency)
        vmax_std_ti = max(std_attention_L1_ti.max(), std_attention_L2_ti.max())
        if vmax_std_ti == 0:
            vmax_std_ti = 1

        im_std1_ti = axes_avg_compare_ti[1, 0].imshow(std_attention_L1_ti, cmap='viridis', vmin=0, vmax=vmax_std_ti, aspect='auto')
        axes_avg_compare_ti[1, 0].set_yticks(range(num_presentations))
        axes_avg_compare_ti[1, 0].set_yticklabels(presentation_labels, fontsize=7)
        axes_avg_compare_ti[1, 0].set_xticks(range(len(test_labels)))
        axes_avg_compare_ti[1, 0].set_xticklabels(test_labels, fontsize=6, rotation=45, ha='right')
        axes_avg_compare_ti[1, 0].set_ylabel('Adjacent Pair')
        axes_avg_compare_ti[1, 0].set_xlabel('Test Pair')
        axes_avg_compare_ti[1, 0].set_title('Layer 1 STD Attention (consistency across networks)')
        plt.colorbar(im_std1_ti, ax=axes_avg_compare_ti[1, 0], label='Std')

        im_std2_ti = axes_avg_compare_ti[1, 1].imshow(std_attention_L2_ti, cmap='viridis', vmin=0, vmax=vmax_std_ti, aspect='auto')
        axes_avg_compare_ti[1, 1].set_yticks(range(num_presentations))
        axes_avg_compare_ti[1, 1].set_yticklabels(presentation_labels, fontsize=7)
        axes_avg_compare_ti[1, 1].set_xticks(range(len(test_labels)))
        axes_avg_compare_ti[1, 1].set_xticklabels(test_labels, fontsize=6, rotation=45, ha='right')
        axes_avg_compare_ti[1, 1].set_ylabel('Adjacent Pair')
        axes_avg_compare_ti[1, 1].set_xlabel('Test Pair')
        axes_avg_compare_ti[1, 1].set_title('Layer 2 STD Attention (consistency across networks)')
        plt.colorbar(im_std2_ti, ax=axes_avg_compare_ti[1, 1], label='Std')

        plt.suptitle('TI: Batch-Averaged Attention (Mean and Std)', fontsize=14)
        plt.tight_layout()
        figures["pca_frozen/pw_attention_batch_averaged"] = fig_avg_compare_ti
        plt.close(fig_avg_compare_ti)

    else:
        # Track final layer plastic weights when extra_layers = 0
        # For one network (network_idx = 0), track the coefficient matrix A
        # where P = Σ_ij A[i,j] * v_i * u_j^T with v_i = W @ u_i

        network_idx = 0
        single_items = batch_items[network_idx]  # (num_items, item_size)

        # Define adjacent pairs for TI (all adjacent item pairs)
        adjacent_pairs_ti = [(i, i+1) for i in range(num_items - 1)]

        # Create all presentations (both orderings: winner first, loser first)
        all_presentations = []
        presentation_to_idx = {}
        for pair in adjacent_pairs_ti:
            winner, loser = pair
            all_presentations.append((winner, loser))
            presentation_to_idx[(winner, loser)] = len(all_presentations) - 1
            all_presentations.append((loser, winner))
            presentation_to_idx[(loser, winner)] = len(all_presentations) - 1

        num_presentations = len(all_presentations)

        # Compute embeddings u_i for each presentation (after embedding layer + tanh)
        embeddings_u = []
        with torch.no_grad():
            for (item1_idx, item2_idx) in all_presentations:
                item1_emb = single_items[item1_idx]
                item2_emb = single_items[item2_idx]
                input_vec = np.concatenate([item1_emb, item2_emb])
                input_t = torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0).to(device)
                if hasattr(model, 'embedding_layer'):
                    u = torch.tanh(model.embedding_layer(input_t))
                else:
                    u = input_t
                embeddings_u.append(u.squeeze(0).cpu().numpy())

        embeddings_u = np.array(embeddings_u)

        # Get innate weights W for final layer (fc2)
        W_final = model.fc2.weight.detach().cpu().numpy()

        # Get alpha matrix for final layer
        alpha_final_layer = model.alpha.detach().cpu().numpy()

        # Get hebbian trace multiplier for final layer
        m_hebb = model.hebbian_trace_multiplier.item()

        # Compute v_i = W @ u_i for each presentation
        embeddings_v = (W_final @ embeddings_u.T).T

        # Compute D_alpha matrix
        if alpha_final_layer.ndim >= 2:
            alpha_mean = alpha_final_layer.mean(axis=0)
        else:
            alpha_mean = float(alpha_final_layer)
        D_alpha = (embeddings_u * alpha_mean) @ embeddings_u.T

        # Initialize coefficient matrix A = 0
        A_coeffs = np.zeros((num_presentations, num_presentations))

        # Re-run training to track coefficients
        pw_track = torch.zeros(1, args.hidden_size, args.hidden_size, dtype=torch.float32).to(device)
        epw_track = []  # No extra layers

        single_trials = trials[network_idx:network_idx+1, :, :]
        single_correct = correct_choices[network_idx:network_idx+1, :]

        for trial_idx in range(args.num_train_trials):
            trial_input = single_trials[:, trial_idx, :]
            trial_correct = single_correct[:, trial_idx]

            item_size = args.item_size
            item1_emb_trial = trial_input[0, :item_size].cpu().numpy()
            item2_emb_trial = trial_input[0, item_size:2*item_size].cpu().numpy()

            item1_idx_found = None
            item2_idx_found = None
            for idx in range(num_items):
                if np.allclose(single_items[idx], item1_emb_trial, atol=1e-5):
                    item1_idx_found = idx
                if np.allclose(single_items[idx], item2_emb_trial, atol=1e-5):
                    item2_idx_found = idx

            if item1_idx_found is None or item2_idx_found is None:
                continue

            presentation_key = (item1_idx_found, item2_idx_found)
            if presentation_key not in presentation_to_idx:
                continue

            k = presentation_to_idx[presentation_key]

            with torch.no_grad():
                output_track = model(trial_input, pw_track, trial_correct,
                                    extra_plastic_weights=epw_track, store_embeddings=False)

            # Get neuromodulator for final layer (last value or only value)
            nm_output = output_track.neuromodulator.squeeze()
            eta_t = nm_output.item() if nm_output.dim() == 0 else nm_output[-1].item()

            # Update coefficient matrix
            e_k = np.zeros(num_presentations)
            e_k[k] = 1.0
            # Scale by ~0.9 to approximate tanh compression on outer products in [-1,1]
            tanh_scale = 0.9
            A_coeffs[:, k] += eta_t * m_hebb * tanh_scale * (e_k + A_coeffs @ D_alpha[:, k])

            pw_track = output_track.plastic_weights

        # Create labels for presentations
        item_labels_decomp = [chr(ord('A') + i) for i in range(num_items)]
        presentation_labels = [f"{item_labels_decomp[i1]}{item_labels_decomp[i2]}" for (i1, i2) in all_presentations]

        # Plot 1: Coefficient matrix A
        fig_coeffs, ax_coeffs = plt.subplots(figsize=(10, 8), dpi=150)
        vmax_coeffs = max(abs(A_coeffs.min()), abs(A_coeffs.max()))
        if vmax_coeffs == 0:
            vmax_coeffs = 1
        im_coeffs = ax_coeffs.imshow(A_coeffs, cmap='RdBu_r', vmin=-vmax_coeffs, vmax=vmax_coeffs, aspect='equal')
        ax_coeffs.set_xticks(range(num_presentations))
        ax_coeffs.set_xticklabels(presentation_labels, rotation=45, ha='right', fontsize=8)
        ax_coeffs.set_yticks(range(num_presentations))
        ax_coeffs.set_yticklabels(presentation_labels, fontsize=8)
        ax_coeffs.set_xlabel('j (u_j column)')
        ax_coeffs.set_ylabel('i (v_i row)')
        ax_coeffs.set_title('Coefficient Matrix A - Final Layer\n(P ≈ Σ_ij A[i,j] · v_i · u_j^T)')
        plt.colorbar(im_coeffs, ax=ax_coeffs, label='Coefficient value')
        plt.tight_layout()
        figures["pca_frozen/pw_decomposition_coefficients"] = fig_coeffs
        plt.close(fig_coeffs)

        # Plot 2: Reconstruction error
        P_actual = frozen_plastic_weights[network_idx].detach().cpu().numpy()
        P_reconstructed = np.zeros_like(P_actual)
        for i in range(num_presentations):
            for j in range(num_presentations):
                P_reconstructed += A_coeffs[i, j] * np.outer(embeddings_v[i], embeddings_u[j])

        reconstruction_error = np.linalg.norm(P_actual - P_reconstructed) / np.linalg.norm(P_actual) if np.linalg.norm(P_actual) > 0 else 0.0

        fig_recon, axes_recon = plt.subplots(1, 3, figsize=(18, 5), dpi=150)

        vmax_p = max(abs(P_actual.min()), abs(P_actual.max()), abs(P_reconstructed.min()), abs(P_reconstructed.max()))
        if vmax_p == 0:
            vmax_p = 1

        im0 = axes_recon[0].imshow(P_actual, cmap='RdBu_r', vmin=-vmax_p, vmax=vmax_p, aspect='equal')
        axes_recon[0].set_title('Actual P (final layer)')
        plt.colorbar(im0, ax=axes_recon[0])

        im1 = axes_recon[1].imshow(P_reconstructed, cmap='RdBu_r', vmin=-vmax_p, vmax=vmax_p, aspect='equal')
        axes_recon[1].set_title('Reconstructed P from coefficients')
        plt.colorbar(im1, ax=axes_recon[1])

        P_diff = P_actual - P_reconstructed
        vmax_diff = max(abs(P_diff.min()), abs(P_diff.max()))
        if vmax_diff == 0:
            vmax_diff = 1
        im2 = axes_recon[2].imshow(P_diff, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff, aspect='equal')
        axes_recon[2].set_title(f'Difference (error = {reconstruction_error:.4f})')
        plt.colorbar(im2, ax=axes_recon[2])

        plt.suptitle('Plastic Weight Decomposition Verification (TI) - Final Layer', fontsize=12)
        plt.tight_layout()
        figures["pca_frozen/pw_decomposition_reconstruction"] = fig_recon
        plt.close(fig_recon)

        # --- D_alpha full 3D tensor and mean/std plots ---
        # Compute D_alpha_full as 3D tensor: D_alpha_full[i,j,m] = sum_l alpha[m,l] * u_i[l] * u_j[l]
        if alpha_final_layer.ndim >= 2:
            hidden_size_decomp = alpha_final_layer.shape[0]
            D_alpha_full = np.zeros((num_presentations, num_presentations, hidden_size_decomp))
            for m in range(hidden_size_decomp):
                D_alpha_full[:, :, m] = (embeddings_u * alpha_final_layer[m, :]) @ embeddings_u.T

            D_alpha_mean = D_alpha_full.mean(axis=2)
            D_alpha_std = D_alpha_full.std(axis=2)
        else:
            D_alpha_mean = float(alpha_final_layer) * (embeddings_u @ embeddings_u.T)
            D_alpha_std = np.zeros_like(D_alpha_mean)

        # Plot D_alpha mean and std side by side
        fig_dots, axes_dots = plt.subplots(1, 2, figsize=(16, 7), dpi=150)

        # Left: Mean D_alpha (with symmetric colormap around 0)
        vmax_dalpha = max(abs(D_alpha_mean.min()), abs(D_alpha_mean.max()))
        if vmax_dalpha == 0:
            vmax_dalpha = 1
        im_mean = axes_dots[0].imshow(D_alpha_mean, cmap='RdBu_r', vmin=-vmax_dalpha, vmax=vmax_dalpha, aspect='equal')
        axes_dots[0].set_xticks(range(num_presentations))
        axes_dots[0].set_xticklabels(presentation_labels, rotation=45, ha='right', fontsize=8)
        axes_dots[0].set_yticks(range(num_presentations))
        axes_dots[0].set_yticklabels(presentation_labels, fontsize=8)
        axes_dots[0].set_xlabel('j')
        axes_dots[0].set_ylabel('i')
        axes_dots[0].set_title('Mean D_α[i,j] across output dimensions\n(D_α[i,j,m] = Σ_l α[m,l] · u_i[l] · u_j[l])')
        plt.colorbar(im_mean, ax=axes_dots[0], label='Mean alpha-weighted dot product')

        # Right: Std D_alpha
        im_std = axes_dots[1].imshow(D_alpha_std, cmap='plasma', aspect='equal')
        axes_dots[1].set_xticks(range(num_presentations))
        axes_dots[1].set_xticklabels(presentation_labels, rotation=45, ha='right', fontsize=8)
        axes_dots[1].set_yticks(range(num_presentations))
        axes_dots[1].set_yticklabels(presentation_labels, fontsize=8)
        axes_dots[1].set_xlabel('j')
        axes_dots[1].set_ylabel('i')
        axes_dots[1].set_title('Std D_α[i,j] across output dimensions\n(variation in alpha weighting)')
        plt.colorbar(im_std, ax=axes_dots[1], label='Std of alpha-weighted dot product')

        plt.suptitle('D_alpha Matrix Analysis (TI) - Final Layer', fontsize=12)
        plt.tight_layout()
        figures["pca_frozen/pw_decomposition_d_alpha"] = fig_dots
        plt.close(fig_dots)

        # --- Residual coefficients in expanded basis ---
        # R = P - P_reconstructed, project onto u_j basis: R_coeff[m,j] = (R @ U_pinv^T)[m,j]
        U_matrix = embeddings_u.T  # (hidden_size, num_presentations)
        U_pinv = np.linalg.pinv(U_matrix)  # (num_presentations, hidden_size)
        R_coeff = P_diff @ U_pinv.T  # (hidden_size, num_presentations)

        # Reconstruction with residuals
        P_with_residual = P_reconstructed + R_coeff @ U_matrix.T
        reconstruction_error_with_residual = np.linalg.norm(P_actual - P_with_residual) / np.linalg.norm(P_actual) if np.linalg.norm(P_actual) > 0 else 0.0

        # Plot residual coefficients
        fig_resid, ax_resid = plt.subplots(figsize=(12, 8), dpi=150)
        vmax_resid = max(abs(R_coeff.min()), abs(R_coeff.max()))
        if vmax_resid == 0:
            vmax_resid = 1
        im_resid = ax_resid.imshow(R_coeff, cmap='RdBu_r', vmin=-vmax_resid, vmax=vmax_resid, aspect='auto')
        ax_resid.set_xticks(range(num_presentations))
        ax_resid.set_xticklabels(presentation_labels, rotation=45, ha='right', fontsize=8)
        ax_resid.set_xlabel('j (u_j basis vector)')
        ax_resid.set_ylabel('m (output dimension)')
        ax_resid.set_title('Residual Coefficients R[m,j] - Final Layer\n(what the v_i basis cannot capture, in the e_m ⊗ u_j basis)')
        plt.colorbar(im_resid, ax=ax_resid, label='Residual coefficient')
        plt.tight_layout()
        figures["pca_frozen/pw_decomposition_residual"] = fig_resid
        plt.close(fig_resid)

        # Plot reconstruction comparison: without vs with residuals
        fig_recon_compare, axes_rc = plt.subplots(1, 3, figsize=(18, 5), dpi=150)

        im_rc0 = axes_rc[0].imshow(P_actual, cmap='RdBu_r', vmin=-vmax_p, vmax=vmax_p, aspect='equal')
        axes_rc[0].set_title('Actual P')
        plt.colorbar(im_rc0, ax=axes_rc[0])

        im_rc1 = axes_rc[1].imshow(P_reconstructed, cmap='RdBu_r', vmin=-vmax_p, vmax=vmax_p, aspect='equal')
        axes_rc[1].set_title(f'v_i ⊗ u_j basis only\n(error = {reconstruction_error:.4f})')
        plt.colorbar(im_rc1, ax=axes_rc[1])

        im_rc2 = axes_rc[2].imshow(P_with_residual, cmap='RdBu_r', vmin=-vmax_p, vmax=vmax_p, aspect='equal')
        axes_rc[2].set_title(f'With e_m ⊗ u_j residuals\n(error = {reconstruction_error_with_residual:.4f})')
        plt.colorbar(im_rc2, ax=axes_rc[2])

        plt.suptitle('Reconstruction Comparison: Scalar vs Expanded Basis (TI) - Final Layer', fontsize=12)
        plt.tight_layout()
        figures["pca_frozen/pw_decomposition_reconstruction_comparison"] = fig_recon_compare
        plt.close(fig_recon_compare)

        # Residual norm summary
        fig_resid_summary, axes_resid_summary = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

        # Left: Residual norm per output dimension m
        resid_norm_per_m = np.linalg.norm(R_coeff, axis=1)  # (hidden_size,)
        axes_resid_summary[0].bar(range(len(resid_norm_per_m)), resid_norm_per_m, color='steelblue', alpha=0.7)
        axes_resid_summary[0].set_xlabel('Output dimension m')
        axes_resid_summary[0].set_ylabel('||R[m,:]||')
        axes_resid_summary[0].set_title('Residual norm per output dimension')

        # Right: Residual norm per u_j basis vector
        resid_norm_per_j = np.linalg.norm(R_coeff, axis=0)  # (num_presentations,)
        axes_resid_summary[1].bar(range(num_presentations), resid_norm_per_j, color='darkorange', alpha=0.7)
        axes_resid_summary[1].set_xticks(range(num_presentations))
        axes_resid_summary[1].set_xticklabels(presentation_labels, rotation=45, ha='right', fontsize=8)
        axes_resid_summary[1].set_xlabel('j (u_j basis vector)')
        axes_resid_summary[1].set_ylabel('||R[:,j]||')
        axes_resid_summary[1].set_title('Residual norm per u_j basis vector')

        plt.suptitle('Residual Analysis: Where does the v_i basis fail? - Final Layer', fontsize=12)
        plt.tight_layout()
        figures["pca_frozen/pw_decomposition_residual_summary"] = fig_resid_summary
        plt.close(fig_resid_summary)

        # --- Attention Analysis: How non-adjacent pairs attend to adjacent pairs ---
        # For each non-adjacent test pair, compute attention weights u_j^T @ u_test
        # and visualize the weighted contribution from A coefficients

        # Generate all non-adjacent test pairs (symbolic distance >= 2)
        test_pairs = []
        for sd in range(2, num_items):
            for i in range(num_items - sd):
                # Both presentation orders
                test_pairs.append((i, i + sd, 0))  # winner first, correct=0
                test_pairs.append((i + sd, i, 1))  # loser first, correct=1

        # Compute test pair embeddings
        test_embeddings = []
        test_labels = []
        test_signed_sds = []
        with torch.no_grad():
            for (item1_idx, item2_idx, correct) in test_pairs:
                item1_emb = single_items[item1_idx]
                item2_emb = single_items[item2_idx]
                input_vec = np.concatenate([item1_emb, item2_emb])
                input_t = torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0).to(device)
                if hasattr(model, 'embedding_layer'):
                    u_test = torch.tanh(model.embedding_layer(input_t))
                else:
                    u_test = input_t
                test_embeddings.append(u_test.squeeze(0).cpu().numpy())
                # Use letter labels (A, B, C, ...) instead of numeric indices
                test_labels.append(f"{item_labels_decomp[item1_idx]}{item_labels_decomp[item2_idx]}")
                # Signed SD: positive if winner (lower index) is first
                signed_sd = item2_idx - item1_idx  # positive when lower is first
                test_signed_sds.append(signed_sd)

        test_embeddings = np.array(test_embeddings)  # (num_test_pairs, hidden_size)

        # Compute attention matrix: attention[test_idx, adj_idx] = u_adj^T @ u_test
        attention_matrix = embeddings_u @ test_embeddings.T  # (num_adj, num_test)

        # Compute weighted A contribution for each test pair
        # For each test pair, sum over j: (u_j^T @ u_test) * sum_i(A[i,j])
        A_col_sums = A_coeffs.sum(axis=0)  # Sum over i for each j
        weighted_A_contributions = A_col_sums @ attention_matrix  # (num_test,)

        # Also compute contribution weighted by full A matrix
        # sum_ij A[i,j] * (u_j^T @ u_test)
        full_A_contributions = (A_coeffs.flatten() @ (embeddings_u @ test_embeddings.T).flatten().reshape(num_presentations, -1).T.flatten().reshape(-1, num_presentations)).diagonal() if False else None

        # Plot 1: Attention heatmap (which adjacent pairs does each test pair attend to?)
        fig_attn, ax_attn = plt.subplots(figsize=(14, 8), dpi=150)
        vmax_attn = max(abs(attention_matrix.min()), abs(attention_matrix.max()))
        if vmax_attn == 0:
            vmax_attn = 1
        im_attn = ax_attn.imshow(attention_matrix, cmap='RdBu_r', vmin=-vmax_attn, vmax=vmax_attn, aspect='auto')
        ax_attn.set_yticks(range(num_presentations))
        ax_attn.set_yticklabels(presentation_labels, fontsize=7)
        ax_attn.set_xticks(range(len(test_labels)))
        ax_attn.set_xticklabels(test_labels, fontsize=7, rotation=45, ha='right')
        ax_attn.set_ylabel('Adjacent Pair Embedding (u_j)')
        ax_attn.set_xlabel('Test Pair Embedding (u_test)')
        ax_attn.set_title('Attention: u_j^T @ u_test\n(How much test pairs attend to adjacent pairs)')
        plt.colorbar(im_attn, ax=ax_attn, label='Dot product')
        plt.tight_layout()
        figures["pca_frozen/pw_attention_to_adjacent"] = fig_attn
        plt.close(fig_attn)

        # Plot 2: Attention grouped by item overlap
        # For each test pair, categorize adjacent pairs by overlap type
        fig_overlap, axes_overlap = plt.subplots(2, 3, figsize=(15, 10), dpi=150)

        # Categorize overlaps
        overlap_categories = ['shares_item1', 'shares_item2', 'shares_both', 'shares_none']
        colors_overlap = {'shares_item1': 'blue', 'shares_item2': 'green', 'shares_both': 'purple', 'shares_none': 'gray'}

        # Pick a few representative test pairs
        representative_test_indices = [0, 1, len(test_pairs)//2, len(test_pairs)//2 + 1, -2, -1]
        representative_test_indices = [i % len(test_pairs) for i in representative_test_indices]

        for plot_idx, test_idx in enumerate(representative_test_indices[:6]):
            ax = axes_overlap.flatten()[plot_idx]
            item1_test, item2_test, _ = test_pairs[test_idx]

            # Categorize each adjacent pair
            overlap_data = {cat: [] for cat in overlap_categories}
            overlap_labels_data = {cat: [] for cat in overlap_categories}

            for adj_idx, (item1_adj, item2_adj) in enumerate(all_presentations):
                shares_1 = (item1_adj == item1_test or item2_adj == item1_test)
                shares_2 = (item1_adj == item2_test or item2_adj == item2_test)

                if shares_1 and shares_2:
                    cat = 'shares_both'
                elif shares_1:
                    cat = 'shares_item1'
                elif shares_2:
                    cat = 'shares_item2'
                else:
                    cat = 'shares_none'

                overlap_data[cat].append(attention_matrix[adj_idx, test_idx])
                overlap_labels_data[cat].append(f"{item_labels_decomp[item1_adj]}{item_labels_decomp[item2_adj]}")

            # Plot as grouped bars
            x_pos = 0
            for cat in overlap_categories:
                if overlap_data[cat]:
                    bars = ax.bar(range(x_pos, x_pos + len(overlap_data[cat])),
                                 overlap_data[cat], color=colors_overlap[cat],
                                 alpha=0.7, label=cat if plot_idx == 0 else None)
                    x_pos += len(overlap_data[cat]) + 1

            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            # Use letter labels for test pair
            test_label_1 = item_labels_decomp[item1_test]
            test_label_2 = item_labels_decomp[item2_test]
            ax.set_title(f'Test: {test_label_1}{test_label_2}, SD={test_signed_sds[test_idx]}')
            ax.set_ylabel('Attention (u_j^T @ u_test)')

        axes_overlap[0, 0].legend(fontsize=8)
        plt.suptitle('Attention Grouped by Item Overlap with Test Pair', fontsize=12)
        plt.tight_layout()
        figures["pca_frozen/pw_attention_by_overlap"] = fig_overlap
        plt.close(fig_overlap)

        # Plot 3: Weighted A contribution vs symbolic distance
        fig_weighted, ax_weighted = plt.subplots(figsize=(10, 6), dpi=150)

        # Color by signed symbolic distance
        scatter = ax_weighted.scatter(range(len(test_signed_sds)), weighted_A_contributions,
                                      c=test_signed_sds, cmap='coolwarm', s=50, edgecolors='black')
        ax_weighted.set_xlabel('Test Pair Index')
        ax_weighted.set_ylabel('Weighted A Contribution: Σ_j (u_j^T @ u_test) * Σ_i A[i,j]')
        ax_weighted.set_title('How A Coefficients Weight the Attention')
        ax_weighted.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        plt.colorbar(scatter, ax=ax_weighted, label='Signed Symbolic Distance')

        # Add trend line
        for label, idx, marker in [('Labels', range(len(test_labels)), 'o')]:
            pass  # Could add regression line here

        plt.tight_layout()
        figures["pca_frozen/pw_weighted_A_by_sd"] = fig_weighted
        plt.close(fig_weighted)

        # Plot 4: Mean attention by signed SD
        sd_to_attention = {}
        for test_idx, signed_sd in enumerate(test_signed_sds):
            if signed_sd not in sd_to_attention:
                sd_to_attention[signed_sd] = []
            sd_to_attention[signed_sd].append(weighted_A_contributions[test_idx])

        signed_sds_sorted = sorted(sd_to_attention.keys())
        mean_contributions = [np.mean(sd_to_attention[sd]) for sd in signed_sds_sorted]
        std_contributions = [np.std(sd_to_attention[sd]) for sd in signed_sds_sorted]

        fig_sd_mean, ax_sd_mean = plt.subplots(figsize=(10, 6), dpi=150)
        ax_sd_mean.bar(signed_sds_sorted, mean_contributions, yerr=std_contributions,
                       color='steelblue', edgecolor='black', alpha=0.8, capsize=3)
        ax_sd_mean.set_xlabel('Signed Symbolic Distance')
        ax_sd_mean.set_ylabel('Mean Weighted A Contribution')
        ax_sd_mean.set_title('Mean Weighted A Contribution by Signed Symbolic Distance')
        ax_sd_mean.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        plt.tight_layout()
        figures["pca_frozen/pw_mean_weighted_A_by_sd"] = fig_sd_mean
        plt.close(fig_sd_mean)

    # --- Plastic weight spectrum analysis ---
    # We'll analyze three matrices: pw_mean, alpha_pw, and W_effective
    # Store SVDs for later use in projection plots

    # SVD for raw plastic weights (for backward compatibility)
    U, S, Vh = np.linalg.svd(pw_mean)

    # SVD for alpha-modulated plastic weights
    U_alpha_pw, S_alpha_pw, Vh_alpha_pw = np.linalg.svd(alpha_pw_final)

    # SVD for effective weights
    U_eff, S_eff, Vh_eff = np.linalg.svd(W_effective_final)

    # Eigenvalue decomposition (can be complex for non-symmetric matrices)
    eigenvalues = np.linalg.eigvals(pw_mean)
    eigenvalue_magnitudes = np.abs(eigenvalues)
    eigenvalue_magnitudes_sorted = np.sort(eigenvalue_magnitudes)[::-1]

    # Plot 1: Singular value spectrum
    fig_sv, ax_sv = plt.subplots(figsize=(10, 6), dpi=150)

    ax_sv.bar(range(len(S)), S, color='steelblue', edgecolor='black', alpha=0.8)
    ax_sv.set_xlabel('Singular Value Index')
    ax_sv.set_ylabel('Singular Value')
    ax_sv.set_title('Singular Value Spectrum - Plastic Weights (Final Layer)')
    ax_sv.set_yscale('log')

    # Add cumulative variance explained as secondary y-axis
    ax_sv2 = ax_sv.twinx()
    cumulative_var = np.cumsum(S**2) / np.sum(S**2)
    ax_sv2.plot(range(len(S)), cumulative_var, 'r-', linewidth=2, label='Cumulative Var.')
    ax_sv2.set_ylabel('Cumulative Variance Explained', color='red')
    ax_sv2.tick_params(axis='y', labelcolor='red')
    ax_sv2.set_ylim(0, 1.05)
    ax_sv2.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax_sv2.axhline(y=0.99, color='red', linestyle=':', alpha=0.5, linewidth=1)

    plt.tight_layout()
    figures["pca_frozen/plastic_weights_singular_values_final"] = fig_sv
    plt.close(fig_sv)

    # Plot 1b: Combined singular value spectra for P, α, and α ⊙ P
    if alpha_final.ndim >= 2:
        # SVD for alpha matrix
        _, S_alpha, _ = np.linalg.svd(alpha_final)

        fig_sv_combined, ax_sv_combined = plt.subplots(figsize=(12, 6), dpi=150)

        # Plot all three spectra
        x_indices = np.arange(len(S))
        width = 0.25

        ax_sv_combined.bar(x_indices - width, S, width, color='steelblue', edgecolor='black', alpha=0.8, label='P (Plastic Weights)')
        ax_sv_combined.bar(x_indices, S_alpha, width, color='forestgreen', edgecolor='black', alpha=0.8, label='α (Alpha)')
        ax_sv_combined.bar(x_indices + width, S_alpha_pw, width, color='darkorange', edgecolor='black', alpha=0.8, label='α ⊙ P (Hadamard)')

        ax_sv_combined.set_xlabel('Singular Value Index')
        ax_sv_combined.set_ylabel('Singular Value')
        ax_sv_combined.set_title('Singular Value Spectra: P, α, and α ⊙ P (Final Layer)')
        ax_sv_combined.set_yscale('log')
        ax_sv_combined.legend(loc='upper right')

        # Only show first 20 indices for readability
        if len(S) > 20:
            ax_sv_combined.set_xlim(-0.5, 20.5)

        plt.tight_layout()
        figures["pca_frozen/singular_values_P_alpha_hadamard_final"] = fig_sv_combined
        plt.close(fig_sv_combined)

    # Plot 2: Eigenvalue spectrum (magnitudes)
    fig_eig, ax_eig = plt.subplots(figsize=(10, 6), dpi=150)

    ax_eig.bar(range(len(eigenvalue_magnitudes_sorted)), eigenvalue_magnitudes_sorted,
               color='darkorange', edgecolor='black', alpha=0.8)
    ax_eig.set_xlabel('Eigenvalue Index (sorted by magnitude)')
    ax_eig.set_ylabel('Eigenvalue Magnitude')
    ax_eig.set_title('Eigenvalue Magnitude Spectrum - Plastic Weights (Final Layer)')
    ax_eig.set_yscale('log')

    plt.tight_layout()
    figures["pca_frozen/plastic_weights_eigenvalues_final"] = fig_eig
    plt.close(fig_eig)

    # Spectrum analysis for extra plastic weights
    for layer_idx, epw in enumerate(frozen_extra_plastic_weights):
        epw_mean = epw.detach().cpu().numpy().mean(axis=0)

        U_epw, S_epw, Vh_epw = np.linalg.svd(epw_mean)
        eigenvalues_epw = np.linalg.eigvals(epw_mean)
        eigenvalue_magnitudes_epw = np.sort(np.abs(eigenvalues_epw))[::-1]

        # Singular value spectrum
        fig_sv_epw, ax_sv_epw = plt.subplots(figsize=(10, 6), dpi=150)
        ax_sv_epw.bar(range(len(S_epw)), S_epw, color='steelblue', edgecolor='black', alpha=0.8)
        ax_sv_epw.set_xlabel('Singular Value Index')
        ax_sv_epw.set_ylabel('Singular Value')
        ax_sv_epw.set_title(f'Singular Value Spectrum - Plastic Weights (Hidden Layer {layer_idx + 1})')
        ax_sv_epw.set_yscale('log')

        ax_sv_epw2 = ax_sv_epw.twinx()
        cumulative_var_epw = np.cumsum(S_epw**2) / np.sum(S_epw**2)
        ax_sv_epw2.plot(range(len(S_epw)), cumulative_var_epw, 'r-', linewidth=2)
        ax_sv_epw2.set_ylabel('Cumulative Variance Explained', color='red')
        ax_sv_epw2.tick_params(axis='y', labelcolor='red')
        ax_sv_epw2.set_ylim(0, 1.05)

        plt.tight_layout()
        figures[f"pca_frozen/plastic_weights_singular_values_hidden{layer_idx + 1}"] = fig_sv_epw
        plt.close(fig_sv_epw)

        # Combined singular value spectra for P, α, and α ⊙ P for extra layers
        alpha_extra = alpha_extra_list[layer_idx]
        alpha_pw_extra = alpha_extra * epw_mean  # Hadamard product

        _, S_alpha_extra, _ = np.linalg.svd(alpha_extra)
        _, S_alpha_pw_extra, _ = np.linalg.svd(alpha_pw_extra)

        fig_sv_combined_extra, ax_sv_combined_extra = plt.subplots(figsize=(12, 6), dpi=150)

        x_indices_extra = np.arange(len(S_epw))
        width_extra = 0.25

        ax_sv_combined_extra.bar(x_indices_extra - width_extra, S_epw, width_extra, color='steelblue', edgecolor='black', alpha=0.8, label='P (Plastic Weights)')
        ax_sv_combined_extra.bar(x_indices_extra, S_alpha_extra, width_extra, color='forestgreen', edgecolor='black', alpha=0.8, label='α (Alpha)')
        ax_sv_combined_extra.bar(x_indices_extra + width_extra, S_alpha_pw_extra, width_extra, color='darkorange', edgecolor='black', alpha=0.8, label='α ⊙ P (Hadamard)')

        ax_sv_combined_extra.set_xlabel('Singular Value Index')
        ax_sv_combined_extra.set_ylabel('Singular Value')
        ax_sv_combined_extra.set_title(f'Singular Value Spectra: P, α, and α ⊙ P (Hidden Layer {layer_idx + 1})')
        ax_sv_combined_extra.set_yscale('log')
        ax_sv_combined_extra.legend(loc='upper right')

        if len(S_epw) > 20:
            ax_sv_combined_extra.set_xlim(-0.5, 20.5)

        plt.tight_layout()
        figures[f"pca_frozen/singular_values_P_alpha_hadamard_hidden{layer_idx + 1}"] = fig_sv_combined_extra
        plt.close(fig_sv_combined_extra)

    # Generate trials for each symbolic distance
    # For each SD, we need batch_size trials
    # Each trial is a pair of items with that symbolic distance

    # Collect embeddings per layer
    # embeddings_by_layer[layer_idx] = list of (embedding, signed_sd, pair_idx)
    embeddings_by_layer = {i: [] for i in range(args.extra_layers + 2)}

    # Collect output probabilities and logits for bar charts
    output_data = []  # List of dicts with signed_sd, probability, logit

    pair_counter = 0

    for sd in range(1, num_symbolic_distances + 1):
        # Generate batch_size random pairs with this symbolic distance
        # Pairs: (i, i+sd) for i in range(num_items - sd)
        valid_starts = list(range(num_items - sd))

        # Sample batch_size pairs (with replacement if needed)
        sampled_starts = np.random.choice(valid_starts, size=batch_size, replace=True)

        # Create trial inputs for these pairs
        sd_trials = []
        sd_correct_choices = []
        sd_pair_indices = []

        for batch_idx, start_item in enumerate(sampled_starts):
            item1_idx = start_item
            item2_idx = start_item + sd

            # Get item embeddings from batch_items
            item1_emb = batch_items[batch_idx, item1_idx, :]
            item2_emb = batch_items[batch_idx, item2_idx, :]

            # Randomly decide presentation order
            if np.random.random() < 0.5:
                trial_input = np.concatenate([item1_emb, item2_emb])
                correct_choice = 0  # item1 (lower index) is correct, presented first
                pair_idx = (item1_idx, item2_idx)
            else:
                trial_input = np.concatenate([item2_emb, item1_emb])
                correct_choice = 1  # item1 (lower index) is correct, presented second
                pair_idx = (item2_idx, item1_idx)

            sd_trials.append(trial_input)
            sd_correct_choices.append(correct_choice)
            sd_pair_indices.append(pair_idx)

        sd_trials = torch.tensor(np.array(sd_trials), dtype=torch.float32).to(device)
        sd_correct_choices = torch.tensor(np.array(sd_correct_choices), dtype=torch.float32).to(device)

        # Run inference with frozen weights
        with torch.inference_mode():
            output = model(sd_trials, frozen_plastic_weights.clone(), sd_correct_choices,
                          extra_plastic_weights=[epw.clone() for epw in frozen_extra_plastic_weights],
                          store_embeddings=True)

        # Collect output probabilities and logits
        probs = output.choice.squeeze(-1).detach().cpu().numpy()  # Shape: (batch_size,)
        # Compute logits from probabilities: logit = log(p / (1-p))
        # Clip to avoid log(0)
        probs_clipped = np.clip(probs, 1e-7, 1 - 1e-7)
        logits = np.log(probs_clipped / (1 - probs_clipped))

        for batch_idx in range(batch_size):
            pair_idx = sd_pair_indices[batch_idx]
            item1_idx, item2_idx = pair_idx
            # Signed SD from position 1's perspective
            signed_sd = item1_idx - item2_idx
            output_data.append({
                'signed_sd': signed_sd,
                'probability': probs[batch_idx],
                'logit': logits[batch_idx]
            })

        # Collect embeddings for each layer
        for layer_idx, embedding in enumerate(output.embeddings):
            emb_np = embedding.detach().cpu().numpy()
            hidden_size = emb_np.shape[1]
            half_dim = hidden_size // 2

            # Split into item1 and item2 embeddings
            emb_item1 = emb_np[:, :half_dim]
            emb_item2 = emb_np[:, half_dim:]

            for batch_idx in range(batch_size):
                pair_idx = sd_pair_indices[batch_idx]
                item1_idx, item2_idx = pair_idx

                # Signed SD from each item's perspective
                signed_sd_item1 = item1_idx - item2_idx
                signed_sd_item2 = item2_idx - item1_idx

                embeddings_by_layer[layer_idx].append({
                    'embedding': emb_item1[batch_idx],
                    'signed_sd': signed_sd_item1,
                    'abs_sd': sd,
                    'pair_id': pair_counter + batch_idx,
                    'is_item1': True
                })
                embeddings_by_layer[layer_idx].append({
                    'embedding': emb_item2[batch_idx],
                    'signed_sd': signed_sd_item2,
                    'abs_sd': sd,
                    'pair_id': pair_counter + batch_idx,
                    'is_item1': False
                })

        pair_counter += batch_size

    # Now create plots for each layer
    for layer_idx in range(args.extra_layers + 2):
        layer_data = embeddings_by_layer[layer_idx]

        if not layer_data:
            continue

        # Extract embeddings and metadata
        all_embeddings = np.array([d['embedding'] for d in layer_data])
        all_signed_sds = np.array([d['signed_sd'] for d in layer_data])
        all_pair_ids = np.array([d['pair_id'] for d in layer_data])

        # Fit PCA
        pca = PCA(n_components=2)
        pca_embeddings = pca.fit_transform(all_embeddings)

        # Layer name
        if layer_idx == 0:
            layer_name = "Embedding"
        elif layer_idx < args.extra_layers + 1:
            layer_name = f"Hidden {layer_idx}"
        else:
            layer_name = "Final"

    # --- Additional PCA plots: For each item in each position, show how it varies with partners ---
    # Generate all ordered pairs (i, j) where i != j
    all_ordered_pairs = [(i, j) for i in range(num_items) for j in range(num_items) if i != j]

    # Create trial inputs for all ordered pairs (using first network in batch)
    all_pair_trials = []
    all_pair_correct_choices = []
    all_pair_indices = []

    for (item1_idx, item2_idx) in all_ordered_pairs:
        # Get item embeddings from batch_items (use first network, index 0)
        item1_emb = batch_items[0, item1_idx, :]
        item2_emb = batch_items[0, item2_idx, :]

        # item1 is in position 1, item2 is in position 2
        trial_input = np.concatenate([item1_emb, item2_emb])
        # Correct choice: 0 if item1 < item2, 1 otherwise
        correct_choice = 0 if item1_idx < item2_idx else 1

        all_pair_trials.append(trial_input)
        all_pair_correct_choices.append(correct_choice)
        all_pair_indices.append((item1_idx, item2_idx))

    all_pair_trials = torch.tensor(np.array(all_pair_trials), dtype=torch.float32).to(device)
    all_pair_correct_choices = torch.tensor(np.array(all_pair_correct_choices), dtype=torch.float32).to(device)

    # Run inference with frozen weights from first network
    with torch.inference_mode():
        # Use first network's frozen weights
        single_frozen_pw = frozen_plastic_weights[0:1].expand(len(all_ordered_pairs), -1, -1).clone()
        single_frozen_epw = [epw[0:1].expand(len(all_ordered_pairs), -1, -1).clone() for epw in frozen_extra_plastic_weights]

        output_all_pairs = model(all_pair_trials, single_frozen_pw, all_pair_correct_choices,
                                  extra_plastic_weights=single_frozen_epw, store_embeddings=True)

    # For each layer, create plots for each item in each position
    for layer_idx, embedding in enumerate(output_all_pairs.embeddings):
        emb_np = embedding.detach().cpu().numpy()
        hidden_size = emb_np.shape[1]
        half_dim = hidden_size // 2

        # Split into position 1 and position 2 embeddings
        emb_pos1 = emb_np[:, :half_dim]  # Item in position 1
        emb_pos2 = emb_np[:, half_dim:]  # Item in position 2

        # Layer name
        if layer_idx == 0:
            layer_name = "Embedding"
        elif layer_idx < args.extra_layers + 1:
            layer_name = f"Hidden {layer_idx}"
        else:
            layer_name = "Final"

        # --- Project mean item embeddings onto top singular vectors of weight matrices ---
        # Skip embedding layer (layer_idx == 0) as it has no plastic weights
        if layer_idx == 0:
            continue

        # Get SVDs for different weight matrices for this layer:
        # 1. Raw plastic weights (PW)
        # 2. Alpha × plastic weights (α×PW)
        # 3. Effective weights (W_innate + α×PW)
        if layer_idx == args.extra_layers + 1:
            # Final layer
            layer_pw_name = "Final"
            svd_matrices = {
                'PW': (U, S, Vh),
                'α×PW': (U_alpha_pw, S_alpha_pw, Vh_alpha_pw),
                'Effective': (U_eff, S_eff, Vh_eff)
            }
        else:
            # Hidden layer
            layer_pw_name = f"Hidden {layer_idx}"
            epw_mean_layer = frozen_extra_plastic_weights[layer_idx - 1].detach().cpu().numpy().mean(axis=0)
            alpha_pw_layer = alpha_pw_extra_list[layer_idx - 1]
            W_eff_layer = W_effective_extra_list[layer_idx - 1]

            U_pw_h, S_pw_h, Vh_pw_h = np.linalg.svd(epw_mean_layer)
            U_apw_h, S_apw_h, Vh_apw_h = np.linalg.svd(alpha_pw_layer)
            U_eff_h, S_eff_h, Vh_eff_h = np.linalg.svd(W_eff_layer)

            svd_matrices = {
                'PW': (U_pw_h, S_pw_h, Vh_pw_h),
                'α×PW': (U_apw_h, S_apw_h, Vh_apw_h),
                'Effective': (U_eff_h, S_eff_h, Vh_eff_h)
            }

        # Use effective weights for the main projection analysis
        U_layer, S_layer, Vh_layer = svd_matrices['Effective']

        # Compute mean embeddings for each item in each position
        mean_emb_by_item_pos = {}  # (item_idx, position) -> mean_embedding (half-size)

        for item_idx in range(num_items):
            for position in [1, 2]:
                if position == 1:
                    pair_mask = [i for i, (p1, _) in enumerate(all_pair_indices) if p1 == item_idx]
                    item_embeddings = emb_pos1[pair_mask]
                else:
                    pair_mask = [i for i, (_, p2) in enumerate(all_pair_indices) if p2 == item_idx]
                    item_embeddings = emb_pos2[pair_mask]

                if len(pair_mask) > 0:
                    mean_emb_by_item_pos[(item_idx, position)] = np.mean(item_embeddings, axis=0)

        matrix_names = ['PW', 'α×PW', 'Effective']

        # --- Full pair embeddings projected onto SVs, colored by signed symbolic distance ---
        # Use the FULL embedding (not split) for each pair
        # Compute signed SD for each pair
        pair_signed_sds = np.array([p1 - p2 for (p1, p2) in all_pair_indices])

        # Project full embeddings onto SVs of effective weights
        V_eff = Vh_layer.T  # Right singular vectors
        proj_full_sv1 = emb_np @ V_eff[:, 0]  # Projection onto SV1
        proj_full_sv2 = emb_np @ V_eff[:, 1]  # Projection onto SV2

        # Plot: SV1 vs SV2, colored by signed symbolic distance
        fig_pair_sv, ax_pair_sv = plt.subplots(figsize=(10, 8), dpi=150)

        unique_ssd = sorted(np.unique(pair_signed_sds))
        colors_ssd = plt.cm.coolwarm(np.linspace(0, 1, len(unique_ssd)))
        ssd_to_color = {ssd: colors_ssd[i] for i, ssd in enumerate(unique_ssd)}

        for ssd in unique_ssd:
            mask = pair_signed_sds == ssd
            sign_str = f'+{int(ssd)}' if ssd > 0 else str(int(ssd))
            ax_pair_sv.scatter(proj_full_sv1[mask], proj_full_sv2[mask],
                              c=[ssd_to_color[ssd]], label=f'SD {sign_str}',
                              s=60, alpha=0.7, edgecolors='black', linewidths=0.5)

        ax_pair_sv.set_xlabel(f'Projection onto SV1 (σ={S_layer[0]:.2e})')
        ax_pair_sv.set_ylabel(f'Projection onto SV2 (σ={S_layer[1]:.2e})')
        ax_pair_sv.set_title(f'Full Pair Embeddings onto Effective Weight SVs - {layer_pw_name} Layer')
        ax_pair_sv.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), title='Signed SD')
        plt.subplots_adjust(right=0.85)
        figures[f"pca_frozen/layer{layer_idx}_pair_sv_by_ssd"] = fig_pair_sv
        plt.close(fig_pair_sv)

        # Plot: SV1 projection vs signed symbolic distance (should show linear relationship)
        fig_sv1_ssd, ax_sv1_ssd = plt.subplots(figsize=(10, 6), dpi=150)

        # Compute mean and std for each signed SD
        ssd_means = []
        ssd_stds = []
        for ssd in unique_ssd:
            mask = pair_signed_sds == ssd
            ssd_means.append(np.mean(proj_full_sv1[mask]))
            ssd_stds.append(np.std(proj_full_sv1[mask]))

        # Bar plot
        x_pos = np.arange(len(unique_ssd))
        colors_bar = plt.cm.coolwarm(np.linspace(0, 1, len(unique_ssd)))
        ax_sv1_ssd.bar(x_pos, ssd_means, yerr=ssd_stds, capsize=3,
                       color=colors_bar, edgecolor='black', alpha=0.8)

        ax_sv1_ssd.set_xticks(x_pos)
        ax_sv1_ssd.set_xticklabels([f'{int(ssd):+d}' if ssd != 0 else '0' for ssd in unique_ssd])
        ax_sv1_ssd.set_xlabel('Signed Symbolic Distance')
        ax_sv1_ssd.set_ylabel('Mean Projection onto SV1')
        ax_sv1_ssd.set_title(f'SV1 Projection by Signed SD (Effective Weights) - {layer_pw_name} Layer')

        # Add correlation
        corr_ssd = np.corrcoef(pair_signed_sds, proj_full_sv1)[0, 1]
        ax_sv1_ssd.text(0.02, 0.98, f'r = {corr_ssd:.3f}', transform=ax_sv1_ssd.transAxes,
                        fontsize=12, verticalalignment='top', fontweight='bold')

        plt.tight_layout()
        figures[f"pca_frozen/layer{layer_idx}_sv1_by_ssd"] = fig_sv1_ssd
        plt.close(fig_sv1_ssd)

        # Comparison across matrix types: SV1 vs Signed SD
        fig_ssd_compare, axes_ssd = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

        for mat_idx, mat_name in enumerate(matrix_names):
            U_mat, S_mat, Vh_mat = svd_matrices[mat_name]
            V_mat = Vh_mat.T

            proj_sv1 = emb_np @ V_mat[:, 0]

            # Compute mean for each signed SD
            means = [np.mean(proj_sv1[pair_signed_sds == ssd]) for ssd in unique_ssd]
            stds = [np.std(proj_sv1[pair_signed_sds == ssd]) for ssd in unique_ssd]

            axes_ssd[mat_idx].bar(x_pos, means, yerr=stds, capsize=3,
                                   color=colors_bar, edgecolor='black', alpha=0.8)
            axes_ssd[mat_idx].set_xticks(x_pos)
            axes_ssd[mat_idx].set_xticklabels([f'{int(ssd):+d}' if ssd != 0 else '0' for ssd in unique_ssd], fontsize=8)
            axes_ssd[mat_idx].set_xlabel('Signed SD')
            axes_ssd[mat_idx].set_ylabel('Mean Proj onto SV1')

            # Correlation
            corr = np.corrcoef(pair_signed_sds, proj_sv1)[0, 1]
            axes_ssd[mat_idx].set_title(f'{mat_name}\nr = {corr:.3f}')

        plt.suptitle(f'SV1 Projection by Signed SD - {layer_pw_name} Layer', fontsize=14)
        plt.tight_layout()
        figures[f"pca_frozen/layer{layer_idx}_sv1_ssd_compare"] = fig_ssd_compare
        plt.close(fig_ssd_compare)

        # --- Multi-SV analysis: Find optimal linear combination that predicts signed SD ---
        from sklearn.linear_model import LinearRegression

        # Try different numbers of top SVs
        max_svs = min(20, len(S_layer))
        r2_by_num_svs = {'PW': [], 'α×PW': [], 'Effective': []}

        # Also track correlations for norm and weighted sum approaches
        corr_norm = {'PW': [], 'α×PW': [], 'Effective': []}
        corr_weighted = {'PW': [], 'α×PW': [], 'Effective': []}

        for mat_name in matrix_names:
            _, S_mat, Vh_mat = svd_matrices[mat_name]
            V_mat = Vh_mat.T

            for k in range(1, max_svs + 1):
                # Project onto top-k SVs
                proj_k = emb_np @ V_mat[:, :k]  # Shape: (num_pairs, k)

                # Approach 1: Linear regression to predict signed SD
                reg = LinearRegression()
                reg.fit(proj_k, pair_signed_sds)
                r2 = reg.score(proj_k, pair_signed_sds)
                r2_by_num_svs[mat_name].append(r2)

                # Approach 2: Norm in subspace ||proj_k||
                # Note: This loses sign, so we correlate with absolute SD
                norm_proj = np.linalg.norm(proj_k, axis=1)
                corr_norm_val = np.corrcoef(np.abs(pair_signed_sds), norm_proj)[0, 1]
                corr_norm[mat_name].append(corr_norm_val)

                # Approach 3: Weighted sum by singular values: Σ σᵢ * projᵢ
                weights = S_mat[:k] / S_mat[:k].sum()  # Normalize weights
                weighted_proj = np.sum(proj_k * weights, axis=1)
                corr_weighted_val = np.corrcoef(pair_signed_sds, weighted_proj)[0, 1]
                corr_weighted[mat_name].append(corr_weighted_val)

        # --- Item Identity Encoding Analysis ---
        # This section analyzes how well SVs encode item identity (not just rank)

        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import silhouette_score

        # Extract item labels for each pair (both items)
        pair_item1_labels = np.array([p[0] for p in all_pair_indices])
        pair_item2_labels = np.array([p[1] for p in all_pair_indices])

        # Use effective weights SVs
        V_eff_id = Vh_layer.T
        max_svs_id = min(20, len(S_layer))

        # Track metrics vs number of SVs
        var_explained_by_k = []
        item1_accuracy_by_k = []
        item2_accuracy_by_k = []
        combined_accuracy_by_k = []
        silhouette_item1_by_k = []
        silhouette_item2_by_k = []

        for k in range(1, max_svs_id + 1):
            # Project onto top-k SVs
            proj_k = emb_np @ V_eff_id[:, :k]

            # 1. Variance explained (reconstruction)
            reconstructed = proj_k @ V_eff_id[:, :k].T
            total_var = np.var(emb_np)
            residual_var = np.var(emb_np - reconstructed)
            var_exp = 1 - residual_var / total_var if total_var > 0 else 0
            var_explained_by_k.append(var_exp)

            # 2. Item classification accuracy using logistic regression
            # Predict item1 identity from projections
            try:
                clf1 = LogisticRegression(max_iter=1000, random_state=42)
                clf1.fit(proj_k, pair_item1_labels)
                acc1 = clf1.score(proj_k, pair_item1_labels)
            except:
                acc1 = 1.0 / num_items  # Chance level
            item1_accuracy_by_k.append(acc1)

            # Predict item2 identity from projections
            try:
                clf2 = LogisticRegression(max_iter=1000, random_state=42)
                clf2.fit(proj_k, pair_item2_labels)
                acc2 = clf2.score(proj_k, pair_item2_labels)
            except:
                acc2 = 1.0 / num_items
            item2_accuracy_by_k.append(acc2)

            # Combined: predict both items (as a tuple label)
            combined_labels = pair_item1_labels * num_items + pair_item2_labels
            try:
                clf_comb = LogisticRegression(max_iter=1000, random_state=42)
                clf_comb.fit(proj_k, combined_labels)
                acc_comb = clf_comb.score(proj_k, combined_labels)
            except:
                acc_comb = 1.0 / (num_items * (num_items - 1))
            combined_accuracy_by_k.append(acc_comb)

            # 3. Silhouette score (cluster separability)
            if k >= 2 and len(np.unique(pair_item1_labels)) > 1:
                try:
                    sil1 = silhouette_score(proj_k, pair_item1_labels)
                except:
                    sil1 = 0
                try:
                    sil2 = silhouette_score(proj_k, pair_item2_labels)
                except:
                    sil2 = 0
            else:
                sil1, sil2 = 0, 0
            silhouette_item1_by_k.append(sil1)
            silhouette_item2_by_k.append(sil2)

        # Chance levels for reference
        chance_single = 1.0 / num_items

        # Plot: Combined comparison - Rank encoding vs Item encoding
        fig_compare_all, axes_cmp = plt.subplots(1, 3, figsize=(18, 5), dpi=150)

        # Left: Rank encoding (R² for signed SD prediction)
        axes_cmp[0].plot(range(1, max_svs_id + 1), r2_by_num_svs['Effective'][:max_svs_id], 'o-',
                         linewidth=2, markersize=8, color='green')
        axes_cmp[0].set_xlabel('Number of Top SVs')
        axes_cmp[0].set_ylabel('R² (Signed SD Prediction)')
        axes_cmp[0].set_title('Rank Encoding')
        axes_cmp[0].set_ylim(0, 1.05)
        axes_cmp[0].set_xticks(range(1, max_svs_id + 1))

        # Middle: Item identity encoding (classification accuracy)
        axes_cmp[1].plot(range(1, max_svs_id + 1), item1_accuracy_by_k, 'o-',
                         linewidth=2, markersize=8, color='blue', label='Item 1')
        axes_cmp[1].plot(range(1, max_svs_id + 1), item2_accuracy_by_k, 's-',
                         linewidth=2, markersize=8, color='red', label='Item 2')
        axes_cmp[1].axhline(y=chance_single, color='gray', linestyle='--', alpha=0.5)
        axes_cmp[1].set_xlabel('Number of Top SVs')
        axes_cmp[1].set_ylabel('Classification Accuracy')
        axes_cmp[1].set_title('Item Identity Encoding')
        axes_cmp[1].set_ylim(0, 1.05)
        axes_cmp[1].set_xticks(range(1, max_svs_id + 1))
        axes_cmp[1].legend()

        # Right: Variance explained
        axes_cmp[2].plot(range(1, max_svs_id + 1), var_explained_by_k, 'o-',
                         linewidth=2, markersize=8, color='purple')
        axes_cmp[2].axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)
        axes_cmp[2].set_xlabel('Number of Top SVs')
        axes_cmp[2].set_ylabel('Variance Explained')
        axes_cmp[2].set_title('Embedding Reconstruction')
        axes_cmp[2].set_ylim(0, 1.05)
        axes_cmp[2].set_xticks(range(1, max_svs_id + 1))

        plt.suptitle(f'Rank vs Item Identity Encoding Comparison - {layer_pw_name} Layer', fontsize=14)
        plt.tight_layout()
        figures[f"pca_frozen/layer{layer_idx}_rank_vs_identity"] = fig_compare_all
        plt.close(fig_compare_all)

        # Plot 5: 2D scatter colored by item identity (for visualization)
        # Show top-2 SV projections colored by item1 and item2 separately
        proj_2d_id = emb_np @ V_eff_id[:, :2]

        fig_id_scatter, axes_id = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

        # Color by item1 (position 1)
        scatter1 = axes_id[0].scatter(proj_2d_id[:, 0], proj_2d_id[:, 1],
                                       c=pair_item1_labels, cmap='tab10',
                                       s=50, alpha=0.7, edgecolors='black', linewidths=0.3)
        axes_id[0].set_xlabel('SV1')
        axes_id[0].set_ylabel('SV2')
        axes_id[0].set_title('Colored by Item in Position 1')
        cbar1 = plt.colorbar(scatter1, ax=axes_id[0])
        cbar1.set_label('Item 1')
        cbar1.set_ticks(range(num_items))
        cbar1.set_ticklabels([chr(ord('A') + i) for i in range(num_items)])

        # Color by item2 (position 2)
        scatter2 = axes_id[1].scatter(proj_2d_id[:, 0], proj_2d_id[:, 1],
                                       c=pair_item2_labels, cmap='tab10',
                                       s=50, alpha=0.7, edgecolors='black', linewidths=0.3)
        axes_id[1].set_xlabel('SV1')
        axes_id[1].set_ylabel('SV2')
        axes_id[1].set_title('Colored by Item in Position 2')
        cbar2 = plt.colorbar(scatter2, ax=axes_id[1])
        cbar2.set_label('Item 2')
        cbar2.set_ticks(range(num_items))
        cbar2.set_ticklabels([chr(ord('A') + i) for i in range(num_items)])

        plt.suptitle(f'Pair Embeddings in SV Space - Item Identity View - {layer_pw_name} Layer', fontsize=12)
        plt.tight_layout()
        figures[f"pca_frozen/layer{layer_idx}_item_identity_scatter"] = fig_id_scatter
        plt.close(fig_id_scatter)

        # Plot 6: Per-item embedding locations in SV space (mean ± std for each item)
        fig_item_means, axes_im = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

        colors_items = plt.cm.tab10(np.linspace(0, 1, num_items))

        for pos_idx, (pos, item_labels) in enumerate([(1, pair_item1_labels), (2, pair_item2_labels)]):
            ax = axes_im[pos_idx]

            for item_idx in range(num_items):
                mask = item_labels == item_idx
                if np.sum(mask) > 0:
                    item_projs = proj_2d_id[mask]
                    mean_proj = item_projs.mean(axis=0)
                    std_proj = item_projs.std(axis=0)

                    # Plot mean with error ellipse approximation (just error bars for simplicity)
                    item_letter = chr(ord('A') + item_idx)
                    ax.errorbar(mean_proj[0], mean_proj[1], xerr=std_proj[0], yerr=std_proj[1],
                               fmt='o', color=colors_items[item_idx], markersize=12,
                               capsize=3, label=item_letter, alpha=0.8)
                    ax.annotate(item_letter, (mean_proj[0], mean_proj[1]),
                               xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')

            ax.set_xlabel('SV1')
            ax.set_ylabel('SV2')
            ax.set_title(f'Mean Item Locations (Position {pos})')
            ax.legend(loc='upper right', ncol=2)

        plt.suptitle(f'Mean Item Embeddings in SV Space - {layer_pw_name} Layer', fontsize=12)
        plt.tight_layout()
        figures[f"pca_frozen/layer{layer_idx}_item_means_sv"] = fig_item_means
        plt.close(fig_item_means)

    # --- Evolution plots: Show how plastic weight representation develops during training ---
    # Use the pw_checkpoints saved during the training phase
    checkpoint_keys = sorted(pw_checkpoints.keys())

    if len(checkpoint_keys) > 1:
        # For each checkpoint, run inference on all test pairs and analyze the representation
        evolution_r2_by_checkpoint = []
        evolution_corr_by_checkpoint = []
        evolution_singular_values = []
        evolution_proj_data = []  # For 3D visualization at each checkpoint

        from sklearn.linear_model import LinearRegression

        for ckpt_trial in checkpoint_keys:
            ckpt_pw, ckpt_epw = pw_checkpoints[ckpt_trial]

            # Run inference with checkpoint weights on all test pairs
            with torch.inference_mode():
                # Use first network's checkpoint weights
                single_ckpt_pw = ckpt_pw[0:1].expand(len(all_ordered_pairs), -1, -1).clone()
                single_ckpt_epw = [epw[0:1].expand(len(all_ordered_pairs), -1, -1).clone() for epw in ckpt_epw]

                output_ckpt = model(all_pair_trials, single_ckpt_pw, all_pair_correct_choices,
                                    extra_plastic_weights=single_ckpt_epw, store_embeddings=True)

            # Get embeddings from final layer
            emb_ckpt = output_ckpt.embeddings[-1].detach().cpu().numpy()

            # Compute effective weights for this checkpoint (final layer)
            pw_ckpt_np = ckpt_pw[0].detach().cpu().numpy()
            alpha_ckpt = model.alpha.detach().cpu().numpy()
            W_innate_ckpt = model.fc2.weight.detach().cpu().numpy()
            alpha_pw_ckpt = alpha_ckpt * pw_ckpt_np
            W_effective_ckpt = W_innate_ckpt + alpha_pw_ckpt

            # SVD of effective weights
            _, S_ckpt, Vh_ckpt = np.linalg.svd(W_effective_ckpt)
            evolution_singular_values.append(S_ckpt[:20] if len(S_ckpt) >= 20 else S_ckpt)

            V_ckpt = Vh_ckpt.T

            # Project onto top-5 SVs
            k_evo = min(5, len(S_ckpt))
            proj_ckpt = emb_ckpt @ V_ckpt[:, :k_evo]

            # Linear regression to predict signed SD
            reg_ckpt = LinearRegression()
            reg_ckpt.fit(proj_ckpt, pair_signed_sds)
            r2_ckpt = reg_ckpt.score(proj_ckpt, pair_signed_sds)
            evolution_r2_by_checkpoint.append(r2_ckpt)

            # Correlation of SV1 with signed SD
            corr_sv1 = np.corrcoef(proj_ckpt[:, 0], pair_signed_sds)[0, 1]
            evolution_corr_by_checkpoint.append(corr_sv1)

            # Store 3D projection data
            proj_3d_ckpt = emb_ckpt @ V_ckpt[:, :3] if len(S_ckpt) >= 3 else proj_ckpt
            evolution_proj_data.append({
                'proj_3d': proj_3d_ckpt,
                'S': S_ckpt,
                'trial': ckpt_trial
            })

        # Plot 7: Heatmap evolution of plastic weights
        n_heatmaps = min(len(checkpoint_keys), 5)  # Limit to 5 heatmaps for space
        heatmap_indices = np.linspace(0, len(checkpoint_keys) - 1, n_heatmaps, dtype=int)

        fig_evo_hm, axes_evo_hm = plt.subplots(1, n_heatmaps, figsize=(4 * n_heatmaps, 4), dpi=100)
        if n_heatmaps == 1:
            axes_evo_hm = [axes_evo_hm]

        # Find global min/max for consistent colorscale
        all_pw_vals = []
        for idx in heatmap_indices:
            ckpt_trial = checkpoint_keys[idx]
            ckpt_pw, _ = pw_checkpoints[ckpt_trial]
            all_pw_vals.append(ckpt_pw[0].detach().cpu().numpy())
        vmin = min(pw.min() for pw in all_pw_vals)
        vmax = max(pw.max() for pw in all_pw_vals)
        vabs = max(abs(vmin), abs(vmax))

        for i, idx in enumerate(heatmap_indices):
            ckpt_trial = checkpoint_keys[idx]
            ckpt_pw, _ = pw_checkpoints[ckpt_trial]
            pw_np = ckpt_pw[0].detach().cpu().numpy()

            im = axes_evo_hm[i].imshow(pw_np, cmap='RdBu_r', vmin=-vabs, vmax=vabs, aspect='auto')

            title_trial = f'Trial {ckpt_trial}' if ckpt_trial < args.num_train_trials else 'Final'
            axes_evo_hm[i].set_title(title_trial)
            axes_evo_hm[i].set_xlabel('Input Dim')
            axes_evo_hm[i].set_ylabel('Output Dim')

        plt.suptitle('Plastic Weight Heatmap Evolution', fontsize=12)
        fig_evo_hm.colorbar(im, ax=axes_evo_hm, shrink=0.8, label='Weight Value')
        plt.tight_layout()
        figures["pca_frozen/evolution_pw_heatmaps"] = fig_evo_hm
        plt.close(fig_evo_hm)

    # --- Item-readout correlation analysis ---
    # For each item, run it through the network with the other position zeroed
    # Compute correlation between each layer's embedding and the readout weights
    # Aggregate across multiple networks

    num_layers_ti = args.extra_layers + 2
    num_networks_to_analyze_ti = min(batch_size, 100)

    # Storage for item-readout correlations
    # Structure: position -> layer_idx -> item_idx -> list of correlations
    item_readout_correlations_ti = {
        'item1': {layer_idx: {item_idx: [] for item_idx in range(num_items)} for layer_idx in range(num_layers_ti)},
        'item2': {layer_idx: {item_idx: [] for item_idx in range(num_items)} for layer_idx in range(num_layers_ti)},
    }

    # Storage for item-readout dot products (raw, not normalized)
    item_readout_dotproducts_ti = {
        'item1': {layer_idx: {item_idx: [] for item_idx in range(num_items)} for layer_idx in range(num_layers_ti)},
        'item2': {layer_idx: {item_idx: [] for item_idx in range(num_items)} for layer_idx in range(num_layers_ti)},
    }

    # Storage for pair-readout correlations for TI (joint representation)
    # Structure: layer_idx -> list of (num_items, num_items) matrices, one per network
    pair_readout_correlations_ti = {layer_idx: [] for layer_idx in range(num_layers_ti)}
    pair_readout_dotproducts_ti = {layer_idx: [] for layer_idx in range(num_layers_ti)}

    # Storage for single-item embeddings for PCA analysis
    single_item_embeddings_ti = {
        'item1': {layer_idx: [] for layer_idx in range(num_layers_ti)},
        'item2': {layer_idx: [] for layer_idx in range(num_layers_ti)},
    }

    readout_weights_ti = model.choice.weight.detach().cpu().numpy().squeeze()  # (hidden_size,)

    print(f"Computing item-readout correlations for {num_networks_to_analyze_ti} networks...")

    with torch.no_grad():
        for network_idx in range(num_networks_to_analyze_ti):
            single_items_ti = batch_items[network_idx]  # (num_items, item_size)
            single_pw_ti = frozen_plastic_weights[network_idx]  # (hidden_size, hidden_size)
            single_epw_ti = [epw[network_idx] for epw in frozen_extra_plastic_weights]

            # Expand plastic weights for inference
            pw_expanded_ti = single_pw_ti.unsqueeze(0).expand(batch_size, -1, -1).clone()
            epw_expanded_ti = [epw.unsqueeze(0).expand(batch_size, -1, -1).clone() for epw in single_epw_ti]

            for item_idx in range(num_items):
                item_emb = single_items_ti[item_idx]  # (item_size,)
                zero_emb = np.zeros_like(item_emb)

                # Position 1: item in first position, zeros in second
                input_pos1 = np.concatenate([item_emb, zero_emb])
                # Position 2: zeros in first position, item in second
                input_pos2 = np.concatenate([zero_emb, item_emb])

                for position, input_vec in [('item1', input_pos1), ('item2', input_pos2)]:
                    # Create batch input
                    input_batch = np.zeros((batch_size, len(input_vec)))
                    input_batch[0] = input_vec
                    input_t = torch.tensor(input_batch, dtype=torch.float32).to(device)
                    dummy_correct = torch.zeros(batch_size, dtype=torch.float32).to(device)

                    # Run through network
                    output_single = model(input_t, pw_expanded_ti, dummy_correct,
                                         extra_plastic_weights=epw_expanded_ti, store_embeddings=True)

                    # For each layer, compute correlation and dot product with readout weights
                    for layer_idx, embedding in enumerate(output_single.embeddings):
                        emb_vec = embedding[0].detach().cpu().numpy()  # (hidden_size,)

                        # Compute Pearson correlation
                        if np.std(emb_vec) > 1e-10 and np.std(readout_weights_ti) > 1e-10:
                            corr = np.corrcoef(emb_vec, readout_weights_ti)[0, 1]
                        else:
                            corr = 0.0

                        # Compute raw dot product
                        dot_prod = np.dot(emb_vec, readout_weights_ti)

                        item_readout_correlations_ti[position][layer_idx][item_idx].append(corr)
                        item_readout_dotproducts_ti[position][layer_idx][item_idx].append(dot_prod)

                        # Store embedding for PCA analysis
                        single_item_embeddings_ti[position][layer_idx].append({
                            'embedding': emb_vec,
                            'item_idx': item_idx,
                            'network_idx': network_idx,
                        })

            # --- Pair-readout correlation analysis for TI (joint representation) ---
            # Generate all ordered pairs and run through network
            # Initialize correlation and dot product matrices for this network
            corr_matrices_ti = {layer_idx: np.full((num_items, num_items), np.nan) for layer_idx in range(num_layers_ti)}
            dot_matrices_ti = {layer_idx: np.full((num_items, num_items), np.nan) for layer_idx in range(num_layers_ti)}

            # Generate all ordered pairs (i, j) where i != j
            ti_pair_inputs = []
            ti_pair_indices = []
            for i in range(num_items):
                for j in range(num_items):
                    if i != j:
                        item_i_emb = single_items_ti[i]
                        item_j_emb = single_items_ti[j]
                        pair_input = np.concatenate([item_i_emb, item_j_emb])
                        ti_pair_inputs.append(pair_input)
                        ti_pair_indices.append((i, j))

            # Run all pairs through network in batches
            num_ti_pairs = len(ti_pair_inputs)
            for batch_start in range(0, num_ti_pairs, batch_size):
                batch_end = min(batch_start + batch_size, num_ti_pairs)
                actual_batch_size = batch_end - batch_start

                # Prepare batch
                batch_inputs = np.zeros((batch_size, len(ti_pair_inputs[0])))
                batch_inputs[:actual_batch_size] = np.array(ti_pair_inputs[batch_start:batch_end])
                batch_inputs_t = torch.tensor(batch_inputs, dtype=torch.float32).to(device)
                dummy_correct_ti = torch.zeros(batch_size, dtype=torch.float32).to(device)

                # Run through network
                output_pairs_ti = model(batch_inputs_t, pw_expanded_ti, dummy_correct_ti,
                                       extra_plastic_weights=epw_expanded_ti, store_embeddings=True)

                # Extract correlations and dot products for each pair
                for idx_in_batch in range(actual_batch_size):
                    pair_idx = batch_start + idx_in_batch
                    item_i, item_j = ti_pair_indices[pair_idx]

                    for layer_idx, embedding in enumerate(output_pairs_ti.embeddings):
                        emb_vec = embedding[idx_in_batch].detach().cpu().numpy()

                        # Compute Pearson correlation
                        if np.std(emb_vec) > 1e-10 and np.std(readout_weights_ti) > 1e-10:
                            corr = np.corrcoef(emb_vec, readout_weights_ti)[0, 1]
                        else:
                            corr = 0.0

                        # Compute raw dot product
                        dot_prod = np.dot(emb_vec, readout_weights_ti)

                        corr_matrices_ti[layer_idx][item_i, item_j] = corr
                        dot_matrices_ti[layer_idx][item_i, item_j] = dot_prod

            # Store matrices for this network
            for layer_idx in range(num_layers_ti):
                pair_readout_correlations_ti[layer_idx].append(corr_matrices_ti[layer_idx])
                pair_readout_dotproducts_ti[layer_idx].append(dot_matrices_ti[layer_idx])

    # Create plots
    item_labels_ti = [chr(ord('A') + i) for i in range(num_items)]
    layer_names_ti = ['Embedding'] + [f'Hidden {i+1}' for i in range(args.extra_layers)] + ['Final']

    for layer_idx in range(num_layers_ti):
        layer_name = layer_names_ti[layer_idx] if layer_idx < len(layer_names_ti) else f'Layer {layer_idx}'

        for position in ['item1', 'item2']:
            position_label = 'Position 1 (other zeroed)' if position == 'item1' else 'Position 2 (other zeroed)'

            # Compute mean and SE for each item
            mean_corrs = []
            se_corrs = []
            for item_idx in range(num_items):
                corrs = item_readout_correlations_ti[position][layer_idx][item_idx]
                if len(corrs) > 0:
                    mean_corrs.append(np.mean(corrs))
                    se_corrs.append(np.std(corrs) / np.sqrt(len(corrs)))
                else:
                    mean_corrs.append(0)
                    se_corrs.append(0)

            mean_corrs = np.array(mean_corrs)
            se_corrs = np.array(se_corrs)

            # Create bar plot
            fig_corr, ax_corr = plt.subplots(figsize=(10, 6), dpi=150)

            x_pos = np.arange(num_items)
            # Use gradient colors based on item rank
            colors_items = plt.cm.viridis(np.linspace(0, 1, num_items))

            ax_corr.bar(x_pos, mean_corrs, yerr=se_corrs, capsize=5,
                        color=colors_items, edgecolor='black', alpha=0.8)

            ax_corr.axhline(y=0, color='gray', linestyle='-', linewidth=1)
            ax_corr.set_xticks(x_pos)
            ax_corr.set_xticklabels(item_labels_ti[:num_items])
            ax_corr.set_xlabel('Item (by rank)')
            ax_corr.set_ylabel('Correlation with Readout Weights')
            ax_corr.set_title(f'TI: {layer_name} - Item-Readout Correlation\n{position_label} (n={num_networks_to_analyze_ti} networks, mean ± SE)')

            plt.tight_layout()
            figures[f"pca_frozen/layer{layer_idx}_item_readout_corr_{position}"] = fig_corr
            plt.close(fig_corr)

        # Combined plot showing both positions
        fig_combined, ax_combined = plt.subplots(figsize=(12, 6), dpi=150)

        x_pos = np.arange(num_items)
        width = 0.35

        mean_corrs_p1 = []
        se_corrs_p1 = []
        for item_idx in range(num_items):
            corrs = item_readout_correlations_ti['item1'][layer_idx][item_idx]
            mean_corrs_p1.append(np.mean(corrs) if len(corrs) > 0 else 0)
            se_corrs_p1.append(np.std(corrs) / np.sqrt(len(corrs)) if len(corrs) > 0 else 0)

        mean_corrs_p2 = []
        se_corrs_p2 = []
        for item_idx in range(num_items):
            corrs = item_readout_correlations_ti['item2'][layer_idx][item_idx]
            mean_corrs_p2.append(np.mean(corrs) if len(corrs) > 0 else 0)
            se_corrs_p2.append(np.std(corrs) / np.sqrt(len(corrs)) if len(corrs) > 0 else 0)

        ax_combined.bar(x_pos - width/2, mean_corrs_p1, width, yerr=se_corrs_p1, capsize=3,
                        label='Position 1', color='tab:blue', alpha=0.8)
        ax_combined.bar(x_pos + width/2, mean_corrs_p2, width, yerr=se_corrs_p2, capsize=3,
                        label='Position 2', color='tab:orange', alpha=0.8)

        ax_combined.axhline(y=0, color='gray', linestyle='-', linewidth=1)
        ax_combined.set_xticks(x_pos)
        ax_combined.set_xticklabels(item_labels_ti[:num_items])
        ax_combined.set_xlabel('Item (by rank)')
        ax_combined.set_ylabel('Correlation with Readout Weights')
        ax_combined.set_title(f'TI: {layer_name} - Item-Readout Correlation by Position\n(n={num_networks_to_analyze_ti} networks, mean ± SE)')
        ax_combined.legend()

        plt.tight_layout()
        figures[f"pca_frozen/layer{layer_idx}_item_readout_corr_combined"] = fig_combined
        plt.close(fig_combined)

    # --- Pair-readout correlation heatmaps for TI ---
    # For each layer, plot heatmaps of correlation with readout for each ordered pair
    for layer_idx in range(num_layers_ti):
        layer_name = layer_names_ti[layer_idx] if layer_idx < len(layer_names_ti) else f'Layer {layer_idx}'

        # Stack matrices across networks and compute mean/std
        corr_matrices = np.array(pair_readout_correlations_ti[layer_idx])  # (num_networks, num_items, num_items)
        dot_matrices = np.array(pair_readout_dotproducts_ti[layer_idx])

        # Compute mean and std (ignoring NaN for diagonal)
        mean_corr = np.nanmean(corr_matrices, axis=0)
        std_corr = np.nanstd(corr_matrices, axis=0)
        mean_dot = np.nanmean(dot_matrices, axis=0)
        std_dot = np.nanstd(dot_matrices, axis=0)

        # --- Correlation Mean Heatmap ---
        fig_corr_mean, ax_corr_mean = plt.subplots(figsize=(10, 8), dpi=150)
        vabs_corr = max(np.nanmax(np.abs(mean_corr)), 0.01)
        im_corr = ax_corr_mean.imshow(mean_corr, cmap='RdBu_r', vmin=-vabs_corr, vmax=vabs_corr, aspect='equal')
        ax_corr_mean.set_xticks(range(num_items))
        ax_corr_mean.set_yticks(range(num_items))
        ax_corr_mean.set_xticklabels(item_labels_ti)
        ax_corr_mean.set_yticklabels(item_labels_ti)
        ax_corr_mean.set_xlabel('Item 2 (second position)')
        ax_corr_mean.set_ylabel('Item 1 (first position)')
        ax_corr_mean.set_title(f'TI: {layer_name} - Pair-Readout Correlation (Mean)\n(n={num_networks_to_analyze_ti} networks)')

        # Add text annotations
        for i in range(num_items):
            for j in range(num_items):
                if not np.isnan(mean_corr[i, j]):
                    text_color = 'white' if np.abs(mean_corr[i, j]) > vabs_corr * 0.6 else 'black'
                    ax_corr_mean.text(j, i, f'{mean_corr[i, j]:.2f}', ha='center', va='center',
                                      fontsize=8, color=text_color)

        plt.colorbar(im_corr, ax=ax_corr_mean, label='Correlation')
        plt.tight_layout()
        figures[f"pca_frozen/layer{layer_idx}_pair_readout_corr_mean"] = fig_corr_mean
        plt.close(fig_corr_mean)

        # --- Correlation Std Heatmap ---
        fig_corr_std, ax_corr_std = plt.subplots(figsize=(10, 8), dpi=150)
        im_corr_std = ax_corr_std.imshow(std_corr, cmap='viridis', aspect='equal')
        ax_corr_std.set_xticks(range(num_items))
        ax_corr_std.set_yticks(range(num_items))
        ax_corr_std.set_xticklabels(item_labels_ti)
        ax_corr_std.set_yticklabels(item_labels_ti)
        ax_corr_std.set_xlabel('Item 2 (second position)')
        ax_corr_std.set_ylabel('Item 1 (first position)')
        ax_corr_std.set_title(f'TI: {layer_name} - Pair-Readout Correlation (Std)\n(n={num_networks_to_analyze_ti} networks)')

        # Add text annotations
        for i in range(num_items):
            for j in range(num_items):
                if not np.isnan(std_corr[i, j]):
                    ax_corr_std.text(j, i, f'{std_corr[i, j]:.2f}', ha='center', va='center',
                                     fontsize=8, color='white')

        plt.colorbar(im_corr_std, ax=ax_corr_std, label='Std Dev')
        plt.tight_layout()
        figures[f"pca_frozen/layer{layer_idx}_pair_readout_corr_std"] = fig_corr_std
        plt.close(fig_corr_std)

    # --- Weight Matrix Orthogonality Analysis ---
    # Check if innate weights at each layer are orthogonal to the readout
    # and analyze the structure of each layer's weight matrix

    print("Analyzing weight matrix orthogonality...")

    # Get readout weights (choice layer)
    readout_weights = model.choice.weight.detach().cpu().numpy().squeeze()  # (hidden_size,)
    readout_norm = np.linalg.norm(readout_weights)
    readout_unit = readout_weights / readout_norm

    # Collect weight matrices for analysis
    weight_matrices = {}

    # Layer 0: Embedding layer (input_size -> hidden_size)
    if hasattr(model, 'embedding_layer'):
        weight_matrices['Embedding'] = model.embedding_layer.weight.detach().cpu().numpy()  # (hidden_size, input_size)

    # Extra hidden layers (hidden_size -> hidden_size)
    for i, layer in enumerate(model.extra_hidden_layers):
        weight_matrices[f'Hidden {i+1}'] = layer.weight.detach().cpu().numpy()  # (hidden_size, hidden_size)

    # Final layer fc2 (hidden_size -> hidden_size)
    weight_matrices['Final (fc2)'] = model.fc2.weight.detach().cpu().numpy()  # (hidden_size, hidden_size)

    # Create figure for orthogonality analysis
    num_weight_matrices = len(weight_matrices)
    fig_orth, axes_orth = plt.subplots(2, num_weight_matrices, figsize=(5*num_weight_matrices, 10), dpi=150)
    if num_weight_matrices == 1:
        axes_orth = axes_orth.reshape(2, 1)

    for idx, (name, W) in enumerate(weight_matrices.items()):
        # W shape: (output_dim, input_dim)
        # Each row is the weights feeding INTO a particular output neuron

        # --- Row 1: Histogram of row-to-readout dot products ---
        # For square matrices where both dims match hidden_size, compute dot product of each row with readout
        if W.shape[0] == len(readout_weights) and W.shape[1] == len(readout_weights):
            row_readout_dots = W @ readout_unit  # (output_dim,)
            axes_orth[0, idx].hist(row_readout_dots, bins=30, edgecolor='black', alpha=0.7)
            axes_orth[0, idx].axvline(x=0, color='red', linestyle='--', linewidth=2)
            axes_orth[0, idx].set_xlabel('Dot product with readout (unit)')
            axes_orth[0, idx].set_ylabel('Count')
            axes_orth[0, idx].set_title(f'{name}\nRow-Readout Dot Products\nmean={np.mean(row_readout_dots):.4f}, std={np.std(row_readout_dots):.4f}')
        else:
            # For non-square matrices (embedding), show message
            axes_orth[0, idx].text(0.5, 0.5, f'Input dim ({W.shape[1]}) ≠\nOutput dim ({W.shape[0]})\n(Non-square: cannot compute\nrow-readout dot)',
                                  ha='center', va='center', fontsize=10, transform=axes_orth[0, idx].transAxes)
            axes_orth[0, idx].set_title(f'{name}\nRow-Readout Dot Products')

        # --- Row 2: Singular value spectrum ---
        U, S, Vh = np.linalg.svd(W, full_matrices=False)
        axes_orth[1, idx].bar(range(min(len(S), 50)), S[:50], alpha=0.7)
        axes_orth[1, idx].set_xlabel('Singular value index')
        axes_orth[1, idx].set_ylabel('Singular value')
        axes_orth[1, idx].set_title(f'{name}\nSingular Values\n(condition number: {S[0]/S[-1]:.2f})')

    plt.suptitle('Weight Matrix Orthogonality Analysis', fontsize=14)
    plt.tight_layout()
    figures["pca_frozen/weight_orthogonality_analysis"] = fig_orth
    plt.close(fig_orth)

    # --- Additional: Check if layer 1 output space is orthogonal to readout ---
    # Compute projection of layer 1 weights onto readout direction
    if args.extra_layers > 0:
        layer1_W = model.extra_hidden_layers[0].weight.detach().cpu().numpy()  # (hidden_size, hidden_size)

        # Each row of W represents the weights into one output neuron
        # The column space of W^T (= row space of W) is the space of possible outputs

        # Project each row onto readout direction
        projections_onto_readout = layer1_W @ readout_unit  # (hidden_size,)

        # Orthogonal component
        orthogonal_component_norms = []
        for row in layer1_W:
            proj = np.dot(row, readout_unit) * readout_unit
            orth = row - proj
            orthogonal_component_norms.append(np.linalg.norm(orth) / np.linalg.norm(row))

        fig_layer1_detail, axes_l1 = plt.subplots(1, 3, figsize=(15, 5), dpi=150)

        # Plot 1: Projection magnitudes onto readout
        axes_l1[0].bar(range(len(projections_onto_readout)), projections_onto_readout, alpha=0.7)
        axes_l1[0].axhline(y=0, color='red', linestyle='--')
        axes_l1[0].set_xlabel('Output neuron index')
        axes_l1[0].set_ylabel('Projection onto readout')
        axes_l1[0].set_title(f'Layer 1 Row Projections onto Readout\nmean={np.mean(projections_onto_readout):.4f}, std={np.std(projections_onto_readout):.4f}')

        # Plot 2: Fraction of each row orthogonal to readout
        axes_l1[1].hist(orthogonal_component_norms, bins=30, edgecolor='black', alpha=0.7)
        axes_l1[1].axvline(x=np.mean(orthogonal_component_norms), color='red', linestyle='--',
                          label=f'mean={np.mean(orthogonal_component_norms):.3f}')
        axes_l1[1].set_xlabel('||orthogonal component|| / ||full row||')
        axes_l1[1].set_ylabel('Count')
        axes_l1[1].set_title('Fraction of Layer 1 Rows Orthogonal to Readout')
        axes_l1[1].legend()

        # Plot 3: Compare layer 1 vs fc2 (layer 2)
        fc2_W = model.fc2.weight.detach().cpu().numpy()
        fc2_projections = fc2_W @ readout_unit

        axes_l1[2].hist(projections_onto_readout, bins=30, alpha=0.6, label=f'Layer 1 (mean={np.mean(projections_onto_readout):.3f})', edgecolor='black')
        axes_l1[2].hist(fc2_projections, bins=30, alpha=0.6, label=f'FC2/Final (mean={np.mean(fc2_projections):.3f})', edgecolor='black')
        axes_l1[2].axvline(x=0, color='red', linestyle='--')
        axes_l1[2].set_xlabel('Projection onto readout')
        axes_l1[2].set_ylabel('Count')
        axes_l1[2].set_title('Layer 1 vs FC2: Row Projections onto Readout')
        axes_l1[2].legend()

        plt.suptitle('Layer 1 Weight Matrix Detailed Analysis', fontsize=12)
        plt.tight_layout()
        figures["pca_frozen/layer1_weight_detail"] = fig_layer1_detail
        plt.close(fig_layer1_detail)

        # Print summary statistics
        print(f"\n=== Layer 1 Weight Orthogonality Summary ===")
        print(f"Layer 1 row projections onto readout: mean={np.mean(projections_onto_readout):.4f}, std={np.std(projections_onto_readout):.4f}")
        print(f"FC2 row projections onto readout: mean={np.mean(fc2_projections):.4f}, std={np.std(fc2_projections):.4f}")
        print(f"Layer 1 fraction orthogonal to readout: mean={np.mean(orthogonal_component_norms):.4f}")

        # Check if layer 1 weights are themselves orthogonal (W @ W^T ≈ I)
        WWT = layer1_W @ layer1_W.T
        WWT_diag = np.diag(WWT)
        WWT_offdiag = WWT - np.diag(WWT_diag)
        print(f"Layer 1 W @ W^T diagonal mean: {np.mean(WWT_diag):.4f}, off-diagonal mean: {np.mean(np.abs(WWT_offdiag)):.6f}")

    model.train()
    return figures


def symbolic_distance_plot(symbolic_distance_bookkeeping, episode_range, num_items, num_test_trials):
    total_episodes = len(symbolic_distance_bookkeeping[0])
    batch_size = len(symbolic_distance_bookkeeping)
    trial_size = len(symbolic_distance_bookkeeping[0][0])

    # Over the last episode_range episodes, for each batch index, for each pair of items, track the correctness of the model's predictions
    # We will then get the mean accuracy for each pair of items
    # Then we will average and get the std of accuracy over all batch indices and plot that
    correctness = {
        i: {
            (j, k): [] for j in range(0, num_items) for k in range(j+1, num_items)
        } for i in range(batch_size)
    }
    for batch_index in range(batch_size):
        for episode_num in range(1, episode_range+1):
            for trial_num in range(trial_size - num_test_trials, trial_size):
                item_1 = symbolic_distance_bookkeeping[batch_index][-episode_num][trial_num]["item_1"]
                item_2 = symbolic_distance_bookkeeping[batch_index][-episode_num][trial_num]["item_2"]
                if item_1 > item_2:
                    item_1, item_2 = item_2, item_1
                model_correctness = symbolic_distance_bookkeeping[batch_index][-episode_num][trial_num]["model_output"] == symbolic_distance_bookkeeping[batch_index][-episode_num][trial_num]["correct_choice"]
                correctness[batch_index][(item_1, item_2)].append(model_correctness)
    avg_correctness = {(i, j): [] for i in range(num_items) for j in range(i+1, num_items)}
    for i in range(batch_size):
        for j in range(num_items):
            for k in range(j+1, num_items):
                avg_correctness[(j, k)].append(np.mean(correctness[i][(j, k)]))
    # Compute median and IQR (25th and 75th percentiles)
    batch_correctness_stats = {
        (i, j): [
            np.median(avg_correctness[(i, j)]),
            np.percentile(avg_correctness[(i, j)], 25),
            np.percentile(avg_correctness[(i, j)], 75)
        ] for i in range(num_items) for j in range(i+1, num_items)
    }

    # reorganize batch_correctness_stats by symbolic distance
    batch_correctness_stats_by_symbolic_distance = {
        i: {} for i in range(1, num_items)
    }

    for j in range(num_items):
        for k in range(j+1, num_items):
            batch_correctness_stats_by_symbolic_distance[np.abs(j-k)][(j, k)] = batch_correctness_stats[(j, k)]

    # For plotting dimensions. Space each symbolic distance by some fixed amount.
    # Space each pair of items within symbolic distance by some fixed amount.

    fixed_symbolic_distance_space = 1.5  # Space between symbolic distance groups (in plot units)
    fixed_pair_space = 0.5  # Space between pairs within a group (in plot units)

    num_symbolic_distances = num_items - 1
    total_pairs = len(list(batch_correctness_stats.keys()))

    # Calculate total width in plot units, then scale to figure inches
    total_width_units = (num_symbolic_distances + 1) * fixed_symbolic_distance_space + total_pairs * fixed_pair_space
    fig_width = max(8, total_width_units * 0.6)  # Scale to reasonable figure size in inches

    # Assume for now at most 26 items so we can use letters. 0 = A, 1 = B, etc.
    pair_label_mapping = {
        (i, j): chr(i + ord('A')) + chr(j + ord('A')) for i in range(num_items) for j in range(i + 1, num_items)
    }

    fig, ax = plt.subplots(figsize=(fig_width, 6), dpi=300)

    # Add random chance baseline
    ax.axhline(y=0.5, color='lightgray', linestyle=':', linewidth=1, zorder=0)

    # Generate distinct colors for each symbolic distance
    colors = plt.cm.tab10(np.linspace(0, 1, num_symbolic_distances))

    # Track x positions and labels for axis setup
    all_x_positions = []
    all_pair_labels = []
    symbolic_distance_centers = []
    symbolic_distance_labels = []

    current_x = fixed_symbolic_distance_space  # Start with left margin

    for sd_idx, symbolic_distance in enumerate(range(1, num_items)):
        pairs = batch_correctness_stats_by_symbolic_distance[symbolic_distance]
        sorted_pairs = sorted(pairs.keys(), key=lambda x: x[0])

        x_positions = []
        medians = []
        q25s = []
        q75s = []
        pair_labels = []

        for pair_idx, pair in enumerate(sorted_pairs):
            x_pos = current_x + pair_idx * fixed_pair_space
            x_positions.append(x_pos)
            medians.append(pairs[pair][0])
            q25s.append(pairs[pair][1])
            q75s.append(pairs[pair][2])
            pair_labels.append(pair_label_mapping[pair])

        x_positions = np.array(x_positions)
        medians = np.array(medians)
        q25s = np.array(q25s)
        q75s = np.array(q75s)

        # Plot line (add marker for last symbolic distance since it has only one point)
        color = colors[sd_idx]
        is_last_sd = (symbolic_distance == num_items - 1)
        marker = 'o' if is_last_sd else ''
        ax.plot(x_positions, medians, color=color, linewidth=2, marker=marker, markersize=6, label=f'SD {symbolic_distance}')

        # Plot IQR band (same color, translucent, no border)
        ax.fill_between(x_positions, q25s, q75s, color=color, alpha=0.3, edgecolor='none')

        # Track for x-axis labels
        all_x_positions.extend(x_positions.tolist())
        all_pair_labels.extend(pair_labels)

        # Center of this symbolic distance group
        center = (x_positions[0] + x_positions[-1]) / 2
        symbolic_distance_centers.append(center)
        symbolic_distance_labels.append(str(symbolic_distance))

        # Move to next symbolic distance group
        current_x = x_positions[-1] + fixed_symbolic_distance_space

    # Set up two-tiered x-axis
    # Upper tier (closer to plot): pair labels
    ax.set_xticks(all_x_positions)
    ax.set_xticklabels(all_pair_labels, fontsize=8)

    # Lower tier (further from plot): symbolic distance labels
    # Create a secondary x-axis below the main one
    ax2 = ax.secondary_xaxis('bottom')
    ax2.set_xticks(symbolic_distance_centers)
    ax2.set_xticklabels(symbolic_distance_labels, fontsize=10, fontweight='bold')
    ax2.tick_params(length=0, pad=25)  # No tick marks, pad to push labels down
    ax2.spines['bottom'].set_visible(False)

    # Labels and title
    ax.set_title(f'Symbolic Distance Accuracy, Episodes {total_episodes - episode_range} to {total_episodes-1}')
    ax.set_ylabel('Correctness')
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    # Add xlabel below the symbolic distance labels
    fig.text(0.5, 0.02, 'Symbolic Distance', ha='center', fontsize=11)

    # Set x limits with margins
    ax.set_xlim(0, current_x)

    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for two-tier x-axis
    plt.close()
    return fig


def ai_heatmap_plot(zero_shot_trials, metadata):
    """
    Create a heatmap for associative inference zero-shot results.

    Args:
        zero_shot_trials: dict mapping (item1_id, item2_id) -> list of 0/1 (incorrect/correct)
        metadata: dict with num_groups, num_items_per_group, total_items

    Returns:
        fig: matplotlib figure
    """
    from matplotlib.colors import LinearSegmentedColormap

    num_groups = metadata['num_groups']
    num_items_per_group = metadata['num_items_per_group']
    total_items = metadata['total_items']

    # Build accuracy matrix (rows = item1, cols = item2)
    accuracy_matrix = np.full((total_items, total_items), np.nan)

    for (i, j), results in zero_shot_trials.items():
        if len(results) > 0:
            accuracy_matrix[i, j] = np.mean(results)

    # Create blue-white-red diverging colormap
    colors_cmap = [(0.0, 0.0, 1.0), (1.0, 1.0, 1.0), (1.0, 0.0, 0.0)]  # Blue -> White -> Red
    blue_white_red_cmap = LinearSegmentedColormap.from_list('blue_white_red', colors_cmap, N=256)

    # Create figure
    fig_size = max(6, total_items * 0.5 + 2)
    fig, ax = plt.subplots(figsize=(fig_size + 1.5, fig_size), dpi=150)

    # Plot heatmap (diagonal included unless exclude_same_item is set)
    im = ax.imshow(accuracy_matrix, cmap=blue_white_red_cmap, vmin=0, vmax=1, aspect='equal')

    # Mask diagonal (same-item pairs) with black if exclude_same_item is True
    exclude_same_item = metadata.get('exclude_same_item', False)
    if exclude_same_item:
        for i in range(total_items):
            rect = plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=True, facecolor='black', edgecolor='none')
            ax.add_patch(rect)

    # Set up tick labels: 0 to num_items_per_group-1, repeated for each group
    tick_positions = np.arange(total_items)
    tick_labels = [str(i % num_items_per_group) for i in range(total_items)]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=9)

    # Add group brackets and labels
    # For x-axis (bottom) - using data coordinates with clip_on=False
    for g in range(num_groups):
        start = g * num_items_per_group - 0.3
        end = (g + 1) * num_items_per_group - 1 + 0.3
        center = g * num_items_per_group + (num_items_per_group - 1) / 2

        # Bracket below x-axis (in data coordinates, below the plot)
        bracket_y_top = total_items + 0.1
        bracket_y_bottom = total_items + 0.4

        # Left vertical part
        ax.plot([start, start], [bracket_y_top, bracket_y_bottom],
                color='black', linewidth=1.5, clip_on=False)
        # Horizontal part
        ax.plot([start, end], [bracket_y_bottom, bracket_y_bottom],
                color='black', linewidth=1.5, clip_on=False)
        # Right vertical part
        ax.plot([end, end], [bracket_y_top, bracket_y_bottom],
                color='black', linewidth=1.5, clip_on=False)

        # Group label
        ax.text(center, bracket_y_bottom + 0.15, f'Group {g}',
                ha='center', va='top', fontsize=10, fontweight='bold', clip_on=False)

    # For y-axis (left) - using data coordinates with clip_on=False
    for g in range(num_groups):
        start = g * num_items_per_group - 0.3
        end = (g + 1) * num_items_per_group - 1 + 0.3
        center = g * num_items_per_group + (num_items_per_group - 1) / 2

        # Bracket to the left of y-axis
        bracket_x_right = -0.6
        bracket_x_left = -1.0

        # Top horizontal part
        ax.plot([bracket_x_left, bracket_x_right], [start, start],
                color='black', linewidth=1.5, clip_on=False)
        # Vertical part
        ax.plot([bracket_x_left, bracket_x_left], [start, end],
                color='black', linewidth=1.5, clip_on=False)
        # Bottom horizontal part
        ax.plot([bracket_x_left, bracket_x_right], [end, end],
                color='black', linewidth=1.5, clip_on=False)

        # Group label
        ax.text(bracket_x_left - 0.3, center, f'Group {g}',
                ha='right', va='center', fontsize=10, fontweight='bold', rotation=90, clip_on=False)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Accuracy', fontsize=11)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    # Labels
    ax.set_xlabel('Item 2', fontsize=11, labelpad=35)
    ax.set_ylabel('Item 1', fontsize=11, labelpad=45)
    ax.set_title('Associative Inference Zero-Shot Accuracy', fontsize=12, pad=10)

    # Adjust limits to show full cells
    ax.set_xlim(-0.5, total_items - 0.5)
    ax.set_ylim(total_items - 0.5, -0.5)

    plt.subplots_adjust(left=0.22, bottom=0.22, right=0.88, top=0.92)
    plt.close()
    return fig


def zero_shot_symbolic_distance_plot(zero_shot_trials, num_items, title='Zero-Shot Symbolic Distance Accuracy'):
    """
    Create a symbolic distance plot for zero-shot trial results from full_eval_ll.

    Args:
        zero_shot_trials: dict with keys (i, j) and values as lists of correctness (0/1)
        num_items: number of items used
        title: plot title (default: 'Zero-Shot Symbolic Distance Accuracy')
    """
    # Compute average correctness for each pair
    pair_avg_correctness = {}
    for pair, correctness_list in zero_shot_trials.items():
        if len(correctness_list) > 0:
            pair_avg_correctness[pair] = np.mean(correctness_list)
        else:
            pair_avg_correctness[pair] = np.nan

    # Organize by symbolic distance
    correctness_by_symbolic_distance = {i: {} for i in range(1, num_items)}

    for (i, j), avg in pair_avg_correctness.items():
        symbolic_distance = abs(j - i)
        if symbolic_distance > 0 and symbolic_distance < num_items:
            correctness_by_symbolic_distance[symbolic_distance][(i, j)] = avg

    # Plotting setup
    fixed_symbolic_distance_space = 1.5
    fixed_pair_space = 0.5

    num_symbolic_distances = num_items - 1
    total_pairs = len([p for p in pair_avg_correctness.keys() if not np.isnan(pair_avg_correctness[p])])

    total_width_units = (num_symbolic_distances + 1) * fixed_symbolic_distance_space + total_pairs * fixed_pair_space
    fig_width = max(8, total_width_units * 0.6)

    pair_label_mapping = {
        (i, j): chr(i + ord('A')) + chr(j + ord('A')) for i in range(num_items) for j in range(i + 1, num_items)
    }

    fig, ax = plt.subplots(figsize=(fig_width, 6), dpi=300)

    # Add random chance baseline
    ax.axhline(y=0.5, color='lightgray', linestyle=':', linewidth=1, zorder=0)

    # Generate distinct colors for each symbolic distance
    colors = plt.cm.tab10(np.linspace(0, 1, num_symbolic_distances))

    all_x_positions = []
    all_pair_labels = []
    symbolic_distance_centers = []
    symbolic_distance_labels = []

    current_x = fixed_symbolic_distance_space

    for sd_idx, symbolic_distance in enumerate(range(1, num_items)):
        pairs = correctness_by_symbolic_distance[symbolic_distance]
        sorted_pairs = sorted(pairs.keys(), key=lambda x: x[0])

        if len(sorted_pairs) == 0:
            continue

        x_positions = []
        means = []
        pair_labels = []

        for pair_idx, pair in enumerate(sorted_pairs):
            if not np.isnan(pairs[pair]):
                x_pos = current_x + pair_idx * fixed_pair_space
                x_positions.append(x_pos)
                means.append(pairs[pair])
                pair_labels.append(pair_label_mapping.get(pair, f"{pair[0]}{pair[1]}"))

        if len(x_positions) == 0:
            continue

        x_positions = np.array(x_positions)
        means = np.array(means)

        # Plot line with markers
        color = colors[sd_idx]
        is_last_sd = (symbolic_distance == num_items - 1)
        marker = 'o' if is_last_sd else ''
        ax.plot(x_positions, means, color=color, linewidth=2, marker=marker, markersize=6, label=f'SD {symbolic_distance}')

        all_x_positions.extend(x_positions.tolist())
        all_pair_labels.extend(pair_labels)

        center = (x_positions[0] + x_positions[-1]) / 2
        symbolic_distance_centers.append(center)
        symbolic_distance_labels.append(str(symbolic_distance))

        current_x = x_positions[-1] + fixed_symbolic_distance_space

    # Set up two-tiered x-axis
    ax.set_xticks(all_x_positions)
    ax.set_xticklabels(all_pair_labels, fontsize=8)

    ax2 = ax.secondary_xaxis('bottom')
    ax2.set_xticks(symbolic_distance_centers)
    ax2.set_xticklabels(symbolic_distance_labels, fontsize=10, fontweight='bold')
    ax2.tick_params(length=0, pad=25)
    ax2.spines['bottom'].set_visible(False)

    ax.set_title(title)
    ax.set_ylabel('Correctness')
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    fig.text(0.5, 0.02, 'Symbolic Distance', ha='center', fontsize=11)

    ax.set_xlim(0, current_x)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.close()
    return fig


def training_accuracy_by_trial_plot(accuracy_dict, title='Training Accuracy by Trial'):
    """
    Line plot of mean training accuracy by trial number.

    Args:
        accuracy_dict: dict mapping label -> array of shape (num_trials,) with mean accuracy per trial
        title: plot title
    """
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(accuracy_dict), 1)))

    for idx, (label, acc) in enumerate(accuracy_dict.items()):
        trial_nums = np.arange(1, len(acc) + 1)
        ax.plot(trial_nums, acc, color=colors[idx], linewidth=2, marker='o', markersize=4, label=label)

    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Chance')
    ax.set_xlabel('Trial Number')
    ax.set_ylabel('Mean Accuracy')
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(1, max(len(a) for a in accuracy_dict.values()) + 1))
    ax.set_title(title)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.close()
    return fig


def pair_logit_by_trial_heatmap(logit_mean, logit_sem, row_labels, trial_labels,
                                title='Pair Logits by Trial', ylabel='Probed Pair'):
    """
    Heatmap of logits measured after each training trial.

    Args:
        logit_mean: array (num_trials, num_rows) — mean logit per trial per row
        logit_sem: array (num_trials, num_rows) — SEM of logit
        row_labels: list of str for the y-axis (e.g. ['AB','BA',...] or ['A','B',...])
        trial_labels: list of str length num_trials — label for each training trial
        title: plot title
        ylabel: label for y-axis
    """
    num_trials, num_rows = logit_mean.shape

    # Transpose so y=row, x=trial
    data = logit_mean.T  # (num_rows, num_trials)
    sem_data = logit_sem.T

    vmax = np.nanmax(np.abs(data)) if not np.all(np.isnan(data)) else 0.1
    vmax = max(vmax, 0.01)

    fig, ax = plt.subplots(figsize=(max(8, num_trials * 1.2), max(6, num_rows * 0.5)), dpi=150)

    im = ax.imshow(data, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    ax.set_yticks(np.arange(num_rows))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(num_trials))
    ax.set_xticklabels([f"T{i+1}\n{trial_labels[i]}" for i in range(num_trials)], fontsize=8)
    ax.set_xlabel('Training Trial')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Mean Logit')

    for i in range(num_rows):
        for j in range(num_trials):
            val = data[i, j]
            sem = sem_data[i, j]
            if not np.isnan(val):
                text_color = 'white' if abs(val) > vmax * 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}\n±{sem:.2f}', ha='center', va='center',
                       color=text_color, fontsize=5)

    plt.tight_layout()
    plt.close()
    return fig


def item_dot_product_heatmaps(dot_mean_list, dot_sem_list, num_items, trial_labels,
                              title_prefix='Item Dot Products at Layer 1', num_episodes=None):
    """
    Create a series of 2N×2N dot product heatmaps (pos1+pos2 items) for each training stage.

    Args:
        dot_mean_list: list of (2*num_items, 2*num_items) mean arrays
            Index 0 = before training, index t = after trial t
        dot_sem_list: same shape, SEM arrays
        num_items: number of items
        trial_labels: list of str for each training trial
        title_prefix: prefix for plot titles
        num_episodes: number of episodes (for subtitle)
    Returns:
        dict of figure_key -> fig
    """
    figures = {}
    item_labels = [chr(ord('A') + i) for i in range(num_items)]
    tick_labels = [f"{l} L" for l in item_labels] + [f"{l} R" for l in item_labels]
    n_str = f'\nn={num_episodes} episodes' if num_episodes else ''

    for idx, (mean_mat, sem_mat) in enumerate(zip(dot_mean_list, dot_sem_list)):
        if idx == 0:
            stage = "Before Training"
            key_suffix = "pre_train"
        else:
            stage = f"After Trial {idx} ({trial_labels[idx-1]})"
            key_suffix = f"trial{idx}"

        dim = mean_mat.shape[0]
        vmax = np.nanmax(np.abs(mean_mat)) if not np.all(np.isnan(mean_mat)) else 0.1
        vmax = max(vmax, 0.01)

        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        im = ax.imshow(mean_mat, cmap='RdBu_r', aspect='equal', vmin=-vmax, vmax=vmax)
        ax.set_xticks(np.arange(dim))
        ax.set_xticklabels(tick_labels, fontsize=7, rotation=45, ha='right')
        ax.set_yticks(np.arange(dim))
        ax.set_yticklabels(tick_labels, fontsize=7)
        ax.set_title(f'{title_prefix} - {stage}{n_str}')
        plt.colorbar(im, ax=ax, label='Mean Dot Product')

        # Draw separator lines between pos1 and pos2 blocks
        ax.axhline(y=num_items - 0.5, color='black', linewidth=1.5)
        ax.axvline(x=num_items - 0.5, color='black', linewidth=1.5)

        for i in range(dim):
            for j in range(dim):
                val = mean_mat[i, j]
                if not np.isnan(val):
                    text_color = 'white' if abs(val) > vmax * 0.5 else 'black'
                    ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                           color=text_color, fontsize=4)

        plt.tight_layout()
        figures[key_suffix] = fig
        plt.close(fig)

    return figures


def delta_symbolic_distance_plot(baseline_trials, ablated_trials, num_items, title='Delta Accuracy (Baseline - Ablated)'):
    """
    Create a symbolic distance plot showing the difference between baseline and ablated accuracy.

    Positive values = baseline was better (ablation hurt performance)
    Negative values = ablation was better (ablation helped performance)

    Args:
        baseline_trials: dict with keys (i, j) and values as lists of correctness (0/1)
        ablated_trials: dict with keys (i, j) and values as lists of correctness (0/1)
        num_items: number of items used
        title: plot title
    """
    # Compute average correctness for each pair in both conditions
    baseline_avg = {}
    for pair, correctness_list in baseline_trials.items():
        if len(correctness_list) > 0:
            baseline_avg[pair] = np.mean(correctness_list)
        else:
            baseline_avg[pair] = np.nan

    ablated_avg = {}
    for pair, correctness_list in ablated_trials.items():
        if len(correctness_list) > 0:
            ablated_avg[pair] = np.mean(correctness_list)
        else:
            ablated_avg[pair] = np.nan

    # Compute delta (baseline - ablated)
    delta_by_pair = {}
    for pair in baseline_avg.keys():
        if pair in ablated_avg and not np.isnan(baseline_avg[pair]) and not np.isnan(ablated_avg[pair]):
            delta_by_pair[pair] = baseline_avg[pair] - ablated_avg[pair]
        else:
            delta_by_pair[pair] = np.nan

    # Organize by symbolic distance
    delta_by_symbolic_distance = {i: {} for i in range(1, num_items)}

    for (i, j), delta in delta_by_pair.items():
        symbolic_distance = abs(j - i)
        if symbolic_distance > 0 and symbolic_distance < num_items:
            delta_by_symbolic_distance[symbolic_distance][(i, j)] = delta

    # Plotting setup
    fixed_symbolic_distance_space = 1.5
    fixed_pair_space = 0.5

    num_symbolic_distances = num_items - 1
    total_pairs = len([p for p in delta_by_pair.keys() if not np.isnan(delta_by_pair[p])])

    total_width_units = (num_symbolic_distances + 1) * fixed_symbolic_distance_space + total_pairs * fixed_pair_space
    fig_width = max(8, total_width_units * 0.6)

    pair_label_mapping = {
        (i, j): chr(i + ord('A')) + chr(j + ord('A')) for i in range(num_items) for j in range(i + 1, num_items)
    }

    fig, ax = plt.subplots(figsize=(fig_width, 6), dpi=300)

    # Add zero baseline (no change) - solid line
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, zorder=0)
    # Add reference lines at ±0.25, ±0.5, ±0.75
    for ref_val in [0.25, 0.5, 0.75]:
        ax.axhline(y=ref_val, color='lightgray', linestyle=':', linewidth=0.5, zorder=0)
        ax.axhline(y=-ref_val, color='lightgray', linestyle=':', linewidth=0.5, zorder=0)

    # Generate distinct colors for each symbolic distance
    colors = plt.cm.tab10(np.linspace(0, 1, num_symbolic_distances))

    all_x_positions = []
    all_pair_labels = []
    symbolic_distance_centers = []
    symbolic_distance_labels = []

    current_x = fixed_symbolic_distance_space

    for sd_idx, symbolic_distance in enumerate(range(1, num_items)):
        pairs = delta_by_symbolic_distance[symbolic_distance]
        sorted_pairs = sorted(pairs.keys(), key=lambda x: x[0])

        if len(sorted_pairs) == 0:
            continue

        x_positions = []
        deltas = []
        pair_labels = []

        for pair_idx, pair in enumerate(sorted_pairs):
            if not np.isnan(pairs[pair]):
                x_pos = current_x + pair_idx * fixed_pair_space
                x_positions.append(x_pos)
                deltas.append(pairs[pair])
                pair_labels.append(pair_label_mapping.get(pair, f"{pair[0]}{pair[1]}"))

        if len(x_positions) == 0:
            continue

        x_positions = np.array(x_positions)
        deltas = np.array(deltas)

        # Plot line with markers (like original symbolic distance plot)
        color = colors[sd_idx]
        is_last_sd = (symbolic_distance == num_items - 1)
        marker = 'o' if is_last_sd else ''
        ax.plot(x_positions, deltas, color=color, linewidth=2, marker=marker, markersize=6, label=f'SD {symbolic_distance}')

        all_x_positions.extend(x_positions.tolist())
        all_pair_labels.extend(pair_labels)

        center = (x_positions[0] + x_positions[-1]) / 2
        symbolic_distance_centers.append(center)
        symbolic_distance_labels.append(str(symbolic_distance))

        current_x = x_positions[-1] + fixed_symbolic_distance_space

    # Set up two-tiered x-axis
    ax.set_xticks(all_x_positions)
    ax.set_xticklabels(all_pair_labels, fontsize=8)

    ax2 = ax.secondary_xaxis('bottom')
    ax2.set_xticks(symbolic_distance_centers)
    ax2.set_xticklabels(symbolic_distance_labels, fontsize=10, fontweight='bold')
    ax2.tick_params(length=0, pad=25)
    ax2.spines['bottom'].set_visible(False)

    ax.set_title(title)
    ax.set_ylabel('Δ Accuracy (Baseline - Ablated)')

    # Fixed y-axis from -1.0 to 1.0
    ax.set_ylim(-1.0, 1.0)
    ax.set_yticks(np.arange(-1.0, 1.1, 0.25))

    fig.text(0.5, 0.02, 'Symbolic Distance', ha='center', fontsize=11)

    ax.set_xlim(0, current_x)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.close()
    return fig


def plot_correlation_evolution_ti(args, model):
    """
    Plot how item-readout correlations evolve throughout training for TI task.

    For each layer with plastic weights, creates 2 figures (item1 position, item2 position),
    each with 6 network subplots.
    Each subplot shows 8 lines (one per item) tracking correlation with readout over trials.
    Error trials are marked with vertical red dotted lines labeled with the pair.

    Returns:
        dict of figures for wandb logging
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    num_networks = 6
    num_items = args.item_range[-1] - 1
    num_train_trials = args.num_train_trials

    # Generate batch items
    batch_items = generate_batch_items(num_items, args.item_size, num_networks, change_items_throughout_batch=True)

    # Generate training trials
    trials, correct_choices, pair_indices, num_train_trials = generate_batch_trials_ti(
        batch_items, num_train_trials, 0, arbitrary=args.arbitrary
    )

    trials = torch.tensor(trials, dtype=torch.float32).to(device)
    correct_choices_t = torch.tensor(correct_choices, dtype=torch.float32).to(device)

    # Get readout weights
    readout_weights = model.choice.weight.detach().cpu().numpy().squeeze()

    # Initialize plastic weights
    plastic_weights = torch.zeros(num_networks, args.hidden_size, args.hidden_size,
                                   dtype=torch.float32, requires_grad=False).to(device)
    extra_plastic_weights = [torch.zeros(num_networks, args.hidden_size, args.hidden_size,
                                          dtype=torch.float32, requires_grad=False).to(device)
                             for _ in range(args.extra_layers)]

    # Determine layers with plastic weights
    # Embedding layer (index 0) - no plastic weights
    # Hidden layers (indices 1 to extra_layers) - have plastic weights if extra_layers > 0
    # Final layer (index extra_layers + 1) - has plastic weights
    plastic_layer_indices = []
    plastic_layer_names = []
    if args.extra_layers > 0:
        for i in range(args.extra_layers):
            plastic_layer_indices.append(i + 1)  # Hidden layers start at index 1
            plastic_layer_names.append(f'Hidden {i + 1}')
    plastic_layer_indices.append(args.extra_layers + 1)  # Final layer
    plastic_layer_names.append('Final')

    # Storage for correlations: layer -> position -> network -> item -> list of corrs
    correlations = {
        layer_idx: {
            'item1': {net_idx: {item_idx: [] for item_idx in range(num_items)} for net_idx in range(num_networks)},
            'item2': {net_idx: {item_idx: [] for item_idx in range(num_items)} for net_idx in range(num_networks)}
        }
        for layer_idx in plastic_layer_indices
    }

    # Storage for error info
    error_trials = {net_idx: [] for net_idx in range(num_networks)}  # list of (trial_idx, pair_label)

    # Item labels
    item_labels = [chr(ord('A') + i) for i in range(num_items)]

    # Run through training trials
    for trial_idx in range(num_train_trials):
        batch_trial = trials[:, trial_idx, :]
        batch_correct_choice = correct_choices_t[:, trial_idx]

        # Before updating weights, compute correlations for each item at each position
        # For each network, test each item in position 1 and position 2
        for net_idx in range(num_networks):
            single_items = batch_items[net_idx]  # (num_items, item_size)
            single_pw = plastic_weights[net_idx:net_idx+1]
            single_epw = [epw[net_idx:net_idx+1] for epw in extra_plastic_weights]

            for item_idx in range(num_items):
                item_emb = single_items[item_idx]
                zero_emb = np.zeros_like(item_emb)

                # Item in position 1
                input_pos1 = np.concatenate([item_emb, zero_emb])
                input_pos1_t = torch.tensor(input_pos1, dtype=torch.float32).unsqueeze(0).to(device)
                dummy_correct = torch.zeros(1, dtype=torch.float32).to(device)

                with torch.inference_mode():
                    output_pos1 = model(input_pos1_t, single_pw, dummy_correct,
                                       extra_plastic_weights=single_epw, store_embeddings=True)

                # Get embeddings for each plastic layer
                for layer_idx in plastic_layer_indices:
                    emb_pos1 = output_pos1.embeddings[layer_idx][0].detach().cpu().numpy()

                    # Compute correlation
                    if np.std(emb_pos1) > 1e-10:
                        corr_pos1 = np.corrcoef(emb_pos1, readout_weights)[0, 1]
                    else:
                        corr_pos1 = 0.0
                    correlations[layer_idx]['item1'][net_idx][item_idx].append(corr_pos1)

                # Item in position 2
                input_pos2 = np.concatenate([zero_emb, item_emb])
                input_pos2_t = torch.tensor(input_pos2, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.inference_mode():
                    output_pos2 = model(input_pos2_t, single_pw, dummy_correct,
                                       extra_plastic_weights=single_epw, store_embeddings=True)

                for layer_idx in plastic_layer_indices:
                    emb_pos2 = output_pos2.embeddings[layer_idx][0].detach().cpu().numpy()

                    if np.std(emb_pos2) > 1e-10:
                        corr_pos2 = np.corrcoef(emb_pos2, readout_weights)[0, 1]
                    else:
                        corr_pos2 = 0.0
                    correlations[layer_idx]['item2'][net_idx][item_idx].append(corr_pos2)

        # Now run the actual trial to update weights
        with torch.inference_mode():
            output = model(batch_trial, plastic_weights, batch_correct_choice,
                          extra_plastic_weights=extra_plastic_weights)

        # Check for errors and record
        choice_sampled = output.sampled_choices.squeeze(-1)
        for net_idx in range(num_networks):
            if choice_sampled[net_idx].item() != batch_correct_choice[net_idx].item():
                # This was an error
                high_item = int(pair_indices[net_idx, trial_idx, 0])
                low_item = int(pair_indices[net_idx, trial_idx, 1])
                # Determine actual presentation order based on correct_choice
                # correct_choice = 1 means higher item is in position 1
                # correct_choice = 0 means higher item is in position 2
                if correct_choices[net_idx, trial_idx] == 1:
                    pos1_item, pos2_item = high_item, low_item
                else:
                    pos1_item, pos2_item = low_item, high_item
                pair_label = f"{item_labels[pos1_item]}{item_labels[pos2_item]}"
                error_trials[net_idx].append((trial_idx, pair_label))

        plastic_weights = output.plastic_weights
        extra_plastic_weights = output.extra_plastic_weights

    # Compute final correlations after the last trial (after all weight updates)
    for net_idx in range(num_networks):
        single_items = batch_items[net_idx]
        single_pw = plastic_weights[net_idx:net_idx+1]
        single_epw = [epw[net_idx:net_idx+1] for epw in extra_plastic_weights]

        for item_idx in range(num_items):
            item_emb = single_items[item_idx]
            zero_emb = np.zeros_like(item_emb)

            # Item in position 1
            input_pos1 = np.concatenate([item_emb, zero_emb])
            input_pos1_t = torch.tensor(input_pos1, dtype=torch.float32).unsqueeze(0).to(device)
            dummy_correct = torch.zeros(1, dtype=torch.float32).to(device)

            with torch.inference_mode():
                output_pos1 = model(input_pos1_t, single_pw, dummy_correct,
                                   extra_plastic_weights=single_epw, store_embeddings=True)

            for layer_idx in plastic_layer_indices:
                emb_pos1 = output_pos1.embeddings[layer_idx][0].detach().cpu().numpy()
                if np.std(emb_pos1) > 1e-10:
                    corr_pos1 = np.corrcoef(emb_pos1, readout_weights)[0, 1]
                else:
                    corr_pos1 = 0.0
                correlations[layer_idx]['item1'][net_idx][item_idx].append(corr_pos1)

            # Item in position 2
            input_pos2 = np.concatenate([zero_emb, item_emb])
            input_pos2_t = torch.tensor(input_pos2, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.inference_mode():
                output_pos2 = model(input_pos2_t, single_pw, dummy_correct,
                                   extra_plastic_weights=single_epw, store_embeddings=True)

            for layer_idx in plastic_layer_indices:
                emb_pos2 = output_pos2.embeddings[layer_idx][0].detach().cpu().numpy()
                if np.std(emb_pos2) > 1e-10:
                    corr_pos2 = np.corrcoef(emb_pos2, readout_weights)[0, 1]
                else:
                    corr_pos2 = 0.0
                correlations[layer_idx]['item2'][net_idx][item_idx].append(corr_pos2)

    # Total number of x points (trials + 1 final measurement)
    num_x_points = num_train_trials + 1

    # Create figures
    figures = {}

    # Color map for items
    colors = plt.cm.viridis(np.linspace(0, 1, num_items))

    # Create figures only for the final layer (hidden layers are uninformative)
    for layer_idx, layer_name in zip(plastic_layer_indices, plastic_layer_names):
        if layer_name != 'Final':
            continue

        # Figure for Item 1 position
        fig1, axes1 = plt.subplots(2, 3, figsize=(18, 10), dpi=150)
        axes1 = axes1.flatten()

        for net_idx in range(num_networks):
            ax = axes1[net_idx]

            # Plot each item's correlation over trials
            for item_idx in range(num_items):
                corrs = correlations[layer_idx]['item1'][net_idx][item_idx]
                ax.plot(range(num_x_points), corrs, color=colors[item_idx],
                       label=item_labels[item_idx], linewidth=1.5, alpha=0.8)

            # Mark error trials
            for trial_idx, pair_label in error_trials[net_idx]:
                ax.axvline(x=trial_idx, color='red', linestyle=':', linewidth=1, alpha=0.7)
                # Add label at top
                y_max = max([max(correlations[layer_idx]['item1'][net_idx][i]) for i in range(num_items)] + [0.1])
                ax.text(trial_idx, y_max * 1.05,
                       pair_label, fontsize=6, color='red', ha='center', va='bottom', rotation=90)

            ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
            ax.set_xlabel('Trial')
            ax.set_ylabel('Correlation with Readout')
            ax.set_title(f'Network {net_idx + 1} ({len(error_trials[net_idx])} errors)')

            # Set x ticks to even integers
            even_ticks = [i for i in range(0, num_x_points, 2)]
            ax.set_xticks(even_ticks)
            ax.set_xticklabels([str(t) for t in even_ticks])

            if net_idx == 0:
                ax.legend(loc='upper left', fontsize=7, ncol=2)

        fig1.suptitle(f'TI {layer_name}: Item-Readout Correlation Evolution (Item in Position 1)', fontsize=14)
        plt.tight_layout()
        figures[f'correlation_evolution/ti_{layer_name.lower().replace(" ", "_")}_item1_position'] = fig1
        plt.close(fig1)

        # Figure for Item 2 position
        fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10), dpi=150)
        axes2 = axes2.flatten()

        for net_idx in range(num_networks):
            ax = axes2[net_idx]

            for item_idx in range(num_items):
                corrs = correlations[layer_idx]['item2'][net_idx][item_idx]
                ax.plot(range(num_x_points), corrs, color=colors[item_idx],
                       label=item_labels[item_idx], linewidth=1.5, alpha=0.8)

            # Mark error trials
            for trial_idx, pair_label in error_trials[net_idx]:
                ax.axvline(x=trial_idx, color='red', linestyle=':', linewidth=1, alpha=0.7)
                y_max = max([max(correlations[layer_idx]['item2'][net_idx][i]) for i in range(num_items)] + [0.1])
                ax.text(trial_idx, y_max * 1.05,
                       pair_label, fontsize=6, color='red', ha='center', va='bottom', rotation=90)

            ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
            ax.set_xlabel('Trial')
            ax.set_ylabel('Correlation with Readout')
            ax.set_title(f'Network {net_idx + 1} ({len(error_trials[net_idx])} errors)')

            # Set x ticks to even integers
            even_ticks = [i for i in range(0, num_x_points, 2)]
            ax.set_xticks(even_ticks)
            ax.set_xticklabels([str(t) for t in even_ticks])

            if net_idx == 0:
                ax.legend(loc='upper left', fontsize=7, ncol=2)

        fig2.suptitle(f'TI {layer_name}: Item-Readout Correlation Evolution (Item in Position 2)', fontsize=14)
        plt.tight_layout()
        figures[f'correlation_evolution/ti_{layer_name.lower().replace(" ", "_")}_item2_position'] = fig2
        plt.close(fig2)

    model.train()
    return figures


def plot_correlation_evolution_ll(args, model):
    """
    Plot how item-readout correlations evolve throughout training for LL task.

    For each layer with plastic weights, creates 2 figures (item1 position, item2 position),
    each with 6 network subplots.
    Each subplot shows 8 lines (one per item) tracking correlation with readout over trials.
    Error trials are marked with vertical red dotted lines labeled with the pair.

    Returns:
        dict of figures for wandb logging
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    num_networks = 6
    num_items = 8  # Fixed for list linking
    num_train_trials = args.num_trials_list_1 + args.num_trials_list_2 + args.num_trials_linking_pair

    # Generate batch items
    batch_items = generate_batch_items(num_items, args.item_size, num_networks, change_items_throughout_batch=True)

    # Generate training trials (no test trials needed)
    trials, correct_choices, pair_indices = generate_batch_trials_ll(
        batch_items, args.num_trials_list_1, args.num_trials_list_2,
        args.num_trials_linking_pair, 0,
        put_linking_trials_first=args.put_linking_trials_first,
        randomize_list_order=args.randomize_list_order
    )

    trials = torch.tensor(trials, dtype=torch.float32).to(device)
    correct_choices_t = torch.tensor(correct_choices, dtype=torch.float32).to(device)

    # Get readout weights
    readout_weights = model.choice.weight.detach().cpu().numpy().squeeze()

    # Initialize plastic weights
    plastic_weights = torch.zeros(num_networks, args.hidden_size, args.hidden_size,
                                   dtype=torch.float32, requires_grad=False).to(device)
    extra_plastic_weights = [torch.zeros(num_networks, args.hidden_size, args.hidden_size,
                                          dtype=torch.float32, requires_grad=False).to(device)
                             for _ in range(args.extra_layers)]

    # Determine layers with plastic weights
    plastic_layer_indices = []
    plastic_layer_names = []
    if args.extra_layers > 0:
        for i in range(args.extra_layers):
            plastic_layer_indices.append(i + 1)
            plastic_layer_names.append(f'Hidden {i + 1}')
    plastic_layer_indices.append(args.extra_layers + 1)
    plastic_layer_names.append('Final')

    # Storage for correlations: layer -> position -> network -> item -> list of corrs
    correlations = {
        layer_idx: {
            'item1': {net_idx: {item_idx: [] for item_idx in range(num_items)} for net_idx in range(num_networks)},
            'item2': {net_idx: {item_idx: [] for item_idx in range(num_items)} for net_idx in range(num_networks)}
        }
        for layer_idx in plastic_layer_indices
    }

    # Storage for error info
    error_trials = {net_idx: [] for net_idx in range(num_networks)}

    # Item labels for LL: A B C D | E F G H (with D > E being the linking pair)
    item_labels = [chr(ord('A') + i) for i in range(num_items)]

    # Run through training trials
    for trial_idx in range(num_train_trials):
        batch_trial = trials[:, trial_idx, :]
        batch_correct_choice = correct_choices_t[:, trial_idx]

        # Compute correlations for each item at each position
        for net_idx in range(num_networks):
            single_items = batch_items[net_idx]
            single_pw = plastic_weights[net_idx:net_idx+1]
            single_epw = [epw[net_idx:net_idx+1] for epw in extra_plastic_weights]

            for item_idx in range(num_items):
                item_emb = single_items[item_idx]
                zero_emb = np.zeros_like(item_emb)

                # Item in position 1
                input_pos1 = np.concatenate([item_emb, zero_emb])
                input_pos1_t = torch.tensor(input_pos1, dtype=torch.float32).unsqueeze(0).to(device)
                dummy_correct = torch.zeros(1, dtype=torch.float32).to(device)

                with torch.inference_mode():
                    output_pos1 = model(input_pos1_t, single_pw, dummy_correct,
                                       extra_plastic_weights=single_epw, store_embeddings=True)

                for layer_idx in plastic_layer_indices:
                    emb_pos1 = output_pos1.embeddings[layer_idx][0].detach().cpu().numpy()

                    if np.std(emb_pos1) > 1e-10:
                        corr_pos1 = np.corrcoef(emb_pos1, readout_weights)[0, 1]
                    else:
                        corr_pos1 = 0.0
                    correlations[layer_idx]['item1'][net_idx][item_idx].append(corr_pos1)

                # Item in position 2
                input_pos2 = np.concatenate([zero_emb, item_emb])
                input_pos2_t = torch.tensor(input_pos2, dtype=torch.float32).unsqueeze(0).to(device)

                with torch.inference_mode():
                    output_pos2 = model(input_pos2_t, single_pw, dummy_correct,
                                       extra_plastic_weights=single_epw, store_embeddings=True)

                for layer_idx in plastic_layer_indices:
                    emb_pos2 = output_pos2.embeddings[layer_idx][0].detach().cpu().numpy()

                    if np.std(emb_pos2) > 1e-10:
                        corr_pos2 = np.corrcoef(emb_pos2, readout_weights)[0, 1]
                    else:
                        corr_pos2 = 0.0
                    correlations[layer_idx]['item2'][net_idx][item_idx].append(corr_pos2)

        # Run actual trial to update weights
        with torch.inference_mode():
            output = model(batch_trial, plastic_weights, batch_correct_choice,
                          extra_plastic_weights=extra_plastic_weights)

        # Check for errors
        choice_sampled = output.sampled_choices.squeeze(-1)
        for net_idx in range(num_networks):
            if choice_sampled[net_idx].item() != batch_correct_choice[net_idx].item():
                high_item = int(pair_indices[net_idx, trial_idx, 0])
                low_item = int(pair_indices[net_idx, trial_idx, 1])
                # Determine actual presentation order based on correct_choice
                # correct_choice = 1 means higher item is in position 1
                # correct_choice = 0 means higher item is in position 2
                if correct_choices[net_idx, trial_idx] == 1:
                    pos1_item, pos2_item = high_item, low_item
                else:
                    pos1_item, pos2_item = low_item, high_item
                pair_label = f"{item_labels[pos1_item]}{item_labels[pos2_item]}"
                error_trials[net_idx].append((trial_idx, pair_label))

        plastic_weights = output.plastic_weights
        extra_plastic_weights = output.extra_plastic_weights

    # Compute final correlations after the last trial (after all weight updates)
    for net_idx in range(num_networks):
        single_items = batch_items[net_idx]
        single_pw = plastic_weights[net_idx:net_idx+1]
        single_epw = [epw[net_idx:net_idx+1] for epw in extra_plastic_weights]

        for item_idx in range(num_items):
            item_emb = single_items[item_idx]
            zero_emb = np.zeros_like(item_emb)

            # Item in position 1
            input_pos1 = np.concatenate([item_emb, zero_emb])
            input_pos1_t = torch.tensor(input_pos1, dtype=torch.float32).unsqueeze(0).to(device)
            dummy_correct = torch.zeros(1, dtype=torch.float32).to(device)

            with torch.inference_mode():
                output_pos1 = model(input_pos1_t, single_pw, dummy_correct,
                                   extra_plastic_weights=single_epw, store_embeddings=True)

            for layer_idx in plastic_layer_indices:
                emb_pos1 = output_pos1.embeddings[layer_idx][0].detach().cpu().numpy()
                if np.std(emb_pos1) > 1e-10:
                    corr_pos1 = np.corrcoef(emb_pos1, readout_weights)[0, 1]
                else:
                    corr_pos1 = 0.0
                correlations[layer_idx]['item1'][net_idx][item_idx].append(corr_pos1)

            # Item in position 2
            input_pos2 = np.concatenate([zero_emb, item_emb])
            input_pos2_t = torch.tensor(input_pos2, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.inference_mode():
                output_pos2 = model(input_pos2_t, single_pw, dummy_correct,
                                   extra_plastic_weights=single_epw, store_embeddings=True)

            for layer_idx in plastic_layer_indices:
                emb_pos2 = output_pos2.embeddings[layer_idx][0].detach().cpu().numpy()
                if np.std(emb_pos2) > 1e-10:
                    corr_pos2 = np.corrcoef(emb_pos2, readout_weights)[0, 1]
                else:
                    corr_pos2 = 0.0
                correlations[layer_idx]['item2'][net_idx][item_idx].append(corr_pos2)

    # Total number of x points (trials + 1 final measurement)
    num_x_points = num_train_trials + 1

    # Create figures
    figures = {}

    # Color map - use different colors for list 1 (0-3) and list 2 (4-7)
    colors_list1 = plt.cm.Blues(np.linspace(0.4, 0.9, 4))
    colors_list2 = plt.cm.Oranges(np.linspace(0.4, 0.9, 4))
    colors = np.vstack([colors_list1, colors_list2])

    # Create figures only for the final layer (hidden layers are uninformative)
    for layer_idx, layer_name in zip(plastic_layer_indices, plastic_layer_names):
        if layer_name != 'Final':
            continue

        # Figure for Item 1 position
        fig1, axes1 = plt.subplots(2, 3, figsize=(18, 10), dpi=150)
        axes1 = axes1.flatten()

        for net_idx in range(num_networks):
            ax = axes1[net_idx]

            for item_idx in range(num_items):
                corrs = correlations[layer_idx]['item1'][net_idx][item_idx]
                ax.plot(range(num_x_points), corrs, color=colors[item_idx],
                       label=item_labels[item_idx], linewidth=1.5, alpha=0.8)

            # Mark error trials
            for trial_idx, pair_label in error_trials[net_idx]:
                ax.axvline(x=trial_idx, color='red', linestyle=':', linewidth=1, alpha=0.7)
                y_max = max([max(correlations[layer_idx]['item1'][net_idx][i]) for i in range(num_items)] + [0.1])
                ax.text(trial_idx, y_max * 1.05,
                       pair_label, fontsize=6, color='red', ha='center', va='bottom', rotation=90)

            ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
            ax.set_xlabel('Trial')
            ax.set_ylabel('Correlation with Readout')
            ax.set_title(f'Network {net_idx + 1} ({len(error_trials[net_idx])} errors)')

            # Set x ticks to even integers
            even_ticks = [i for i in range(0, num_x_points, 2)]
            ax.set_xticks(even_ticks)
            ax.set_xticklabels([str(t) for t in even_ticks])

            if net_idx == 0:
                ax.legend(loc='upper left', fontsize=7, ncol=2)

        fig1.suptitle(f'LL {layer_name}: Item-Readout Correlation Evolution (Item in Position 1)', fontsize=14)
        plt.tight_layout()
        figures[f'correlation_evolution/ll_{layer_name.lower().replace(" ", "_")}_item1_position'] = fig1
        plt.close(fig1)

        # Figure for Item 2 position
        fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10), dpi=150)
        axes2 = axes2.flatten()

        for net_idx in range(num_networks):
            ax = axes2[net_idx]

            for item_idx in range(num_items):
                corrs = correlations[layer_idx]['item2'][net_idx][item_idx]
                ax.plot(range(num_x_points), corrs, color=colors[item_idx],
                       label=item_labels[item_idx], linewidth=1.5, alpha=0.8)

            # Mark error trials
            for trial_idx, pair_label in error_trials[net_idx]:
                ax.axvline(x=trial_idx, color='red', linestyle=':', linewidth=1, alpha=0.7)
                y_max = max([max(correlations[layer_idx]['item2'][net_idx][i]) for i in range(num_items)] + [0.1])
                ax.text(trial_idx, y_max * 1.05,
                       pair_label, fontsize=6, color='red', ha='center', va='bottom', rotation=90)

            ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
            ax.set_xlabel('Trial')
            ax.set_ylabel('Correlation with Readout')
            ax.set_title(f'Network {net_idx + 1} ({len(error_trials[net_idx])} errors)')

            # Set x ticks to even integers
            even_ticks = [i for i in range(0, num_x_points, 2)]
            ax.set_xticks(even_ticks)
            ax.set_xticklabels([str(t) for t in even_ticks])

            if net_idx == 0:
                ax.legend(loc='upper left', fontsize=7, ncol=2)

        fig2.suptitle(f'LL {layer_name}: Item-Readout Correlation Evolution (Item in Position 2)', fontsize=14)
        plt.tight_layout()
        figures[f'correlation_evolution/ll_{layer_name.lower().replace(" ", "_")}_item2_position'] = fig2
        plt.close(fig2)

    model.train()
    return figures


def plot_list_linking_analysis(args, model):
    """
    Create rank vs identity encoding analysis plots for list linking task.

    This performs similar analysis to plot_pca_frozen_by_symbolic_distance but
    adapted for the list linking task structure:
    - 8 items total: List 1 (ABCD, items 0-3) and List 2 (EFGH, items 4-7)
    - Training: adjacent pairs within each list + linking pair (D>E, items 3>4)
    - Testing: cross-list comparisons (items from list 1 vs items from list 2)

    For list linking, the "signed symbolic distance" is defined as:
    - For same-list pairs: item1_idx - item2_idx (standard)
    - For cross-list pairs: The global rank difference (list1 items 0-3, list2 items 4-7)
      So A(0) vs E(4) has signed_sd = 0-4 = -4

    Returns a dict of figures for wandb logging.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    batch_size = args.batch_size * 4
    num_items = 8  # Fixed for list linking: 4 items per list

    # Generate batch items (different items per network for robust averaging)
    batch_items = generate_batch_items(num_items, args.item_size, batch_size, change_items_throughout_batch=True)

    # Set list linking parameters (use args if available, else defaults)
    num_trials_list_1 = getattr(args, 'num_trials_list_1', 20)
    num_trials_list_2 = getattr(args, 'num_trials_list_2', 20)
    num_trials_linking_pair = getattr(args, 'num_trials_linking_pair', 10)
    put_linking_trials_first = getattr(args, 'put_linking_trials_first', False)
    randomize_list_order = getattr(args, 'randomize_list_order', False)

    # Generate list linking trials (only training, no test trials yet)
    trials, correct_choices, pair_indices_ll = generate_batch_trials_ll(
        batch_items,
        num_trials_list_1,
        num_trials_list_2,
        num_trials_linking_pair,
        num_test_trials=0,  # No test trials during training phase
        put_linking_trials_first=put_linking_trials_first,
        randomize_list_order=randomize_list_order
    )

    num_train_trials = num_trials_list_1 + num_trials_list_2 + num_trials_linking_pair

    trials = torch.tensor(trials, dtype=torch.float32).to(device)
    correct_choices_tensor = torch.tensor(correct_choices, dtype=torch.float32).to(device)

    # Initialize plastic weights
    plastic_weights = torch.zeros(batch_size, args.hidden_size, args.hidden_size,
                                   dtype=torch.float32, requires_grad=False).to(device)
    extra_plastic_weights = [torch.zeros(batch_size, args.hidden_size, args.hidden_size,
                                          dtype=torch.float32, requires_grad=False).to(device)
                             for _ in range(args.extra_layers)]

    # --- Neuromodulator tracking storage for LL ---
    neuromodulator_history_ll = []
    reward_history_ll = []
    choice_history_ll = []

    # Run training phase to build up plastic weights
    for trial in range(num_train_trials):
        batch_trial = trials[:, trial, :]
        batch_correct_choice = correct_choices_tensor[:, trial]

        with torch.inference_mode():
            output = model(batch_trial, plastic_weights, batch_correct_choice,
                          extra_plastic_weights=extra_plastic_weights, store_embeddings=False)

        # Track neuromodulator values
        nm_values_ll = output.neuromodulator.detach().cpu().numpy()
        neuromodulator_history_ll.append(nm_values_ll)

        # Track actual rewards from model output
        reward_history_ll.append(output.reward.squeeze().detach().cpu().numpy())

        # Track model's choice (sigmoid output)
        choice_history_ll.append(output.choice.squeeze().detach().cpu().numpy())

        plastic_weights = output.plastic_weights
        extra_plastic_weights = output.extra_plastic_weights

    # Freeze plastic weights
    frozen_plastic_weights = plastic_weights.clone()
    frozen_extra_plastic_weights = [epw.clone() for epw in extra_plastic_weights]

    figures = {}

    # --- Alpha × PW Histograms for LL ---
    # Helper function to create histogram
    def create_histogram_ll(values, title, filename, bins=100):
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        ax.hist(values.flatten(), bins=bins, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='zero')
        ax.axvline(x=values.mean(), color='green', linestyle='-', alpha=0.7, label=f'mean={values.mean():.4f}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
        ax.set_title(f'{title}\n(n={values.size:,}, std={values.std():.4f})')
        ax.legend()
        plt.tight_layout()
        figures[filename] = fig
        plt.close(fig)

    # Get alpha parameters from model
    alpha_final_ll = model.alpha.detach().cpu().numpy()
    alpha_extra_list_ll = [model.alpha_extra[i].detach().cpu().numpy() for i in range(args.extra_layers)]

    # Compute mean plastic weights for LL
    pw_mean_ll = frozen_plastic_weights.detach().cpu().numpy().mean(axis=0)
    alpha_pw_final_ll = alpha_final_ll * pw_mean_ll

    # Final layer histogram
    create_histogram_ll(alpha_pw_final_ll, 'Alpha × PW (LL) - Final Layer', 'list_linking/alpha_pw_final_histogram')

    # Extra layers histograms
    for layer_idx, epw in enumerate(frozen_extra_plastic_weights):
        epw_mean_ll = epw.detach().cpu().numpy().mean(axis=0)
        alpha_pw_extra_ll = alpha_extra_list_ll[layer_idx] * epw_mean_ll
        create_histogram_ll(alpha_pw_extra_ll, f'Alpha × PW (LL) - Hidden Layer {layer_idx + 1}',
                           f'list_linking/alpha_pw_hidden{layer_idx + 1}_histogram')

    # --- Neuromodulator Analysis Plots for LL ---
    neuromodulator_history_ll = np.array(neuromodulator_history_ll)
    num_train_trials_ll = neuromodulator_history_ll.shape[0]

    # Use the first (or only) neuromodulator
    # Squeeze to remove any extra dimensions and get shape: (num_train_trials, batch_size)
    nm_main_ll = np.squeeze(neuromodulator_history_ll)
    if nm_main_ll.ndim == 3:
        nm_main_ll = nm_main_ll[:, :, 0]  # First neuromodulator if multiple
    elif nm_main_ll.ndim == 1:
        nm_main_ll = nm_main_ll[:, np.newaxis]  # Add batch dimension if missing

    # Item labels for LL
    item_labels_ll_nm = [chr(ord('A') + i) for i in range(num_items)]

    # --- Plot 1: Heatmap of individual networks (first 10) x trials ---
    # Now includes neuromodulator, readout (sigmoid), and actual reward (±1) with pair labels
    num_networks_to_show_ll = min(10, batch_size)

    # Convert reward history to array
    reward_history_ll_np = np.array(reward_history_ll)  # (num_train_trials, batch_size) or (num_train_trials, batch_size, ...)
    # Ensure 2D shape (num_train_trials, batch_size)
    # First squeeze any dimensions of size 1
    reward_history_ll_np = np.squeeze(reward_history_ll_np)
    # If still more than 2D, take the first slice of extra dimensions
    while reward_history_ll_np.ndim > 2:
        reward_history_ll_np = reward_history_ll_np[..., 0]
    if reward_history_ll_np.ndim == 1:
        reward_history_ll_np = reward_history_ll_np[:, np.newaxis]

    # Convert choice history to array
    choice_history_ll_np = np.array(choice_history_ll)  # (num_train_trials, batch_size)
    choice_history_ll_np = np.squeeze(choice_history_ll_np)
    while choice_history_ll_np.ndim > 2:
        choice_history_ll_np = choice_history_ll_np[..., 0]
    if choice_history_ll_np.ndim == 1:
        choice_history_ll_np = choice_history_ll_np[:, np.newaxis]

    # Create figure with 3 subplots stacked vertically
    fig_nm_heatmap_ll, axes_nm_heatmap = plt.subplots(3, 1, figsize=(14, 12), dpi=150)

    # Panel 1: Neuromodulator values
    nm_subset_ll = nm_main_ll[:, :num_networks_to_show_ll].T  # (num_networks, num_trials)
    vmax_nm_ll = max(abs(nm_subset_ll.min()), abs(nm_subset_ll.max()))
    if vmax_nm_ll == 0:
        vmax_nm_ll = 1
    im_nm_ll = axes_nm_heatmap[0].imshow(nm_subset_ll, cmap='RdBu_r', vmin=-vmax_nm_ll, vmax=vmax_nm_ll, aspect='auto')
    axes_nm_heatmap[0].set_ylabel('Network')
    axes_nm_heatmap[0].set_yticks(range(num_networks_to_show_ll))
    axes_nm_heatmap[0].set_yticklabels([f'Net {i}' for i in range(num_networks_to_show_ll)])
    axes_nm_heatmap[0].set_title('Neuromodulator Values')
    plt.colorbar(im_nm_ll, ax=axes_nm_heatmap[0], label='Neuromodulator')

    # Add neuromodulator values as black text inside boxes
    for net_idx in range(num_networks_to_show_ll):
        for trial_idx in range(num_train_trials_ll):
            nm_val = nm_subset_ll[net_idx, trial_idx]
            axes_nm_heatmap[0].text(trial_idx, net_idx, f'{nm_val:.2f}',
                                    ha='center', va='center', fontsize=5, color='black', fontweight='bold')

    # Panel 2: Readout values (sigmoid output) - model's confidence
    choice_subset_ll = choice_history_ll_np[:, :num_networks_to_show_ll].T  # (num_networks, num_trials)
    im_choice_ll = axes_nm_heatmap[1].imshow(choice_subset_ll, cmap='RdBu_r', vmin=0, vmax=1, aspect='auto')
    axes_nm_heatmap[1].set_ylabel('Network')
    axes_nm_heatmap[1].set_yticks(range(num_networks_to_show_ll))
    axes_nm_heatmap[1].set_yticklabels([f'Net {i}' for i in range(num_networks_to_show_ll)])
    axes_nm_heatmap[1].set_title('Readout (Sigmoid) - P(choose position 1)')
    plt.colorbar(im_choice_ll, ax=axes_nm_heatmap[1], label='P(pos 1)')

    # Add readout values as black text inside boxes
    for net_idx in range(num_networks_to_show_ll):
        for trial_idx in range(num_train_trials_ll):
            choice_val = choice_subset_ll[net_idx, trial_idx]
            axes_nm_heatmap[1].text(trial_idx, net_idx, f'{choice_val:.2f}',
                                    ha='center', va='center', fontsize=5, color='black', fontweight='bold')

    # Panel 3: Actual reward (±1) with pair labels as white text
    reward_subset_ll = reward_history_ll_np[:, :num_networks_to_show_ll].T  # (num_networks, num_trials)
    im_reward = axes_nm_heatmap[2].imshow(reward_subset_ll, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    axes_nm_heatmap[2].set_xlabel('Trial')
    axes_nm_heatmap[2].set_ylabel('Network')
    axes_nm_heatmap[2].set_yticks(range(num_networks_to_show_ll))
    axes_nm_heatmap[2].set_yticklabels([f'Net {i}' for i in range(num_networks_to_show_ll)])
    axes_nm_heatmap[2].set_title('Reward & Pair Presented (+1=correct green, -1=incorrect red)')
    plt.colorbar(im_reward, ax=axes_nm_heatmap[2], label='Reward')

    # Add pair labels as white text on the reward heatmap
    for net_idx in range(num_networks_to_show_ll):
        for trial_idx in range(num_train_trials_ll):
            high_item = int(pair_indices_ll[net_idx, trial_idx, 0])
            low_item = int(pair_indices_ll[net_idx, trial_idx, 1])
            # Determine actual presentation order from correct_choice
            if correct_choices[net_idx, trial_idx] == 0:
                pair_label = f"{item_labels_ll_nm[high_item]}{item_labels_ll_nm[low_item]}"
            else:
                pair_label = f"{item_labels_ll_nm[low_item]}{item_labels_ll_nm[high_item]}"
            axes_nm_heatmap[2].text(trial_idx, net_idx, pair_label,
                                    ha='center', va='center', fontsize=5, color='white', fontweight='bold')

    plt.suptitle(f'List Linking: Neuromodulator, Readout, and Reward Analysis\n(first {num_networks_to_show_ll} networks)', fontsize=14)
    plt.tight_layout()
    figures["list_linking/neuromodulator_heatmap"] = fig_nm_heatmap_ll
    plt.close(fig_nm_heatmap_ll)

    # --- Prepare data for neuromodulator by category ---
    # Adjacent pairs for LL:
    # List 1: AB, BC, CD (items 0-1, 1-2, 2-3)
    # Linking: DE (items 3-4)
    # List 2: EF, FG, GH (items 4-5, 5-6, 6-7)
    pair_categories = {
        'List 1': ['AB', 'BC', 'CD'],
        'Linking': ['DE'],
        'List 2': ['EF', 'FG', 'GH']
    }
    all_ll_pairs = ['AB', 'BC', 'CD', 'DE', 'EF', 'FG', 'GH']
    nm_by_pair_ll = {pair_name: [] for pair_name in all_ll_pairs}

    # pair_indices_ll shape: (batch_size, num_trials, 2)
    for trial_idx in range(num_train_trials_ll):
        for network_idx in range(batch_size):
            high_item = pair_indices_ll[network_idx, trial_idx, 0]
            low_item = pair_indices_ll[network_idx, trial_idx, 1]
            # Adjacent pairs have low_item = high_item + 1
            if low_item == high_item + 1 and high_item < num_items - 1:
                pair_name = f"{item_labels_ll_nm[high_item]}{item_labels_ll_nm[low_item]}"
                if pair_name in nm_by_pair_ll:
                    nm_by_pair_ll[pair_name].append(nm_main_ll[trial_idx, network_idx])

    # --- Plot 5: Neuromodulator by category (aggregated) ---
    fig_nm_cat, ax_nm_cat = plt.subplots(figsize=(8, 6), dpi=150)
    cat_names = ['List 1\n(AB,BC,CD)', 'Linking\n(DE)', 'List 2\n(EF,FG,GH)']
    cat_colors = ['tab:blue', 'tab:red', 'tab:green']
    cat_values = [[], [], []]
    for pair in pair_categories['List 1']:
        cat_values[0].extend(nm_by_pair_ll[pair])
    for pair in pair_categories['Linking']:
        cat_values[1].extend(nm_by_pair_ll[pair])
    for pair in pair_categories['List 2']:
        cat_values[2].extend(nm_by_pair_ll[pair])

    cat_means = [np.mean(v) if len(v) > 0 else 0 for v in cat_values]
    cat_stds = [np.std(v) if len(v) > 0 else 0 for v in cat_values]
    cat_counts = [len(v) for v in cat_values]

    bars_cat = ax_nm_cat.bar(range(3), cat_means, yerr=cat_stds, capsize=5,
                              color=cat_colors, edgecolor='black', alpha=0.8)
    ax_nm_cat.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax_nm_cat.set_xticks(range(3))
    ax_nm_cat.set_xticklabels(cat_names)
    ax_nm_cat.set_ylabel('Mean Neuromodulator Value')
    ax_nm_cat.set_title('List Linking: Neuromodulator by Category')

    for i, (bar, count) in enumerate(zip(bars_cat, cat_counts)):
        ax_nm_cat.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'n={count}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    figures["list_linking/neuromodulator_by_category"] = fig_nm_cat
    plt.close(fig_nm_cat)

    # --- Coefficient matrix A for individual networks (to compare with neuromodulator) ---
    # Compute A_coeffs for the same networks shown in neuromodulator_heatmap
    if args.extra_layers > 0:
        # Define adjacent pairs and presentations (same structure as later)
        adjacent_pairs_ll_nm = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7)]
        all_presentations_nm = []
        presentation_to_idx_nm = {}
        for pair in adjacent_pairs_ll_nm:
            winner, loser = pair
            all_presentations_nm.append((winner, loser))
            presentation_to_idx_nm[(winner, loser)] = len(all_presentations_nm) - 1
            all_presentations_nm.append((loser, winner))
            presentation_to_idx_nm[(loser, winner)] = len(all_presentations_nm) - 1
        num_presentations_nm = len(all_presentations_nm)

        # Get model parameters for coefficient tracking
        alpha_first_nm = model.alpha_extra[0].detach().cpu().numpy()
        m_hebb_nm = model.hebbian_trace_multiplier_extra[0].item()

        # Compute A_coeffs for each of the first num_networks_to_show_ll networks
        all_A_coeffs = []
        all_A_col_sums = []

        for net_idx in range(num_networks_to_show_ll):
            single_items_nm = batch_items[net_idx]

            # Compute embeddings for this network's items
            embeddings_u_nm = []
            with torch.no_grad():
                for (item1_idx, item2_idx) in all_presentations_nm:
                    item1_emb = single_items_nm[item1_idx]
                    item2_emb = single_items_nm[item2_idx]
                    input_vec = np.concatenate([item1_emb, item2_emb])
                    input_t = torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0).to(device)
                    if hasattr(model, 'embedding_layer'):
                        u = torch.tanh(model.embedding_layer(input_t))
                    else:
                        u = input_t
                    embeddings_u_nm.append(u.squeeze(0).cpu().numpy())
            embeddings_u_nm = np.array(embeddings_u_nm)

            # Compute D_alpha for this network
            if alpha_first_nm.ndim >= 2:
                alpha_mean_nm = alpha_first_nm.mean(axis=0)
            else:
                alpha_mean_nm = float(alpha_first_nm)
            D_alpha_nm = (embeddings_u_nm * alpha_mean_nm) @ embeddings_u_nm.T

            # Initialize and track coefficients
            A_coeffs_nm = np.zeros((num_presentations_nm, num_presentations_nm))
            pw_track_nm = torch.zeros(1, args.hidden_size, args.hidden_size, dtype=torch.float32).to(device)
            epw_track_nm = [torch.zeros(1, args.hidden_size, args.hidden_size, dtype=torch.float32).to(device)
                           for _ in range(args.extra_layers)]

            single_trials_nm = trials[net_idx:net_idx+1, :, :]
            single_correct_nm = correct_choices_tensor[net_idx:net_idx+1, :]

            for trial_idx in range(num_train_trials):
                trial_input = single_trials_nm[:, trial_idx, :]
                trial_correct = single_correct_nm[:, trial_idx]

                item_size = args.item_size
                item1_emb_trial = trial_input[0, :item_size].cpu().numpy()
                item2_emb_trial = trial_input[0, item_size:2*item_size].cpu().numpy()

                item1_idx_found = None
                item2_idx_found = None
                for idx in range(num_items):
                    if np.allclose(single_items_nm[idx], item1_emb_trial, atol=1e-5):
                        item1_idx_found = idx
                    if np.allclose(single_items_nm[idx], item2_emb_trial, atol=1e-5):
                        item2_idx_found = idx

                if item1_idx_found is None or item2_idx_found is None:
                    with torch.no_grad():
                        output_track = model(trial_input, pw_track_nm, trial_correct,
                                            extra_plastic_weights=epw_track_nm, store_embeddings=False)
                    pw_track_nm = output_track.plastic_weights
                    epw_track_nm = output_track.extra_plastic_weights
                    continue

                presentation_key = (item1_idx_found, item2_idx_found)
                if presentation_key not in presentation_to_idx_nm:
                    with torch.no_grad():
                        output_track = model(trial_input, pw_track_nm, trial_correct,
                                            extra_plastic_weights=epw_track_nm, store_embeddings=False)
                    pw_track_nm = output_track.plastic_weights
                    epw_track_nm = output_track.extra_plastic_weights
                    continue

                k = presentation_to_idx_nm[presentation_key]

                with torch.no_grad():
                    output_track = model(trial_input, pw_track_nm, trial_correct,
                                        extra_plastic_weights=epw_track_nm, store_embeddings=False)

                nm_output = output_track.neuromodulator.squeeze()
                if args.use_extra_neuromodulator and args.extra_layers > 0:
                    eta_t = nm_output[0].item() if nm_output.dim() > 0 else nm_output.item()
                else:
                    eta_t = nm_output.item() if nm_output.dim() == 0 else nm_output[0].item()

                e_k = np.zeros(num_presentations_nm)
                e_k[k] = 1.0
                tanh_scale = 0.9
                A_coeffs_nm[:, k] += eta_t * m_hebb_nm * tanh_scale * (e_k + A_coeffs_nm @ D_alpha_nm[:, k])

                pw_track_nm = output_track.plastic_weights
                epw_track_nm = output_track.extra_plastic_weights

            all_A_coeffs.append(A_coeffs_nm)
            all_A_col_sums.append(A_coeffs_nm.sum(axis=0))  # Sum over rows for each column

        # Create figure showing A_col_sums for each network
        # This shows which adjacent pairs accumulated coefficients
        presentation_labels_nm = []
        for (i1, i2) in all_presentations_nm:
            presentation_labels_nm.append(f"{item_labels_ll_nm[i1]}{item_labels_ll_nm[i2]}")

        fig_coeffs_networks, axes_coeffs = plt.subplots(2, 1, figsize=(14, 10), dpi=150)

        # Panel 1: A_col_sums heatmap (networks x presentations)
        A_col_sums_matrix = np.array(all_A_col_sums)  # (num_networks, num_presentations)
        vmax_col = max(abs(A_col_sums_matrix.min()), abs(A_col_sums_matrix.max()))
        if vmax_col == 0:
            vmax_col = 1
        im_cols = axes_coeffs[0].imshow(A_col_sums_matrix, cmap='RdBu_r', vmin=-vmax_col, vmax=vmax_col, aspect='auto')
        axes_coeffs[0].set_xlabel('Adjacent Pair Presentation')
        axes_coeffs[0].set_ylabel('Network')
        axes_coeffs[0].set_xticks(range(num_presentations_nm))
        axes_coeffs[0].set_xticklabels(presentation_labels_nm, rotation=45, ha='right', fontsize=8)
        axes_coeffs[0].set_yticks(range(num_networks_to_show_ll))
        axes_coeffs[0].set_yticklabels([f'Net {i}' for i in range(num_networks_to_show_ll)])
        axes_coeffs[0].set_title('A_col_sums = Σ_i A[i,j] for each presentation j\n(Total coefficient weight per adjacent pair)')
        plt.colorbar(im_cols, ax=axes_coeffs[0], label='Column Sum')

        # Add vertical lines to separate pair types
        axes_coeffs[0].axvline(x=5.5, color='black', linestyle=':', linewidth=1)  # After CD presentations
        axes_coeffs[0].axvline(x=7.5, color='red', linestyle='-', linewidth=2)    # After DE presentations
        axes_coeffs[0].axvline(x=9.5, color='black', linestyle=':', linewidth=1)  # After EF presentations

        # Panel 2: Full A_coeffs matrices for first 3 networks side by side
        num_to_show_detail = min(3, num_networks_to_show_ll)
        # Create inset axes for detailed coefficient matrices
        axes_coeffs[1].axis('off')
        for i in range(num_to_show_detail):
            left = 0.05 + i * 0.32
            ax_inset = fig_coeffs_networks.add_axes([left, 0.05, 0.28, 0.35])
            vmax_a = max(abs(all_A_coeffs[i].min()), abs(all_A_coeffs[i].max()))
            if vmax_a == 0:
                vmax_a = 1
            im_a = ax_inset.imshow(all_A_coeffs[i], cmap='RdBu_r', vmin=-vmax_a, vmax=vmax_a, aspect='equal')
            ax_inset.set_xticks(range(0, num_presentations_nm, 2))
            ax_inset.set_xticklabels([presentation_labels_nm[j] for j in range(0, num_presentations_nm, 2)], rotation=45, ha='right', fontsize=6)
            ax_inset.set_yticks(range(0, num_presentations_nm, 2))
            ax_inset.set_yticklabels([presentation_labels_nm[j] for j in range(0, num_presentations_nm, 2)], fontsize=6)
            ax_inset.set_title(f'Net {i}: A[i,j]', fontsize=10)
            plt.colorbar(im_a, ax=ax_inset, fraction=0.046)

        plt.suptitle('List Linking: Coefficient Matrices by Network\n(Compare with neuromodulator_heatmap to see correlation)', fontsize=12)
        figures["list_linking/pw_decomposition_coefficients_by_network"] = fig_coeffs_networks
        plt.close(fig_coeffs_networks)

    # --- Plastic Weight Decomposition Analysis (for first hidden layer only) ---
    # Only do this if there are extra layers
    if args.extra_layers > 0:
        # For one network (network_idx = 0), track the coefficient matrix A
        # where P = Σ_ij A[i,j] * v_i * u_j^T with v_i = W @ u_i

        network_idx = 0
        single_items = batch_items[network_idx]  # (num_items, item_size)

        # Define adjacent pairs for list linking (including linking pair)
        # List 1: (0,1), (1,2), (2,3)
        # Linking: (3,4)
        # List 2: (4,5), (5,6), (6,7)
        adjacent_pairs_ll = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7)]

        # Create all presentations (both orderings: winner first, loser first)
        all_presentations = []
        presentation_to_idx = {}
        for pair in adjacent_pairs_ll:
            winner, loser = pair  # winner < loser in rank
            # Winner first (correct choice = 0)
            all_presentations.append((winner, loser))
            presentation_to_idx[(winner, loser)] = len(all_presentations) - 1
            # Loser first (correct choice = 1)
            all_presentations.append((loser, winner))
            presentation_to_idx[(loser, winner)] = len(all_presentations) - 1

        num_presentations = len(all_presentations)  # Should be 14

        # Compute embeddings u_i for each presentation (after embedding layer + tanh)
        embeddings_u = []
        with torch.no_grad():
            for (item1_idx, item2_idx) in all_presentations:
                item1_emb = single_items[item1_idx]
                item2_emb = single_items[item2_idx]
                input_vec = np.concatenate([item1_emb, item2_emb])
                input_t = torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0).to(device)
                # Run through embedding layer + tanh
                if hasattr(model, 'embedding_layer'):
                    u = torch.tanh(model.embedding_layer(input_t))  # (1, hidden_size)
                else:
                    u = input_t
                embeddings_u.append(u.squeeze(0).cpu().numpy())

        embeddings_u = np.array(embeddings_u)  # (num_presentations, hidden_size)

        # Get innate weights W for first hidden layer
        W_first_hidden = model.extra_hidden_layers[0].weight.detach().cpu().numpy()  # (hidden_size, hidden_size)

        # Get alpha matrix for first hidden layer
        alpha_first = model.alpha_extra[0].detach().cpu().numpy()  # (hidden_size, hidden_size)

        # Get hebbian trace multiplier for first hidden layer
        m_hebb = model.hebbian_trace_multiplier_extra[0].item()

        # Compute v_i = W @ u_i for each presentation
        embeddings_v = (W_first_hidden @ embeddings_u.T).T  # (num_presentations, hidden_size)

        # Compute D_alpha_full as 3D tensor: D_alpha_full[i,j,m] = sum_l alpha[m,l] * u_i[l] * u_j[l]
        # This captures the full alpha matrix without averaging
        if alpha_first.ndim >= 2:
            hidden_size_decomp = alpha_first.shape[0]
            D_alpha_full = np.zeros((num_presentations, num_presentations, hidden_size_decomp))
            for m in range(hidden_size_decomp):
                # For each output dimension m, compute the alpha[m,:]-weighted dot products
                D_alpha_full[:, :, m] = (embeddings_u * alpha_first[m, :]) @ embeddings_u.T

            # Compute mean and std across output dimensions
            D_alpha_mean = D_alpha_full.mean(axis=2)  # (num_presentations, num_presentations)
            D_alpha_std = D_alpha_full.std(axis=2)    # (num_presentations, num_presentations)
        else:
            # Scalar alpha: D_alpha = alpha * (U @ U^T)
            D_alpha_mean = float(alpha_first) * (embeddings_u @ embeddings_u.T)
            D_alpha_std = np.zeros_like(D_alpha_mean)

        # For coefficient tracking, we still use the mean approximation
        D_alpha = D_alpha_mean

        # Initialize coefficient matrix A = 0
        A_coeffs = np.zeros((num_presentations, num_presentations))

        # Re-run training to track coefficients
        # Reset plastic weights for tracking
        pw_track = torch.zeros(1, args.hidden_size, args.hidden_size, dtype=torch.float32).to(device)
        epw_track = [torch.zeros(1, args.hidden_size, args.hidden_size, dtype=torch.float32).to(device)
                     for _ in range(args.extra_layers)]

        # Get trials for this single network
        single_trials = trials[network_idx:network_idx+1, :, :]  # (1, num_trials, input_size)
        single_correct = correct_choices_tensor[network_idx:network_idx+1, :]  # (1, num_trials)

        for trial_idx in range(num_train_trials):
            trial_input = single_trials[:, trial_idx, :]  # (1, input_size)
            trial_correct = single_correct[:, trial_idx]  # (1,)

            # Identify which presentation this is
            # Extract the two item embeddings from the trial input
            item_size = args.item_size
            item1_emb_trial = trial_input[0, :item_size].cpu().numpy()
            item2_emb_trial = trial_input[0, item_size:2*item_size].cpu().numpy()

            # Find which items these correspond to
            item1_idx_found = None
            item2_idx_found = None
            for idx in range(num_items):
                if np.allclose(single_items[idx], item1_emb_trial, atol=1e-5):
                    item1_idx_found = idx
                if np.allclose(single_items[idx], item2_emb_trial, atol=1e-5):
                    item2_idx_found = idx

            if item1_idx_found is None or item2_idx_found is None:
                continue  # Skip if we can't identify the items

            # Check if this is an adjacent pair presentation
            presentation_key = (item1_idx_found, item2_idx_found)
            if presentation_key not in presentation_to_idx:
                continue  # Not an adjacent pair, skip

            k = presentation_to_idx[presentation_key]  # Embedding index

            # Run forward pass to get neuromodulator
            with torch.no_grad():
                output_track = model(trial_input, pw_track, trial_correct,
                                    extra_plastic_weights=epw_track, store_embeddings=False)

            # Get neuromodulator for first extra layer
            # The neuromodulator output has shape (batch, num_neuromodulators)
            # For use_extra_neuromodulator=True, first entries are for extra layers
            nm_output = output_track.neuromodulator.squeeze()
            use_extra_nm = getattr(args, 'use_extra_neuromodulator', False)
            if use_extra_nm and args.extra_layers > 0:
                # First neuromodulator is for first extra layer
                eta_t = nm_output[0].item() if nm_output.dim() > 0 else nm_output.item()
            else:
                eta_t = nm_output.item() if nm_output.dim() == 0 else nm_output[0].item()

            # Update coefficient matrix with full update rule:
            # A[:,k] += eta_t * m * (e_k + A @ D_alpha[:,k])
            # where m = hebbian_trace_multiplier, D_alpha accounts for alpha modulation
            e_k = np.zeros(num_presentations)
            e_k[k] = 1.0
            # Scale by ~0.9 to approximate tanh compression on outer products in [-1,1]
            tanh_scale = 0.9
            A_coeffs[:, k] += eta_t * m_hebb * tanh_scale * (e_k + A_coeffs @ D_alpha[:, k])

            # Update plastic weights for next iteration
            pw_track = output_track.plastic_weights
            epw_track = output_track.extra_plastic_weights

        # Create labels for presentations
        item_labels_decomp = [chr(ord('A') + i) for i in range(num_items)]
        presentation_labels = []
        for (item1_idx, item2_idx) in all_presentations:
            presentation_labels.append(f"{item_labels_decomp[item1_idx]}{item_labels_decomp[item2_idx]}")

        # Plot 1: Coefficient matrix A
        fig_coeffs, ax_coeffs = plt.subplots(figsize=(10, 8), dpi=150)
        vmax_coeffs = max(abs(A_coeffs.min()), abs(A_coeffs.max()))
        if vmax_coeffs == 0:
            vmax_coeffs = 1
        im_coeffs = ax_coeffs.imshow(A_coeffs, cmap='RdBu_r', vmin=-vmax_coeffs, vmax=vmax_coeffs, aspect='equal')
        ax_coeffs.set_xticks(range(num_presentations))
        ax_coeffs.set_xticklabels(presentation_labels, rotation=45, ha='right', fontsize=8)
        ax_coeffs.set_yticks(range(num_presentations))
        ax_coeffs.set_yticklabels(presentation_labels, fontsize=8)
        ax_coeffs.set_xlabel('j (u_j column)')
        ax_coeffs.set_ylabel('i (v_i row)')
        ax_coeffs.set_title('Coefficient Matrix A\n(P ≈ Σ_ij A[i,j] · v_i · u_j^T)')
        plt.colorbar(im_coeffs, ax=ax_coeffs, label='Coefficient value')
        plt.tight_layout()
        figures["list_linking/pw_decomposition_coefficients"] = fig_coeffs
        plt.close(fig_coeffs)

        # Plot 2: D_alpha matrix - Mean and Std side by side
        fig_dots, axes_dots = plt.subplots(1, 2, figsize=(16, 7), dpi=150)

        # Left: Mean D_alpha (with symmetric colormap around 0)
        vmax_dalpha = max(abs(D_alpha_mean.min()), abs(D_alpha_mean.max()))
        if vmax_dalpha == 0:
            vmax_dalpha = 1
        im_mean = axes_dots[0].imshow(D_alpha_mean, cmap='RdBu_r', vmin=-vmax_dalpha, vmax=vmax_dalpha, aspect='equal')
        axes_dots[0].set_xticks(range(num_presentations))
        axes_dots[0].set_xticklabels(presentation_labels, rotation=45, ha='right', fontsize=8)
        axes_dots[0].set_yticks(range(num_presentations))
        axes_dots[0].set_yticklabels(presentation_labels, fontsize=8)
        axes_dots[0].set_xlabel('j')
        axes_dots[0].set_ylabel('i')
        axes_dots[0].set_title('D_alpha Mean\n(averaged over output dimensions)')
        plt.colorbar(im_mean, ax=axes_dots[0], label='Mean alpha-weighted dot product')

        # Right: Std D_alpha
        im_std = axes_dots[1].imshow(D_alpha_std, cmap='plasma', aspect='equal')
        axes_dots[1].set_xticks(range(num_presentations))
        axes_dots[1].set_xticklabels(presentation_labels, rotation=45, ha='right', fontsize=8)
        axes_dots[1].set_yticks(range(num_presentations))
        axes_dots[1].set_yticklabels(presentation_labels, fontsize=8)
        axes_dots[1].set_xlabel('j')
        axes_dots[1].set_ylabel('i')
        axes_dots[1].set_title('D_alpha Std\n(variation across output dimensions)')
        plt.colorbar(im_std, ax=axes_dots[1], label='Std of alpha-weighted dot product')

        plt.suptitle('D_alpha Matrix: Mean vs Std across Output Dimensions', fontsize=12)
        plt.tight_layout()
        figures["list_linking/pw_decomposition_d_alpha"] = fig_dots
        plt.close(fig_dots)

        # Plot 3: Reconstruction error
        # Reconstruct P from coefficients and compare to actual P
        P_actual = frozen_extra_plastic_weights[0][network_idx].detach().cpu().numpy()
        P_reconstructed = np.zeros_like(P_actual)
        for i in range(num_presentations):
            for j in range(num_presentations):
                P_reconstructed += A_coeffs[i, j] * np.outer(embeddings_v[i], embeddings_u[j])

        reconstruction_error = np.linalg.norm(P_actual - P_reconstructed) / np.linalg.norm(P_actual)

        fig_recon, axes_recon = plt.subplots(1, 3, figsize=(18, 5), dpi=150)

        vmax_p = max(abs(P_actual.min()), abs(P_actual.max()), abs(P_reconstructed.min()), abs(P_reconstructed.max()))
        if vmax_p == 0:
            vmax_p = 1

        im0 = axes_recon[0].imshow(P_actual, cmap='RdBu_r', vmin=-vmax_p, vmax=vmax_p, aspect='equal')
        axes_recon[0].set_title('Actual P (first hidden layer)')
        plt.colorbar(im0, ax=axes_recon[0])

        im1 = axes_recon[1].imshow(P_reconstructed, cmap='RdBu_r', vmin=-vmax_p, vmax=vmax_p, aspect='equal')
        axes_recon[1].set_title('Reconstructed P from coefficients')
        plt.colorbar(im1, ax=axes_recon[1])

        P_diff = P_actual - P_reconstructed
        vmax_diff = max(abs(P_diff.min()), abs(P_diff.max()))
        if vmax_diff == 0:
            vmax_diff = 1
        im2 = axes_recon[2].imshow(P_diff, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff, aspect='equal')
        axes_recon[2].set_title(f'Difference (error = {reconstruction_error:.4f})')
        plt.colorbar(im2, ax=axes_recon[2])

        plt.suptitle('Plastic Weight Decomposition Verification', fontsize=12)
        plt.tight_layout()
        figures["list_linking/pw_decomposition_reconstruction"] = fig_recon
        plt.close(fig_recon)

        # Plot 4: Residual coefficients in the expanded basis {e_m ⊗ u_j}
        # Compute residual: R = P_actual - P_reconstructed
        # Then express R in the {e_m ⊗ u_j} basis: R[m,:] = sum_j R_coeff[m,j] * u_j^T
        # R_coeff[m,j] = R[m,:] projected onto u_j direction
        U_matrix = embeddings_u.T  # (hidden_size, num_presentations)
        U_pinv = np.linalg.pinv(U_matrix)  # (num_presentations, hidden_size)

        # R_coeff[m,j] = (R @ U @ (U^T U)^{-1})[m,j] but simpler with pinv of U^T
        # R[m,:] @ U_pinv.T gives the coefficients for row m
        R_coeff = P_diff @ U_pinv.T  # (hidden_size, num_presentations)

        # Compute reconstruction with residuals added
        P_with_residual = P_reconstructed + R_coeff @ U_matrix.T
        reconstruction_error_with_residual = np.linalg.norm(P_actual - P_with_residual) / np.linalg.norm(P_actual)

        # Plot residual coefficients heatmap
        fig_resid, ax_resid = plt.subplots(figsize=(12, 8), dpi=150)
        vmax_resid = max(abs(R_coeff.min()), abs(R_coeff.max()))
        if vmax_resid == 0:
            vmax_resid = 1
        im_resid = ax_resid.imshow(R_coeff, cmap='RdBu_r', vmin=-vmax_resid, vmax=vmax_resid, aspect='auto')
        ax_resid.set_xticks(range(num_presentations))
        ax_resid.set_xticklabels(presentation_labels, rotation=45, ha='right', fontsize=8)
        ax_resid.set_xlabel('j (u_j basis vector)')
        ax_resid.set_ylabel('m (output dimension)')
        ax_resid.set_title('Residual Coefficients R[m,j]\n(what the v_i basis cannot capture, in the e_m ⊗ u_j basis)')
        plt.colorbar(im_resid, ax=ax_resid, label='Residual coefficient')
        plt.tight_layout()
        figures["list_linking/pw_decomposition_residual"] = fig_resid
        plt.close(fig_resid)

        # Plot reconstruction comparison: without vs with residuals
        fig_recon_compare, axes_rc = plt.subplots(1, 3, figsize=(18, 5), dpi=150)

        im_rc0 = axes_rc[0].imshow(P_actual, cmap='RdBu_r', vmin=-vmax_p, vmax=vmax_p, aspect='equal')
        axes_rc[0].set_title('Actual P')
        plt.colorbar(im_rc0, ax=axes_rc[0])

        im_rc1 = axes_rc[1].imshow(P_reconstructed, cmap='RdBu_r', vmin=-vmax_p, vmax=vmax_p, aspect='equal')
        axes_rc[1].set_title(f'v_i ⊗ u_j basis only\n(error = {reconstruction_error:.4f})')
        plt.colorbar(im_rc1, ax=axes_rc[1])

        im_rc2 = axes_rc[2].imshow(P_with_residual, cmap='RdBu_r', vmin=-vmax_p, vmax=vmax_p, aspect='equal')
        axes_rc[2].set_title(f'With e_m ⊗ u_j residuals\n(error = {reconstruction_error_with_residual:.4f})')
        plt.colorbar(im_rc2, ax=axes_rc[2])

        plt.suptitle('Reconstruction Comparison: Scalar vs Expanded Basis', fontsize=12)
        plt.tight_layout()
        figures["list_linking/pw_decomposition_reconstruction_comparison"] = fig_recon_compare
        plt.close(fig_recon_compare)

        # Also plot the norm of residual per output dimension and per u_j
        fig_resid_summary, axes_resid_summary = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

        # Left: Residual norm per output dimension m
        resid_norm_per_m = np.linalg.norm(R_coeff, axis=1)  # (hidden_size,)
        axes_resid_summary[0].bar(range(len(resid_norm_per_m)), resid_norm_per_m, color='steelblue', alpha=0.7)
        axes_resid_summary[0].set_xlabel('Output dimension m')
        axes_resid_summary[0].set_ylabel('||R[m,:]||')
        axes_resid_summary[0].set_title('Residual norm per output dimension')

        # Right: Residual norm per u_j basis vector
        resid_norm_per_j = np.linalg.norm(R_coeff, axis=0)  # (num_presentations,)
        axes_resid_summary[1].bar(range(num_presentations), resid_norm_per_j, color='darkorange', alpha=0.7)
        axes_resid_summary[1].set_xticks(range(num_presentations))
        axes_resid_summary[1].set_xticklabels(presentation_labels, rotation=45, ha='right', fontsize=8)
        axes_resid_summary[1].set_xlabel('j (u_j basis vector)')
        axes_resid_summary[1].set_ylabel('||R[:,j]||')
        axes_resid_summary[1].set_title('Residual norm per u_j basis vector')

        plt.suptitle('Residual Analysis: Where does the v_i basis fail?', fontsize=12)
        plt.tight_layout()
        figures["list_linking/pw_decomposition_residual_summary"] = fig_resid_summary
        plt.close(fig_resid_summary)

        # ===================================================================================
        # LAYER 2 (Final Layer) Decomposition Analysis for List Linking
        # P₂ = Σᵢⱼ B[i,j] · ṽᵢ ⊗ ũⱼ where ũⱼ = layer-1 activations, ṽᵢ = W₂ @ ũᵢ
        # ===================================================================================

        # Compute ũⱼ = layer-1 activations for each presentation
        embeddings_u_tilde = []
        with torch.no_grad():
            pw_layer1_final = frozen_extra_plastic_weights[0][network_idx:network_idx+1]

            for (item1_idx, item2_idx) in all_presentations:
                item1_emb = single_items[item1_idx]
                item2_emb = single_items[item2_idx]
                input_vec = np.concatenate([item1_emb, item2_emb])
                input_t = torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0).to(device)

                # Layer 0: embedding layer
                if hasattr(model, 'embedding_layer'):
                    u = torch.tanh(model.embedding_layer(input_t))
                else:
                    u = input_t

                # Layer 1: first extra hidden layer with plastic weights
                alpha_layer1 = model.alpha_extra[0]
                W_layer1 = model.extra_hidden_layers[0].weight
                b_layer1 = model.extra_hidden_layers[0].bias
                # Include bias term: h1 = tanh(W @ u + bias + (α * P1) @ u)
                innate = W_layer1 @ u.T
                if b_layer1 is not None:
                    innate = innate + b_layer1.unsqueeze(1)
                h1 = torch.tanh(innate + (alpha_layer1 * pw_layer1_final.squeeze(0)) @ u.T)
                h1 = h1.T

                embeddings_u_tilde.append(h1.squeeze(0).cpu().numpy())

        embeddings_u_tilde = np.array(embeddings_u_tilde)

        # Get layer-2 (final layer) parameters
        W_layer2 = model.fc2.weight.detach().cpu().numpy()
        alpha_layer2 = model.alpha.detach().cpu().numpy()
        m_hebb_layer2 = model.hebbian_trace_multiplier.item()

        # Compute ṽᵢ = W₂ @ ũᵢ
        embeddings_v_tilde = (W_layer2 @ embeddings_u_tilde.T).T

        # Compute D_alpha_full for layer 2
        if alpha_layer2.ndim >= 2:
            hidden_size_decomp_l2 = alpha_layer2.shape[0]
            D_alpha_full_layer2 = np.zeros((num_presentations, num_presentations, hidden_size_decomp_l2))
            for m in range(hidden_size_decomp_l2):
                D_alpha_full_layer2[:, :, m] = (embeddings_u_tilde * alpha_layer2[m, :]) @ embeddings_u_tilde.T

            D_alpha_mean_layer2 = D_alpha_full_layer2.mean(axis=2)
            D_alpha_std_layer2 = D_alpha_full_layer2.std(axis=2)
        else:
            # Scalar alpha: D_alpha = alpha * (U_tilde @ U_tilde^T)
            D_alpha_mean_layer2 = float(alpha_layer2) * (embeddings_u_tilde @ embeddings_u_tilde.T)
            D_alpha_std_layer2 = np.zeros_like(D_alpha_mean_layer2)
        D_alpha_layer2 = D_alpha_mean_layer2

        # Initialize coefficient matrix B = 0 for layer 2
        B_coeffs = np.zeros((num_presentations, num_presentations))

        # Re-run training to track layer-2 coefficients with TIME-VARYING ũ basis
        pw_track_l2 = torch.zeros(1, args.hidden_size, args.hidden_size, dtype=torch.float32).to(device)
        epw_track_l2 = [torch.zeros(1, args.hidden_size, args.hidden_size, dtype=torch.float32).to(device)
                        for _ in range(args.extra_layers)]

        # Also accumulate the actual reconstruction using correct vectors at each step
        P_reconstructed_l2_timevarying = np.zeros((args.hidden_size, args.hidden_size))
        # Also track reconstruction WITH plastic contribution and tanh
        P_reconstructed_l2_with_plastic = np.zeros((args.hidden_size, args.hidden_size))

        for trial_idx in range(num_train_trials):
            trial_input = single_trials[:, trial_idx, :]
            trial_correct = single_correct[:, trial_idx]

            item_size = args.item_size
            item1_emb_trial = trial_input[0, :item_size].cpu().numpy()
            item2_emb_trial = trial_input[0, item_size:2*item_size].cpu().numpy()

            item1_idx_found = None
            item2_idx_found = None
            for idx in range(num_items):
                if np.allclose(single_items[idx], item1_emb_trial, atol=1e-5):
                    item1_idx_found = idx
                if np.allclose(single_items[idx], item2_emb_trial, atol=1e-5):
                    item2_idx_found = idx

            if item1_idx_found is None or item2_idx_found is None:
                continue

            presentation_key = (item1_idx_found, item2_idx_found)
            if presentation_key not in presentation_to_idx:
                continue

            k = presentation_to_idx[presentation_key]

            # Compute time-varying ũⱼ⁽ᵗ⁾ for ALL presentations using current P₁
            with torch.no_grad():
                P1_current = epw_track_l2[0].squeeze(0)
                embeddings_u_tilde_t = []
                for (item1_idx, item2_idx) in all_presentations:
                    item1_emb = single_items[item1_idx]
                    item2_emb = single_items[item2_idx]
                    input_vec = np.concatenate([item1_emb, item2_emb])
                    input_t = torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0).to(device)
                    if hasattr(model, 'embedding_layer'):
                        u = torch.tanh(model.embedding_layer(input_t))
                    else:
                        u = input_t
                    alpha_layer1 = model.alpha_extra[0]
                    W_layer1 = model.extra_hidden_layers[0].weight
                    b_layer1 = model.extra_hidden_layers[0].bias
                    # Include bias term: h1 = tanh(W @ u + bias + (α * P1) @ u)
                    innate = W_layer1 @ u.T
                    if b_layer1 is not None:
                        innate = innate + b_layer1.unsqueeze(1)
                    h1 = torch.tanh(innate + (alpha_layer1 * P1_current) @ u.T)
                    embeddings_u_tilde_t.append(h1.T.squeeze(0).cpu().numpy())
                embeddings_u_tilde_t = np.array(embeddings_u_tilde_t)

            # Compute time-varying D_alpha using current ũ basis
            if alpha_layer2.ndim >= 2:
                D_alpha_t = (embeddings_u_tilde_t * alpha_layer2.mean(axis=0)) @ embeddings_u_tilde_t.T
            else:
                D_alpha_t = float(alpha_layer2) * (embeddings_u_tilde_t @ embeddings_u_tilde_t.T)

            with torch.no_grad():
                output_track_l2 = model(trial_input, pw_track_l2, trial_correct,
                                        extra_plastic_weights=epw_track_l2, store_embeddings=False)

            nm_output = output_track_l2.neuromodulator.squeeze()
            if nm_output.dim() == 0:
                eta_t_layer2 = nm_output.item()
            else:
                eta_t_layer2 = nm_output[-1].item() if len(nm_output) > 1 else nm_output.item()

            # Update coefficient matrix B for layer 2 using TIME-VARYING D_alpha
            e_k = np.zeros(num_presentations)
            e_k[k] = 1.0
            tanh_scale = 0.9
            B_coeffs[:, k] += eta_t_layer2 * m_hebb_layer2 * tanh_scale * (e_k + B_coeffs @ D_alpha_t[:, k])

            # Get current ũₖ and compute v contributions
            u_tilde_k_t = embeddings_u_tilde_t[k]

            # Innate contribution: W2 @ ũ
            v_innate = W_layer2 @ u_tilde_k_t

            # Plastic contribution: (α * P2) @ ũ - using current P2 BEFORE update
            P2_current = pw_track_l2.squeeze(0).cpu().numpy()
            v_plastic = (alpha_layer2 * P2_current) @ u_tilde_k_t

            # Full v = innate + plastic
            v_full = v_innate + v_plastic

            # OLD reconstruction (without plastic, without tanh on outer)
            P_reconstructed_l2_timevarying += eta_t_layer2 * m_hebb_layer2 * np.outer(v_innate, u_tilde_k_t)

            # NEW reconstruction (with plastic contribution AND tanh on outer product)
            outer_product = np.outer(v_full, u_tilde_k_t)
            P_reconstructed_l2_with_plastic += eta_t_layer2 * np.tanh(outer_product) * m_hebb_layer2

            pw_track_l2 = output_track_l2.plastic_weights
            epw_track_l2 = output_track_l2.extra_plastic_weights

        # --- Layer 2 Plots for List Linking ---

        # Plot 1: Coefficient matrix B for layer 2
        fig_coeffs_l2, ax_coeffs_l2 = plt.subplots(figsize=(10, 8), dpi=150)
        vmax_coeffs_l2 = max(abs(B_coeffs.min()), abs(B_coeffs.max()))
        if vmax_coeffs_l2 == 0:
            vmax_coeffs_l2 = 1
        im_coeffs_l2 = ax_coeffs_l2.imshow(B_coeffs, cmap='RdBu_r', vmin=-vmax_coeffs_l2, vmax=vmax_coeffs_l2, aspect='equal')
        ax_coeffs_l2.set_xticks(range(num_presentations))
        ax_coeffs_l2.set_xticklabels(presentation_labels, rotation=45, ha='right', fontsize=8)
        ax_coeffs_l2.set_yticks(range(num_presentations))
        ax_coeffs_l2.set_yticklabels(presentation_labels, fontsize=8)
        ax_coeffs_l2.set_xlabel('j (ũ_j column)')
        ax_coeffs_l2.set_ylabel('i (ṽ_i row)')
        ax_coeffs_l2.set_title('Coefficient Matrix B - Layer 2 (Final)\n(P₂ ≈ Σᵢⱼ B[i,j] · ṽᵢ · ũⱼᵀ)')
        plt.colorbar(im_coeffs_l2, ax=ax_coeffs_l2, label='Coefficient value')
        plt.tight_layout()
        figures["list_linking/pw_decomposition_coefficients_layer2"] = fig_coeffs_l2
        plt.close(fig_coeffs_l2)

        # Plot 2: D_alpha mean and std for layer 2
        fig_dots_l2, axes_dots_l2 = plt.subplots(1, 2, figsize=(16, 7), dpi=150)

        vmax_dalpha_l2 = max(abs(D_alpha_mean_layer2.min()), abs(D_alpha_mean_layer2.max()))
        if vmax_dalpha_l2 == 0:
            vmax_dalpha_l2 = 1
        im_mean_l2 = axes_dots_l2[0].imshow(D_alpha_mean_layer2, cmap='RdBu_r', vmin=-vmax_dalpha_l2, vmax=vmax_dalpha_l2, aspect='equal')
        axes_dots_l2[0].set_xticks(range(num_presentations))
        axes_dots_l2[0].set_xticklabels(presentation_labels, rotation=45, ha='right', fontsize=8)
        axes_dots_l2[0].set_yticks(range(num_presentations))
        axes_dots_l2[0].set_yticklabels(presentation_labels, fontsize=8)
        axes_dots_l2[0].set_xlabel('j')
        axes_dots_l2[0].set_ylabel('i')
        axes_dots_l2[0].set_title('Mean D_α²[i,j] across output dimensions\n(D_α²[i,j,m] = Σₗ α₂[m,l] · ũᵢ[l] · ũⱼ[l])')
        plt.colorbar(im_mean_l2, ax=axes_dots_l2[0], label='Mean alpha-weighted dot product')

        im_std_l2 = axes_dots_l2[1].imshow(D_alpha_std_layer2, cmap='plasma', aspect='equal')
        axes_dots_l2[1].set_xticks(range(num_presentations))
        axes_dots_l2[1].set_xticklabels(presentation_labels, rotation=45, ha='right', fontsize=8)
        axes_dots_l2[1].set_yticks(range(num_presentations))
        axes_dots_l2[1].set_yticklabels(presentation_labels, fontsize=8)
        axes_dots_l2[1].set_xlabel('j')
        axes_dots_l2[1].set_ylabel('i')
        axes_dots_l2[1].set_title('Std D_α²[i,j] across output dimensions\n(variation in alpha weighting)')
        plt.colorbar(im_std_l2, ax=axes_dots_l2[1], label='Std of alpha-weighted dot product')

        plt.suptitle('D_alpha Matrix Analysis - Layer 2 (using ũ embeddings)', fontsize=12)
        plt.tight_layout()
        figures["list_linking/pw_decomposition_d_alpha_layer2"] = fig_dots_l2
        plt.close(fig_dots_l2)

        # Plot 3: Reconstruction error for layer 2 (comparing fixed basis vs time-varying)
        P_actual_l2 = frozen_plastic_weights[network_idx].detach().cpu().numpy()

        # Fixed basis reconstruction (using final ũ)
        P_reconstructed_l2_fixed = np.zeros_like(P_actual_l2)
        for i in range(num_presentations):
            for j in range(num_presentations):
                P_reconstructed_l2_fixed += B_coeffs[i, j] * np.outer(embeddings_v_tilde[i], embeddings_u_tilde[j])

        reconstruction_error_l2_fixed = np.linalg.norm(P_actual_l2 - P_reconstructed_l2_fixed) / np.linalg.norm(P_actual_l2) if np.linalg.norm(P_actual_l2) > 0 else 0.0

        # Time-varying reconstruction error (innate only, no tanh)
        reconstruction_error_l2_timevarying = np.linalg.norm(P_actual_l2 - P_reconstructed_l2_timevarying) / np.linalg.norm(P_actual_l2) if np.linalg.norm(P_actual_l2) > 0 else 0.0

        # NEW: Time-varying with plastic contribution AND tanh
        reconstruction_error_l2_with_plastic = np.linalg.norm(P_actual_l2 - P_reconstructed_l2_with_plastic) / np.linalg.norm(P_actual_l2) if np.linalg.norm(P_actual_l2) > 0 else 0.0

        # DEBUG: Log reconstruction errors for comparison
        logger.info("=" * 60)
        logger.info("LAYER 2: RECONSTRUCTION ERROR COMPARISON")
        logger.info("=" * 60)
        logger.info(f"  Fixed basis (final ũ, B coeffs):     error = {reconstruction_error_l2_fixed:.4f}")
        logger.info(f"  Time-varying (W2@ũ only, no tanh):   error = {reconstruction_error_l2_timevarying:.4f}")
        logger.info(f"  Time-varying WITH plastic + tanh:    error = {reconstruction_error_l2_with_plastic:.4f}")
        logger.info("=" * 60)

        # Create 2x3 plot to show all reconstructions
        fig_recon_l2, axes_recon_l2 = plt.subplots(2, 3, figsize=(18, 12), dpi=150)

        vmax_p_l2 = max(abs(P_actual_l2.min()), abs(P_actual_l2.max()),
                        abs(P_reconstructed_l2_fixed.min()), abs(P_reconstructed_l2_fixed.max()),
                        abs(P_reconstructed_l2_timevarying.min()), abs(P_reconstructed_l2_timevarying.max()),
                        abs(P_reconstructed_l2_with_plastic.min()), abs(P_reconstructed_l2_with_plastic.max()))
        if vmax_p_l2 == 0:
            vmax_p_l2 = 1

        # Row 1: Actual, Fixed basis, Time-varying (innate only)
        im0_l2 = axes_recon_l2[0, 0].imshow(P_actual_l2, cmap='RdBu_r', vmin=-vmax_p_l2, vmax=vmax_p_l2, aspect='equal')
        axes_recon_l2[0, 0].set_title('Actual P₂ (final layer)')
        plt.colorbar(im0_l2, ax=axes_recon_l2[0, 0])

        im1_l2 = axes_recon_l2[0, 1].imshow(P_reconstructed_l2_fixed, cmap='RdBu_r', vmin=-vmax_p_l2, vmax=vmax_p_l2, aspect='equal')
        axes_recon_l2[0, 1].set_title(f'Fixed basis (final ũ)\n(error = {reconstruction_error_l2_fixed:.4f})')
        plt.colorbar(im1_l2, ax=axes_recon_l2[0, 1])

        im2_l2 = axes_recon_l2[0, 2].imshow(P_reconstructed_l2_timevarying, cmap='RdBu_r', vmin=-vmax_p_l2, vmax=vmax_p_l2, aspect='equal')
        axes_recon_l2[0, 2].set_title(f'Time-varying (W2@ũ only)\n(error = {reconstruction_error_l2_timevarying:.4f})')
        plt.colorbar(im2_l2, ax=axes_recon_l2[0, 2])

        # Row 2: With plastic+tanh, and residuals
        im3_l2 = axes_recon_l2[1, 0].imshow(P_reconstructed_l2_with_plastic, cmap='RdBu_r', vmin=-vmax_p_l2, vmax=vmax_p_l2, aspect='equal')
        axes_recon_l2[1, 0].set_title(f'WITH plastic + tanh\n(error = {reconstruction_error_l2_with_plastic:.4f})')
        plt.colorbar(im3_l2, ax=axes_recon_l2[1, 0])

        P_diff_l2 = P_actual_l2 - P_reconstructed_l2_with_plastic
        vmax_diff_l2 = max(abs(P_diff_l2.min()), abs(P_diff_l2.max()))
        if vmax_diff_l2 == 0:
            vmax_diff_l2 = 1

        im4_l2 = axes_recon_l2[1, 1].imshow(P_diff_l2, cmap='RdBu_r', vmin=-vmax_diff_l2, vmax=vmax_diff_l2, aspect='equal')
        axes_recon_l2[1, 1].set_title(f'Residual (actual - with_plastic)\n(error = {reconstruction_error_l2_with_plastic:.4f})')
        plt.colorbar(im4_l2, ax=axes_recon_l2[1, 1])

        # Show difference between timevarying and with_plastic to see impact of plastic term
        diff_methods = P_reconstructed_l2_with_plastic - P_reconstructed_l2_timevarying
        vmax_diff_methods = max(abs(diff_methods.min()), abs(diff_methods.max()))
        if vmax_diff_methods == 0:
            vmax_diff_methods = 1
        im5_l2 = axes_recon_l2[1, 2].imshow(diff_methods, cmap='RdBu_r', vmin=-vmax_diff_methods, vmax=vmax_diff_methods, aspect='equal')
        axes_recon_l2[1, 2].set_title('Difference: with_plastic - innate_only\n(effect of (αP2)@ũ + tanh)')
        plt.colorbar(im5_l2, ax=axes_recon_l2[1, 2])

        plt.suptitle('Layer 2 Reconstruction Comparison (LL)', fontsize=12)
        plt.tight_layout()
        figures["list_linking/pw_decomposition_reconstruction_layer2"] = fig_recon_l2
        plt.close(fig_recon_l2)

        # Use the best reconstruction (with plastic) for subsequent analysis
        P_reconstructed_l2 = P_reconstructed_l2_with_plastic
        reconstruction_error_l2 = reconstruction_error_l2_with_plastic

        # Plot 4: Residual coefficients for layer 2
        U_matrix_l2 = embeddings_u_tilde.T
        U_pinv_l2 = np.linalg.pinv(U_matrix_l2)
        R_coeff_l2 = P_diff_l2 @ U_pinv_l2.T

        P_with_residual_l2 = P_reconstructed_l2 + R_coeff_l2 @ U_matrix_l2.T
        reconstruction_error_with_residual_l2 = np.linalg.norm(P_actual_l2 - P_with_residual_l2) / np.linalg.norm(P_actual_l2) if np.linalg.norm(P_actual_l2) > 0 else 0.0

        fig_resid_l2, ax_resid_l2 = plt.subplots(figsize=(12, 8), dpi=150)
        vmax_resid_l2 = max(abs(R_coeff_l2.min()), abs(R_coeff_l2.max()))
        if vmax_resid_l2 == 0:
            vmax_resid_l2 = 1
        im_resid_l2 = ax_resid_l2.imshow(R_coeff_l2, cmap='RdBu_r', vmin=-vmax_resid_l2, vmax=vmax_resid_l2, aspect='auto')
        ax_resid_l2.set_xticks(range(num_presentations))
        ax_resid_l2.set_xticklabels(presentation_labels, rotation=45, ha='right', fontsize=8)
        ax_resid_l2.set_xlabel('j (ũ_j basis vector)')
        ax_resid_l2.set_ylabel('m (output dimension)')
        ax_resid_l2.set_title('Residual Coefficients R[m,j] - Layer 2\n(what the ṽ_i basis cannot capture)')
        plt.colorbar(im_resid_l2, ax=ax_resid_l2, label='Residual coefficient')
        plt.tight_layout()
        figures["list_linking/pw_decomposition_residual_layer2"] = fig_resid_l2
        plt.close(fig_resid_l2)

        # Plot 5: Reconstruction comparison for layer 2
        fig_recon_compare_l2, axes_rc_l2 = plt.subplots(1, 3, figsize=(18, 5), dpi=150)

        im_rc0_l2 = axes_rc_l2[0].imshow(P_actual_l2, cmap='RdBu_r', vmin=-vmax_p_l2, vmax=vmax_p_l2, aspect='equal')
        axes_rc_l2[0].set_title('Actual P₂')
        plt.colorbar(im_rc0_l2, ax=axes_rc_l2[0])

        im_rc1_l2 = axes_rc_l2[1].imshow(P_reconstructed_l2, cmap='RdBu_r', vmin=-vmax_p_l2, vmax=vmax_p_l2, aspect='equal')
        axes_rc_l2[1].set_title(f'ṽ_i ⊗ ũ_j basis only\n(error = {reconstruction_error_l2:.4f})')
        plt.colorbar(im_rc1_l2, ax=axes_rc_l2[1])

        im_rc2_l2 = axes_rc_l2[2].imshow(P_with_residual_l2, cmap='RdBu_r', vmin=-vmax_p_l2, vmax=vmax_p_l2, aspect='equal')
        axes_rc_l2[2].set_title(f'With e_m ⊗ ũ_j residuals\n(error = {reconstruction_error_with_residual_l2:.4f})')
        plt.colorbar(im_rc2_l2, ax=axes_rc_l2[2])

        plt.suptitle('Reconstruction Comparison: Scalar vs Expanded Basis - Layer 2 (LL)', fontsize=12)
        plt.tight_layout()
        figures["list_linking/pw_decomposition_reconstruction_comparison_layer2"] = fig_recon_compare_l2
        plt.close(fig_recon_compare_l2)

        # Plot 6: Residual norm summary for layer 2
        fig_resid_summary_l2, axes_resid_summary_l2 = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

        resid_norm_per_m_l2 = np.linalg.norm(R_coeff_l2, axis=1)
        axes_resid_summary_l2[0].bar(range(len(resid_norm_per_m_l2)), resid_norm_per_m_l2, color='steelblue', alpha=0.7)
        axes_resid_summary_l2[0].set_xlabel('Output dimension m')
        axes_resid_summary_l2[0].set_ylabel('||R[m,:]||')
        axes_resid_summary_l2[0].set_title('Residual norm per output dimension')

        resid_norm_per_j_l2 = np.linalg.norm(R_coeff_l2, axis=0)
        axes_resid_summary_l2[1].bar(range(num_presentations), resid_norm_per_j_l2, color='darkorange', alpha=0.7)
        axes_resid_summary_l2[1].set_xticks(range(num_presentations))
        axes_resid_summary_l2[1].set_xticklabels(presentation_labels, rotation=45, ha='right', fontsize=8)
        axes_resid_summary_l2[1].set_xlabel('j (ũ_j basis vector)')
        axes_resid_summary_l2[1].set_ylabel('||R[:,j]||')
        axes_resid_summary_l2[1].set_title('Residual norm per ũ_j basis vector')

        plt.suptitle('Residual Analysis - Layer 2: Where does the ṽ_i basis fail?', fontsize=12)
        plt.tight_layout()
        figures["list_linking/pw_decomposition_residual_summary_layer2"] = fig_resid_summary_l2
        plt.close(fig_resid_summary_l2)

        # =====================================================================
        # ATTENTION ANALYSIS: How non-adjacent pairs attend to adjacent pairs
        # For List Linking, we have:
        # - Within-list non-adjacent pairs (e.g., AC, BD in list 1; EG, FH in list 2)
        # - Cross-list pairs (e.g., AE, AF, BF, CG, etc.)
        # =====================================================================

        # Generate all non-adjacent test pairs (symbolic distance >= 2)
        # For LL: items 0-3 are list 1, items 4-7 are list 2
        test_pairs = []
        for i in range(num_items):
            for j in range(num_items):
                if i == j:
                    continue
                # Check if this is NOT an adjacent pair
                if (i, j) not in presentation_to_idx and (j, i) not in presentation_to_idx:
                    # This is a non-adjacent pair
                    # Determine correct choice: lower index wins (correct=0 if winner is first)
                    winner = min(i, j)
                    loser = max(i, j)
                    if i == winner:
                        correct = 0  # winner first
                    else:
                        correct = 1  # loser first
                    test_pairs.append((i, j, correct))

        # Compute test pair embeddings (u_test) and Layer-1 activations (ũ_test)
        test_embeddings_u = []
        test_embeddings_u_tilde = []
        test_labels = []
        test_signed_sds = []
        test_is_cross_list = []

        with torch.no_grad():
            # Use frozen P1 for computing ũ_test
            pw_layer1_final = frozen_extra_plastic_weights[0][network_idx:network_idx+1].squeeze(0)

            for (item1_idx, item2_idx, correct) in test_pairs:
                item1_emb = single_items[item1_idx]
                item2_emb = single_items[item2_idx]
                input_vec = np.concatenate([item1_emb, item2_emb])
                input_t = torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0).to(device)

                # Layer 0: embedding layer -> u_test
                if hasattr(model, 'embedding_layer'):
                    u_test = torch.tanh(model.embedding_layer(input_t))
                else:
                    u_test = input_t
                test_embeddings_u.append(u_test.squeeze(0).cpu().numpy())

                # Layer 1: compute ũ_test = tanh(W @ u + bias + (α * P1) @ u)
                alpha_layer1 = model.alpha_extra[0]
                W_layer1 = model.extra_hidden_layers[0].weight
                b_layer1 = model.extra_hidden_layers[0].bias
                innate = W_layer1 @ u_test.T
                if b_layer1 is not None:
                    innate = innate + b_layer1.unsqueeze(1)
                u_tilde_test = torch.tanh(innate + (alpha_layer1 * pw_layer1_final) @ u_test.T)
                test_embeddings_u_tilde.append(u_tilde_test.T.squeeze(0).cpu().numpy())

                # Labels and metadata
                test_labels.append(f"{item_labels_decomp[item1_idx]}{item_labels_decomp[item2_idx]}")
                signed_sd = item2_idx - item1_idx
                test_signed_sds.append(signed_sd)

                # Cross-list: one item from list 1 (0-3), one from list 2 (4-7)
                is_cross = (item1_idx < 4) != (item2_idx < 4)
                test_is_cross_list.append(is_cross)

        test_embeddings_u = np.array(test_embeddings_u)
        test_embeddings_u_tilde = np.array(test_embeddings_u_tilde)

        # --- Layer 1 Attention Analysis ---
        attention_matrix_L1 = embeddings_u @ test_embeddings_u.T

        # --- Layer 2 Attention Analysis ---
        attention_matrix_L2 = embeddings_u_tilde @ test_embeddings_u_tilde.T

        # =====================================================================
        # Plot 1: Layer 1 Attention Heatmap (u-space)
        # =====================================================================
        fig_attn_L1, ax_attn_L1 = plt.subplots(figsize=(16, 8), dpi=150)
        vmax_attn_L1 = max(abs(attention_matrix_L1.min()), abs(attention_matrix_L1.max()))
        if vmax_attn_L1 == 0:
            vmax_attn_L1 = 1
        im_attn_L1 = ax_attn_L1.imshow(attention_matrix_L1, cmap='RdBu_r', vmin=-vmax_attn_L1, vmax=vmax_attn_L1, aspect='auto')
        ax_attn_L1.set_yticks(range(num_presentations))
        ax_attn_L1.set_yticklabels(presentation_labels, fontsize=7)
        ax_attn_L1.set_xticks(range(len(test_labels)))
        ax_attn_L1.set_xticklabels(test_labels, fontsize=6, rotation=45, ha='right')
        ax_attn_L1.set_ylabel('Adjacent Pair Embedding (u_j)')
        ax_attn_L1.set_xlabel('Test Pair Embedding (u_test)')
        ax_attn_L1.set_title('Layer 1 Attention: u_j^T @ u_test\n(Embedding-space similarity)')
        plt.colorbar(im_attn_L1, ax=ax_attn_L1, label='Dot product')
        plt.tight_layout()
        figures["list_linking/pw_attention_layer1"] = fig_attn_L1
        plt.close(fig_attn_L1)

        # =====================================================================
        # Plot 2: Layer 2 Attention Heatmap (ũ-space)
        # =====================================================================
        fig_attn_L2, ax_attn_L2 = plt.subplots(figsize=(16, 8), dpi=150)
        vmax_attn_L2 = max(abs(attention_matrix_L2.min()), abs(attention_matrix_L2.max()))
        if vmax_attn_L2 == 0:
            vmax_attn_L2 = 1
        im_attn_L2 = ax_attn_L2.imshow(attention_matrix_L2, cmap='RdBu_r', vmin=-vmax_attn_L2, vmax=vmax_attn_L2, aspect='auto')
        ax_attn_L2.set_yticks(range(num_presentations))
        ax_attn_L2.set_yticklabels(presentation_labels, fontsize=7)
        ax_attn_L2.set_xticks(range(len(test_labels)))
        ax_attn_L2.set_xticklabels(test_labels, fontsize=6, rotation=45, ha='right')
        ax_attn_L2.set_ylabel('Adjacent Pair Activation (ũ_j)')
        ax_attn_L2.set_xlabel('Test Pair Activation (ũ_test)')
        ax_attn_L2.set_title('Layer 2 Attention: ũ_j^T @ ũ_test\n(Hidden activation similarity - after plastic layer 1)')
        plt.colorbar(im_attn_L2, ax=ax_attn_L2, label='Dot product')
        plt.tight_layout()
        figures["list_linking/pw_attention_layer2"] = fig_attn_L2
        plt.close(fig_attn_L2)

        # =====================================================================
        # Plot 3: Layer 1 vs Layer 2 Attention Comparison
        # =====================================================================
        fig_attn_compare, axes_attn_compare = plt.subplots(1, 2, figsize=(22, 8), dpi=150)

        vmax_compare = max(vmax_attn_L1, vmax_attn_L2)

        im_c1 = axes_attn_compare[0].imshow(attention_matrix_L1, cmap='RdBu_r', vmin=-vmax_compare, vmax=vmax_compare, aspect='auto')
        axes_attn_compare[0].set_yticks(range(num_presentations))
        axes_attn_compare[0].set_yticklabels(presentation_labels, fontsize=7)
        axes_attn_compare[0].set_xticks(range(len(test_labels)))
        axes_attn_compare[0].set_xticklabels(test_labels, fontsize=6, rotation=45, ha='right')
        axes_attn_compare[0].set_ylabel('Adjacent Pair')
        axes_attn_compare[0].set_xlabel('Test Pair')
        axes_attn_compare[0].set_title('Layer 1: u_j^T @ u_test\n(Before plastic layer)')
        plt.colorbar(im_c1, ax=axes_attn_compare[0], label='Dot product')

        im_c2 = axes_attn_compare[1].imshow(attention_matrix_L2, cmap='RdBu_r', vmin=-vmax_compare, vmax=vmax_compare, aspect='auto')
        axes_attn_compare[1].set_yticks(range(num_presentations))
        axes_attn_compare[1].set_yticklabels(presentation_labels, fontsize=7)
        axes_attn_compare[1].set_xticks(range(len(test_labels)))
        axes_attn_compare[1].set_xticklabels(test_labels, fontsize=6, rotation=45, ha='right')
        axes_attn_compare[1].set_ylabel('Adjacent Pair')
        axes_attn_compare[1].set_xlabel('Test Pair')
        axes_attn_compare[1].set_title('Layer 2: ũ_j^T @ ũ_test\n(After plastic layer 1)')
        plt.colorbar(im_c2, ax=axes_attn_compare[1], label='Dot product')

        plt.suptitle('List Linking: Attention Comparison - How does Layer 1 plasticity transform attention?', fontsize=12)
        plt.tight_layout()
        figures["list_linking/pw_attention_comparison"] = fig_attn_compare
        plt.close(fig_attn_compare)

        # =====================================================================
        # Plot 4: Attention by Pair Type (Cross-list vs Within-list)
        # =====================================================================
        # Separate test pairs into cross-list and within-list
        cross_list_indices = [i for i, is_cross in enumerate(test_is_cross_list) if is_cross]
        within_list_indices = [i for i, is_cross in enumerate(test_is_cross_list) if not is_cross]

        fig_by_type, axes_by_type = plt.subplots(2, 2, figsize=(18, 12), dpi=150)

        # Top row: Cross-list attention
        if cross_list_indices:
            cross_attn_L1 = attention_matrix_L1[:, cross_list_indices]
            cross_attn_L2 = attention_matrix_L2[:, cross_list_indices]
            cross_labels = [test_labels[i] for i in cross_list_indices]

            vmax_cross = max(abs(cross_attn_L1.min()), abs(cross_attn_L1.max()),
                            abs(cross_attn_L2.min()), abs(cross_attn_L2.max()))
            if vmax_cross == 0:
                vmax_cross = 1

            im_cross1 = axes_by_type[0, 0].imshow(cross_attn_L1, cmap='RdBu_r', vmin=-vmax_cross, vmax=vmax_cross, aspect='auto')
            axes_by_type[0, 0].set_yticks(range(num_presentations))
            axes_by_type[0, 0].set_yticklabels(presentation_labels, fontsize=7)
            axes_by_type[0, 0].set_xticks(range(len(cross_labels)))
            axes_by_type[0, 0].set_xticklabels(cross_labels, fontsize=6, rotation=45, ha='right')
            axes_by_type[0, 0].set_title('CROSS-LIST: Layer 1 Attention')
            plt.colorbar(im_cross1, ax=axes_by_type[0, 0])

            im_cross2 = axes_by_type[0, 1].imshow(cross_attn_L2, cmap='RdBu_r', vmin=-vmax_cross, vmax=vmax_cross, aspect='auto')
            axes_by_type[0, 1].set_yticks(range(num_presentations))
            axes_by_type[0, 1].set_yticklabels(presentation_labels, fontsize=7)
            axes_by_type[0, 1].set_xticks(range(len(cross_labels)))
            axes_by_type[0, 1].set_xticklabels(cross_labels, fontsize=6, rotation=45, ha='right')
            axes_by_type[0, 1].set_title('CROSS-LIST: Layer 2 Attention')
            plt.colorbar(im_cross2, ax=axes_by_type[0, 1])

        # Bottom row: Within-list attention
        if within_list_indices:
            within_attn_L1 = attention_matrix_L1[:, within_list_indices]
            within_attn_L2 = attention_matrix_L2[:, within_list_indices]
            within_labels = [test_labels[i] for i in within_list_indices]

            vmax_within = max(abs(within_attn_L1.min()), abs(within_attn_L1.max()),
                             abs(within_attn_L2.min()), abs(within_attn_L2.max()))
            if vmax_within == 0:
                vmax_within = 1

            im_within1 = axes_by_type[1, 0].imshow(within_attn_L1, cmap='RdBu_r', vmin=-vmax_within, vmax=vmax_within, aspect='auto')
            axes_by_type[1, 0].set_yticks(range(num_presentations))
            axes_by_type[1, 0].set_yticklabels(presentation_labels, fontsize=7)
            axes_by_type[1, 0].set_xticks(range(len(within_labels)))
            axes_by_type[1, 0].set_xticklabels(within_labels, fontsize=6, rotation=45, ha='right')
            axes_by_type[1, 0].set_title('WITHIN-LIST: Layer 1 Attention')
            plt.colorbar(im_within1, ax=axes_by_type[1, 0])

            im_within2 = axes_by_type[1, 1].imshow(within_attn_L2, cmap='RdBu_r', vmin=-vmax_within, vmax=vmax_within, aspect='auto')
            axes_by_type[1, 1].set_yticks(range(num_presentations))
            axes_by_type[1, 1].set_yticklabels(presentation_labels, fontsize=7)
            axes_by_type[1, 1].set_xticks(range(len(within_labels)))
            axes_by_type[1, 1].set_xticklabels(within_labels, fontsize=6, rotation=45, ha='right')
            axes_by_type[1, 1].set_title('WITHIN-LIST: Layer 2 Attention')
            plt.colorbar(im_within2, ax=axes_by_type[1, 1])

        plt.suptitle('List Linking: Attention by Pair Type (Cross-list vs Within-list)', fontsize=12)
        plt.tight_layout()
        figures["list_linking/pw_attention_by_pair_type"] = fig_by_type
        plt.close(fig_by_type)

        # =====================================================================
        # Plot 5: Weighted Contribution Analysis
        # =====================================================================
        A_col_sums = A_coeffs.sum(axis=0)
        B_col_sums = B_coeffs.sum(axis=0)

        weighted_contrib_L1 = A_col_sums @ attention_matrix_L1
        weighted_contrib_L2 = B_col_sums @ attention_matrix_L2

        fig_contrib, axes_contrib = plt.subplots(2, 2, figsize=(16, 12), dpi=150)

        # Color by cross-list status
        colors_cross = ['red' if is_cross else 'blue' for is_cross in test_is_cross_list]

        # Top left: Layer 1 weighted contribution
        for i, (label, contrib, color) in enumerate(zip(test_labels, weighted_contrib_L1, colors_cross)):
            axes_contrib[0, 0].scatter(i, contrib, c=color, s=50, edgecolors='black')
            axes_contrib[0, 0].annotate(label, (i, contrib), fontsize=5, ha='center', va='bottom')
        axes_contrib[0, 0].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        axes_contrib[0, 0].set_xlabel('Test pair index')
        axes_contrib[0, 0].set_ylabel('Weighted contribution')
        axes_contrib[0, 0].set_title('Layer 1: Coefficient-weighted attention\n(red=cross-list, blue=within-list)')

        # Top right: Layer 2 weighted contribution
        for i, (label, contrib, color) in enumerate(zip(test_labels, weighted_contrib_L2, colors_cross)):
            axes_contrib[0, 1].scatter(i, contrib, c=color, s=50, edgecolors='black')
            axes_contrib[0, 1].annotate(label, (i, contrib), fontsize=5, ha='center', va='bottom')
        axes_contrib[0, 1].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        axes_contrib[0, 1].set_xlabel('Test pair index')
        axes_contrib[0, 1].set_ylabel('Weighted contribution')
        axes_contrib[0, 1].set_title('Layer 2: Coefficient-weighted attention\n(red=cross-list, blue=within-list)')

        # Bottom left: L1 vs L2 scatter
        for i, (label, is_cross) in enumerate(zip(test_labels, test_is_cross_list)):
            color = 'red' if is_cross else 'blue'
            axes_contrib[1, 0].scatter(weighted_contrib_L1[i], weighted_contrib_L2[i], c=color, s=50, edgecolors='black')
            axes_contrib[1, 0].annotate(label, (weighted_contrib_L1[i], weighted_contrib_L2[i]), fontsize=5)
        axes_contrib[1, 0].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        axes_contrib[1, 0].axvline(x=0, color='black', linestyle='--', linewidth=0.5)
        axes_contrib[1, 0].set_xlabel('Layer 1 weighted contribution')
        axes_contrib[1, 0].set_ylabel('Layer 2 weighted contribution')
        axes_contrib[1, 0].set_title('Layer 1 vs Layer 2 contributions')

        # Bottom right: Mean contribution by pair type
        cross_L1 = [weighted_contrib_L1[i] for i in cross_list_indices] if cross_list_indices else []
        cross_L2 = [weighted_contrib_L2[i] for i in cross_list_indices] if cross_list_indices else []
        within_L1 = [weighted_contrib_L1[i] for i in within_list_indices] if within_list_indices else []
        within_L2 = [weighted_contrib_L2[i] for i in within_list_indices] if within_list_indices else []

        x_pos = np.arange(2)
        width = 0.35

        L1_means = [np.mean(cross_L1) if cross_L1 else 0, np.mean(within_L1) if within_L1 else 0]
        L1_stds = [np.std(cross_L1) if cross_L1 else 0, np.std(within_L1) if within_L1 else 0]
        L2_means = [np.mean(cross_L2) if cross_L2 else 0, np.mean(within_L2) if within_L2 else 0]
        L2_stds = [np.std(cross_L2) if cross_L2 else 0, np.std(within_L2) if within_L2 else 0]

        axes_contrib[1, 1].bar(x_pos - width/2, L1_means, width, yerr=L1_stds, label='Layer 1', alpha=0.7, capsize=3)
        axes_contrib[1, 1].bar(x_pos + width/2, L2_means, width, yerr=L2_stds, label='Layer 2', alpha=0.7, capsize=3)
        axes_contrib[1, 1].set_xticks(x_pos)
        axes_contrib[1, 1].set_xticklabels(['Cross-list', 'Within-list'])
        axes_contrib[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        axes_contrib[1, 1].set_ylabel('Mean weighted contribution')
        axes_contrib[1, 1].set_title('Contribution by Pair Type')
        axes_contrib[1, 1].legend()

        plt.suptitle('List Linking: Weighted Contribution Analysis', fontsize=12)
        plt.tight_layout()
        figures["list_linking/pw_attention_weighted_contribution"] = fig_contrib
        plt.close(fig_contrib)

        # =====================================================================
        # Plot 6: Per-adjacent-pair contribution breakdown
        # =====================================================================
        individual_contrib_L1 = A_col_sums[:, np.newaxis] * attention_matrix_L1
        individual_contrib_L2 = B_col_sums[:, np.newaxis] * attention_matrix_L2

        fig_breakdown, axes_breakdown = plt.subplots(1, 2, figsize=(20, 8), dpi=150)

        vmax_breakdown = max(abs(individual_contrib_L1.min()), abs(individual_contrib_L1.max()),
                             abs(individual_contrib_L2.min()), abs(individual_contrib_L2.max()))
        if vmax_breakdown == 0:
            vmax_breakdown = 1

        im_bd1 = axes_breakdown[0].imshow(individual_contrib_L1, cmap='RdBu_r', vmin=-vmax_breakdown, vmax=vmax_breakdown, aspect='auto')
        axes_breakdown[0].set_yticks(range(num_presentations))
        axes_breakdown[0].set_yticklabels(presentation_labels, fontsize=7)
        axes_breakdown[0].set_xticks(range(len(test_labels)))
        axes_breakdown[0].set_xticklabels(test_labels, fontsize=5, rotation=45, ha='right')
        axes_breakdown[0].set_ylabel('Adjacent Pair')
        axes_breakdown[0].set_xlabel('Test Pair')
        axes_breakdown[0].set_title('Layer 1: A_col[j] * (u_j^T @ u_test)')
        plt.colorbar(im_bd1, ax=axes_breakdown[0], label='Contribution')

        im_bd2 = axes_breakdown[1].imshow(individual_contrib_L2, cmap='RdBu_r', vmin=-vmax_breakdown, vmax=vmax_breakdown, aspect='auto')
        axes_breakdown[1].set_yticks(range(num_presentations))
        axes_breakdown[1].set_yticklabels(presentation_labels, fontsize=7)
        axes_breakdown[1].set_xticks(range(len(test_labels)))
        axes_breakdown[1].set_xticklabels(test_labels, fontsize=5, rotation=45, ha='right')
        axes_breakdown[1].set_ylabel('Adjacent Pair')
        axes_breakdown[1].set_xlabel('Test Pair')
        axes_breakdown[1].set_title('Layer 2: B_col[j] * (ũ_j^T @ ũ_test)')
        plt.colorbar(im_bd2, ax=axes_breakdown[1], label='Contribution')

        plt.suptitle('List Linking: Per-Adjacent-Pair Contribution Breakdown', fontsize=12)
        plt.tight_layout()
        figures["list_linking/pw_attention_contribution_breakdown"] = fig_breakdown
        plt.close(fig_breakdown)

        # =====================================================================
        # Plot 7: Attention transformation by Layer 1 plasticity
        # =====================================================================
        attention_diff = attention_matrix_L2 - attention_matrix_L1

        fig_attn_diff, axes_attn_diff = plt.subplots(1, 3, figsize=(22, 6), dpi=150)

        vmax_attn_all = max(vmax_attn_L1, vmax_attn_L2)
        vmax_diff = max(abs(attention_diff.min()), abs(attention_diff.max()))
        if vmax_diff == 0:
            vmax_diff = 1

        im_diff1 = axes_attn_diff[0].imshow(attention_matrix_L1, cmap='RdBu_r', vmin=-vmax_attn_all, vmax=vmax_attn_all, aspect='auto')
        axes_attn_diff[0].set_yticks(range(num_presentations))
        axes_attn_diff[0].set_yticklabels(presentation_labels, fontsize=6)
        axes_attn_diff[0].set_xticks(range(len(test_labels)))
        axes_attn_diff[0].set_xticklabels(test_labels, fontsize=5, rotation=45, ha='right')
        axes_attn_diff[0].set_title('Before: u_j^T @ u_test')
        plt.colorbar(im_diff1, ax=axes_attn_diff[0])

        im_diff2 = axes_attn_diff[1].imshow(attention_matrix_L2, cmap='RdBu_r', vmin=-vmax_attn_all, vmax=vmax_attn_all, aspect='auto')
        axes_attn_diff[1].set_yticks(range(num_presentations))
        axes_attn_diff[1].set_yticklabels(presentation_labels, fontsize=6)
        axes_attn_diff[1].set_xticks(range(len(test_labels)))
        axes_attn_diff[1].set_xticklabels(test_labels, fontsize=5, rotation=45, ha='right')
        axes_attn_diff[1].set_title('After: ũ_j^T @ ũ_test')
        plt.colorbar(im_diff2, ax=axes_attn_diff[1])

        im_diff3 = axes_attn_diff[2].imshow(attention_diff, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff, aspect='auto')
        axes_attn_diff[2].set_yticks(range(num_presentations))
        axes_attn_diff[2].set_yticklabels(presentation_labels, fontsize=6)
        axes_attn_diff[2].set_xticks(range(len(test_labels)))
        axes_attn_diff[2].set_xticklabels(test_labels, fontsize=5, rotation=45, ha='right')
        axes_attn_diff[2].set_title('Difference: (ũ_j^T @ ũ_test) - (u_j^T @ u_test)')
        plt.colorbar(im_diff3, ax=axes_attn_diff[2])

        plt.suptitle('List Linking: How Layer 1 Plasticity Transforms Attention Patterns', fontsize=12)
        plt.tight_layout()
        figures["list_linking/pw_attention_transformation"] = fig_attn_diff
        plt.close(fig_attn_diff)

        # =====================================================================
        # Plot 8: Linking Pair Analysis - How cross-list pairs attend to linking pair
        # The linking pair DE (3,4) and ED (4,3) is crucial for cross-list inference
        # =====================================================================
        # Find linking pair indices in adjacent presentations
        linking_pair_indices = []
        for idx, (i, j) in enumerate(all_presentations):
            if (i == 3 and j == 4) or (i == 4 and j == 3):
                linking_pair_indices.append(idx)

        if linking_pair_indices and cross_list_indices:
            fig_linking, axes_linking = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

            # Get attention from linking pair to all cross-list test pairs
            linking_attn_L1 = attention_matrix_L1[linking_pair_indices, :][:, cross_list_indices]
            linking_attn_L2 = attention_matrix_L2[linking_pair_indices, :][:, cross_list_indices]

            linking_labels = [presentation_labels[i] for i in linking_pair_indices]
            cross_test_labels = [test_labels[i] for i in cross_list_indices]

            vmax_linking = max(abs(linking_attn_L1.min()), abs(linking_attn_L1.max()),
                              abs(linking_attn_L2.min()), abs(linking_attn_L2.max()))
            if vmax_linking == 0:
                vmax_linking = 1

            im_link1 = axes_linking[0].imshow(linking_attn_L1, cmap='RdBu_r', vmin=-vmax_linking, vmax=vmax_linking, aspect='auto')
            axes_linking[0].set_yticks(range(len(linking_labels)))
            axes_linking[0].set_yticklabels(linking_labels, fontsize=10)
            axes_linking[0].set_xticks(range(len(cross_test_labels)))
            axes_linking[0].set_xticklabels(cross_test_labels, fontsize=7, rotation=45, ha='right')
            axes_linking[0].set_ylabel('Linking Pair')
            axes_linking[0].set_xlabel('Cross-list Test Pair')
            axes_linking[0].set_title('Layer 1: Linking pair attention to cross-list pairs')
            plt.colorbar(im_link1, ax=axes_linking[0])

            im_link2 = axes_linking[1].imshow(linking_attn_L2, cmap='RdBu_r', vmin=-vmax_linking, vmax=vmax_linking, aspect='auto')
            axes_linking[1].set_yticks(range(len(linking_labels)))
            axes_linking[1].set_yticklabels(linking_labels, fontsize=10)
            axes_linking[1].set_xticks(range(len(cross_test_labels)))
            axes_linking[1].set_xticklabels(cross_test_labels, fontsize=7, rotation=45, ha='right')
            axes_linking[1].set_ylabel('Linking Pair')
            axes_linking[1].set_xlabel('Cross-list Test Pair')
            axes_linking[1].set_title('Layer 2: Linking pair attention to cross-list pairs')
            plt.colorbar(im_link2, ax=axes_linking[1])

            plt.suptitle('List Linking: Linking Pair (DE/ED) Attention to Cross-List Pairs', fontsize=12)
            plt.tight_layout()
            figures["list_linking/pw_attention_linking_pair"] = fig_linking
            plt.close(fig_linking)

        # =====================================================================
        # BATCH-AVERAGED ATTENTION ANALYSIS
        # Compute attention matrices averaged across all networks in the batch
        # =====================================================================
        num_networks = batch_items.shape[0]

        # Accumulators for attention matrices
        all_attention_L1 = []
        all_attention_L2 = []
        all_linking_attention_L1 = []
        all_linking_attention_L2 = []

        for net_idx in range(num_networks):
            net_items = batch_items[net_idx]
            net_pw_layer1 = frozen_extra_plastic_weights[0][net_idx]  # (H, H)

            # Compute embeddings for adjacent pairs
            net_embeddings_u = []
            net_embeddings_u_tilde = []
            with torch.no_grad():
                for (item1_idx, item2_idx) in all_presentations:
                    item1_emb = net_items[item1_idx]
                    item2_emb = net_items[item2_idx]
                    input_vec = np.concatenate([item1_emb, item2_emb])
                    input_t = torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0).to(device)

                    # u = embedding layer output
                    if hasattr(model, 'embedding_layer'):
                        u = torch.tanh(model.embedding_layer(input_t))
                    else:
                        u = input_t
                    net_embeddings_u.append(u.squeeze(0).cpu().numpy())

                    # ũ = layer 1 output with plastic weights
                    alpha_layer1 = model.alpha_extra[0]
                    W_layer1 = model.extra_hidden_layers[0].weight
                    b_layer1 = model.extra_hidden_layers[0].bias
                    innate = W_layer1 @ u.T
                    if b_layer1 is not None:
                        innate = innate + b_layer1.unsqueeze(1)
                    u_tilde = torch.tanh(innate + (alpha_layer1 * net_pw_layer1) @ u.T)
                    net_embeddings_u_tilde.append(u_tilde.T.squeeze(0).cpu().numpy())

            net_embeddings_u = np.array(net_embeddings_u)
            net_embeddings_u_tilde = np.array(net_embeddings_u_tilde)

            # Compute embeddings for test pairs
            net_test_embeddings_u = []
            net_test_embeddings_u_tilde = []
            with torch.no_grad():
                for (item1_idx, item2_idx, correct) in test_pairs:
                    item1_emb = net_items[item1_idx]
                    item2_emb = net_items[item2_idx]
                    input_vec = np.concatenate([item1_emb, item2_emb])
                    input_t = torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0).to(device)

                    if hasattr(model, 'embedding_layer'):
                        u_test = torch.tanh(model.embedding_layer(input_t))
                    else:
                        u_test = input_t
                    net_test_embeddings_u.append(u_test.squeeze(0).cpu().numpy())

                    alpha_layer1 = model.alpha_extra[0]
                    W_layer1 = model.extra_hidden_layers[0].weight
                    b_layer1 = model.extra_hidden_layers[0].bias
                    innate = W_layer1 @ u_test.T
                    if b_layer1 is not None:
                        innate = innate + b_layer1.unsqueeze(1)
                    u_tilde_test = torch.tanh(innate + (alpha_layer1 * net_pw_layer1) @ u_test.T)
                    net_test_embeddings_u_tilde.append(u_tilde_test.T.squeeze(0).cpu().numpy())

            net_test_embeddings_u = np.array(net_test_embeddings_u)
            net_test_embeddings_u_tilde = np.array(net_test_embeddings_u_tilde)

            # Compute attention matrices for this network
            net_attention_L1 = net_embeddings_u @ net_test_embeddings_u.T
            net_attention_L2 = net_embeddings_u_tilde @ net_test_embeddings_u_tilde.T

            all_attention_L1.append(net_attention_L1)
            all_attention_L2.append(net_attention_L2)

            # Extract linking pair attention for this network
            if linking_pair_indices and cross_list_indices:
                net_linking_L1 = net_attention_L1[linking_pair_indices, :][:, cross_list_indices]
                net_linking_L2 = net_attention_L2[linking_pair_indices, :][:, cross_list_indices]
                all_linking_attention_L1.append(net_linking_L1)
                all_linking_attention_L2.append(net_linking_L2)

        # Convert to arrays and compute statistics
        all_attention_L1 = np.array(all_attention_L1)  # (num_networks, num_adj, num_test)
        all_attention_L2 = np.array(all_attention_L2)

        mean_attention_L1 = all_attention_L1.mean(axis=0)
        std_attention_L1 = all_attention_L1.std(axis=0)
        mean_attention_L2 = all_attention_L2.mean(axis=0)
        std_attention_L2 = all_attention_L2.std(axis=0)

        # =====================================================================
        # Plot: Batch-Averaged Attention Comparison (L1 vs L2)
        # =====================================================================
        fig_avg_compare, axes_avg_compare = plt.subplots(2, 2, figsize=(22, 16), dpi=150)

        vmax_avg = max(abs(mean_attention_L1.min()), abs(mean_attention_L1.max()),
                       abs(mean_attention_L2.min()), abs(mean_attention_L2.max()))
        if vmax_avg == 0:
            vmax_avg = 1

        # Top row: Mean attention
        im_avg1 = axes_avg_compare[0, 0].imshow(mean_attention_L1, cmap='RdBu_r', vmin=-vmax_avg, vmax=vmax_avg, aspect='auto')
        axes_avg_compare[0, 0].set_yticks(range(num_presentations))
        axes_avg_compare[0, 0].set_yticklabels(presentation_labels, fontsize=7)
        axes_avg_compare[0, 0].set_xticks(range(len(test_labels)))
        axes_avg_compare[0, 0].set_xticklabels(test_labels, fontsize=5, rotation=45, ha='right')
        axes_avg_compare[0, 0].set_ylabel('Adjacent Pair')
        axes_avg_compare[0, 0].set_xlabel('Test Pair')
        axes_avg_compare[0, 0].set_title(f'Layer 1 MEAN Attention (n={num_networks} networks)')
        plt.colorbar(im_avg1, ax=axes_avg_compare[0, 0], label='Mean dot product')

        im_avg2 = axes_avg_compare[0, 1].imshow(mean_attention_L2, cmap='RdBu_r', vmin=-vmax_avg, vmax=vmax_avg, aspect='auto')
        axes_avg_compare[0, 1].set_yticks(range(num_presentations))
        axes_avg_compare[0, 1].set_yticklabels(presentation_labels, fontsize=7)
        axes_avg_compare[0, 1].set_xticks(range(len(test_labels)))
        axes_avg_compare[0, 1].set_xticklabels(test_labels, fontsize=5, rotation=45, ha='right')
        axes_avg_compare[0, 1].set_ylabel('Adjacent Pair')
        axes_avg_compare[0, 1].set_xlabel('Test Pair')
        axes_avg_compare[0, 1].set_title(f'Layer 2 MEAN Attention (n={num_networks} networks)')
        plt.colorbar(im_avg2, ax=axes_avg_compare[0, 1], label='Mean dot product')

        # Bottom row: Std attention (to see consistency)
        vmax_std = max(std_attention_L1.max(), std_attention_L2.max())
        if vmax_std == 0:
            vmax_std = 1

        im_std1 = axes_avg_compare[1, 0].imshow(std_attention_L1, cmap='viridis', vmin=0, vmax=vmax_std, aspect='auto')
        axes_avg_compare[1, 0].set_yticks(range(num_presentations))
        axes_avg_compare[1, 0].set_yticklabels(presentation_labels, fontsize=7)
        axes_avg_compare[1, 0].set_xticks(range(len(test_labels)))
        axes_avg_compare[1, 0].set_xticklabels(test_labels, fontsize=5, rotation=45, ha='right')
        axes_avg_compare[1, 0].set_ylabel('Adjacent Pair')
        axes_avg_compare[1, 0].set_xlabel('Test Pair')
        axes_avg_compare[1, 0].set_title('Layer 1 STD Attention (consistency across networks)')
        plt.colorbar(im_std1, ax=axes_avg_compare[1, 0], label='Std')

        im_std2 = axes_avg_compare[1, 1].imshow(std_attention_L2, cmap='viridis', vmin=0, vmax=vmax_std, aspect='auto')
        axes_avg_compare[1, 1].set_yticks(range(num_presentations))
        axes_avg_compare[1, 1].set_yticklabels(presentation_labels, fontsize=7)
        axes_avg_compare[1, 1].set_xticks(range(len(test_labels)))
        axes_avg_compare[1, 1].set_xticklabels(test_labels, fontsize=5, rotation=45, ha='right')
        axes_avg_compare[1, 1].set_ylabel('Adjacent Pair')
        axes_avg_compare[1, 1].set_xlabel('Test Pair')
        axes_avg_compare[1, 1].set_title('Layer 2 STD Attention (consistency across networks)')
        plt.colorbar(im_std2, ax=axes_avg_compare[1, 1], label='Std')

        plt.suptitle('List Linking: Batch-Averaged Attention (Mean and Std)', fontsize=14)
        plt.tight_layout()
        figures["list_linking/pw_attention_batch_averaged"] = fig_avg_compare
        plt.close(fig_avg_compare)

        # =====================================================================
        # Plot: Batch-Averaged Linking Pair Attention
        # =====================================================================
        if all_linking_attention_L1:
            all_linking_attention_L1 = np.array(all_linking_attention_L1)
            all_linking_attention_L2 = np.array(all_linking_attention_L2)

            mean_linking_L1 = all_linking_attention_L1.mean(axis=0)
            std_linking_L1 = all_linking_attention_L1.std(axis=0)
            mean_linking_L2 = all_linking_attention_L2.mean(axis=0)
            std_linking_L2 = all_linking_attention_L2.std(axis=0)

            fig_avg_linking, axes_avg_linking = plt.subplots(2, 2, figsize=(16, 10), dpi=150)

            linking_labels = [presentation_labels[i] for i in linking_pair_indices]
            cross_test_labels = [test_labels[i] for i in cross_list_indices]

            vmax_link = max(abs(mean_linking_L1.min()), abs(mean_linking_L1.max()),
                           abs(mean_linking_L2.min()), abs(mean_linking_L2.max()))
            if vmax_link == 0:
                vmax_link = 1

            # Top row: Mean
            im_link_avg1 = axes_avg_linking[0, 0].imshow(mean_linking_L1, cmap='RdBu_r', vmin=-vmax_link, vmax=vmax_link, aspect='auto')
            axes_avg_linking[0, 0].set_yticks(range(len(linking_labels)))
            axes_avg_linking[0, 0].set_yticklabels(linking_labels, fontsize=10)
            axes_avg_linking[0, 0].set_xticks(range(len(cross_test_labels)))
            axes_avg_linking[0, 0].set_xticklabels(cross_test_labels, fontsize=6, rotation=45, ha='right')
            axes_avg_linking[0, 0].set_ylabel('Linking Pair')
            axes_avg_linking[0, 0].set_xlabel('Cross-list Test Pair')
            axes_avg_linking[0, 0].set_title(f'Layer 1 MEAN (n={num_networks})')
            plt.colorbar(im_link_avg1, ax=axes_avg_linking[0, 0])

            im_link_avg2 = axes_avg_linking[0, 1].imshow(mean_linking_L2, cmap='RdBu_r', vmin=-vmax_link, vmax=vmax_link, aspect='auto')
            axes_avg_linking[0, 1].set_yticks(range(len(linking_labels)))
            axes_avg_linking[0, 1].set_yticklabels(linking_labels, fontsize=10)
            axes_avg_linking[0, 1].set_xticks(range(len(cross_test_labels)))
            axes_avg_linking[0, 1].set_xticklabels(cross_test_labels, fontsize=6, rotation=45, ha='right')
            axes_avg_linking[0, 1].set_ylabel('Linking Pair')
            axes_avg_linking[0, 1].set_xlabel('Cross-list Test Pair')
            axes_avg_linking[0, 1].set_title(f'Layer 2 MEAN (n={num_networks})')
            plt.colorbar(im_link_avg2, ax=axes_avg_linking[0, 1])

            # Bottom row: Std
            vmax_link_std = max(std_linking_L1.max(), std_linking_L2.max())
            if vmax_link_std == 0:
                vmax_link_std = 1

            im_link_std1 = axes_avg_linking[1, 0].imshow(std_linking_L1, cmap='viridis', vmin=0, vmax=vmax_link_std, aspect='auto')
            axes_avg_linking[1, 0].set_yticks(range(len(linking_labels)))
            axes_avg_linking[1, 0].set_yticklabels(linking_labels, fontsize=10)
            axes_avg_linking[1, 0].set_xticks(range(len(cross_test_labels)))
            axes_avg_linking[1, 0].set_xticklabels(cross_test_labels, fontsize=6, rotation=45, ha='right')
            axes_avg_linking[1, 0].set_ylabel('Linking Pair')
            axes_avg_linking[1, 0].set_xlabel('Cross-list Test Pair')
            axes_avg_linking[1, 0].set_title('Layer 1 STD')
            plt.colorbar(im_link_std1, ax=axes_avg_linking[1, 0])

            im_link_std2 = axes_avg_linking[1, 1].imshow(std_linking_L2, cmap='viridis', vmin=0, vmax=vmax_link_std, aspect='auto')
            axes_avg_linking[1, 1].set_yticks(range(len(linking_labels)))
            axes_avg_linking[1, 1].set_yticklabels(linking_labels, fontsize=10)
            axes_avg_linking[1, 1].set_xticks(range(len(cross_test_labels)))
            axes_avg_linking[1, 1].set_xticklabels(cross_test_labels, fontsize=6, rotation=45, ha='right')
            axes_avg_linking[1, 1].set_ylabel('Linking Pair')
            axes_avg_linking[1, 1].set_xlabel('Cross-list Test Pair')
            axes_avg_linking[1, 1].set_title('Layer 2 STD')
            plt.colorbar(im_link_std2, ax=axes_avg_linking[1, 1])

            plt.suptitle('List Linking: Batch-Averaged Linking Pair (DE/ED) Attention', fontsize=14)
            plt.tight_layout()
            figures["list_linking/pw_attention_linking_pair_batch_averaged"] = fig_avg_linking
            plt.close(fig_avg_linking)

    else:
        # Track final layer plastic weights when extra_layers = 0
        # For one network (network_idx = 0), track the coefficient matrix A
        # where P = Σ_ij A[i,j] * v_i * u_j^T with v_i = W @ u_i

        network_idx = 0
        single_items = batch_items[network_idx]

        # Define adjacent pairs for list linking (including linking pair)
        adjacent_pairs_ll = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7)]

        # Create all presentations (both orderings)
        all_presentations = []
        presentation_to_idx = {}
        for pair in adjacent_pairs_ll:
            winner, loser = pair
            all_presentations.append((winner, loser))
            presentation_to_idx[(winner, loser)] = len(all_presentations) - 1
            all_presentations.append((loser, winner))
            presentation_to_idx[(loser, winner)] = len(all_presentations) - 1

        num_presentations = len(all_presentations)

        # Compute embeddings u_i for each presentation
        embeddings_u = []
        with torch.no_grad():
            for (item1_idx, item2_idx) in all_presentations:
                item1_emb = single_items[item1_idx]
                item2_emb = single_items[item2_idx]
                input_vec = np.concatenate([item1_emb, item2_emb])
                input_t = torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0).to(device)
                if hasattr(model, 'embedding_layer'):
                    u = torch.tanh(model.embedding_layer(input_t))
                else:
                    u = input_t
                embeddings_u.append(u.squeeze(0).cpu().numpy())

        embeddings_u = np.array(embeddings_u)

        # Get innate weights W for final layer (fc2)
        W_final = model.fc2.weight.detach().cpu().numpy()

        # Get alpha matrix for final layer
        alpha_final_layer = model.alpha.detach().cpu().numpy()

        # Get hebbian trace multiplier for final layer
        m_hebb = model.hebbian_trace_multiplier.item()

        # Compute v_i = W @ u_i for each presentation
        embeddings_v = (W_final @ embeddings_u.T).T

        # Compute D_alpha matrix
        if alpha_final_layer.ndim >= 2:
            alpha_mean = alpha_final_layer.mean(axis=0)
        else:
            alpha_mean = float(alpha_final_layer)
        D_alpha = (embeddings_u * alpha_mean) @ embeddings_u.T

        # Initialize coefficient matrix A = 0
        A_coeffs = np.zeros((num_presentations, num_presentations))

        # Re-run training to track coefficients
        pw_track = torch.zeros(1, args.hidden_size, args.hidden_size, dtype=torch.float32).to(device)
        epw_track = []  # No extra layers

        single_trials = trials[network_idx:network_idx+1, :, :]
        single_correct = correct_choices_tensor[network_idx:network_idx+1, :]

        for trial_idx in range(num_train_trials):
            trial_input = single_trials[:, trial_idx, :]
            trial_correct = single_correct[:, trial_idx]

            item_size = args.item_size
            item1_emb_trial = trial_input[0, :item_size].cpu().numpy()
            item2_emb_trial = trial_input[0, item_size:2*item_size].cpu().numpy()

            item1_idx_found = None
            item2_idx_found = None
            for idx in range(num_items):
                if np.allclose(single_items[idx], item1_emb_trial, atol=1e-5):
                    item1_idx_found = idx
                if np.allclose(single_items[idx], item2_emb_trial, atol=1e-5):
                    item2_idx_found = idx

            if item1_idx_found is None or item2_idx_found is None:
                continue

            presentation_key = (item1_idx_found, item2_idx_found)
            if presentation_key not in presentation_to_idx:
                continue

            k = presentation_to_idx[presentation_key]

            with torch.no_grad():
                output_track = model(trial_input, pw_track, trial_correct,
                                    extra_plastic_weights=epw_track, store_embeddings=False)

            # Get neuromodulator for final layer (last value or only value)
            nm_output = output_track.neuromodulator.squeeze()
            eta_t = nm_output.item() if nm_output.dim() == 0 else nm_output[-1].item()

            # Update coefficient matrix
            e_k = np.zeros(num_presentations)
            e_k[k] = 1.0
            # Scale by ~0.9 to approximate tanh compression on outer products in [-1,1]
            tanh_scale = 0.9
            A_coeffs[:, k] += eta_t * m_hebb * tanh_scale * (e_k + A_coeffs @ D_alpha[:, k])

            pw_track = output_track.plastic_weights

        # Create labels for presentations
        item_labels_decomp = [chr(ord('A') + i) for i in range(num_items)]
        presentation_labels = [f"{item_labels_decomp[i1]}{item_labels_decomp[i2]}" for (i1, i2) in all_presentations]

        # Plot 1: Coefficient matrix A
        fig_coeffs, ax_coeffs = plt.subplots(figsize=(10, 8), dpi=150)
        vmax_coeffs = max(abs(A_coeffs.min()), abs(A_coeffs.max()))
        if vmax_coeffs == 0:
            vmax_coeffs = 1
        im_coeffs = ax_coeffs.imshow(A_coeffs, cmap='RdBu_r', vmin=-vmax_coeffs, vmax=vmax_coeffs, aspect='equal')
        ax_coeffs.set_xticks(range(num_presentations))
        ax_coeffs.set_xticklabels(presentation_labels, rotation=45, ha='right', fontsize=8)
        ax_coeffs.set_yticks(range(num_presentations))
        ax_coeffs.set_yticklabels(presentation_labels, fontsize=8)
        ax_coeffs.set_xlabel('j (u_j column)')
        ax_coeffs.set_ylabel('i (v_i row)')
        ax_coeffs.set_title('Coefficient Matrix A - Final Layer\n(P ≈ Σ_ij A[i,j] · v_i · u_j^T)')
        plt.colorbar(im_coeffs, ax=ax_coeffs, label='Coefficient value')
        plt.tight_layout()
        figures["list_linking/pw_decomposition_coefficients"] = fig_coeffs
        plt.close(fig_coeffs)

        # Plot 2: Reconstruction error
        P_actual = frozen_plastic_weights[network_idx].detach().cpu().numpy()
        P_reconstructed = np.zeros_like(P_actual)
        for i in range(num_presentations):
            for j in range(num_presentations):
                P_reconstructed += A_coeffs[i, j] * np.outer(embeddings_v[i], embeddings_u[j])

        reconstruction_error = np.linalg.norm(P_actual - P_reconstructed) / np.linalg.norm(P_actual) if np.linalg.norm(P_actual) > 0 else 0.0

        fig_recon, axes_recon = plt.subplots(1, 3, figsize=(18, 5), dpi=150)

        vmax_p = max(abs(P_actual.min()), abs(P_actual.max()), abs(P_reconstructed.min()), abs(P_reconstructed.max()))
        if vmax_p == 0:
            vmax_p = 1

        im0 = axes_recon[0].imshow(P_actual, cmap='RdBu_r', vmin=-vmax_p, vmax=vmax_p, aspect='equal')
        axes_recon[0].set_title('Actual P (final layer)')
        plt.colorbar(im0, ax=axes_recon[0])

        im1 = axes_recon[1].imshow(P_reconstructed, cmap='RdBu_r', vmin=-vmax_p, vmax=vmax_p, aspect='equal')
        axes_recon[1].set_title('Reconstructed P from coefficients')
        plt.colorbar(im1, ax=axes_recon[1])

        P_diff = P_actual - P_reconstructed
        vmax_diff = max(abs(P_diff.min()), abs(P_diff.max()))
        if vmax_diff == 0:
            vmax_diff = 1
        im2 = axes_recon[2].imshow(P_diff, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff, aspect='equal')
        axes_recon[2].set_title(f'Difference (error = {reconstruction_error:.4f})')
        plt.colorbar(im2, ax=axes_recon[2])

        plt.suptitle('Plastic Weight Decomposition Verification (LL) - Final Layer', fontsize=12)
        plt.tight_layout()
        figures["list_linking/pw_decomposition_reconstruction"] = fig_recon
        plt.close(fig_recon)

        # --- D_alpha full 3D tensor and mean/std plots ---
        # Compute D_alpha_full as 3D tensor: D_alpha_full[i,j,m] = sum_l alpha[m,l] * u_i[l] * u_j[l]
        if alpha_final_layer.ndim >= 2:
            hidden_size_decomp = alpha_final_layer.shape[0]
            D_alpha_full = np.zeros((num_presentations, num_presentations, hidden_size_decomp))
            for m in range(hidden_size_decomp):
                D_alpha_full[:, :, m] = (embeddings_u * alpha_final_layer[m, :]) @ embeddings_u.T

            D_alpha_mean = D_alpha_full.mean(axis=2)
            D_alpha_std = D_alpha_full.std(axis=2)
        else:
            D_alpha_mean = float(alpha_final_layer) * (embeddings_u @ embeddings_u.T)
            D_alpha_std = np.zeros_like(D_alpha_mean)

        # Plot D_alpha mean and std side by side
        fig_dots, axes_dots = plt.subplots(1, 2, figsize=(16, 7), dpi=150)

        # Left: Mean D_alpha (with symmetric colormap around 0)
        vmax_dalpha = max(abs(D_alpha_mean.min()), abs(D_alpha_mean.max()))
        if vmax_dalpha == 0:
            vmax_dalpha = 1
        im_mean = axes_dots[0].imshow(D_alpha_mean, cmap='RdBu_r', vmin=-vmax_dalpha, vmax=vmax_dalpha, aspect='equal')
        axes_dots[0].set_xticks(range(num_presentations))
        axes_dots[0].set_xticklabels(presentation_labels, rotation=45, ha='right', fontsize=8)
        axes_dots[0].set_yticks(range(num_presentations))
        axes_dots[0].set_yticklabels(presentation_labels, fontsize=8)
        axes_dots[0].set_xlabel('j')
        axes_dots[0].set_ylabel('i')
        axes_dots[0].set_title('Mean D_α[i,j] across output dimensions\n(D_α[i,j,m] = Σ_l α[m,l] · u_i[l] · u_j[l])')
        plt.colorbar(im_mean, ax=axes_dots[0], label='Mean alpha-weighted dot product')

        # Right: Std D_alpha
        im_std = axes_dots[1].imshow(D_alpha_std, cmap='plasma', aspect='equal')
        axes_dots[1].set_xticks(range(num_presentations))
        axes_dots[1].set_xticklabels(presentation_labels, rotation=45, ha='right', fontsize=8)
        axes_dots[1].set_yticks(range(num_presentations))
        axes_dots[1].set_yticklabels(presentation_labels, fontsize=8)
        axes_dots[1].set_xlabel('j')
        axes_dots[1].set_ylabel('i')
        axes_dots[1].set_title('Std D_α[i,j] across output dimensions\n(variation in alpha weighting)')
        plt.colorbar(im_std, ax=axes_dots[1], label='Std of alpha-weighted dot product')

        plt.suptitle('D_alpha Matrix Analysis (LL) - Final Layer', fontsize=12)
        plt.tight_layout()
        figures["list_linking/pw_decomposition_d_alpha"] = fig_dots
        plt.close(fig_dots)

        # --- Residual coefficients in expanded basis ---
        # R = P - P_reconstructed, project onto u_j basis: R_coeff[m,j] = (R @ U_pinv^T)[m,j]
        U_matrix = embeddings_u.T  # (hidden_size, num_presentations)
        U_pinv = np.linalg.pinv(U_matrix)  # (num_presentations, hidden_size)
        R_coeff = P_diff @ U_pinv.T  # (hidden_size, num_presentations)

        # Reconstruction with residuals
        P_with_residual = P_reconstructed + R_coeff @ U_matrix.T
        reconstruction_error_with_residual = np.linalg.norm(P_actual - P_with_residual) / np.linalg.norm(P_actual) if np.linalg.norm(P_actual) > 0 else 0.0

        # Plot residual coefficients
        fig_resid, ax_resid = plt.subplots(figsize=(12, 8), dpi=150)
        vmax_resid = max(abs(R_coeff.min()), abs(R_coeff.max()))
        if vmax_resid == 0:
            vmax_resid = 1
        im_resid = ax_resid.imshow(R_coeff, cmap='RdBu_r', vmin=-vmax_resid, vmax=vmax_resid, aspect='auto')
        ax_resid.set_xticks(range(num_presentations))
        ax_resid.set_xticklabels(presentation_labels, rotation=45, ha='right', fontsize=8)
        ax_resid.set_xlabel('j (u_j basis vector)')
        ax_resid.set_ylabel('m (output dimension)')
        ax_resid.set_title('Residual Coefficients R[m,j] - Final Layer\n(what the v_i basis cannot capture, in the e_m ⊗ u_j basis)')
        plt.colorbar(im_resid, ax=ax_resid, label='Residual coefficient')
        plt.tight_layout()
        figures["list_linking/pw_decomposition_residual"] = fig_resid
        plt.close(fig_resid)

        # Plot reconstruction comparison: without vs with residuals
        fig_recon_compare, axes_rc = plt.subplots(1, 3, figsize=(18, 5), dpi=150)

        im_rc0 = axes_rc[0].imshow(P_actual, cmap='RdBu_r', vmin=-vmax_p, vmax=vmax_p, aspect='equal')
        axes_rc[0].set_title('Actual P')
        plt.colorbar(im_rc0, ax=axes_rc[0])

        im_rc1 = axes_rc[1].imshow(P_reconstructed, cmap='RdBu_r', vmin=-vmax_p, vmax=vmax_p, aspect='equal')
        axes_rc[1].set_title(f'v_i ⊗ u_j basis only\n(error = {reconstruction_error:.4f})')
        plt.colorbar(im_rc1, ax=axes_rc[1])

        im_rc2 = axes_rc[2].imshow(P_with_residual, cmap='RdBu_r', vmin=-vmax_p, vmax=vmax_p, aspect='equal')
        axes_rc[2].set_title(f'With e_m ⊗ u_j residuals\n(error = {reconstruction_error_with_residual:.4f})')
        plt.colorbar(im_rc2, ax=axes_rc[2])

        plt.suptitle('Reconstruction Comparison: Scalar vs Expanded Basis (LL) - Final Layer', fontsize=12)
        plt.tight_layout()
        figures["list_linking/pw_decomposition_reconstruction_comparison"] = fig_recon_compare
        plt.close(fig_recon_compare)

        # Residual norm summary
        fig_resid_summary, axes_resid_summary = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

        # Left: Residual norm per output dimension m
        resid_norm_per_m = np.linalg.norm(R_coeff, axis=1)  # (hidden_size,)
        axes_resid_summary[0].bar(range(len(resid_norm_per_m)), resid_norm_per_m, color='steelblue', alpha=0.7)
        axes_resid_summary[0].set_xlabel('Output dimension m')
        axes_resid_summary[0].set_ylabel('||R[m,:]||')
        axes_resid_summary[0].set_title('Residual norm per output dimension')

        # Right: Residual norm per u_j basis vector
        resid_norm_per_j = np.linalg.norm(R_coeff, axis=0)  # (num_presentations,)
        axes_resid_summary[1].bar(range(num_presentations), resid_norm_per_j, color='darkorange', alpha=0.7)
        axes_resid_summary[1].set_xticks(range(num_presentations))
        axes_resid_summary[1].set_xticklabels(presentation_labels, rotation=45, ha='right', fontsize=8)
        axes_resid_summary[1].set_xlabel('j (u_j basis vector)')
        axes_resid_summary[1].set_ylabel('||R[:,j]||')
        axes_resid_summary[1].set_title('Residual norm per u_j basis vector')

        plt.suptitle('Residual Analysis: Where does the v_i basis fail? - Final Layer', fontsize=12)
        plt.tight_layout()
        figures["list_linking/pw_decomposition_residual_summary"] = fig_resid_summary
        plt.close(fig_resid_summary)

    # Get model parameters (innate weights, alpha) - shared across networks
    alpha_param = model.alpha.detach().cpu().numpy()
    W_innate_final = model.fc2.weight.detach().cpu().numpy()
    alpha_extra_list = [model.alpha_extra[i].detach().cpu().numpy() for i in range(args.extra_layers)]
    W_innate_extra_list = [model.extra_hidden_layers[i].weight.detach().cpu().numpy() for i in range(args.extra_layers)]

    # --- Singular Value Spectra for P, α, and α ⊙ P ---
    # Compute mean plastic weights across batch
    pw_mean_ll = frozen_plastic_weights.detach().cpu().numpy().mean(axis=0)
    alpha_pw_ll = alpha_param * pw_mean_ll  # Hadamard product

    # SVD for all three matrices (final layer)
    _, S_pw_ll, _ = np.linalg.svd(pw_mean_ll)
    _, S_alpha_pw_ll, _ = np.linalg.svd(alpha_pw_ll)

    if alpha_param.ndim >= 2:
        _, S_alpha_ll, _ = np.linalg.svd(alpha_param)

        # Combined singular value plot for final layer
        fig_sv_ll, ax_sv_ll = plt.subplots(figsize=(12, 6), dpi=150)

        x_indices_ll = np.arange(len(S_pw_ll))
        width_ll = 0.25

        ax_sv_ll.bar(x_indices_ll - width_ll, S_pw_ll, width_ll, color='steelblue', edgecolor='black', alpha=0.8, label='P (Plastic Weights)')
        ax_sv_ll.bar(x_indices_ll, S_alpha_ll, width_ll, color='forestgreen', edgecolor='black', alpha=0.8, label='α (Alpha)')
        ax_sv_ll.bar(x_indices_ll + width_ll, S_alpha_pw_ll, width_ll, color='darkorange', edgecolor='black', alpha=0.8, label='α ⊙ P (Hadamard)')

        ax_sv_ll.set_xlabel('Singular Value Index')
        ax_sv_ll.set_ylabel('Singular Value')
        ax_sv_ll.set_title('Singular Value Spectra: P, α, and α ⊙ P (Final Layer)')
        ax_sv_ll.set_yscale('log')
        ax_sv_ll.legend(loc='upper right')

        if len(S_pw_ll) > 20:
            ax_sv_ll.set_xlim(-0.5, 20.5)

        plt.tight_layout()
        figures["list_linking/singular_values_P_alpha_hadamard_final"] = fig_sv_ll
        plt.close(fig_sv_ll)

    # Combined singular value plots for extra layers
    for layer_idx, epw in enumerate(frozen_extra_plastic_weights):
        epw_mean_ll = epw.detach().cpu().numpy().mean(axis=0)
        alpha_extra_ll = alpha_extra_list[layer_idx]
        alpha_pw_extra_ll = alpha_extra_ll * epw_mean_ll  # Hadamard product

        if alpha_extra_ll.ndim >= 2:
            _, S_epw_ll, _ = np.linalg.svd(epw_mean_ll)
            _, S_alpha_extra_ll, _ = np.linalg.svd(alpha_extra_ll)
            _, S_alpha_pw_extra_ll, _ = np.linalg.svd(alpha_pw_extra_ll)

            fig_sv_extra_ll, ax_sv_extra_ll = plt.subplots(figsize=(12, 6), dpi=150)

            x_indices_extra_ll = np.arange(len(S_epw_ll))
            width_extra_ll = 0.25

            ax_sv_extra_ll.bar(x_indices_extra_ll - width_extra_ll, S_epw_ll, width_extra_ll, color='steelblue', edgecolor='black', alpha=0.8, label='P (Plastic Weights)')
            ax_sv_extra_ll.bar(x_indices_extra_ll, S_alpha_extra_ll, width_extra_ll, color='forestgreen', edgecolor='black', alpha=0.8, label='α (Alpha)')
            ax_sv_extra_ll.bar(x_indices_extra_ll + width_extra_ll, S_alpha_pw_extra_ll, width_extra_ll, color='darkorange', edgecolor='black', alpha=0.8, label='α ⊙ P (Hadamard)')

            ax_sv_extra_ll.set_xlabel('Singular Value Index')
            ax_sv_extra_ll.set_ylabel('Singular Value')
            ax_sv_extra_ll.set_title(f'Singular Value Spectra: P, α, and α ⊙ P (Hidden Layer {layer_idx + 1})')
            ax_sv_extra_ll.set_yscale('log')
            ax_sv_extra_ll.legend(loc='upper right')

            if len(S_epw_ll) > 20:
                ax_sv_extra_ll.set_xlim(-0.5, 20.5)

            plt.tight_layout()
            figures[f"list_linking/singular_values_P_alpha_hadamard_hidden{layer_idx + 1}"] = fig_sv_extra_ll
            plt.close(fig_sv_extra_ll)

    # --- Define test pairs (same structure for all networks) ---
    cross_list_pairs = [(i, j) for i in range(4) for j in range(4, 8)]
    within_list_pairs = ([(i, j) for i in range(4) for j in range(i + 1, 4)] +
                         [(i, j) for i in range(4, 8) for j in range(i + 1, 8)])
    all_test_pairs = cross_list_pairs + within_list_pairs
    num_pairs = len(all_test_pairs)  # 16 cross + 12 within = 28 pairs
    num_test_samples_per_network = num_pairs * 2  # 2 presentation orders per pair = 56

    # --- Storage for per-network metrics ---
    num_networks_to_analyze = min(batch_size, 100)  # Use up to 100 networks for statistics
    max_svs = 20
    num_layers = args.extra_layers + 2

    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.model_selection import cross_val_score

    # Cross-validation settings
    cv_folds = 5  # 5-fold CV

    # Initialize storage: layer_idx -> metric_name -> list of (max_svs,) arrays
    layer_network_metrics = {}
    for layer_idx in range(1, num_layers):  # Skip layer 0 (embedding)
        layer_network_metrics[layer_idx] = {
            'r2_all': [],
            'r2_cross': [],
            'r2_within': [],
            'item_acc': [],
            'var_exp': [],
            # Different weight types
            'r2_pw': [],
            'r2_apw': [],
            'r2_eff': [],
            # Decomposition comparison
            'r2_right_sv': [],
            'r2_left_sv': [],
            'r2_eigen': [],
            'var_right_sv': [],
            'var_left_sv': [],
            'var_eigen': [],
        }

    # Also collect behavioral data across networks
    all_networks_output_data = []

    # Storage for per-layer readout predictions (passing embeddings through remaining layers)
    # Structure: layer_idx -> list of dicts with 'signed_sd', 'probability', 'logit', 'is_cross_list'
    layer_readout_data = {layer_idx: [] for layer_idx in range(num_layers)}

    # Storage for item-readout correlation analysis
    # Structure: position -> layer_idx -> item_idx -> list of correlations (one per network)
    # position is 'item1' or 'item2'
    item_readout_correlations = {
        'item1': {layer_idx: {item_idx: [] for item_idx in range(num_items)} for layer_idx in range(num_layers)},
        'item2': {layer_idx: {item_idx: [] for item_idx in range(num_items)} for layer_idx in range(num_layers)},
    }

    # Storage for item-readout dot products (raw, not normalized)
    item_readout_dotproducts = {
        'item1': {layer_idx: {item_idx: [] for item_idx in range(num_items)} for layer_idx in range(num_layers)},
        'item2': {layer_idx: {item_idx: [] for item_idx in range(num_items)} for layer_idx in range(num_layers)},
    }

    # Storage for pair-readout correlations (joint representation)
    # Structure: layer_idx -> list of (num_items, num_items) matrices, one per network
    pair_readout_correlations = {layer_idx: [] for layer_idx in range(num_layers)}
    pair_readout_dotproducts = {layer_idx: [] for layer_idx in range(num_layers)}

    # Storage for single-item embeddings for PCA analysis
    # Structure: position -> layer_idx -> list of (embedding, item_idx, network_idx)
    single_item_embeddings = {
        'item1': {layer_idx: [] for layer_idx in range(num_layers)},
        'item2': {layer_idx: [] for layer_idx in range(num_layers)},
    }

    # --- Process each network independently ---
    print(f"Analyzing {num_networks_to_analyze} networks independently...")
    for network_idx in range(num_networks_to_analyze):
        single_items = batch_items[network_idx]  # (num_items, item_size)
        single_pw = frozen_plastic_weights[network_idx]  # (hidden_size, hidden_size)
        single_epw = [epw[network_idx] for epw in frozen_extra_plastic_weights]

        # Generate test trials for this network
        test_trials = []
        test_correct_choices = []
        test_pair_indices = []

        for pair in all_test_pairs:
            item1_idx, item2_idx = pair
            item1_emb = single_items[item1_idx]
            item2_emb = single_items[item2_idx]

            # Presentation order 1: item1 first
            trial_input_1 = np.concatenate([item1_emb, item2_emb])
            test_trials.append(trial_input_1)
            test_correct_choices.append(0)
            test_pair_indices.append((item1_idx, item2_idx))

            # Presentation order 2: item2 first
            trial_input_2 = np.concatenate([item2_emb, item1_emb])
            test_trials.append(trial_input_2)
            test_correct_choices.append(1)
            test_pair_indices.append((item2_idx, item1_idx))

        # Prepare batch for inference
        batch_trials_np = np.array(test_trials)
        batch_correct_np = np.array(test_correct_choices)

        # Pad to batch_size if needed
        if num_test_samples_per_network < batch_size:
            pad_size = batch_size - num_test_samples_per_network
            batch_trials_np = np.concatenate([batch_trials_np, np.zeros((pad_size, batch_trials_np.shape[1]))], axis=0)
            batch_correct_np = np.concatenate([batch_correct_np, np.zeros(pad_size)], axis=0)

        batch_trials_t = torch.tensor(batch_trials_np, dtype=torch.float32).to(device)
        batch_correct_t = torch.tensor(batch_correct_np, dtype=torch.float32).to(device)

        # Expand single network's plastic weights
        pw_expanded = single_pw.unsqueeze(0).expand(batch_size, -1, -1).clone()
        epw_expanded = [epw.unsqueeze(0).expand(batch_size, -1, -1).clone() for epw in single_epw]

        with torch.inference_mode():
            output = model(batch_trials_t, pw_expanded, batch_correct_t,
                          extra_plastic_weights=epw_expanded, store_embeddings=True)

        # Collect output data (only first num_test_samples_per_network)
        probs = output.choice.squeeze(-1).detach().cpu().numpy()[:num_test_samples_per_network]
        probs_clipped = np.clip(probs, 1e-7, 1 - 1e-7)
        logits = np.log(probs_clipped / (1 - probs_clipped))

        network_output_data = []
        for i in range(num_test_samples_per_network):
            item1_idx, item2_idx = test_pair_indices[i]
            signed_sd = item1_idx - item2_idx
            is_cross_list = (item1_idx < 4 and item2_idx >= 4) or (item1_idx >= 4 and item2_idx < 4)
            network_output_data.append({
                'signed_sd': signed_sd,
                'probability': probs[i],
                'logit': logits[i],
                'is_cross_list': is_cross_list,
            })
        all_networks_output_data.append(network_output_data)

        # --- Compute per-layer readouts by passing embeddings through remaining layers ---
        # For each layer's embedding, pass through remaining layers to get choice probability
        # Need to use no_grad and clone tensors to avoid inference mode issues
        with torch.no_grad():
            for layer_idx, embedding in enumerate(output.embeddings):
                # Clone to get normal tensors (inference tensors can't be saved for backward)
                emb_tensor = embedding[:num_test_samples_per_network].clone()

                # Start with this layer's embedding and pass through remaining layers
                h = emb_tensor  # Current hidden state

                # Apply remaining extra hidden layers (if any)
                for remaining_layer_idx in range(layer_idx, args.extra_layers):
                    # Get the layer and its plastic weights
                    layer_module = model.extra_hidden_layers[remaining_layer_idx]
                    alpha_extra = model.alpha_extra[remaining_layer_idx]
                    epw = single_epw[remaining_layer_idx]

                    # Innate contribution
                    innate = layer_module(h)
                    # Plastic contribution: einsum('bhi,bi->bh', alpha * pw, h)
                    plastic = torch.einsum('hi,bi->bh', alpha_extra * epw, h)
                    # Apply activation
                    h = torch.tanh(innate + plastic)

                # Apply final layer (fc2 + plastic weights)
                innate_final = model.fc2(h)
                plastic_final = torch.einsum('hi,bi->bh', model.alpha * single_pw, h)
                final_hidden = torch.tanh(innate_final + plastic_final)

                # Apply choice layer to get probability
                layer_choice = torch.sigmoid(model.choice(final_hidden))
                layer_probs = layer_choice.squeeze(-1).detach().cpu().numpy()
                layer_probs_clipped = np.clip(layer_probs, 1e-7, 1 - 1e-7)
                layer_logits = np.log(layer_probs_clipped / (1 - layer_probs_clipped))

                # Store readout data for this layer
                for i in range(num_test_samples_per_network):
                    item1_idx, item2_idx = test_pair_indices[i]
                    signed_sd = item1_idx - item2_idx
                    is_cross_list = (item1_idx < 4 and item2_idx >= 4) or (item1_idx >= 4 and item2_idx < 4)
                    layer_readout_data[layer_idx].append({
                        'signed_sd': signed_sd,
                        'probability': layer_probs[i],
                        'logit': layer_logits[i],
                        'is_cross_list': is_cross_list,
                    })

        # --- Item-readout correlation analysis ---
        # For each item, run it through the network with the other position zeroed
        # Then compute correlation between each layer's embedding and the readout weights
        readout_weights = model.choice.weight.detach().cpu().numpy().squeeze()  # (hidden_size,)

        with torch.no_grad():
            for item_idx in range(num_items):
                item_emb = single_items[item_idx]  # (item_size,)
                zero_emb = np.zeros_like(item_emb)

                # Position 1: item in first position, zeros in second
                input_pos1 = np.concatenate([item_emb, zero_emb])
                # Position 2: zeros in first position, item in second
                input_pos2 = np.concatenate([zero_emb, item_emb])

                for position, input_vec in [('item1', input_pos1), ('item2', input_pos2)]:
                    # Create batch input (single sample, padded to batch_size)
                    input_batch = np.zeros((batch_size, len(input_vec)))
                    input_batch[0] = input_vec
                    input_t = torch.tensor(input_batch, dtype=torch.float32).to(device)
                    dummy_correct = torch.zeros(batch_size, dtype=torch.float32).to(device)

                    # Run through network
                    output_single = model(input_t, pw_expanded, dummy_correct,
                                         extra_plastic_weights=epw_expanded, store_embeddings=True)

                    # For each layer, compute correlation and dot product with readout weights
                    for layer_idx, embedding in enumerate(output_single.embeddings):
                        emb_vec = embedding[0].detach().cpu().numpy()  # (hidden_size,)

                        # Compute Pearson correlation between embedding and readout weights
                        if np.std(emb_vec) > 1e-10 and np.std(readout_weights) > 1e-10:
                            corr = np.corrcoef(emb_vec, readout_weights)[0, 1]
                        else:
                            corr = 0.0

                        # Compute raw dot product
                        dot_prod = np.dot(emb_vec, readout_weights)

                        item_readout_correlations[position][layer_idx][item_idx].append(corr)
                        item_readout_dotproducts[position][layer_idx][item_idx].append(dot_prod)

                        # Store embedding for PCA analysis
                        single_item_embeddings[position][layer_idx].append({
                            'embedding': emb_vec,
                            'item_idx': item_idx,
                            'network_idx': network_idx,
                        })

        # --- Pair-readout correlation analysis (joint representation) ---
        # For each ordered pair (i, j), compute correlation between embedding and readout weights
        # Build a (num_items, num_items) matrix for each layer
        for layer_idx, embedding in enumerate(output.embeddings):
            # Initialize correlation and dot product matrices for this network
            corr_matrix = np.full((num_items, num_items), np.nan)
            dot_matrix = np.full((num_items, num_items), np.nan)

            # Process each test trial
            for trial_idx in range(num_test_samples_per_network):
                item1_idx, item2_idx = test_pair_indices[trial_idx]
                emb_vec = embedding[trial_idx].detach().cpu().numpy()

                # Compute Pearson correlation
                if np.std(emb_vec) > 1e-10 and np.std(readout_weights) > 1e-10:
                    corr = np.corrcoef(emb_vec, readout_weights)[0, 1]
                else:
                    corr = 0.0

                # Compute raw dot product
                dot_prod = np.dot(emb_vec, readout_weights)

                # Store in matrix (item1 = row/y-axis, item2 = col/x-axis)
                corr_matrix[item1_idx, item2_idx] = corr
                dot_matrix[item1_idx, item2_idx] = dot_prod

            pair_readout_correlations[layer_idx].append(corr_matrix)
            pair_readout_dotproducts[layer_idx].append(dot_matrix)

        # Prepare per-trial metadata
        pair_signed_sds = np.array([test_pair_indices[i][0] - test_pair_indices[i][1] for i in range(num_test_samples_per_network)])
        pair_is_cross_list = np.array([
            (test_pair_indices[i][0] < 4 and test_pair_indices[i][1] >= 4) or
            (test_pair_indices[i][0] >= 4 and test_pair_indices[i][1] < 4)
            for i in range(num_test_samples_per_network)
        ])
        pair_item1_indices = np.array([test_pair_indices[i][0] for i in range(num_test_samples_per_network)])

        cross_indices = np.where(pair_is_cross_list)[0]
        within_indices = np.where(~pair_is_cross_list)[0]

        # Get this network's weight matrices
        pw_single_np = single_pw.detach().cpu().numpy()
        single_epw_np = [epw.detach().cpu().numpy() for epw in single_epw]

        # Process each layer
        for layer_idx, embedding in enumerate(output.embeddings):
            if layer_idx == 0:
                continue  # Skip embedding layer

            emb_np = embedding.detach().cpu().numpy()[:num_test_samples_per_network]

            # Get weight matrices for this layer
            if layer_idx == args.extra_layers + 1:
                # Final layer
                pw_layer = pw_single_np
                alpha_layer = alpha_param
                W_innate_layer = W_innate_final
            else:
                # Hidden layer
                pw_layer = single_epw_np[layer_idx - 1]
                alpha_layer = alpha_extra_list[layer_idx - 1]
                W_innate_layer = W_innate_extra_list[layer_idx - 1]

            alpha_pw_layer = alpha_layer * pw_layer
            W_eff_layer = W_innate_layer + alpha_pw_layer

            # SVD of each weight type (only need V for projection)
            _, _, Vh_pw = np.linalg.svd(pw_layer)
            _, _, Vh_apw = np.linalg.svd(alpha_pw_layer)
            U_eff, S_eff, Vh_eff = np.linalg.svd(W_eff_layer)

            V_pw = Vh_pw.T
            V_apw = Vh_apw.T
            V_eff = Vh_eff.T

            # Eigendecomposition for comparison
            eigenvalues, eigenvectors = np.linalg.eig(W_eff_layer)
            eig_order = np.argsort(np.abs(eigenvalues))[::-1]
            eigenvectors_sorted = eigenvectors[:, eig_order]
            eigenvectors_real = np.real(eigenvectors_sorted)

            max_k = min(max_svs, len(S_eff))

            # Compute metrics for each k
            r2_all_k, r2_cross_k, r2_within_k = [], [], []
            item_acc_k, var_exp_k = [], []
            r2_pw_k, r2_apw_k, r2_eff_k = [], [], []
            r2_right_k, r2_left_k, r2_eig_k = [], [], []
            var_right_k, var_left_k, var_eig_k = [], [], []

            for k in range(1, max_k + 1):
                # Project onto top-k right SVs of effective weights
                proj_eff = emb_np @ V_eff[:, :k]

                # R² for all pairs (rank encoding) - using cross-validation
                if len(proj_eff) > cv_folds and len(proj_eff) > k + 1:
                    try:
                        reg = LinearRegression()
                        cv_scores = cross_val_score(reg, proj_eff, pair_signed_sds, cv=cv_folds, scoring='r2')
                        r2 = np.mean(cv_scores)
                        r2 = max(0, r2)  # Clip negative R² to 0
                    except:
                        r2 = 0
                else:
                    r2 = 0
                r2_all_k.append(r2)

                # R² for cross-list pairs - using cross-validation
                if len(cross_indices) > cv_folds and len(cross_indices) > k + 1:
                    try:
                        reg_c = LinearRegression()
                        cv_scores_c = cross_val_score(reg_c, proj_eff[cross_indices], pair_signed_sds[cross_indices], cv=cv_folds, scoring='r2')
                        r2_c = np.mean(cv_scores_c)
                        r2_c = max(0, r2_c)
                    except:
                        r2_c = 0
                else:
                    r2_c = 0
                r2_cross_k.append(r2_c)

                # R² for within-list pairs - using cross-validation
                if len(within_indices) > cv_folds and len(within_indices) > k + 1:
                    try:
                        reg_w = LinearRegression()
                        cv_scores_w = cross_val_score(reg_w, proj_eff[within_indices], pair_signed_sds[within_indices], cv=cv_folds, scoring='r2')
                        r2_w = np.mean(cv_scores_w)
                        r2_w = max(0, r2_w)
                    except:
                        r2_w = 0
                else:
                    r2_w = 0
                r2_within_k.append(r2_w)

                # Item classification accuracy - using cross-validation
                try:
                    clf = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
                    cv_scores_acc = cross_val_score(clf, proj_eff, pair_item1_indices, cv=cv_folds, scoring='accuracy')
                    acc = np.mean(cv_scores_acc)
                except:
                    acc = 1.0 / num_items
                item_acc_k.append(acc)

                # Variance explained (this is not a prediction task, so no CV needed)
                recon = proj_eff @ V_eff[:, :k].T
                total_var = np.var(emb_np)
                residual_var = np.var(emb_np - recon)
                var_exp = 1 - residual_var / total_var if total_var > 0 else 0
                var_exp_k.append(var_exp)

                # R² for different weight types - using cross-validation
                proj_pw = emb_np @ V_pw[:, :k]
                proj_apw = emb_np @ V_apw[:, :k]

                try:
                    reg_pw = LinearRegression()
                    cv_scores_pw = cross_val_score(reg_pw, proj_pw, pair_signed_sds, cv=cv_folds, scoring='r2')
                    r2_pw_k.append(max(0, np.mean(cv_scores_pw)))
                except:
                    r2_pw_k.append(0)

                try:
                    reg_apw = LinearRegression()
                    cv_scores_apw = cross_val_score(reg_apw, proj_apw, pair_signed_sds, cv=cv_folds, scoring='r2')
                    r2_apw_k.append(max(0, np.mean(cv_scores_apw)))
                except:
                    r2_apw_k.append(0)

                r2_eff_k.append(r2)  # Same as r2_all_k

                # Decomposition comparison: Right SVs, Left SVs, Eigenvectors - using cross-validation
                proj_right = emb_np @ V_eff[:, :k]
                proj_left = emb_np @ U_eff[:, :k]
                proj_eig = emb_np @ eigenvectors_real[:, :k]

                try:
                    reg_r = LinearRegression()
                    cv_scores_r = cross_val_score(reg_r, proj_right, pair_signed_sds, cv=cv_folds, scoring='r2')
                    r2_right_k.append(max(0, np.mean(cv_scores_r)))
                except:
                    r2_right_k.append(0)

                try:
                    reg_l = LinearRegression()
                    cv_scores_l = cross_val_score(reg_l, proj_left, pair_signed_sds, cv=cv_folds, scoring='r2')
                    r2_left_k.append(max(0, np.mean(cv_scores_l)))
                except:
                    r2_left_k.append(0)

                try:
                    reg_e = LinearRegression()
                    cv_scores_e = cross_val_score(reg_e, proj_eig, pair_signed_sds, cv=cv_folds, scoring='r2')
                    r2_eig_k.append(max(0, np.mean(cv_scores_e)))
                except:
                    r2_eig_k.append(0)

                # Variance explained for each decomposition (no CV needed)
                recon_r = proj_right @ V_eff[:, :k].T
                recon_l = proj_left @ U_eff[:, :k].T
                recon_e = proj_eig @ eigenvectors_real[:, :k].T

                var_right_k.append(1 - np.var(emb_np - recon_r) / total_var if total_var > 0 else 0)
                var_left_k.append(1 - np.var(emb_np - recon_l) / total_var if total_var > 0 else 0)
                var_eig_k.append(1 - np.var(emb_np - recon_e) / total_var if total_var > 0 else 0)

            # Pad to max_svs if needed
            while len(r2_all_k) < max_svs:
                r2_all_k.append(r2_all_k[-1] if r2_all_k else 0)
                r2_cross_k.append(r2_cross_k[-1] if r2_cross_k else 0)
                r2_within_k.append(r2_within_k[-1] if r2_within_k else 0)
                item_acc_k.append(item_acc_k[-1] if item_acc_k else 0)
                var_exp_k.append(var_exp_k[-1] if var_exp_k else 0)
                r2_pw_k.append(r2_pw_k[-1] if r2_pw_k else 0)
                r2_apw_k.append(r2_apw_k[-1] if r2_apw_k else 0)
                r2_eff_k.append(r2_eff_k[-1] if r2_eff_k else 0)
                r2_right_k.append(r2_right_k[-1] if r2_right_k else 0)
                r2_left_k.append(r2_left_k[-1] if r2_left_k else 0)
                r2_eig_k.append(r2_eig_k[-1] if r2_eig_k else 0)
                var_right_k.append(var_right_k[-1] if var_right_k else 0)
                var_left_k.append(var_left_k[-1] if var_left_k else 0)
                var_eig_k.append(var_eig_k[-1] if var_eig_k else 0)

            # Store metrics for this network
            layer_network_metrics[layer_idx]['r2_all'].append(r2_all_k)
            layer_network_metrics[layer_idx]['r2_cross'].append(r2_cross_k)
            layer_network_metrics[layer_idx]['r2_within'].append(r2_within_k)
            layer_network_metrics[layer_idx]['item_acc'].append(item_acc_k)
            layer_network_metrics[layer_idx]['var_exp'].append(var_exp_k)
            layer_network_metrics[layer_idx]['r2_pw'].append(r2_pw_k)
            layer_network_metrics[layer_idx]['r2_apw'].append(r2_apw_k)
            layer_network_metrics[layer_idx]['r2_eff'].append(r2_eff_k)
            layer_network_metrics[layer_idx]['r2_right_sv'].append(r2_right_k)
            layer_network_metrics[layer_idx]['r2_left_sv'].append(r2_left_k)
            layer_network_metrics[layer_idx]['r2_eigen'].append(r2_eig_k)
            layer_network_metrics[layer_idx]['var_right_sv'].append(var_right_k)
            layer_network_metrics[layer_idx]['var_left_sv'].append(var_left_k)
            layer_network_metrics[layer_idx]['var_eigen'].append(var_eig_k)

    print(f"Finished analyzing {num_networks_to_analyze} networks. Creating plots...")

    # --- Create plots with mean ± stderr across networks ---
    for layer_idx in range(1, num_layers):
        if layer_idx < args.extra_layers + 1:
            layer_pw_name = f"Hidden Layer {layer_idx}"
        else:
            layer_pw_name = "Final Layer"

        # Convert to numpy arrays: (num_networks, max_svs)
        r2_all = np.array(layer_network_metrics[layer_idx]['r2_all'])
        r2_cross = np.array(layer_network_metrics[layer_idx]['r2_cross'])
        r2_within = np.array(layer_network_metrics[layer_idx]['r2_within'])
        item_acc = np.array(layer_network_metrics[layer_idx]['item_acc'])
        var_exp = np.array(layer_network_metrics[layer_idx]['var_exp'])

        # Compute mean and stderr
        mean_r2_all = np.mean(r2_all, axis=0)
        stderr_r2_all = np.std(r2_all, axis=0) / np.sqrt(num_networks_to_analyze)
        mean_r2_cross = np.mean(r2_cross, axis=0)
        stderr_r2_cross = np.std(r2_cross, axis=0) / np.sqrt(num_networks_to_analyze)
        mean_r2_within = np.mean(r2_within, axis=0)
        stderr_r2_within = np.std(r2_within, axis=0) / np.sqrt(num_networks_to_analyze)
        mean_item_acc = np.mean(item_acc, axis=0)
        stderr_item_acc = np.std(item_acc, axis=0) / np.sqrt(num_networks_to_analyze)
        mean_var_exp = np.mean(var_exp, axis=0)
        stderr_var_exp = np.std(var_exp, axis=0) / np.sqrt(num_networks_to_analyze)

        x_vals = np.array(range(1, max_svs + 1))

        # --- Plot 1: Rank vs Identity Encoding Comparison (main plot with error bands) ---
        fig_compare, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)

        # Left: Rank encoding (R²) with error band
        axes[0].plot(x_vals, mean_r2_all, 'o-', linewidth=2, markersize=6, color='green')
        axes[0].fill_between(x_vals, mean_r2_all - stderr_r2_all, mean_r2_all + stderr_r2_all,
                             color='green', alpha=0.2)
        axes[0].set_xlabel('Number of Top SVs')
        axes[0].set_ylabel('R² (Signed SD Prediction)')
        axes[0].set_title(f'Rank Encoding (5-fold CV)\n(n={num_networks_to_analyze} networks)')
        axes[0].set_ylim(0, 1.05)
        axes[0].set_xticks(x_vals[::2])  # Show every other tick
        axes[0].axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)

        # Middle: Item identity encoding with error band
        chance_item = 1.0 / num_items
        axes[1].plot(x_vals, mean_item_acc, 'o-', linewidth=2, markersize=6, color='blue', label='Item Identity')
        axes[1].fill_between(x_vals, mean_item_acc - stderr_item_acc, mean_item_acc + stderr_item_acc,
                             color='blue', alpha=0.2)
        axes[1].axhline(y=chance_item, color='gray', linestyle='--', alpha=0.5, label=f'Chance: {chance_item:.2f}')
        axes[1].set_xlabel('Number of Top SVs')
        axes[1].set_ylabel('Classification Accuracy')
        axes[1].set_title(f'Item Identity (5-fold CV)\n(n={num_networks_to_analyze} networks)')
        axes[1].set_ylim(0, 1.05)
        axes[1].set_xticks(x_vals[::2])
        axes[1].legend()

        # Right: Variance explained with error band
        axes[2].plot(x_vals, mean_var_exp, 'o-', linewidth=2, markersize=6, color='purple')
        axes[2].fill_between(x_vals, mean_var_exp - stderr_var_exp, mean_var_exp + stderr_var_exp,
                             color='purple', alpha=0.2)
        axes[2].axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)
        axes[2].set_xlabel('Number of Top SVs')
        axes[2].set_ylabel('Variance Explained')
        axes[2].set_title(f'Embedding Reconstruction\n(n={num_networks_to_analyze} networks)')
        axes[2].set_ylim(0, 1.05)
        axes[2].set_xticks(x_vals[::2])

        plt.suptitle(f'List Linking: Rank vs Item Identity Encoding - {layer_pw_name}', fontsize=14)
        plt.tight_layout()
        figures[f"list_linking/layer{layer_idx}_rank_vs_identity"] = fig_compare
        plt.close(fig_compare)

        # --- Plot 2: R² by pair type (cross-list vs within-list) with error bands ---
        fig_r2_split, ax_r2 = plt.subplots(figsize=(12, 5), dpi=150)

        ax_r2.plot(x_vals, mean_r2_cross, 'o-', linewidth=2, markersize=6, color='red', label='Cross-List Pairs')
        ax_r2.fill_between(x_vals, mean_r2_cross - stderr_r2_cross, mean_r2_cross + stderr_r2_cross,
                           color='red', alpha=0.2)
        ax_r2.plot(x_vals, mean_r2_within, 's-', linewidth=2, markersize=6, color='blue', label='Within-List Pairs')
        ax_r2.fill_between(x_vals, mean_r2_within - stderr_r2_within, mean_r2_within + stderr_r2_within,
                           color='blue', alpha=0.2)
        ax_r2.plot(x_vals, mean_r2_all, '^-', linewidth=2, markersize=6, color='green', alpha=0.7, label='All Pairs')
        ax_r2.fill_between(x_vals, mean_r2_all - stderr_r2_all, mean_r2_all + stderr_r2_all,
                           color='green', alpha=0.1)
        ax_r2.set_xlabel('Number of Top SVs')
        ax_r2.set_ylabel('R² (Signed SD Prediction)')
        ax_r2.set_title(f'Rank Encoding by Pair Type (5-fold CV) - {layer_pw_name}\n(n={num_networks_to_analyze} networks, mean ± SE)')
        ax_r2.set_ylim(0, 1.05)
        ax_r2.set_xticks(x_vals[::2])
        ax_r2.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)
        ax_r2.legend()

        plt.tight_layout()
        figures[f"list_linking/layer{layer_idx}_r2_by_pair_type"] = fig_r2_split
        plt.close(fig_r2_split)

    # --- Bar charts: Output logits and probabilities by signed symbolic distance ---
    # Aggregate output data from all networks
    all_output_signed_sds = []
    all_output_probs = []
    all_output_logits = []
    all_output_is_cross = []

    for network_data in all_networks_output_data:
        for d in network_data:
            all_output_signed_sds.append(d['signed_sd'])
            all_output_probs.append(d['probability'])
            all_output_logits.append(d['logit'])
            all_output_is_cross.append(d['is_cross_list'])

    all_output_signed_sds = np.array(all_output_signed_sds)
    all_output_probs = np.array(all_output_probs)
    all_output_logits = np.array(all_output_logits)
    all_output_is_cross = np.array(all_output_is_cross)

    unique_signed_sds = sorted(np.unique(all_output_signed_sds))

    # Compute mean and standard error for each signed SD
    sd_values = []
    mean_probs = []
    std_err_probs = []
    mean_logits = []
    std_err_logits = []

    for ssd in unique_signed_sds:
        mask = all_output_signed_sds == ssd
        probs_for_sd = all_output_probs[mask]
        logits_for_sd = all_output_logits[mask]

        sd_values.append(ssd)
        mean_probs.append(np.mean(probs_for_sd))
        std_err_probs.append(np.std(probs_for_sd) / np.sqrt(len(probs_for_sd)) if len(probs_for_sd) > 0 else 0)
        mean_logits.append(np.mean(logits_for_sd))
        std_err_logits.append(np.std(logits_for_sd) / np.sqrt(len(logits_for_sd)) if len(logits_for_sd) > 0 else 0)

    sd_values = np.array(sd_values)
    mean_probs = np.array(mean_probs)
    std_err_probs = np.array(std_err_probs)
    mean_logits = np.array(mean_logits)
    std_err_logits = np.array(std_err_logits)

    # --- Separate bar charts for cross-list vs within-list pairs ---
    # Cross-list pairs only
    cross_mask = all_output_is_cross
    if np.sum(cross_mask) > 0:
        cross_signed_sds = all_output_signed_sds[cross_mask]
        cross_probs = all_output_probs[cross_mask]
        cross_logits = all_output_logits[cross_mask]

        unique_cross_sds = sorted(np.unique(cross_signed_sds))

        cross_sd_values = []
        cross_mean_probs = []
        cross_std_probs = []
        cross_mean_logits = []
        cross_std_logits = []

        for ssd in unique_cross_sds:
            mask = cross_signed_sds == ssd
            cross_sd_values.append(ssd)
            cross_mean_probs.append(np.mean(cross_probs[mask]))
            cross_std_probs.append(np.std(cross_probs[mask]) / np.sqrt(np.sum(mask)) if np.sum(mask) > 0 else 0)
            cross_mean_logits.append(np.mean(cross_logits[mask]))
            cross_std_logits.append(np.std(cross_logits[mask]) / np.sqrt(np.sum(mask)) if np.sum(mask) > 0 else 0)

        # Cross-list logits
        fig_cross_logits, ax_cross_logits = plt.subplots(figsize=(12, 6), dpi=150)

        x_pos_cross = np.arange(len(cross_sd_values))
        colors_cross = plt.cm.coolwarm(np.linspace(0, 1, len(cross_sd_values)))

        ax_cross_logits.bar(x_pos_cross, cross_mean_logits, yerr=cross_std_logits, capsize=5,
                            color=colors_cross, edgecolor='black', alpha=0.8)
        ax_cross_logits.axhline(y=0, color='gray', linestyle='-', linewidth=1)
        ax_cross_logits.set_xticks(x_pos_cross)
        ax_cross_logits.set_xticklabels([f'{int(sd):+d}' if sd != 0 else '0' for sd in cross_sd_values])
        ax_cross_logits.set_xlabel('Signed Symbolic Distance')
        ax_cross_logits.set_ylabel('Mean Logit')
        ax_cross_logits.set_title(f'List Linking: Output Logit - CROSS-LIST Pairs Only\n(n={num_networks_to_analyze} networks, mean ± SE)')

        plt.tight_layout()
        figures["list_linking/output_logit_cross_list"] = fig_cross_logits
        plt.close(fig_cross_logits)

        # Cross-list probabilities
        fig_cross_probs, ax_cross_probs = plt.subplots(figsize=(12, 6), dpi=150)

        ax_cross_probs.bar(x_pos_cross, cross_mean_probs, yerr=cross_std_probs, capsize=5,
                           color=colors_cross, edgecolor='black', alpha=0.8)
        ax_cross_probs.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='Chance')
        ax_cross_probs.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        ax_cross_probs.axhline(y=1, color='gray', linestyle='-', linewidth=0.5)
        ax_cross_probs.set_xticks(x_pos_cross)
        ax_cross_probs.set_xticklabels([f'{int(sd):+d}' if sd != 0 else '0' for sd in cross_sd_values])
        ax_cross_probs.set_xlabel('Signed Symbolic Distance')
        ax_cross_probs.set_ylabel('Mean Probability (Choice 0)')
        ax_cross_probs.set_title(f'List Linking: Output Probability - CROSS-LIST Pairs Only\n(n={num_networks_to_analyze} networks, mean ± SE)')
        ax_cross_probs.set_ylim(0, 1)

        plt.tight_layout()
        figures["list_linking/output_prob_cross_list"] = fig_cross_probs
        plt.close(fig_cross_probs)

    # Within-list pairs only
    within_mask = ~all_output_is_cross
    if np.sum(within_mask) > 0:
        within_signed_sds = all_output_signed_sds[within_mask]
        within_probs = all_output_probs[within_mask]
        within_logits = all_output_logits[within_mask]

        unique_within_sds = sorted(np.unique(within_signed_sds))

        within_sd_values = []
        within_mean_probs = []
        within_std_probs = []
        within_mean_logits = []
        within_std_logits = []

        for ssd in unique_within_sds:
            mask = within_signed_sds == ssd
            within_sd_values.append(ssd)
            within_mean_probs.append(np.mean(within_probs[mask]))
            within_std_probs.append(np.std(within_probs[mask]) / np.sqrt(np.sum(mask)) if np.sum(mask) > 0 else 0)
            within_mean_logits.append(np.mean(within_logits[mask]))
            within_std_logits.append(np.std(within_logits[mask]) / np.sqrt(np.sum(mask)) if np.sum(mask) > 0 else 0)

        # Within-list logits
        fig_within_logits, ax_within_logits = plt.subplots(figsize=(12, 6), dpi=150)

        x_pos_within = np.arange(len(within_sd_values))
        colors_within = plt.cm.coolwarm(np.linspace(0, 1, len(within_sd_values)))

        ax_within_logits.bar(x_pos_within, within_mean_logits, yerr=within_std_logits, capsize=5,
                             color=colors_within, edgecolor='black', alpha=0.8)
        ax_within_logits.axhline(y=0, color='gray', linestyle='-', linewidth=1)
        ax_within_logits.set_xticks(x_pos_within)
        ax_within_logits.set_xticklabels([f'{int(sd):+d}' if sd != 0 else '0' for sd in within_sd_values])
        ax_within_logits.set_xlabel('Signed Symbolic Distance')
        ax_within_logits.set_ylabel('Mean Logit')
        ax_within_logits.set_title(f'List Linking: Output Logit - WITHIN-LIST Pairs Only\n(n={num_networks_to_analyze} networks, mean ± SE)')

        plt.tight_layout()
        figures["list_linking/output_logit_within_list"] = fig_within_logits
        plt.close(fig_within_logits)

        # Within-list probabilities
        fig_within_probs, ax_within_probs = plt.subplots(figsize=(12, 6), dpi=150)

        ax_within_probs.bar(x_pos_within, within_mean_probs, yerr=within_std_probs, capsize=5,
                            color=colors_within, edgecolor='black', alpha=0.8)
        ax_within_probs.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='Chance')
        ax_within_probs.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        ax_within_probs.axhline(y=1, color='gray', linestyle='-', linewidth=0.5)
        ax_within_probs.set_xticks(x_pos_within)
        ax_within_probs.set_xticklabels([f'{int(sd):+d}' if sd != 0 else '0' for sd in within_sd_values])
        ax_within_probs.set_xlabel('Signed Symbolic Distance')
        ax_within_probs.set_ylabel('Mean Probability (Choice 0)')
        ax_within_probs.set_title(f'List Linking: Output Probability - WITHIN-LIST Pairs Only\n(n={num_networks_to_analyze} networks, mean ± SE)')
        ax_within_probs.set_ylim(0, 1)

        plt.tight_layout()
        figures["list_linking/output_prob_within_list"] = fig_within_probs
        plt.close(fig_within_probs)

    # --- Per-layer readout bar charts ---
    # For each layer, create bar charts showing logits/probs by signed SD
    layer_names = ['Embedding'] + [f'Hidden {i+1}' for i in range(args.extra_layers)] + ['Final']

    for layer_idx in range(num_layers):
        layer_data = layer_readout_data[layer_idx]
        if not layer_data:
            continue

        layer_name = layer_names[layer_idx] if layer_idx < len(layer_names) else f'Layer {layer_idx}'

        # Extract data
        layer_signed_sds = np.array([d['signed_sd'] for d in layer_data])
        layer_probs = np.array([d['probability'] for d in layer_data])
        layer_logits = np.array([d['logit'] for d in layer_data])
        layer_is_cross = np.array([d['is_cross_list'] for d in layer_data])

        unique_sds = sorted(np.unique(layer_signed_sds))

        # Compute mean and SE for each signed SD
        sd_vals = []
        mean_probs_layer = []
        se_probs_layer = []
        mean_logits_layer = []
        se_logits_layer = []

        for ssd in unique_sds:
            mask = layer_signed_sds == ssd
            probs_ssd = layer_probs[mask]
            logits_ssd = layer_logits[mask]

            sd_vals.append(ssd)
            mean_probs_layer.append(np.mean(probs_ssd))
            se_probs_layer.append(np.std(probs_ssd) / np.sqrt(len(probs_ssd)) if len(probs_ssd) > 0 else 0)
            mean_logits_layer.append(np.mean(logits_ssd))
            se_logits_layer.append(np.std(logits_ssd) / np.sqrt(len(logits_ssd)) if len(logits_ssd) > 0 else 0)

        sd_vals = np.array(sd_vals)
        mean_probs_layer = np.array(mean_probs_layer)
        se_probs_layer = np.array(se_probs_layer)
        mean_logits_layer = np.array(mean_logits_layer)
        se_logits_layer = np.array(se_logits_layer)

        x_pos_layer = np.arange(len(sd_vals))
        colors_layer = plt.cm.coolwarm(np.linspace(0, 1, len(sd_vals)))
        x_labels_layer = [f'{int(sd):+d}' if sd != 0 else '0' for sd in sd_vals]

        # Logit bar chart for this layer
        fig_layer_logits, ax_layer_logits = plt.subplots(figsize=(12, 6), dpi=150)
        ax_layer_logits.bar(x_pos_layer, mean_logits_layer, yerr=se_logits_layer, capsize=5,
                            color=colors_layer, edgecolor='black', alpha=0.8)
        ax_layer_logits.axhline(y=0, color='gray', linestyle='-', linewidth=1)
        ax_layer_logits.set_xticks(x_pos_layer)
        ax_layer_logits.set_xticklabels(x_labels_layer)
        ax_layer_logits.set_xlabel('Signed Symbolic Distance')
        ax_layer_logits.set_ylabel('Mean Logit (via remaining layers)')
        ax_layer_logits.set_title(f'List Linking: {layer_name} Layer Readout - Logit\n(n={num_networks_to_analyze} networks, mean ± SE)')
        plt.tight_layout()
        figures[f"list_linking/layer{layer_idx}_readout_logit"] = fig_layer_logits
        plt.close(fig_layer_logits)

        # Probability bar chart for this layer
        fig_layer_probs, ax_layer_probs = plt.subplots(figsize=(12, 6), dpi=150)
        ax_layer_probs.bar(x_pos_layer, mean_probs_layer, yerr=se_probs_layer, capsize=5,
                           color=colors_layer, edgecolor='black', alpha=0.8)
        ax_layer_probs.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='Chance')
        ax_layer_probs.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        ax_layer_probs.axhline(y=1, color='gray', linestyle='-', linewidth=0.5)
        ax_layer_probs.set_xticks(x_pos_layer)
        ax_layer_probs.set_xticklabels(x_labels_layer)
        ax_layer_probs.set_xlabel('Signed Symbolic Distance')
        ax_layer_probs.set_ylabel('Mean Probability (via remaining layers)')
        ax_layer_probs.set_title(f'List Linking: {layer_name} Layer Readout - Probability\n(n={num_networks_to_analyze} networks, mean ± SE)')
        ax_layer_probs.set_ylim(0, 1)
        plt.tight_layout()
        figures[f"list_linking/layer{layer_idx}_readout_prob"] = fig_layer_probs
        plt.close(fig_layer_probs)

        # Cross-list only for this layer
        cross_mask_layer = layer_is_cross
        if np.sum(cross_mask_layer) > 0:
            cross_sds = layer_signed_sds[cross_mask_layer]
            cross_probs_l = layer_probs[cross_mask_layer]
            cross_logits_l = layer_logits[cross_mask_layer]

            unique_cross_sds_l = sorted(np.unique(cross_sds))
            cross_sd_vals = []
            cross_mean_probs_l = []
            cross_se_probs_l = []
            cross_mean_logits_l = []
            cross_se_logits_l = []

            for ssd in unique_cross_sds_l:
                mask = cross_sds == ssd
                cross_sd_vals.append(ssd)
                cross_mean_probs_l.append(np.mean(cross_probs_l[mask]))
                cross_se_probs_l.append(np.std(cross_probs_l[mask]) / np.sqrt(np.sum(mask)) if np.sum(mask) > 0 else 0)
                cross_mean_logits_l.append(np.mean(cross_logits_l[mask]))
                cross_se_logits_l.append(np.std(cross_logits_l[mask]) / np.sqrt(np.sum(mask)) if np.sum(mask) > 0 else 0)

            x_pos_cross_l = np.arange(len(cross_sd_vals))
            colors_cross_l = plt.cm.coolwarm(np.linspace(0, 1, len(cross_sd_vals)))

            # Cross-list logits
            fig_cross_logits_l, ax_cross_logits_l = plt.subplots(figsize=(12, 6), dpi=150)
            ax_cross_logits_l.bar(x_pos_cross_l, cross_mean_logits_l, yerr=cross_se_logits_l, capsize=5,
                                  color=colors_cross_l, edgecolor='black', alpha=0.8)
            ax_cross_logits_l.axhline(y=0, color='gray', linestyle='-', linewidth=1)
            ax_cross_logits_l.set_xticks(x_pos_cross_l)
            ax_cross_logits_l.set_xticklabels([f'{int(sd):+d}' if sd != 0 else '0' for sd in cross_sd_vals])
            ax_cross_logits_l.set_xlabel('Signed Symbolic Distance')
            ax_cross_logits_l.set_ylabel('Mean Logit (via remaining layers)')
            ax_cross_logits_l.set_title(f'List Linking: {layer_name} Readout - CROSS-LIST Logit\n(n={num_networks_to_analyze} networks)')
            plt.tight_layout()
            figures[f"list_linking/layer{layer_idx}_readout_logit_cross"] = fig_cross_logits_l
            plt.close(fig_cross_logits_l)

            # Cross-list probs
            fig_cross_probs_l, ax_cross_probs_l = plt.subplots(figsize=(12, 6), dpi=150)
            ax_cross_probs_l.bar(x_pos_cross_l, cross_mean_probs_l, yerr=cross_se_probs_l, capsize=5,
                                 color=colors_cross_l, edgecolor='black', alpha=0.8)
            ax_cross_probs_l.axhline(y=0.5, color='gray', linestyle='--', linewidth=1)
            ax_cross_probs_l.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
            ax_cross_probs_l.axhline(y=1, color='gray', linestyle='-', linewidth=0.5)
            ax_cross_probs_l.set_xticks(x_pos_cross_l)
            ax_cross_probs_l.set_xticklabels([f'{int(sd):+d}' if sd != 0 else '0' for sd in cross_sd_vals])
            ax_cross_probs_l.set_xlabel('Signed Symbolic Distance')
            ax_cross_probs_l.set_ylabel('Mean Probability (via remaining layers)')
            ax_cross_probs_l.set_title(f'List Linking: {layer_name} Readout - CROSS-LIST Probability\n(n={num_networks_to_analyze} networks)')
            ax_cross_probs_l.set_ylim(0, 1)
            plt.tight_layout()
            figures[f"list_linking/layer{layer_idx}_readout_prob_cross"] = fig_cross_probs_l
            plt.close(fig_cross_probs_l)

    # --- Item-readout correlation plots ---
    # For each layer, plot item index vs mean correlation with readout weights
    item_labels = [chr(ord('A') + i) for i in range(num_items)]  # A, B, C, D, E, F, G, H

    for layer_idx in range(num_layers):
        layer_name = layer_names[layer_idx] if layer_idx < len(layer_names) else f'Layer {layer_idx}'

        for position in ['item1', 'item2']:
            position_label = 'Position 1 (other zeroed)' if position == 'item1' else 'Position 2 (other zeroed)'

            # Compute mean and SE for each item
            mean_corrs = []
            se_corrs = []
            for item_idx in range(num_items):
                corrs = item_readout_correlations[position][layer_idx][item_idx]
                if len(corrs) > 0:
                    mean_corrs.append(np.mean(corrs))
                    se_corrs.append(np.std(corrs) / np.sqrt(len(corrs)))
                else:
                    mean_corrs.append(0)
                    se_corrs.append(0)

            mean_corrs = np.array(mean_corrs)
            se_corrs = np.array(se_corrs)

            # Create bar plot
            fig_corr, ax_corr = plt.subplots(figsize=(10, 6), dpi=150)

            x_pos = np.arange(num_items)
            # Color by list membership (items 0-3 = List 1, items 4-7 = List 2)
            colors_items = ['tab:blue'] * 4 + ['tab:orange'] * 4

            bars = ax_corr.bar(x_pos, mean_corrs, yerr=se_corrs, capsize=5,
                               color=colors_items, edgecolor='black', alpha=0.8)

            ax_corr.axhline(y=0, color='gray', linestyle='-', linewidth=1)
            ax_corr.set_xticks(x_pos)
            ax_corr.set_xticklabels(item_labels)
            ax_corr.set_xlabel('Item')
            ax_corr.set_ylabel('Correlation with Readout Weights')
            ax_corr.set_title(f'List Linking: {layer_name} - Item-Readout Correlation\n{position_label} (n={num_networks_to_analyze} networks, mean ± SE)')

            # Add legend for list membership
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='tab:blue', label='List 1 (ABCD)'),
                               Patch(facecolor='tab:orange', label='List 2 (EFGH)')]
            ax_corr.legend(handles=legend_elements, loc='best')

            plt.tight_layout()
            figures[f"list_linking/layer{layer_idx}_item_readout_corr_{position}"] = fig_corr
            plt.close(fig_corr)

        # Also create a combined plot showing both positions
        fig_combined, ax_combined = plt.subplots(figsize=(12, 6), dpi=150)

        x_pos = np.arange(num_items)
        width = 0.35

        # Position 1 data
        mean_corrs_p1 = []
        se_corrs_p1 = []
        for item_idx in range(num_items):
            corrs = item_readout_correlations['item1'][layer_idx][item_idx]
            mean_corrs_p1.append(np.mean(corrs) if len(corrs) > 0 else 0)
            se_corrs_p1.append(np.std(corrs) / np.sqrt(len(corrs)) if len(corrs) > 0 else 0)

        # Position 2 data
        mean_corrs_p2 = []
        se_corrs_p2 = []
        for item_idx in range(num_items):
            corrs = item_readout_correlations['item2'][layer_idx][item_idx]
            mean_corrs_p2.append(np.mean(corrs) if len(corrs) > 0 else 0)
            se_corrs_p2.append(np.std(corrs) / np.sqrt(len(corrs)) if len(corrs) > 0 else 0)

        ax_combined.bar(x_pos - width/2, mean_corrs_p1, width, yerr=se_corrs_p1, capsize=3,
                        label='Position 1', color='tab:blue', alpha=0.8)
        ax_combined.bar(x_pos + width/2, mean_corrs_p2, width, yerr=se_corrs_p2, capsize=3,
                        label='Position 2', color='tab:orange', alpha=0.8)

        ax_combined.axhline(y=0, color='gray', linestyle='-', linewidth=1)
        ax_combined.set_xticks(x_pos)
        ax_combined.set_xticklabels(item_labels)
        ax_combined.set_xlabel('Item')
        ax_combined.set_ylabel('Correlation with Readout Weights')
        ax_combined.set_title(f'List Linking: {layer_name} - Item-Readout Correlation by Position\n(n={num_networks_to_analyze} networks, mean ± SE)')
        ax_combined.legend()

        plt.tight_layout()
        figures[f"list_linking/layer{layer_idx}_item_readout_corr_combined"] = fig_combined
        plt.close(fig_combined)

    # --- Pair-readout correlation heatmaps ---
    # For each layer, plot heatmaps of correlation with readout for each ordered pair
    for layer_idx in range(num_layers):
        layer_name = layer_names[layer_idx] if layer_idx < len(layer_names) else f'Layer {layer_idx}'

        # Stack matrices across networks and compute mean/std
        corr_matrices = np.array(pair_readout_correlations[layer_idx])  # (num_networks, num_items, num_items)
        dot_matrices = np.array(pair_readout_dotproducts[layer_idx])

        # Compute mean and std (ignoring NaN for diagonal)
        mean_corr = np.nanmean(corr_matrices, axis=0)
        std_corr = np.nanstd(corr_matrices, axis=0)
        mean_dot = np.nanmean(dot_matrices, axis=0)
        std_dot = np.nanstd(dot_matrices, axis=0)

        # --- Correlation Mean Heatmap ---
        fig_corr_mean, ax_corr_mean = plt.subplots(figsize=(10, 8), dpi=150)
        vabs_corr = max(np.nanmax(np.abs(mean_corr)), 0.01)
        im_corr = ax_corr_mean.imshow(mean_corr, cmap='RdBu_r', vmin=-vabs_corr, vmax=vabs_corr, aspect='equal')
        ax_corr_mean.set_xticks(range(num_items))
        ax_corr_mean.set_yticks(range(num_items))
        ax_corr_mean.set_xticklabels(item_labels)
        ax_corr_mean.set_yticklabels(item_labels)
        ax_corr_mean.set_xlabel('Item 2 (second position)')
        ax_corr_mean.set_ylabel('Item 1 (first position)')
        ax_corr_mean.set_title(f'List Linking: {layer_name} - Pair-Readout Correlation (Mean)\n(n={num_networks_to_analyze} networks)')

        # Add list boundary lines
        ax_corr_mean.axhline(y=3.5, color='black', linestyle='-', linewidth=2)
        ax_corr_mean.axvline(x=3.5, color='black', linestyle='-', linewidth=2)

        # Add text annotations
        for i in range(num_items):
            for j in range(num_items):
                if not np.isnan(mean_corr[i, j]):
                    text_color = 'white' if np.abs(mean_corr[i, j]) > vabs_corr * 0.6 else 'black'
                    ax_corr_mean.text(j, i, f'{mean_corr[i, j]:.2f}', ha='center', va='center',
                                      fontsize=8, color=text_color)

        plt.colorbar(im_corr, ax=ax_corr_mean, label='Correlation')
        plt.tight_layout()
        figures[f"list_linking/layer{layer_idx}_pair_readout_corr_mean"] = fig_corr_mean
        plt.close(fig_corr_mean)

        # --- Correlation Std Heatmap ---
        fig_corr_std, ax_corr_std = plt.subplots(figsize=(10, 8), dpi=150)
        im_corr_std = ax_corr_std.imshow(std_corr, cmap='viridis', aspect='equal')
        ax_corr_std.set_xticks(range(num_items))
        ax_corr_std.set_yticks(range(num_items))
        ax_corr_std.set_xticklabels(item_labels)
        ax_corr_std.set_yticklabels(item_labels)
        ax_corr_std.set_xlabel('Item 2 (second position)')
        ax_corr_std.set_ylabel('Item 1 (first position)')
        ax_corr_std.set_title(f'List Linking: {layer_name} - Pair-Readout Correlation (Std)\n(n={num_networks_to_analyze} networks)')

        # Add list boundary lines
        ax_corr_std.axhline(y=3.5, color='white', linestyle='-', linewidth=2)
        ax_corr_std.axvline(x=3.5, color='white', linestyle='-', linewidth=2)

        # Add text annotations
        for i in range(num_items):
            for j in range(num_items):
                if not np.isnan(std_corr[i, j]):
                    ax_corr_std.text(j, i, f'{std_corr[i, j]:.2f}', ha='center', va='center',
                                     fontsize=8, color='white')

        plt.colorbar(im_corr_std, ax=ax_corr_std, label='Std Dev')
        plt.tight_layout()
        figures[f"list_linking/layer{layer_idx}_pair_readout_corr_std"] = fig_corr_std
        plt.close(fig_corr_std)

    model.train()
    return figures


def plot_innate_weight_analysis(args, model, task='ll'):
    """
    Analyze readout correlations using only innate weights (no plastic contribution).
    Also compute SVD analysis showing projections onto top singular vectors.

    Args:
        args: Training arguments
        model: Trained MLP model
        task: 'ti' for transitive inference, 'll' for list linking

    Returns:
        Dictionary of figures
    """
    from generate_data import generate_batch_items, generate_batch_trials_ti, generate_batch_trials_ll

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    figures = {}

    num_items = 8 if task == 'll' else args.num_items
    num_layers = args.extra_layers + 2  # embedding + extra layers + final

    # Use 128 networks or 4x batch_size, whichever is larger
    num_networks_to_analyze = max(128, 4 * args.batch_size)

    if task == 'll':
        item_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    else:
        item_labels = [chr(ord('A') + i) for i in range(num_items)]

    # Generate items and run trials to build plastic weights
    batch_items = generate_batch_items(num_items, args.item_size, num_networks_to_analyze, change_items_throughout_batch=True)

    if task == 'll':
        trials, correct_choices, pair_indices = generate_batch_trials_ll(
            batch_items, args.num_trials_list_1, args.num_trials_list_2, args.num_trials_linking_pair, num_test_trials=0
        )
    else:
        trials, correct_choices, pair_indices, _ = generate_batch_trials_ti(
            batch_items, args.num_train_trials // (num_items - 1), num_test_trials=0
        )

    trials = torch.tensor(trials, dtype=torch.float32).to(device)
    correct_choices = torch.tensor(correct_choices, dtype=torch.float32).to(device)

    # Initialize plastic weights
    plastic_weights = torch.zeros(num_networks_to_analyze, args.hidden_size, args.hidden_size,
                                  dtype=torch.float32, requires_grad=False).to(device)
    extra_plastic_weights = [torch.zeros(num_networks_to_analyze, args.hidden_size, args.hidden_size,
                                         dtype=torch.float32, requires_grad=False).to(device)
                            for _ in range(args.extra_layers)]

    # Run through training trials to build up plastic weights
    num_trials = trials.shape[1]

    # For list linking, save weights before linking trial DE is shown
    # Trial order: List 1 (num_trials_list_1), List 2 (num_trials_list_2), then DE (num_trials_linking_pair)
    pre_linking_plastic_weights = None
    pre_linking_extra_plastic_weights = None
    if task == 'll':
        # Calculate when DE trials start: after list 1 and list 2 trials
        pre_linking_trial_idx = args.num_trials_list_1 + args.num_trials_list_2  # Index where DE starts
        print(f"List linking trial structure: {args.num_trials_list_1} list1 + {args.num_trials_list_2} list2 + {args.num_trials_linking_pair} linking = {num_trials} total")
        print(f"Will save pre-linking weights after trial {pre_linking_trial_idx - 1} (before DE)")
    else:
        pre_linking_trial_idx = -1  # Not used for TI

    with torch.no_grad():
        for trial_idx in range(num_trials):
            # Save weights before the first linking trial (DE) for list linking
            if task == 'll' and trial_idx == pre_linking_trial_idx:
                pre_linking_plastic_weights = plastic_weights.clone()
                pre_linking_extra_plastic_weights = [epw.clone() for epw in extra_plastic_weights]
                print(f"Saved pre-linking weights at trial {trial_idx}")

            batch_trial = trials[:, trial_idx, :]
            batch_correct_choice = correct_choices[:, trial_idx]
            output = model(batch_trial, plastic_weights, batch_correct_choice,
                          extra_plastic_weights=extra_plastic_weights)
            plastic_weights = output.plastic_weights
            extra_plastic_weights = output.extra_plastic_weights

    frozen_plastic_weights = plastic_weights.clone()
    frozen_extra_plastic_weights = [epw.clone() for epw in extra_plastic_weights]

    # Get readout weights
    readout_weights = model.choice.weight.detach().cpu().numpy().squeeze()  # (hidden_size,)

    # Get innate weight matrices for each layer
    # Layer 0: embedding_layer (input -> hidden)
    # Layer 1 to extra_layers: extra_hidden_layers (hidden -> hidden)
    # Final layer: fc2 (hidden -> hidden)

    innate_weights = []
    layer_names = []

    # Embedding layer
    if hasattr(model, 'embedding_layer'):
        innate_weights.append(model.embedding_layer.weight.detach().cpu().numpy())  # (hidden_size, input_size)
        layer_names.append('Embedding')

    # Extra hidden layers
    for i in range(args.extra_layers):
        innate_weights.append(model.extra_hidden_layers[i].weight.detach().cpu().numpy())  # (hidden_size, hidden_size)
        layer_names.append(f'Hidden {i+1}')

    # Final layer (fc2)
    innate_weights.append(model.fc2.weight.detach().cpu().numpy())  # (hidden_size, hidden_size)
    layer_names.append('Final')

    # Storage for innate-only readout correlations
    innate_readout_correlations = {
        'item1': {layer_idx: {item_idx: [] for item_idx in range(num_items)} for layer_idx in range(num_layers)},
        'item2': {layer_idx: {item_idx: [] for item_idx in range(num_items)} for layer_idx in range(num_layers)},
    }

    # Storage for SVD projections (top k singular vectors)
    num_svd_components = 5
    # Right singular vectors (V) - input directions: h · v_k
    svd_projections_V = {
        'item1': {layer_idx: {k: {item_idx: [] for item_idx in range(num_items)} for k in range(num_svd_components)} for layer_idx in range(1, num_layers)},
        'item2': {layer_idx: {k: {item_idx: [] for item_idx in range(num_items)} for k in range(num_svd_components)} for layer_idx in range(1, num_layers)},
    }
    # Left singular vectors (U) - output directions: (W @ h) · u_k
    svd_projections_U = {
        'item1': {layer_idx: {k: {item_idx: [] for item_idx in range(num_items)} for k in range(num_svd_components)} for layer_idx in range(1, num_layers)},
        'item2': {layer_idx: {k: {item_idx: [] for item_idx in range(num_items)} for k in range(num_svd_components)} for layer_idx in range(1, num_layers)},
    }

    # Storage for projection magnitudes (||W @ h||) - will be converted to signed projections
    projection_magnitudes = {
        'item1': {layer_idx: {item_idx: [] for item_idx in range(num_items)} for layer_idx in range(1, num_layers)},
        'item2': {layer_idx: {item_idx: [] for item_idx in range(num_items)} for layer_idx in range(1, num_layers)},
    }

    # Storage for innate projection vectors (W @ h) - needed for computing signed projections
    innate_projection_vectors = {
        'item1': {layer_idx: {item_idx: [] for item_idx in range(num_items)} for layer_idx in range(1, num_layers)},
        'item2': {layer_idx: {item_idx: [] for item_idx in range(num_items)} for layer_idx in range(1, num_layers)},
    }

    # Storage for full layer outputs (innate + plastic, or just innate for embedding)
    # These are the actual hidden representations at each layer
    full_layer_outputs = {
        'item1': {layer_idx: {item_idx: [] for item_idx in range(num_items)} for layer_idx in range(num_layers)},
        'item2': {layer_idx: {item_idx: [] for item_idx in range(num_items)} for layer_idx in range(num_layers)},
    }

    # Compute SVD of each layer's innate weights (except embedding layer which has different input size)
    svd_results = {}
    for layer_idx in range(1, num_layers):
        W = innate_weights[layer_idx]
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        svd_results[layer_idx] = {'U': U, 'S': S, 'Vt': Vt}

    print(f"Computing innate-only readout correlations for {num_networks_to_analyze} networks ({task.upper()})...")

    # Get alpha values for plastic weight scaling
    alpha_extra_np = [model.alpha_extra[i].detach().cpu().numpy() for i in range(args.extra_layers)]
    alpha_final_np = model.alpha.detach().cpu().numpy()

    with torch.no_grad():
        for network_idx in range(num_networks_to_analyze):
            single_items = batch_items[network_idx]  # (num_items, item_size)
            single_pw = frozen_plastic_weights[network_idx].cpu().numpy()  # (hidden_size, hidden_size)
            single_epw = [epw[network_idx].cpu().numpy() for epw in frozen_extra_plastic_weights]

            for item_idx in range(num_items):
                item_emb = single_items[item_idx]  # (item_size,)
                zero_emb = np.zeros_like(item_emb)

                # Position 1: item in first position, zeros in second
                input_pos1 = np.concatenate([item_emb, zero_emb])
                # Position 2: zeros in first position, item in second
                input_pos2 = np.concatenate([zero_emb, item_emb])

                for position, input_vec in [('item1', input_pos1), ('item2', input_pos2)]:
                    # Layer 0: embedding (no plastic weights here)
                    h = np.tanh(innate_weights[0] @ input_vec)

                    # Store embedding layer output
                    full_layer_outputs[position][0][item_idx].append(h.copy())

                    # Compute correlation with readout for embedding layer
                    if np.std(h) > 1e-10:
                        corr = np.corrcoef(h, readout_weights)[0, 1]
                    else:
                        corr = 0.0
                    innate_readout_correlations[position][0][item_idx].append(corr)

                    # Process through extra hidden layers WITH plastic weights
                    for layer_idx in range(1, num_layers - 1):  # All layers except final
                        extra_layer_idx = layer_idx - 1
                        W_innate = innate_weights[layer_idx]
                        W_plastic = single_epw[extra_layer_idx]
                        alpha = alpha_extra_np[extra_layer_idx]

                        # Full forward pass: innate + plastic (no bias)
                        innate_contrib = W_innate @ h
                        plastic_contrib = (alpha * W_plastic) @ h
                        h_full = np.tanh(innate_contrib + plastic_contrib)

                        # Store full layer output (innate + plastic)
                        full_layer_outputs[position][layer_idx][item_idx].append(h_full.copy())

                        # But also compute innate-only for correlation analysis
                        h_innate_only = np.tanh(innate_contrib)

                        # Store innate projection vector (for computing signed projection later)
                        innate_projection_vectors[position][layer_idx][item_idx].append(innate_contrib.copy())

                        # Compute projections onto top singular vectors
                        Vt = svd_results[layer_idx]['Vt']
                        U = svd_results[layer_idx]['U']
                        for k in range(num_svd_components):
                            # Right singular vectors (V) - input directions: h · v_k
                            v_k = Vt[k, :]
                            proj_v_k = np.dot(h, v_k)  # h is the input to this layer
                            svd_projections_V[position][layer_idx][k][item_idx].append(proj_v_k)

                            # Left singular vectors (U) - output directions: (W @ h) · u_k
                            u_k = U[:, k]
                            proj_u_k = np.dot(innate_contrib, u_k)  # innate_contrib = W @ h
                            svd_projections_U[position][layer_idx][k][item_idx].append(proj_u_k)

                        # Compute correlation with readout (innate-only output)
                        if np.std(h_innate_only) > 1e-10:
                            corr = np.corrcoef(h_innate_only, readout_weights)[0, 1]
                        else:
                            corr = 0.0
                        innate_readout_correlations[position][layer_idx][item_idx].append(corr)

                        # Continue forward pass with full (plastic-enriched) representation
                        h = h_full

                    # Final layer: compute innate-only readout correlation
                    # h now contains the output of layer 1 (with plastic weights)
                    final_layer_idx = num_layers - 1
                    W_innate_final = innate_weights[final_layer_idx]

                    # Compute full final layer output (innate + plastic, no bias)
                    innate_contrib_final = W_innate_final @ h
                    plastic_contrib_final = (alpha_final_np * single_pw) @ h
                    h_final_full = np.tanh(innate_contrib_final + plastic_contrib_final)

                    # Store full final layer output
                    full_layer_outputs[position][final_layer_idx][item_idx].append(h_final_full.copy())

                    # Also compute innate-only for correlation analysis
                    h_final_innate_only = np.tanh(innate_contrib_final)

                    # Store innate projection vector (for computing signed projection later)
                    innate_projection_vectors[position][final_layer_idx][item_idx].append(innate_contrib_final.copy())

                    # SVD projections
                    Vt = svd_results[final_layer_idx]['Vt']
                    U = svd_results[final_layer_idx]['U']
                    for k in range(num_svd_components):
                        # Right singular vectors (V) - input directions: h · v_k
                        v_k = Vt[k, :]
                        proj_v_k = np.dot(h, v_k)
                        svd_projections_V[position][final_layer_idx][k][item_idx].append(proj_v_k)

                        # Left singular vectors (U) - output directions: (W @ h) · u_k
                        u_k = U[:, k]
                        proj_u_k = np.dot(innate_contrib_final, u_k)
                        svd_projections_U[position][final_layer_idx][k][item_idx].append(proj_u_k)

                    # Correlation with readout (innate-only at final layer)
                    if np.std(h_final_innate_only) > 1e-10:
                        corr = np.corrcoef(h_final_innate_only, readout_weights)[0, 1]
                    else:
                        corr = 0.0
                    innate_readout_correlations[position][final_layer_idx][item_idx].append(corr)

    # --- Compute signed projections using PC1 as reference ---
    # For each layer, compute PC1 of all innate projection vectors, then compute signed projection
    print(f"Computing signed projections using PC1 as reference ({task.upper()})...")

    for layer_idx in range(1, num_layers):
        # Collect all vectors for this layer (across both positions and all items)
        all_vectors = []
        for position in ['item1', 'item2']:
            for item_idx in range(num_items):
                vectors = innate_projection_vectors[position][layer_idx][item_idx]
                all_vectors.extend(vectors)

        if len(all_vectors) > 0:
            # Stack into matrix and compute PCA
            all_vectors_matrix = np.array(all_vectors)  # (n_samples, hidden_size)
            # Center the data
            mean_vec = np.mean(all_vectors_matrix, axis=0)
            centered = all_vectors_matrix - mean_vec
            # Compute covariance and get first eigenvector (PC1)
            cov_matrix = np.cov(centered, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            # Sort by eigenvalue (descending) and get PC1
            sort_idx = np.argsort(eigenvalues)[::-1]
            pc1 = eigenvectors[:, sort_idx[0]]  # First principal component
            pc1_norm = np.linalg.norm(pc1)

            # Compute signed projections for each item
            for position in ['item1', 'item2']:
                for item_idx in range(num_items):
                    vectors = innate_projection_vectors[position][layer_idx][item_idx]
                    signed_projs = []
                    for vec in vectors:
                        # Signed projection: (pc1 · vec) / ||pc1||
                        signed_proj = np.dot(pc1, vec) / pc1_norm if pc1_norm > 1e-10 else 0.0
                        signed_projs.append(signed_proj)
                    projection_magnitudes[position][layer_idx][item_idx] = signed_projs

    # --- Compute signed projections for full layer outputs (innate + plastic) ---
    # For each layer, compute PC1 of all full layer outputs, then compute signed projection
    print(f"Computing signed projections of full outputs using PC1 as reference ({task.upper()})...")

    # Storage for signed projections of full outputs
    full_output_signed_projections = {
        'item1': {layer_idx: {item_idx: [] for item_idx in range(num_items)} for layer_idx in range(num_layers)},
        'item2': {layer_idx: {item_idx: [] for item_idx in range(num_items)} for layer_idx in range(num_layers)},
    }

    full_output_pc1_vectors = {}  # layer_idx -> PC1 vector (for external use)

    for layer_idx in range(num_layers):
        # Collect all vectors for this layer (across both positions and all items)
        all_vectors = []
        for position in ['item1', 'item2']:
            for item_idx in range(num_items):
                vectors = full_layer_outputs[position][layer_idx][item_idx]
                all_vectors.extend(vectors)

        if len(all_vectors) > 0:
            # Stack into matrix and compute PCA
            all_vectors_matrix = np.array(all_vectors)  # (n_samples, hidden_size)
            # Center the data
            mean_vec = np.mean(all_vectors_matrix, axis=0)
            centered = all_vectors_matrix - mean_vec
            # Compute covariance and get first eigenvector (PC1)
            cov_matrix = np.cov(centered, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            # Sort by eigenvalue (descending) and get PC1
            sort_idx = np.argsort(eigenvalues)[::-1]
            pc1 = eigenvectors[:, sort_idx[0]]  # First principal component
            pc1_norm = np.linalg.norm(pc1)

            full_output_pc1_vectors[layer_idx] = pc1.copy()

            # Compute signed projections for each item
            for position in ['item1', 'item2']:
                for item_idx in range(num_items):
                    vectors = full_layer_outputs[position][layer_idx][item_idx]
                    signed_projs = []
                    for vec in vectors:
                        # Signed projection: (pc1 · vec) / ||pc1||
                        signed_proj = np.dot(pc1, vec) / pc1_norm if pc1_norm > 1e-10 else 0.0
                        signed_projs.append(signed_proj)
                    full_output_signed_projections[position][layer_idx][item_idx] = signed_projs

    task_prefix = "list_linking" if task == 'll' else "pca_frozen"

    # --- Plot 2: Projection magnitudes for each layer (except embedding) ---
    for layer_idx in range(1, num_layers):
        layer_name = layer_names[layer_idx]

        fig_mag, ax_mag = plt.subplots(figsize=(12, 6), dpi=150)

        width = 0.35
        x_pos = np.arange(num_items)

        mean_mags_p1 = []
        se_mags_p1 = []
        for item_idx in range(num_items):
            mags = projection_magnitudes['item1'][layer_idx][item_idx]
            mean_mags_p1.append(np.mean(mags) if len(mags) > 0 else 0)
            se_mags_p1.append(np.std(mags) / np.sqrt(len(mags)) if len(mags) > 0 else 0)

        mean_mags_p2 = []
        se_mags_p2 = []
        for item_idx in range(num_items):
            mags = projection_magnitudes['item2'][layer_idx][item_idx]
            mean_mags_p2.append(np.mean(mags) if len(mags) > 0 else 0)
            se_mags_p2.append(np.std(mags) / np.sqrt(len(mags)) if len(mags) > 0 else 0)

        ax_mag.bar(x_pos - width/2, mean_mags_p1, width, yerr=se_mags_p1, capsize=3,
                  label='Position 1', color='tab:blue', alpha=0.8)
        ax_mag.bar(x_pos + width/2, mean_mags_p2, width, yerr=se_mags_p2, capsize=3,
                  label='Position 2', color='tab:orange', alpha=0.8)

        ax_mag.set_xlabel('Item')
        ax_mag.set_ylabel('Signed Projection onto PC1')
        ax_mag.set_title(f'{task.upper()}: {layer_name} - Innate Projection (Signed by PC1)\n(n={num_networks_to_analyze} networks)')
        ax_mag.set_xticks(x_pos)
        ax_mag.set_xticklabels(item_labels[:num_items])
        ax_mag.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax_mag.legend()

        plt.tight_layout()
        figures[f"{task_prefix}/layer{layer_idx}_innate_projection_magnitude"] = fig_mag
        plt.close(fig_mag)

    # --- Plot 2b: Signed projections of full layer outputs (innate + plastic) onto PC1 ---
    for layer_idx in range(num_layers):
        layer_name = layer_names[layer_idx]

        fig_full, ax_full = plt.subplots(figsize=(12, 6), dpi=150)

        width = 0.35
        x_pos = np.arange(num_items)

        mean_proj_p1 = []
        se_proj_p1 = []
        for item_idx in range(num_items):
            projs = full_output_signed_projections['item1'][layer_idx][item_idx]
            mean_proj_p1.append(np.mean(projs) if len(projs) > 0 else 0)
            se_proj_p1.append(np.std(projs) / np.sqrt(len(projs)) if len(projs) > 0 else 0)

        mean_proj_p2 = []
        se_proj_p2 = []
        for item_idx in range(num_items):
            projs = full_output_signed_projections['item2'][layer_idx][item_idx]
            mean_proj_p2.append(np.mean(projs) if len(projs) > 0 else 0)
            se_proj_p2.append(np.std(projs) / np.sqrt(len(projs)) if len(projs) > 0 else 0)

        ax_full.bar(x_pos - width/2, mean_proj_p1, width, yerr=se_proj_p1, capsize=3,
                   label='Position 1', color='tab:blue', alpha=0.8)
        ax_full.bar(x_pos + width/2, mean_proj_p2, width, yerr=se_proj_p2, capsize=3,
                   label='Position 2', color='tab:orange', alpha=0.8)

        ax_full.set_xlabel('Item')
        ax_full.set_ylabel('Signed Projection onto PC1')
        layer_desc = "(innate only)" if layer_idx == 0 else "(innate + plastic)"
        ax_full.set_title(f'{task.upper()}: {layer_name} - Full Output Projection {layer_desc}\n(n={num_networks_to_analyze} networks)')
        ax_full.set_xticks(x_pos)
        ax_full.set_xticklabels(item_labels[:num_items])
        ax_full.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax_full.legend()

        plt.tight_layout()
        figures[f"{task_prefix}/layer{layer_idx}_full_output_pc1_projection"] = fig_full
        plt.close(fig_full)

    # --- Plot 2c: Pre-linking variant for list linking (after 20 trials, before DE) ---
    if task == 'll' and pre_linking_plastic_weights is not None:
        print(f"Computing pre-linking projections for list linking (after {pre_linking_trial_idx} trials, before DE)...")

        # Storage for pre-linking full layer outputs
        pre_linking_full_layer_outputs = {
            'item1': {layer_idx: {item_idx: [] for item_idx in range(num_items)} for layer_idx in range(num_layers)},
            'item2': {layer_idx: {item_idx: [] for item_idx in range(num_items)} for layer_idx in range(num_layers)},
        }

        # Compute full layer outputs with pre-linking weights
        with torch.no_grad():
            for network_idx in range(num_networks_to_analyze):
                single_items = batch_items[network_idx]
                single_pw = pre_linking_plastic_weights[network_idx].cpu().numpy()
                single_epw = [epw[network_idx].cpu().numpy() for epw in pre_linking_extra_plastic_weights]

                for item_idx in range(num_items):
                    item_emb = single_items[item_idx]
                    zero_emb = np.zeros_like(item_emb)

                    input_pos1 = np.concatenate([item_emb, zero_emb])
                    input_pos2 = np.concatenate([zero_emb, item_emb])

                    for position, input_vec in [('item1', input_pos1), ('item2', input_pos2)]:
                        # Layer 0: embedding (no plastic weights)
                        h = np.tanh(innate_weights[0] @ input_vec)
                        pre_linking_full_layer_outputs[position][0][item_idx].append(h.copy())

                        # Process through extra hidden layers WITH pre-linking plastic weights
                        for layer_idx in range(1, num_layers - 1):
                            extra_layer_idx = layer_idx - 1
                            W_innate = innate_weights[layer_idx]
                            W_plastic = single_epw[extra_layer_idx]
                            alpha = alpha_extra_np[extra_layer_idx]

                            innate_contrib = W_innate @ h
                            plastic_contrib = (alpha * W_plastic) @ h
                            h_full = np.tanh(innate_contrib + plastic_contrib)

                            pre_linking_full_layer_outputs[position][layer_idx][item_idx].append(h_full.copy())
                            h = h_full

                        # Final layer
                        final_layer_idx = num_layers - 1
                        W_innate_final = innate_weights[final_layer_idx]
                        innate_contrib_final = W_innate_final @ h
                        plastic_contrib_final = (alpha_final_np * single_pw) @ h
                        h_final_full = np.tanh(innate_contrib_final + plastic_contrib_final)

                        pre_linking_full_layer_outputs[position][final_layer_idx][item_idx].append(h_final_full.copy())

        # Compute signed projections for pre-linking outputs
        pre_linking_signed_projections = {
            'item1': {layer_idx: {item_idx: [] for item_idx in range(num_items)} for layer_idx in range(num_layers)},
            'item2': {layer_idx: {item_idx: [] for item_idx in range(num_items)} for layer_idx in range(num_layers)},
        }

        for layer_idx in range(num_layers):
            all_vectors = []
            for position in ['item1', 'item2']:
                for item_idx in range(num_items):
                    vectors = pre_linking_full_layer_outputs[position][layer_idx][item_idx]
                    all_vectors.extend(vectors)

            if len(all_vectors) > 0:
                all_vectors_matrix = np.array(all_vectors)
                mean_vec = np.mean(all_vectors_matrix, axis=0)
                centered = all_vectors_matrix - mean_vec
                cov_matrix = np.cov(centered, rowvar=False)
                eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
                sort_idx = np.argsort(eigenvalues)[::-1]
                pc1 = eigenvectors[:, sort_idx[0]]
                pc1_norm = np.linalg.norm(pc1)

                for position in ['item1', 'item2']:
                    for item_idx in range(num_items):
                        vectors = pre_linking_full_layer_outputs[position][layer_idx][item_idx]
                        signed_projs = []
                        for vec in vectors:
                            signed_proj = np.dot(pc1, vec) / pc1_norm if pc1_norm > 1e-10 else 0.0
                            signed_projs.append(signed_proj)
                        pre_linking_signed_projections[position][layer_idx][item_idx] = signed_projs

        # Create pre-linking plots
        for layer_idx in range(num_layers):
            layer_name = layer_names[layer_idx]

            fig_pre, ax_pre = plt.subplots(figsize=(12, 6), dpi=150)

            width = 0.35
            x_pos = np.arange(num_items)

            mean_proj_p1 = []
            se_proj_p1 = []
            for item_idx in range(num_items):
                projs = pre_linking_signed_projections['item1'][layer_idx][item_idx]
                mean_proj_p1.append(np.mean(projs) if len(projs) > 0 else 0)
                se_proj_p1.append(np.std(projs) / np.sqrt(len(projs)) if len(projs) > 0 else 0)

            mean_proj_p2 = []
            se_proj_p2 = []
            for item_idx in range(num_items):
                projs = pre_linking_signed_projections['item2'][layer_idx][item_idx]
                mean_proj_p2.append(np.mean(projs) if len(projs) > 0 else 0)
                se_proj_p2.append(np.std(projs) / np.sqrt(len(projs)) if len(projs) > 0 else 0)

            ax_pre.bar(x_pos - width/2, mean_proj_p1, width, yerr=se_proj_p1, capsize=3,
                      label='Position 1', color='tab:blue', alpha=0.8)
            ax_pre.bar(x_pos + width/2, mean_proj_p2, width, yerr=se_proj_p2, capsize=3,
                      label='Position 2', color='tab:orange', alpha=0.8)

            ax_pre.set_xlabel('Item')
            ax_pre.set_ylabel('Signed Projection onto PC1')
            layer_desc = "(innate only)" if layer_idx == 0 else "(innate + plastic)"
            ax_pre.set_title(f'LL PRE-LINKING: {layer_name} - Full Output Projection {layer_desc}\n(after {pre_linking_trial_idx} trials, BEFORE DE; n={num_networks_to_analyze} networks)')
            ax_pre.set_xticks(x_pos)
            ax_pre.set_xticklabels(item_labels[:num_items])
            ax_pre.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax_pre.legend()

            plt.tight_layout()
            figures[f"{task_prefix}/layer{layer_idx}_full_output_pc1_projection_pre_linking"] = fig_pre
            plt.close(fig_pre)

    # --- Plot 3: SVD projections onto top right singular vectors (V - input directions) ---
    for layer_idx in range(1, num_layers):
        layer_name = layer_names[layer_idx]
        S = svd_results[layer_idx]['S']

        for k in range(min(num_svd_components, 3)):  # Plot top 3 singular vectors
            fig_svd, ax_svd = plt.subplots(figsize=(12, 6), dpi=150)

            width = 0.35
            x_pos = np.arange(num_items)

            mean_proj_p1 = []
            se_proj_p1 = []
            for item_idx in range(num_items):
                projs = svd_projections_V['item1'][layer_idx][k][item_idx]
                mean_proj_p1.append(np.mean(projs) if len(projs) > 0 else 0)
                se_proj_p1.append(np.std(projs) / np.sqrt(len(projs)) if len(projs) > 0 else 0)

            mean_proj_p2 = []
            se_proj_p2 = []
            for item_idx in range(num_items):
                projs = svd_projections_V['item2'][layer_idx][k][item_idx]
                mean_proj_p2.append(np.mean(projs) if len(projs) > 0 else 0)
                se_proj_p2.append(np.std(projs) / np.sqrt(len(projs)) if len(projs) > 0 else 0)

            ax_svd.bar(x_pos - width/2, mean_proj_p1, width, yerr=se_proj_p1, capsize=3,
                      label='Position 1', color='tab:blue', alpha=0.8)
            ax_svd.bar(x_pos + width/2, mean_proj_p2, width, yerr=se_proj_p2, capsize=3,
                      label='Position 2', color='tab:orange', alpha=0.8)

            # Calculate variance explained by this singular value
            var_explained = (S[k]**2 / np.sum(S**2)) * 100

            ax_svd.set_xlabel('Item')
            ax_svd.set_ylabel(f'Projection onto v_{k+1} (input dir)')
            ax_svd.set_title(f'{task.upper()}: {layer_name} - Input Projection h·v_{k+1}\n(σ={S[k]:.2f}, {var_explained:.1f}% var, n={num_networks_to_analyze} networks)')
            ax_svd.set_xticks(x_pos)
            ax_svd.set_xticklabels(item_labels[:num_items])
            ax_svd.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax_svd.legend()

            plt.tight_layout()
            figures[f"{task_prefix}/layer{layer_idx}_svd_V_projection_v{k+1}"] = fig_svd
            plt.close(fig_svd)

    # --- Plot 4: SVD projections onto top left singular vectors (U - output directions) ---
    for layer_idx in range(1, num_layers):
        layer_name = layer_names[layer_idx]
        S = svd_results[layer_idx]['S']

        for k in range(min(num_svd_components, 3)):  # Plot top 3 singular vectors
            fig_svd, ax_svd = plt.subplots(figsize=(12, 6), dpi=150)

            width = 0.35
            x_pos = np.arange(num_items)

            mean_proj_p1 = []
            se_proj_p1 = []
            for item_idx in range(num_items):
                projs = svd_projections_U['item1'][layer_idx][k][item_idx]
                mean_proj_p1.append(np.mean(projs) if len(projs) > 0 else 0)
                se_proj_p1.append(np.std(projs) / np.sqrt(len(projs)) if len(projs) > 0 else 0)

            mean_proj_p2 = []
            se_proj_p2 = []
            for item_idx in range(num_items):
                projs = svd_projections_U['item2'][layer_idx][k][item_idx]
                mean_proj_p2.append(np.mean(projs) if len(projs) > 0 else 0)
                se_proj_p2.append(np.std(projs) / np.sqrt(len(projs)) if len(projs) > 0 else 0)

            ax_svd.bar(x_pos - width/2, mean_proj_p1, width, yerr=se_proj_p1, capsize=3,
                      label='Position 1', color='tab:blue', alpha=0.8)
            ax_svd.bar(x_pos + width/2, mean_proj_p2, width, yerr=se_proj_p2, capsize=3,
                      label='Position 2', color='tab:orange', alpha=0.8)

            # Calculate variance explained by this singular value
            var_explained = (S[k]**2 / np.sum(S**2)) * 100

            ax_svd.set_xlabel('Item')
            ax_svd.set_ylabel(f'Projection onto u_{k+1} (output dir)')
            ax_svd.set_title(f'{task.upper()}: {layer_name} - Output Projection (Wh)·u_{k+1}\n(σ={S[k]:.2f}, {var_explained:.1f}% var, n={num_networks_to_analyze} networks)')
            ax_svd.set_xticks(x_pos)
            ax_svd.set_xticklabels(item_labels[:num_items])
            ax_svd.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax_svd.legend()

            plt.tight_layout()
            figures[f"{task_prefix}/layer{layer_idx}_svd_U_projection_u{k+1}"] = fig_svd
            plt.close(fig_svd)

    # --- Plot 5: Summary of rank correlations for V (input directions) ---
    for layer_idx in range(1, num_layers):
        layer_name = layer_names[layer_idx]

        fig_rank_corr_V, ax_rank_corr_V = plt.subplots(figsize=(10, 6), dpi=150)

        ranks = np.arange(num_items)  # Item rank (0 = best, num_items-1 = worst)

        sv_rank_corrs_p1 = []
        sv_rank_corrs_p2 = []

        for k in range(num_svd_components):
            # Get mean projections for each item
            mean_proj_p1 = [np.mean(svd_projections_V['item1'][layer_idx][k][item_idx])
                           for item_idx in range(num_items)]
            mean_proj_p2 = [np.mean(svd_projections_V['item2'][layer_idx][k][item_idx])
                           for item_idx in range(num_items)]

            # Correlate with rank
            corr_p1 = np.corrcoef(mean_proj_p1, ranks)[0, 1] if np.std(mean_proj_p1) > 1e-10 else 0
            corr_p2 = np.corrcoef(mean_proj_p2, ranks)[0, 1] if np.std(mean_proj_p2) > 1e-10 else 0

            sv_rank_corrs_p1.append(corr_p1)
            sv_rank_corrs_p2.append(corr_p2)

        x_sv = np.arange(num_svd_components)
        width = 0.35
        ax_rank_corr_V.bar(x_sv - width/2, sv_rank_corrs_p1, width, label='Position 1', color='tab:blue', alpha=0.8)
        ax_rank_corr_V.bar(x_sv + width/2, sv_rank_corrs_p2, width, label='Position 2', color='tab:orange', alpha=0.8)

        ax_rank_corr_V.set_xlabel('Right Singular Vector Index (V)')
        ax_rank_corr_V.set_ylabel('Correlation with Item Rank')
        ax_rank_corr_V.set_title(f'{task.upper()}: {layer_name} - Input Direction (h·v) vs Rank\n(n={num_networks_to_analyze} networks)')
        ax_rank_corr_V.set_xticks(x_sv)
        ax_rank_corr_V.set_xticklabels([f'v{k+1}' for k in range(num_svd_components)])
        ax_rank_corr_V.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax_rank_corr_V.legend()

        plt.tight_layout()
        figures[f"{task_prefix}/layer{layer_idx}_svd_V_rank_correlation"] = fig_rank_corr_V
        plt.close(fig_rank_corr_V)

    # --- Plot 6: Summary of rank correlations for U (output directions) ---
    for layer_idx in range(1, num_layers):
        layer_name = layer_names[layer_idx]

        fig_rank_corr_U, ax_rank_corr_U = plt.subplots(figsize=(10, 6), dpi=150)

        ranks = np.arange(num_items)

        sv_rank_corrs_p1 = []
        sv_rank_corrs_p2 = []

        for k in range(num_svd_components):
            # Get mean projections for each item
            mean_proj_p1 = [np.mean(svd_projections_U['item1'][layer_idx][k][item_idx])
                           for item_idx in range(num_items)]
            mean_proj_p2 = [np.mean(svd_projections_U['item2'][layer_idx][k][item_idx])
                           for item_idx in range(num_items)]

            # Correlate with rank
            corr_p1 = np.corrcoef(mean_proj_p1, ranks)[0, 1] if np.std(mean_proj_p1) > 1e-10 else 0
            corr_p2 = np.corrcoef(mean_proj_p2, ranks)[0, 1] if np.std(mean_proj_p2) > 1e-10 else 0

            sv_rank_corrs_p1.append(corr_p1)
            sv_rank_corrs_p2.append(corr_p2)

        x_sv = np.arange(num_svd_components)
        width = 0.35
        ax_rank_corr_U.bar(x_sv - width/2, sv_rank_corrs_p1, width, label='Position 1', color='tab:blue', alpha=0.8)
        ax_rank_corr_U.bar(x_sv + width/2, sv_rank_corrs_p2, width, label='Position 2', color='tab:orange', alpha=0.8)

        ax_rank_corr_U.set_xlabel('Left Singular Vector Index (U)')
        ax_rank_corr_U.set_ylabel('Correlation with Item Rank')
        ax_rank_corr_U.set_title(f'{task.upper()}: {layer_name} - Output Direction (Wh·u) vs Rank\n(n={num_networks_to_analyze} networks)')
        ax_rank_corr_U.set_xticks(x_sv)
        ax_rank_corr_U.set_xticklabels([f'u{k+1}' for k in range(num_svd_components)])
        ax_rank_corr_U.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax_rank_corr_U.legend()

        plt.tight_layout()
        figures[f"{task_prefix}/layer{layer_idx}_svd_U_rank_correlation"] = fig_rank_corr_U
        plt.close(fig_rank_corr_U)

    model.train()
    return figures, full_output_pc1_vectors


def plot_pair_pca_by_choice(args, model, task='ll'):
    """
    Create PCA plots of pair embeddings at each layer, colored by:
    1. Model's choice (0 or 1)
    2. Correct answer (0 or 1)

    Args:
        args: Training arguments
        model: Trained MLP model
        task: 'ti' for transitive inference, 'll' for list linking

    Returns:
        Dictionary of figures
    """
    from generate_data import generate_batch_items, generate_batch_trials_ti, generate_batch_trials_ll
    from sklearn.decomposition import PCA

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    figures = {}

    num_items = 8 if task == 'll' else args.num_items
    num_layers = args.extra_layers + 2  # embedding + extra layers + final
    num_networks = min(128, 4 * args.batch_size)

    layer_names = ['Embedding']
    for i in range(args.extra_layers):
        layer_names.append(f'Hidden {i+1}')
    layer_names.append('Final')

    task_prefix = "list_linking" if task == 'll' else "pca_frozen"

    # Generate items and trials
    batch_items = generate_batch_items(num_items, args.item_size, num_networks, change_items_throughout_batch=True)

    if task == 'll':
        trials, correct_choices, pair_indices = generate_batch_trials_ll(
            batch_items, args.num_trials_list_1, args.num_trials_list_2, args.num_trials_linking_pair, num_test_trials=0
        )
    else:
        trials, correct_choices, pair_indices, _ = generate_batch_trials_ti(
            batch_items, args.num_train_trials // (num_items - 1), num_test_trials=0
        )

    trials = torch.tensor(trials, dtype=torch.float32).to(device)
    correct_choices_t = torch.tensor(correct_choices, dtype=torch.float32).to(device)

    # Initialize plastic weights
    plastic_weights = torch.zeros(num_networks, args.hidden_size, args.hidden_size,
                                  dtype=torch.float32, requires_grad=False).to(device)
    extra_plastic_weights = [torch.zeros(num_networks, args.hidden_size, args.hidden_size,
                                         dtype=torch.float32, requires_grad=False).to(device)
                            for _ in range(args.extra_layers)]

    # Storage for embeddings, choices, and correct answers
    # layer_idx -> list of (embedding, model_choice, correct_choice)
    all_embeddings = {layer_idx: [] for layer_idx in range(num_layers)}
    all_model_choices = []
    all_correct_choices = []

    num_trials = trials.shape[1]

    print(f"Computing pair embeddings for {num_networks} networks, {num_trials} trials each ({task.upper()})...")

    with torch.no_grad():
        for trial_idx in range(num_trials):
            batch_trial = trials[:, trial_idx, :]
            batch_correct_choice = correct_choices_t[:, trial_idx]

            # Run forward pass with embedding storage
            output = model(batch_trial, plastic_weights, batch_correct_choice,
                          extra_plastic_weights=extra_plastic_weights, store_embeddings=True)

            # Store embeddings for each layer
            for layer_idx, embedding in enumerate(output.embeddings):
                emb_np = embedding.detach().cpu().numpy()  # (num_networks, hidden_size)
                all_embeddings[layer_idx].append(emb_np)

            # Store choices
            model_choices = output.sampled_choices.squeeze(-1).detach().cpu().numpy()  # (num_networks,)
            all_model_choices.append(model_choices)
            all_correct_choices.append(correct_choices[:, trial_idx])

            # Update plastic weights for next trial
            plastic_weights = output.plastic_weights
            extra_plastic_weights = output.extra_plastic_weights

    # Concatenate all embeddings: (num_trials, num_networks, hidden_size) -> (num_trials * num_networks, hidden_size)
    for layer_idx in range(num_layers):
        all_embeddings[layer_idx] = np.vstack(all_embeddings[layer_idx])

    all_model_choices = np.concatenate(all_model_choices, axis=0)
    all_correct_choices = np.concatenate(all_correct_choices, axis=0)

    # Create PCA plots for each layer
    for layer_idx in range(num_layers):
        layer_name = layer_names[layer_idx]
        embeddings = all_embeddings[layer_idx]

        # Fit PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)

        # Plot 1: Colored by model's choice
        fig_choice, ax_choice = plt.subplots(figsize=(10, 8), dpi=150)

        # Model chose 0 (position 1)
        mask_0 = all_model_choices == 0
        ax_choice.scatter(embeddings_2d[mask_0, 0], embeddings_2d[mask_0, 1],
                         c='tab:blue', label='Model chose pos 1', alpha=0.5, s=10)
        # Model chose 1 (position 2)
        mask_1 = all_model_choices == 1
        ax_choice.scatter(embeddings_2d[mask_1, 0], embeddings_2d[mask_1, 1],
                         c='tab:orange', label='Model chose pos 2', alpha=0.5, s=10)

        ax_choice.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax_choice.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax_choice.set_title(f'{task.upper()}: {layer_name} - Pair Embeddings by Model Choice\n(n={num_networks} networks, {num_trials} trials)')
        ax_choice.legend()
        plt.tight_layout()
        figures[f"{task_prefix}/layer{layer_idx}_pair_pca_by_model_choice"] = fig_choice
        plt.close(fig_choice)

        # Plot 2: Colored by correct answer
        fig_correct, ax_correct = plt.subplots(figsize=(10, 8), dpi=150)

        # Correct answer is 0 (position 1)
        mask_correct_0 = all_correct_choices == 0
        ax_correct.scatter(embeddings_2d[mask_correct_0, 0], embeddings_2d[mask_correct_0, 1],
                          c='tab:green', label='Correct: pos 1', alpha=0.5, s=10)
        # Correct answer is 1 (position 2)
        mask_correct_1 = all_correct_choices == 1
        ax_correct.scatter(embeddings_2d[mask_correct_1, 0], embeddings_2d[mask_correct_1, 1],
                          c='tab:red', label='Correct: pos 2', alpha=0.5, s=10)

        ax_correct.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
        ax_correct.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
        ax_correct.set_title(f'{task.upper()}: {layer_name} - Pair Embeddings by Correct Answer\n(n={num_networks} networks, {num_trials} trials)')
        ax_correct.legend()
        plt.tight_layout()
        figures[f"{task_prefix}/layer{layer_idx}_pair_pca_by_correct_answer"] = fig_correct
        plt.close(fig_correct)

    model.train()
    return figures


def plot_trial_violin_by_reward(args, model, task='ll'):
    """
    Create violin plots showing the distribution of readout values across networks
    for each of the first 10 trials, broken down by reward (+1 for correct, -1 for incorrect).

    Args:
        args: Training arguments
        model: Trained MLP model
        task: 'ti' for transitive inference, 'll' for list linking

    Returns:
        Dictionary of figures
    """
    from generate_data import generate_batch_items, generate_batch_trials_ti, generate_batch_trials_ll

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    figures = {}

    num_items = 8 if task == 'll' else args.num_items
    num_networks = 4 * args.batch_size

    task_prefix = "list_linking" if task == 'll' else "pca_frozen"

    # Generate items and trials
    batch_items = generate_batch_items(num_items, args.item_size, num_networks, change_items_throughout_batch=True)

    if task == 'll':
        trials, correct_choices, pair_indices = generate_batch_trials_ll(
            batch_items, args.num_trials_list_1, args.num_trials_list_2, args.num_trials_linking_pair, num_test_trials=0
        )
        num_trials_to_plot = args.num_trials_list_1
    else:
        # Ensure at least 10 trials for TI violin plots
        num_ti_trials = max(10, args.num_train_trials)
        trials, correct_choices, pair_indices, _ = generate_batch_trials_ti(
            batch_items, num_ti_trials, num_test_trials=0
        )
        num_trials_to_plot = 10

    trials = torch.tensor(trials, dtype=torch.float32).to(device)
    correct_choices_t = torch.tensor(correct_choices, dtype=torch.float32).to(device)

    # Initialize plastic weights
    plastic_weights = torch.zeros(num_networks, args.hidden_size, args.hidden_size,
                                  dtype=torch.float32, requires_grad=False).to(device)
    extra_plastic_weights = [torch.zeros(num_networks, args.hidden_size, args.hidden_size,
                                         dtype=torch.float32, requires_grad=False).to(device)
                            for _ in range(args.extra_layers)]
    trial_neuromodulators = []  # List of (num_networks,) arrays
    trial_rewards = []   # List of (num_networks,) arrays with +1 or -1

    print(f"Computing trial-by-trial neuromodulator values for {num_networks} networks, {num_trials_to_plot} trials ({task.upper()})...")

    with torch.no_grad():
        for trial_idx in range(num_trials_to_plot):
            batch_trial = trials[:, trial_idx, :]
            batch_correct_choice = correct_choices_t[:, trial_idx]

            # Run forward pass
            output = model(batch_trial, plastic_weights, batch_correct_choice,
                          extra_plastic_weights=extra_plastic_weights)

            # Get neuromodulator values
            nm_values = output.neuromodulator.squeeze().detach().cpu().numpy()  # (num_networks,) or (num_networks, num_nm)
            if nm_values.ndim > 1:
                nm_values = nm_values[:, 0]  # Take first neuromodulator if multiple

            # Get model's sampled choices
            model_choices = output.sampled_choices.squeeze(-1).detach().cpu().numpy()  # (num_networks,)

            # Compute rewards: +1 if correct, -1 if incorrect
            correct_np = correct_choices[:, trial_idx]  # (num_networks,)
            rewards = np.where(model_choices == correct_np, 1.0, -1.0)

            trial_neuromodulators.append(nm_values)
            trial_rewards.append(rewards)

            # Update plastic weights for next trial
            plastic_weights = output.plastic_weights
            extra_plastic_weights = output.extra_plastic_weights

    # Create violin plots for each trial
    for trial_idx in range(num_trials_to_plot):
        nm_vals = trial_neuromodulators[trial_idx]
        rewards = trial_rewards[trial_idx]

        # Split by reward
        correct_mask = rewards == 1.0
        incorrect_mask = rewards == -1.0

        nm_correct = nm_vals[correct_mask]
        nm_incorrect = nm_vals[incorrect_mask]

        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

        # Prepare data for violin plot
        data_to_plot = []
        positions = []
        labels = []

        if len(nm_incorrect) > 1:
            data_to_plot.append(nm_incorrect)
            positions.append(0)
            labels.append(f'-1\n(n={len(nm_incorrect)})')

        if len(nm_correct) > 1:
            data_to_plot.append(nm_correct)
            positions.append(1)
            labels.append(f'+1\n(n={len(nm_correct)})')

        if len(data_to_plot) >= 1:
            parts = ax.violinplot(data_to_plot, positions=positions, showmeans=False, showextrema=False, showmedians=False)

            # Color the violins
            colors = ['tab:red', 'tab:green']
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[positions[i]])
                pc.set_alpha(0.7)

            # Add median and IQR lines manually
            for i, (data, pos) in enumerate(zip(data_to_plot, positions)):
                q1, median, q3 = np.percentile(data, [25, 50, 75])
                # Median line (thicker)
                ax.hlines(median, pos - 0.15, pos + 0.15, color='white', linewidth=2, zorder=3)
                # IQR lines (thinner)
                ax.hlines(q1, pos - 0.1, pos + 0.1, color='white', linewidth=1, zorder=3)
                ax.hlines(q3, pos - 0.1, pos + 0.1, color='white', linewidth=1, zorder=3)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels)
        ax.set_xlabel('Reward')
        ax.set_ylabel('Neuromodulator Value')
        ax.set_title(f'{task.upper()}: Trial {trial_idx + 1} - Neuromodulator Distribution by Reward\n({num_networks} networks)')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        plt.tight_layout()
        figures[f"{task_prefix}/trial{trial_idx + 1}_neuromodulator_by_reward"] = fig
        plt.close(fig)

    # Also create a combined plot showing all 10 trials
    fig_combined, ax_combined = plt.subplots(figsize=(16, 8), dpi=150)

    all_data = []
    all_positions = []
    all_colors = []

    for trial_idx in range(num_trials_to_plot):
        nm_vals = trial_neuromodulators[trial_idx]
        rewards = trial_rewards[trial_idx]

        correct_mask = rewards == 1.0
        incorrect_mask = rewards == -1.0

        nm_correct = nm_vals[correct_mask]
        nm_incorrect = nm_vals[incorrect_mask]

        base_pos = trial_idx * 3  # 3 units per trial (2 violins + 1 gap)

        if len(nm_incorrect) > 1:
            all_data.append(nm_incorrect)
            all_positions.append(base_pos)
            all_colors.append('tab:red')

        if len(nm_correct) > 1:
            all_data.append(nm_correct)
            all_positions.append(base_pos + 1)
            all_colors.append('tab:green')

    if len(all_data) > 0:
        parts = ax_combined.violinplot(all_data, positions=all_positions, showmeans=False, showextrema=False, showmedians=False, widths=0.8)

        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(all_colors[i])
            pc.set_alpha(0.7)

        # Add median and IQR lines manually
        for i, (data, pos) in enumerate(zip(all_data, all_positions)):
            q1, median, q3 = np.percentile(data, [25, 50, 75])
            # Median line (thicker)
            ax_combined.hlines(median, pos - 0.3, pos + 0.3, color='white', linewidth=2, zorder=3)
            # IQR lines (thinner)
            ax_combined.hlines(q1, pos - 0.2, pos + 0.2, color='white', linewidth=1, zorder=3)
            ax_combined.hlines(q3, pos - 0.2, pos + 0.2, color='white', linewidth=1, zorder=3)

    # Set x-axis labels to show trial numbers
    trial_positions = [trial_idx * 3 + 0.5 for trial_idx in range(num_trials_to_plot)]
    ax_combined.set_xticks(trial_positions)
    ax_combined.set_xticklabels([f'Trial {i+1}' for i in range(num_trials_to_plot)])
    ax_combined.set_xlabel('Trial')
    ax_combined.set_ylabel('Neuromodulator Value')
    ax_combined.set_title(f'{task.upper()}: Neuromodulator Distribution by Reward Across Trials\n({num_networks} networks, Red=-1/Incorrect, Green=+1/Correct)')
    ax_combined.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='tab:red', alpha=0.7, label='Reward -1 (Incorrect)'),
                       Patch(facecolor='tab:green', alpha=0.7, label='Reward +1 (Correct)')]
    ax_combined.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    figures[f"{task_prefix}/all_trials_neuromodulator_by_reward"] = fig_combined
    plt.close(fig_combined)

    model.train()
    return figures


def plot_neural_activity_by_pair_ti(args, model, num_networks=4):
    """
    Plot neural activity (final hidden vector) for each pair in TI task.

    For 4 individual networks, after training phase:
    - Present each pair (both orders treated separately)
    - Collect the final hidden vector (200 dimensions)
    - Create histograms and bar charts organized by symbolic distance
    - Also create weighted versions (multiplied by readout weights)

    Args:
        args: experiment arguments
        model: trained model
        num_networks: number of individual networks to plot (default 4)

    Returns:
        dict of figures for wandb logging
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    num_items = args.item_range[-1] - 1
    item_labels = [chr(ord('A') + i) for i in range(num_items)]

    # Get readout weights for weighted neural activity plots
    readout_weights = model.choice.weight.detach().cpu().numpy().squeeze()  # (hidden_size,)

    figures = {}

    # Generate batch items for the networks
    batch_items = generate_batch_items(num_items, args.item_size, num_networks, change_items_throughout_batch=True)

    # Generate training trials
    trials, correct_choices, pair_indices, num_train_trials = generate_batch_trials_ti(
        batch_items, args.num_train_trials, 0, arbitrary=args.arbitrary
    )

    trials = torch.tensor(trials, dtype=torch.float32).to(device)
    correct_choices_t = torch.tensor(correct_choices, dtype=torch.float32).to(device)

    # Initialize plastic weights
    plastic_weights = torch.zeros(num_networks, args.hidden_size, args.hidden_size,
                                   dtype=torch.float32, requires_grad=False).to(device)
    extra_plastic_weights = [torch.zeros(num_networks, args.hidden_size, args.hidden_size,
                                          dtype=torch.float32, requires_grad=False).to(device)
                             for _ in range(args.extra_layers)]

    # Run training phase to build up plastic weights
    for trial_idx in range(num_train_trials):
        batch_trial = trials[:, trial_idx, :]
        batch_correct_choice = correct_choices_t[:, trial_idx]

        with torch.inference_mode():
            output = model(batch_trial, plastic_weights, batch_correct_choice,
                          extra_plastic_weights=extra_plastic_weights, store_embeddings=False)

        plastic_weights = output.plastic_weights
        extra_plastic_weights = output.extra_plastic_weights

    # Freeze plastic weights
    frozen_pw = plastic_weights.clone()
    frozen_epw = [epw.clone() for epw in extra_plastic_weights]

    # Generate all pairs with both presentation orders
    # Organize by symbolic distance
    pairs_by_sd = {}
    for sd in range(1, num_items):
        pairs_by_sd[sd] = []
        for high_idx in range(num_items - sd):
            low_idx = high_idx + sd
            # Both presentation orders
            pairs_by_sd[sd].append((high_idx, low_idx, 0))  # high item first, correct choice = 0
            pairs_by_sd[sd].append((low_idx, high_idx, 1))  # low item first, correct choice = 1

    # Collect hidden vectors for each network and pair
    # Structure: {network_idx: {sd: {pair_label: hidden_vector}}}
    hidden_vectors = {net_idx: {sd: {} for sd in range(1, num_items)} for net_idx in range(num_networks)}

    for sd, pairs in pairs_by_sd.items():
        for item1_idx, item2_idx, correct_choice in pairs:
            # Create trial input for all networks
            trial_inputs = []
            for net_idx in range(num_networks):
                item1 = batch_items[net_idx, item1_idx]
                item2 = batch_items[net_idx, item2_idx]
                trial_input = np.concatenate([item1, item2])
                trial_inputs.append(trial_input)

            trial_inputs = torch.tensor(np.array(trial_inputs), dtype=torch.float32).to(device)
            correct_choices_batch = torch.full((num_networks,), correct_choice, dtype=torch.float32).to(device)

            # Run forward pass with frozen weights and store embeddings
            with torch.inference_mode():
                output = model(trial_inputs, frozen_pw.clone(), correct_choices_batch,
                              extra_plastic_weights=[epw.clone() for epw in frozen_epw],
                              store_embeddings=True)

            # Get final hidden vector (last in embeddings list)
            # embeddings is a list where the last element is the final hidden
            final_hidden = output.embeddings[-1].detach().cpu().numpy()  # (num_networks, hidden_size)

            # Store for each network
            pair_label = f"{item_labels[item1_idx]}{item_labels[item2_idx]}"
            for net_idx in range(num_networks):
                hidden_vectors[net_idx][sd][pair_label] = final_hidden[net_idx]

    # Create plots organized by symbolic distance
    for sd in range(1, num_items):
        pairs = list(hidden_vectors[0][sd].keys())
        num_pairs = len(pairs)

        if num_pairs == 0:
            continue

        # --- Histogram plots ---
        # One figure per symbolic distance, with subplots for each network x pair
        fig_hist, axes_hist = plt.subplots(num_networks, num_pairs,
                                            figsize=(3 * num_pairs, 3 * num_networks),
                                            dpi=150, squeeze=False)

        for net_idx in range(num_networks):
            for pair_idx, pair_label in enumerate(pairs):
                ax = axes_hist[net_idx, pair_idx]
                hidden_vec = hidden_vectors[net_idx][sd][pair_label]

                ax.hist(hidden_vec, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
                ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
                ax.axvline(x=np.mean(hidden_vec), color='green', linestyle='-', linewidth=2, label=f'mean={np.mean(hidden_vec):.2f}')

                if net_idx == 0:
                    ax.set_title(f'{pair_label}', fontsize=12, fontweight='bold')
                if pair_idx == 0:
                    ax.set_ylabel(f'Net {net_idx}', fontsize=10)

                ax.set_xlabel('Activation')

        fig_hist.suptitle(f'TI Neural Activity Histograms - Symbolic Distance {sd}\n(Final Hidden Layer, {args.hidden_size} neurons)', fontsize=14)
        plt.tight_layout()
        figures[f"ti_neural_activity/sd{sd}_histogram"] = fig_hist
        plt.close(fig_hist)

        # --- Bar chart plots ---
        # One figure per symbolic distance, with subplots for each network x pair
        fig_bar, axes_bar = plt.subplots(num_networks, num_pairs,
                                          figsize=(max(6, num_pairs * 3), 3 * num_networks),
                                          dpi=150, squeeze=False)

        for net_idx in range(num_networks):
            for pair_idx, pair_label in enumerate(pairs):
                ax = axes_bar[net_idx, pair_idx]
                hidden_vec = hidden_vectors[net_idx][sd][pair_label]

                x = np.arange(len(hidden_vec))
                colors = ['tab:red' if v < 0 else 'tab:blue' for v in hidden_vec]
                ax.bar(x, hidden_vec, color=colors, width=1.0, edgecolor='none')
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

                if net_idx == 0:
                    ax.set_title(f'{pair_label}', fontsize=12, fontweight='bold')
                if pair_idx == 0:
                    ax.set_ylabel(f'Net {net_idx}\nActivation', fontsize=10)

                ax.set_xlabel(f'Neuron (0-{args.hidden_size-1})')
                ax.set_xlim(-0.5, len(hidden_vec) - 0.5)

        fig_bar.suptitle(f'TI Neural Activity Bar Charts - Symbolic Distance {sd}\n(Final Hidden Layer, {args.hidden_size} neurons)', fontsize=14)
        plt.tight_layout()
        figures[f"ti_neural_activity/sd{sd}_barchart"] = fig_bar
        plt.close(fig_bar)

        # --- Weighted Histogram plots (neural activity * readout weights) ---
        fig_hist_w, axes_hist_w = plt.subplots(num_networks, num_pairs,
                                                figsize=(3 * num_pairs, 3 * num_networks),
                                                dpi=150, squeeze=False)

        for net_idx in range(num_networks):
            for pair_idx, pair_label in enumerate(pairs):
                ax = axes_hist_w[net_idx, pair_idx]
                hidden_vec = hidden_vectors[net_idx][sd][pair_label]
                weighted_vec = hidden_vec * readout_weights

                ax.hist(weighted_vec, bins=30, color='darkorange', edgecolor='black', alpha=0.7)
                ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
                ax.axvline(x=np.mean(weighted_vec), color='green', linestyle='-', linewidth=2, label=f'mean={np.mean(weighted_vec):.2f}')
                # Add sum as text in upper right corner
                ax.text(0.95, 0.95, f'sum={np.sum(weighted_vec):.2f}', transform=ax.transAxes,
                        fontsize=8, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                if net_idx == 0:
                    ax.set_title(f'{pair_label}', fontsize=12, fontweight='bold')
                if pair_idx == 0:
                    ax.set_ylabel(f'Net {net_idx}', fontsize=10)

                ax.set_xlabel('Weighted Activation')
                ax.legend(fontsize=6, loc='upper left')

        fig_hist_w.suptitle(f'TI Weighted Neural Activity Histograms - Symbolic Distance {sd}\n(Hidden * Readout Weights, {args.hidden_size} neurons)', fontsize=14)
        plt.tight_layout()
        figures[f"ti_neural_activity/sd{sd}_weighted_histogram"] = fig_hist_w
        plt.close(fig_hist_w)

        # --- Weighted Bar chart plots (neural activity * readout weights) ---
        fig_bar_w, axes_bar_w = plt.subplots(num_networks, num_pairs,
                                              figsize=(max(6, num_pairs * 3), 3 * num_networks),
                                              dpi=150, squeeze=False)

        for net_idx in range(num_networks):
            for pair_idx, pair_label in enumerate(pairs):
                ax = axes_bar_w[net_idx, pair_idx]
                hidden_vec = hidden_vectors[net_idx][sd][pair_label]
                weighted_vec = hidden_vec * readout_weights

                x = np.arange(len(weighted_vec))
                colors = ['tab:red' if v < 0 else 'tab:blue' for v in weighted_vec]
                ax.bar(x, weighted_vec, color=colors, width=1.0, edgecolor='none')
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                # Add sum as text in upper right corner
                ax.text(0.95, 0.95, f'sum={np.sum(weighted_vec):.2f}', transform=ax.transAxes,
                        fontsize=8, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                if net_idx == 0:
                    ax.set_title(f'{pair_label}', fontsize=12, fontweight='bold')
                if pair_idx == 0:
                    ax.set_ylabel(f'Net {net_idx}\nWeighted Act.', fontsize=10)

                ax.set_xlabel(f'Neuron (0-{args.hidden_size-1})')
                ax.set_xlim(-0.5, len(weighted_vec) - 0.5)

        fig_bar_w.suptitle(f'TI Weighted Neural Activity Bar Charts - Symbolic Distance {sd}\n(Hidden * Readout Weights, {args.hidden_size} neurons)', fontsize=14)
        plt.tight_layout()
        figures[f"ti_neural_activity/sd{sd}_weighted_barchart"] = fig_bar_w
        plt.close(fig_bar_w)

    model.train()
    return figures


def plot_neural_activity_by_pair_ll(args, model, num_networks=4):
    """
    Plot neural activity (final hidden vector) for each pair in LL task.

    For 4 individual networks, after training phase:
    - Present each pair (both orders treated separately)
    - Collect the final hidden vector (200 dimensions)
    - Create histograms and bar charts organized by symbolic distance
    - Also create weighted versions (multiplied by readout weights)

    Args:
        args: experiment arguments
        model: trained model
        num_networks: number of individual networks to plot (default 4)

    Returns:
        dict of figures for wandb logging
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    num_items = args.item_range[-1] - 1
    item_labels = [chr(ord('A') + i) for i in range(num_items)]

    # Get readout weights for weighted neural activity plots
    readout_weights = model.choice.weight.detach().cpu().numpy().squeeze()  # (hidden_size,)

    figures = {}

    # Generate batch items for the networks
    batch_items = generate_batch_items(num_items, args.item_size, num_networks, change_items_throughout_batch=True)

    # Generate training trials for list linking
    trials, correct_choices, pair_indices = generate_batch_trials_ll(
        batch_items, args.num_trials_list_1, args.num_trials_list_2,
        args.num_trials_linking_pair, 0,
        put_linking_trials_first=args.put_linking_trials_first,
        randomize_list_order=args.randomize_list_order
    )

    trials = torch.tensor(trials, dtype=torch.float32).to(device)
    correct_choices_t = torch.tensor(correct_choices, dtype=torch.float32).to(device)

    # Initialize plastic weights
    plastic_weights = torch.zeros(num_networks, args.hidden_size, args.hidden_size,
                                   dtype=torch.float32, requires_grad=False).to(device)
    extra_plastic_weights = [torch.zeros(num_networks, args.hidden_size, args.hidden_size,
                                          dtype=torch.float32, requires_grad=False).to(device)
                             for _ in range(args.extra_layers)]

    num_train_trials = args.num_trials_list_1 + args.num_trials_list_2 + args.num_trials_linking_pair

    # Run training phase to build up plastic weights
    for trial_idx in range(num_train_trials):
        batch_trial = trials[:, trial_idx, :]
        batch_correct_choice = correct_choices_t[:, trial_idx]

        with torch.inference_mode():
            output = model(batch_trial, plastic_weights, batch_correct_choice,
                          extra_plastic_weights=extra_plastic_weights, store_embeddings=False)

        plastic_weights = output.plastic_weights
        extra_plastic_weights = output.extra_plastic_weights

    # Freeze plastic weights
    frozen_pw = plastic_weights.clone()
    frozen_epw = [epw.clone() for epw in extra_plastic_weights]

    # Generate all pairs with both presentation orders
    # Organize by symbolic distance
    pairs_by_sd = {}
    for sd in range(1, num_items):
        pairs_by_sd[sd] = []
        for high_idx in range(num_items - sd):
            low_idx = high_idx + sd
            # Both presentation orders
            pairs_by_sd[sd].append((high_idx, low_idx, 0))  # high item first, correct choice = 0
            pairs_by_sd[sd].append((low_idx, high_idx, 1))  # low item first, correct choice = 1

    # Collect hidden vectors for each network and pair
    # Structure: {network_idx: {sd: {pair_label: hidden_vector}}}
    hidden_vectors = {net_idx: {sd: {} for sd in range(1, num_items)} for net_idx in range(num_networks)}

    for sd, pairs in pairs_by_sd.items():
        for item1_idx, item2_idx, correct_choice in pairs:
            # Create trial input for all networks
            trial_inputs = []
            for net_idx in range(num_networks):
                item1 = batch_items[net_idx, item1_idx]
                item2 = batch_items[net_idx, item2_idx]
                trial_input = np.concatenate([item1, item2])
                trial_inputs.append(trial_input)

            trial_inputs = torch.tensor(np.array(trial_inputs), dtype=torch.float32).to(device)
            correct_choices_batch = torch.full((num_networks,), correct_choice, dtype=torch.float32).to(device)

            # Run forward pass with frozen weights and store embeddings
            with torch.inference_mode():
                output = model(trial_inputs, frozen_pw.clone(), correct_choices_batch,
                              extra_plastic_weights=[epw.clone() for epw in frozen_epw],
                              store_embeddings=True)

            # Get final hidden vector (last in embeddings list)
            final_hidden = output.embeddings[-1].detach().cpu().numpy()  # (num_networks, hidden_size)

            # Store for each network
            pair_label = f"{item_labels[item1_idx]}{item_labels[item2_idx]}"
            for net_idx in range(num_networks):
                hidden_vectors[net_idx][sd][pair_label] = final_hidden[net_idx]

    # Create plots organized by symbolic distance
    for sd in range(1, num_items):
        pairs = list(hidden_vectors[0][sd].keys())
        num_pairs = len(pairs)

        if num_pairs == 0:
            continue

        # --- Histogram plots ---
        fig_hist, axes_hist = plt.subplots(num_networks, num_pairs,
                                            figsize=(3 * num_pairs, 3 * num_networks),
                                            dpi=150, squeeze=False)

        for net_idx in range(num_networks):
            for pair_idx, pair_label in enumerate(pairs):
                ax = axes_hist[net_idx, pair_idx]
                hidden_vec = hidden_vectors[net_idx][sd][pair_label]

                ax.hist(hidden_vec, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
                ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
                ax.axvline(x=np.mean(hidden_vec), color='green', linestyle='-', linewidth=2, label=f'mean={np.mean(hidden_vec):.2f}')

                if net_idx == 0:
                    ax.set_title(f'{pair_label}', fontsize=12, fontweight='bold')
                if pair_idx == 0:
                    ax.set_ylabel(f'Net {net_idx}', fontsize=10)

                ax.set_xlabel('Activation')

        fig_hist.suptitle(f'LL Neural Activity Histograms - Symbolic Distance {sd}\n(Final Hidden Layer, {args.hidden_size} neurons)', fontsize=14)
        plt.tight_layout()
        figures[f"ll_neural_activity/sd{sd}_histogram"] = fig_hist
        plt.close(fig_hist)

        # --- Bar chart plots ---
        fig_bar, axes_bar = plt.subplots(num_networks, num_pairs,
                                          figsize=(max(6, num_pairs * 3), 3 * num_networks),
                                          dpi=150, squeeze=False)

        for net_idx in range(num_networks):
            for pair_idx, pair_label in enumerate(pairs):
                ax = axes_bar[net_idx, pair_idx]
                hidden_vec = hidden_vectors[net_idx][sd][pair_label]

                x = np.arange(len(hidden_vec))
                colors = ['tab:red' if v < 0 else 'tab:blue' for v in hidden_vec]
                ax.bar(x, hidden_vec, color=colors, width=1.0, edgecolor='none')
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

                if net_idx == 0:
                    ax.set_title(f'{pair_label}', fontsize=12, fontweight='bold')
                if pair_idx == 0:
                    ax.set_ylabel(f'Net {net_idx}\nActivation', fontsize=10)

                ax.set_xlabel(f'Neuron (0-{args.hidden_size-1})')
                ax.set_xlim(-0.5, len(hidden_vec) - 0.5)

        fig_bar.suptitle(f'LL Neural Activity Bar Charts - Symbolic Distance {sd}\n(Final Hidden Layer, {args.hidden_size} neurons)', fontsize=14)
        plt.tight_layout()
        figures[f"ll_neural_activity/sd{sd}_barchart"] = fig_bar
        plt.close(fig_bar)

        # --- Weighted Histogram plots (neural activity * readout weights) ---
        fig_hist_w, axes_hist_w = plt.subplots(num_networks, num_pairs,
                                                figsize=(3 * num_pairs, 3 * num_networks),
                                                dpi=150, squeeze=False)

        for net_idx in range(num_networks):
            for pair_idx, pair_label in enumerate(pairs):
                ax = axes_hist_w[net_idx, pair_idx]
                hidden_vec = hidden_vectors[net_idx][sd][pair_label]
                weighted_vec = hidden_vec * readout_weights

                ax.hist(weighted_vec, bins=30, color='darkorange', edgecolor='black', alpha=0.7)
                ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
                ax.axvline(x=np.mean(weighted_vec), color='green', linestyle='-', linewidth=2, label=f'mean={np.mean(weighted_vec):.2f}')
                # Add sum as text in upper right corner
                ax.text(0.95, 0.95, f'sum={np.sum(weighted_vec):.2f}', transform=ax.transAxes,
                        fontsize=8, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                if net_idx == 0:
                    ax.set_title(f'{pair_label}', fontsize=12, fontweight='bold')
                if pair_idx == 0:
                    ax.set_ylabel(f'Net {net_idx}', fontsize=10)

                ax.set_xlabel('Weighted Activation')
                ax.legend(fontsize=6, loc='upper left')

        fig_hist_w.suptitle(f'LL Weighted Neural Activity Histograms - Symbolic Distance {sd}\n(Hidden * Readout Weights, {args.hidden_size} neurons)', fontsize=14)
        plt.tight_layout()
        figures[f"ll_neural_activity/sd{sd}_weighted_histogram"] = fig_hist_w
        plt.close(fig_hist_w)

        # --- Weighted Bar chart plots (neural activity * readout weights) ---
        fig_bar_w, axes_bar_w = plt.subplots(num_networks, num_pairs,
                                              figsize=(max(6, num_pairs * 3), 3 * num_networks),
                                              dpi=150, squeeze=False)

        for net_idx in range(num_networks):
            for pair_idx, pair_label in enumerate(pairs):
                ax = axes_bar_w[net_idx, pair_idx]
                hidden_vec = hidden_vectors[net_idx][sd][pair_label]
                weighted_vec = hidden_vec * readout_weights

                x = np.arange(len(weighted_vec))
                colors = ['tab:red' if v < 0 else 'tab:blue' for v in weighted_vec]
                ax.bar(x, weighted_vec, color=colors, width=1.0, edgecolor='none')
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                # Add sum as text in upper right corner
                ax.text(0.95, 0.95, f'sum={np.sum(weighted_vec):.2f}', transform=ax.transAxes,
                        fontsize=8, verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                if net_idx == 0:
                    ax.set_title(f'{pair_label}', fontsize=12, fontweight='bold')
                if pair_idx == 0:
                    ax.set_ylabel(f'Net {net_idx}\nWeighted Act.', fontsize=10)

                ax.set_xlabel(f'Neuron (0-{args.hidden_size-1})')
                ax.set_xlim(-0.5, len(weighted_vec) - 0.5)

        fig_bar_w.suptitle(f'LL Weighted Neural Activity Bar Charts - Symbolic Distance {sd}\n(Hidden * Readout Weights, {args.hidden_size} neurons)', fontsize=14)
        plt.tight_layout()
        figures[f"ll_neural_activity/sd{sd}_weighted_barchart"] = fig_bar_w
        plt.close(fig_bar_w)

    model.train()
    return figures


def plot_rank_neuron_analysis_ti(args, model, num_networks=4):
    """
    Identify rank neurons using Spearman correlation between neuron activation and item rank.

    For TI task:
    - Present each item individually after training (frozen plastic weights)
    - Compute Spearman correlation between each neuron's activation and item rank
    - Analyze at 3 layers: embedding output, extra layer output, final hidden
    - Check overlap between rank neurons and readout-important neurons

    Args:
        args: experiment arguments
        model: trained model
        num_networks: number of networks to analyze

    Returns:
        dict of figures for wandb logging
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    num_items = args.item_range[-1] - 1
    item_labels = [chr(ord('A') + i) for i in range(num_items)]
    item_ranks = np.arange(num_items)  # 0, 1, 2, ... (lower = higher rank)

    figures = {}

    # Get readout weights
    readout_weights = model.choice.weight.detach().cpu().numpy().squeeze()  # (hidden_size,)
    abs_readout = np.abs(readout_weights)

    # Generate batch items for the networks
    batch_items = generate_batch_items(num_items, args.item_size, num_networks, change_items_throughout_batch=True)

    # Generate training trials
    trials, correct_choices, pair_indices, num_train_trials = generate_batch_trials_ti(
        batch_items, args.num_train_trials, 0, arbitrary=args.arbitrary
    )

    trials = torch.tensor(trials, dtype=torch.float32).to(device)
    correct_choices_t = torch.tensor(correct_choices, dtype=torch.float32).to(device)

    # Initialize plastic weights
    plastic_weights = torch.zeros(num_networks, args.hidden_size, args.hidden_size,
                                   dtype=torch.float32, requires_grad=False).to(device)
    extra_plastic_weights = [torch.zeros(num_networks, args.hidden_size, args.hidden_size,
                                          dtype=torch.float32, requires_grad=False).to(device)
                             for _ in range(args.extra_layers)]

    # Run training phase to build up plastic weights
    for trial_idx in range(num_train_trials):
        batch_trial = trials[:, trial_idx, :]
        batch_correct_choice = correct_choices_t[:, trial_idx]

        with torch.inference_mode():
            output = model(batch_trial, plastic_weights, batch_correct_choice,
                          extra_plastic_weights=extra_plastic_weights, store_embeddings=False)

        plastic_weights = output.plastic_weights
        extra_plastic_weights = output.extra_plastic_weights

    # Freeze plastic weights
    frozen_pw = plastic_weights.clone()
    frozen_epw = [epw.clone() for epw in extra_plastic_weights]

    # Present each item individually and collect activations at each layer
    # For single item presentation, we'll pad with zeros for the second item position
    # Storage: layer -> network -> item -> activation vector
    activations = {
        'embedding': {net_idx: {} for net_idx in range(num_networks)},
        'extra': {net_idx: {} for net_idx in range(num_networks)},
        'final': {net_idx: {} for net_idx in range(num_networks)},
    }

    for item_idx in range(num_items):
        # Create input with item in position 1, zeros in position 2
        item_inputs = []
        for net_idx in range(num_networks):
            item = batch_items[net_idx, item_idx]
            # Put item in first position, zeros in second
            item_input = np.concatenate([item, np.zeros_like(item)])
            item_inputs.append(item_input)

        item_inputs = torch.tensor(np.array(item_inputs), dtype=torch.float32).to(device)
        dummy_choice = torch.zeros(num_networks, dtype=torch.float32).to(device)

        with torch.inference_mode():
            output = model(item_inputs, frozen_pw.clone(), dummy_choice,
                          extra_plastic_weights=[epw.clone() for epw in frozen_epw],
                          store_embeddings=True)

        # Extract activations at each layer
        # embeddings[0] = embedding output, embeddings[1] = extra layer output (if exists), embeddings[-1] = final
        embeddings_list = output.embeddings

        for net_idx in range(num_networks):
            activations['embedding'][net_idx][item_idx] = embeddings_list[0][net_idx].detach().cpu().numpy()
            if args.extra_layers > 0:
                activations['extra'][net_idx][item_idx] = embeddings_list[1][net_idx].detach().cpu().numpy()
            activations['final'][net_idx][item_idx] = embeddings_list[-1][net_idx].detach().cpu().numpy()

    # Compute Spearman correlation for each neuron at each layer
    layers_to_analyze = ['embedding', 'final']
    if args.extra_layers > 0:
        layers_to_analyze = ['embedding', 'extra', 'final']

    rank_correlations = {layer: [] for layer in layers_to_analyze}

    for layer in layers_to_analyze:
        # For each network, compute correlation for each neuron
        all_correlations = []
        for net_idx in range(num_networks):
            # Build activation matrix: (num_items, hidden_size)
            act_matrix = np.array([activations[layer][net_idx][item_idx] for item_idx in range(num_items)])

            # Compute Spearman correlation for each neuron
            neuron_correlations = []
            for neuron_idx in range(act_matrix.shape[1]):
                rho, _ = spearmanr(item_ranks, act_matrix[:, neuron_idx])
                if np.isnan(rho):
                    rho = 0.0
                neuron_correlations.append(rho)
            all_correlations.append(neuron_correlations)

        # Average correlations across networks
        rank_correlations[layer] = np.mean(all_correlations, axis=0)

    # --- Plot 1: Distribution of rank correlations at each layer ---
    fig_dist, axes_dist = plt.subplots(1, len(layers_to_analyze), figsize=(5 * len(layers_to_analyze), 5), dpi=150)
    if len(layers_to_analyze) == 1:
        axes_dist = [axes_dist]

    for idx, layer in enumerate(layers_to_analyze):
        ax = axes_dist[idx]
        corrs = rank_correlations[layer]
        ax.hist(corrs, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=np.mean(corrs), color='green', linestyle='-', linewidth=2,
                   label=f'mean={np.mean(corrs):.3f}')
        ax.set_xlabel('Spearman ρ (activation vs rank)')
        ax.set_ylabel('Count')
        ax.set_title(f'{layer.capitalize()} Layer\n(std={np.std(corrs):.3f})')
        ax.legend()
        ax.set_xlim(-1, 1)

    fig_dist.suptitle('TI: Distribution of Rank Correlations by Layer\n(Spearman ρ between neuron activation and item rank)', fontsize=14)
    plt.tight_layout()
    figures["ti_rank_neurons/correlation_distribution"] = fig_dist
    plt.close(fig_dist)

    # --- Plot 2: Sorted bar chart of rank correlations for final layer ---
    final_corrs = rank_correlations['final']
    sorted_idx = np.argsort(final_corrs)
    sorted_corrs = final_corrs[sorted_idx]

    fig_sorted, ax_sorted = plt.subplots(figsize=(12, 5), dpi=150)
    colors = ['tab:red' if c < 0 else 'tab:blue' for c in sorted_corrs]
    ax_sorted.bar(range(len(sorted_corrs)), sorted_corrs, color=colors, width=1.0, edgecolor='none')
    ax_sorted.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax_sorted.set_xlabel('Neuron (sorted by rank correlation)')
    ax_sorted.set_ylabel('Spearman ρ')
    ax_sorted.set_title(f'TI: Final Layer Rank Correlations (Sorted)\n(mean={np.mean(final_corrs):.3f}, std={np.std(final_corrs):.3f})')
    ax_sorted.set_ylim(-1, 1)
    plt.tight_layout()
    figures["ti_rank_neurons/final_layer_sorted"] = fig_sorted
    plt.close(fig_sorted)

    # --- Plot 3: Scatter plot - |readout weight| vs |rank correlation| ---
    abs_rank_corr = np.abs(final_corrs)
    corr_readout_rank = np.corrcoef(abs_readout, abs_rank_corr)[0, 1]

    fig_scatter, ax_scatter = plt.subplots(figsize=(8, 8), dpi=150)
    ax_scatter.scatter(abs_rank_corr, abs_readout, alpha=0.6, s=20)
    ax_scatter.set_xlabel('|Rank Correlation| (Spearman ρ)')
    ax_scatter.set_ylabel('|Readout Weight|')
    ax_scatter.set_title(f'TI: Rank Neurons vs Readout-Important Neurons\n(Correlation: r={corr_readout_rank:.4f})')

    # Add line of best fit
    slope, intercept = np.polyfit(abs_rank_corr, abs_readout, 1)
    x_line = np.array([abs_rank_corr.min(), abs_rank_corr.max()])
    y_line = slope * x_line + intercept
    ax_scatter.plot(x_line, y_line, 'r-', linewidth=2, label=f'Best fit: y={slope:.3f}x+{intercept:.3f}')
    ax_scatter.legend()
    plt.tight_layout()
    figures["ti_rank_neurons/rank_vs_readout_scatter"] = fig_scatter
    plt.close(fig_scatter)

    # --- Plot 4: Overlap analysis - top-k rank neurons vs top-k readout neurons ---
    k_values = [10, 20, 30, 50]
    overlaps = []

    top_rank_neurons = np.argsort(abs_rank_corr)[::-1]  # Descending by |rank corr|
    top_readout_neurons = np.argsort(abs_readout)[::-1]  # Descending by |readout weight|

    for k in k_values:
        top_k_rank = set(top_rank_neurons[:k])
        top_k_readout = set(top_readout_neurons[:k])
        overlap = len(top_k_rank & top_k_readout)
        overlaps.append(overlap / k)  # Fraction overlap

    fig_overlap, ax_overlap = plt.subplots(figsize=(8, 6), dpi=150)
    ax_overlap.bar(range(len(k_values)), overlaps, color='purple', alpha=0.7)
    ax_overlap.set_xticks(range(len(k_values)))
    ax_overlap.set_xticklabels([f'Top {k}' for k in k_values])
    ax_overlap.set_ylabel('Fraction Overlap')
    ax_overlap.set_title('TI: Overlap Between Top Rank Neurons and Top Readout Neurons')
    ax_overlap.set_ylim(0, 1)
    ax_overlap.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random chance')
    # Add values on bars
    for i, (k, ov) in enumerate(zip(k_values, overlaps)):
        ax_overlap.text(i, ov + 0.02, f'{ov:.2f}\n({int(ov*k)}/{k})', ha='center', fontsize=9)
    ax_overlap.legend()
    plt.tight_layout()
    figures["ti_rank_neurons/overlap_analysis"] = fig_overlap
    plt.close(fig_overlap)

    # --- Plot 5: Rank encoding strength across layers ---
    # Use mean |correlation| as a measure of rank encoding strength
    encoding_strengths = [np.mean(np.abs(rank_correlations[layer])) for layer in layers_to_analyze]

    fig_strength, ax_strength = plt.subplots(figsize=(8, 6), dpi=150)
    bars = ax_strength.bar(range(len(layers_to_analyze)), encoding_strengths, color='teal', alpha=0.7)
    ax_strength.set_xticks(range(len(layers_to_analyze)))
    ax_strength.set_xticklabels([l.capitalize() for l in layers_to_analyze])
    ax_strength.set_ylabel('Mean |Spearman ρ|')
    ax_strength.set_title('TI: Rank Encoding Strength Across Layers')
    ax_strength.set_ylim(0, 1)
    for bar, strength in zip(bars, encoding_strengths):
        ax_strength.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{strength:.3f}', ha='center', fontsize=10)
    plt.tight_layout()
    figures["ti_rank_neurons/encoding_strength_by_layer"] = fig_strength
    plt.close(fig_strength)

    model.train()
    return figures


def plot_rank_neuron_analysis_ll(args, model, num_networks=4):
    """
    Identify rank neurons using Spearman correlation for LL task.

    For LL task, we analyze both:
    - Global rank (0-7 for 8 items across both lists)
    - Within-list rank

    Args:
        args: experiment arguments
        model: trained model
        num_networks: number of networks to analyze

    Returns:
        dict of figures for wandb logging
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    num_items = args.item_range[-1] - 1
    item_labels = [chr(ord('A') + i) for i in range(num_items)]
    item_ranks = np.arange(num_items)  # Global rank: 0, 1, 2, ... (lower = higher rank)

    figures = {}

    # Get readout weights
    readout_weights = model.choice.weight.detach().cpu().numpy().squeeze()
    abs_readout = np.abs(readout_weights)

    # Generate batch items for the networks
    batch_items = generate_batch_items(num_items, args.item_size, num_networks, change_items_throughout_batch=True)

    # Generate training trials for list linking
    trials, correct_choices, pair_indices = generate_batch_trials_ll(
        batch_items, args.num_trials_list_1, args.num_trials_list_2,
        args.num_trials_linking_pair, 0,
        put_linking_trials_first=args.put_linking_trials_first,
        randomize_list_order=args.randomize_list_order
    )

    trials = torch.tensor(trials, dtype=torch.float32).to(device)
    correct_choices_t = torch.tensor(correct_choices, dtype=torch.float32).to(device)

    # Initialize plastic weights
    plastic_weights = torch.zeros(num_networks, args.hidden_size, args.hidden_size,
                                   dtype=torch.float32, requires_grad=False).to(device)
    extra_plastic_weights = [torch.zeros(num_networks, args.hidden_size, args.hidden_size,
                                          dtype=torch.float32, requires_grad=False).to(device)
                             for _ in range(args.extra_layers)]

    num_train_trials = args.num_trials_list_1 + args.num_trials_list_2 + args.num_trials_linking_pair

    # Run training phase
    for trial_idx in range(num_train_trials):
        batch_trial = trials[:, trial_idx, :]
        batch_correct_choice = correct_choices_t[:, trial_idx]

        with torch.inference_mode():
            output = model(batch_trial, plastic_weights, batch_correct_choice,
                          extra_plastic_weights=extra_plastic_weights, store_embeddings=False)

        plastic_weights = output.plastic_weights
        extra_plastic_weights = output.extra_plastic_weights

    # Freeze plastic weights
    frozen_pw = plastic_weights.clone()
    frozen_epw = [epw.clone() for epw in extra_plastic_weights]

    # Present each item individually
    activations = {
        'embedding': {net_idx: {} for net_idx in range(num_networks)},
        'extra': {net_idx: {} for net_idx in range(num_networks)},
        'final': {net_idx: {} for net_idx in range(num_networks)},
    }

    for item_idx in range(num_items):
        item_inputs = []
        for net_idx in range(num_networks):
            item = batch_items[net_idx, item_idx]
            item_input = np.concatenate([item, np.zeros_like(item)])
            item_inputs.append(item_input)

        item_inputs = torch.tensor(np.array(item_inputs), dtype=torch.float32).to(device)
        dummy_choice = torch.zeros(num_networks, dtype=torch.float32).to(device)

        with torch.inference_mode():
            output = model(item_inputs, frozen_pw.clone(), dummy_choice,
                          extra_plastic_weights=[epw.clone() for epw in frozen_epw],
                          store_embeddings=True)

        embeddings_list = output.embeddings

        for net_idx in range(num_networks):
            activations['embedding'][net_idx][item_idx] = embeddings_list[0][net_idx].detach().cpu().numpy()
            if args.extra_layers > 0:
                activations['extra'][net_idx][item_idx] = embeddings_list[1][net_idx].detach().cpu().numpy()
            activations['final'][net_idx][item_idx] = embeddings_list[-1][net_idx].detach().cpu().numpy()

    # Compute Spearman correlation for global rank
    layers_to_analyze = ['embedding', 'final']
    if args.extra_layers > 0:
        layers_to_analyze = ['embedding', 'extra', 'final']

    rank_correlations_global = {layer: [] for layer in layers_to_analyze}

    for layer in layers_to_analyze:
        all_correlations = []
        for net_idx in range(num_networks):
            act_matrix = np.array([activations[layer][net_idx][item_idx] for item_idx in range(num_items)])

            neuron_correlations = []
            for neuron_idx in range(act_matrix.shape[1]):
                rho, _ = spearmanr(item_ranks, act_matrix[:, neuron_idx])
                if np.isnan(rho):
                    rho = 0.0
                neuron_correlations.append(rho)
            all_correlations.append(neuron_correlations)

        rank_correlations_global[layer] = np.mean(all_correlations, axis=0)

    # Also compute within-list rank correlations
    # List 1: items 0 to num_items//2 - 1
    # List 2: items num_items//2 to num_items - 1
    half = num_items // 2
    list1_items = list(range(half))
    list2_items = list(range(half, num_items))
    list1_ranks = np.arange(len(list1_items))
    list2_ranks = np.arange(len(list2_items))

    rank_correlations_list1 = {layer: [] for layer in layers_to_analyze}
    rank_correlations_list2 = {layer: [] for layer in layers_to_analyze}

    for layer in layers_to_analyze:
        all_corrs_list1 = []
        all_corrs_list2 = []
        for net_idx in range(num_networks):
            # List 1
            act_list1 = np.array([activations[layer][net_idx][item_idx] for item_idx in list1_items])
            corrs_list1 = []
            for neuron_idx in range(act_list1.shape[1]):
                rho, _ = spearmanr(list1_ranks, act_list1[:, neuron_idx])
                if np.isnan(rho):
                    rho = 0.0
                corrs_list1.append(rho)
            all_corrs_list1.append(corrs_list1)

            # List 2
            act_list2 = np.array([activations[layer][net_idx][item_idx] for item_idx in list2_items])
            corrs_list2 = []
            for neuron_idx in range(act_list2.shape[1]):
                rho, _ = spearmanr(list2_ranks, act_list2[:, neuron_idx])
                if np.isnan(rho):
                    rho = 0.0
                corrs_list2.append(rho)
            all_corrs_list2.append(corrs_list2)

        rank_correlations_list1[layer] = np.mean(all_corrs_list1, axis=0)
        rank_correlations_list2[layer] = np.mean(all_corrs_list2, axis=0)

    # --- Plot 1: Distribution of global rank correlations at each layer ---
    fig_dist, axes_dist = plt.subplots(1, len(layers_to_analyze), figsize=(5 * len(layers_to_analyze), 5), dpi=150)
    if len(layers_to_analyze) == 1:
        axes_dist = [axes_dist]

    for idx, layer in enumerate(layers_to_analyze):
        ax = axes_dist[idx]
        corrs = rank_correlations_global[layer]
        ax.hist(corrs, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax.axvline(x=np.mean(corrs), color='green', linestyle='-', linewidth=2,
                   label=f'mean={np.mean(corrs):.3f}')
        ax.set_xlabel('Spearman ρ (activation vs global rank)')
        ax.set_ylabel('Count')
        ax.set_title(f'{layer.capitalize()} Layer\n(std={np.std(corrs):.3f})')
        ax.legend()
        ax.set_xlim(-1, 1)

    fig_dist.suptitle('LL: Distribution of Global Rank Correlations by Layer', fontsize=14)
    plt.tight_layout()
    figures["ll_rank_neurons/global_correlation_distribution"] = fig_dist
    plt.close(fig_dist)

    # --- Plot 2: Within-list rank correlations comparison ---
    fig_within, axes_within = plt.subplots(2, len(layers_to_analyze),
                                            figsize=(5 * len(layers_to_analyze), 8), dpi=150)

    for idx, layer in enumerate(layers_to_analyze):
        # List 1
        ax1 = axes_within[0, idx] if len(layers_to_analyze) > 1 else axes_within[0]
        corrs1 = rank_correlations_list1[layer]
        ax1.hist(corrs1, bins=30, color='coral', edgecolor='black', alpha=0.7)
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax1.axvline(x=np.mean(corrs1), color='green', linestyle='-', linewidth=2,
                    label=f'mean={np.mean(corrs1):.3f}')
        ax1.set_xlabel('Spearman ρ')
        ax1.set_ylabel('Count')
        ax1.set_title(f'{layer.capitalize()} - List 1')
        ax1.legend()
        ax1.set_xlim(-1, 1)

        # List 2
        ax2 = axes_within[1, idx] if len(layers_to_analyze) > 1 else axes_within[1]
        corrs2 = rank_correlations_list2[layer]
        ax2.hist(corrs2, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax2.axvline(x=np.mean(corrs2), color='green', linestyle='-', linewidth=2,
                    label=f'mean={np.mean(corrs2):.3f}')
        ax2.set_xlabel('Spearman ρ')
        ax2.set_ylabel('Count')
        ax2.set_title(f'{layer.capitalize()} - List 2')
        ax2.legend()
        ax2.set_xlim(-1, 1)

    fig_within.suptitle('LL: Within-List Rank Correlations by Layer', fontsize=14)
    plt.tight_layout()
    figures["ll_rank_neurons/within_list_correlation_distribution"] = fig_within
    plt.close(fig_within)

    # --- Plot 3: Sorted bar chart of global rank correlations for final layer ---
    final_corrs = rank_correlations_global['final']
    sorted_idx = np.argsort(final_corrs)
    sorted_corrs = final_corrs[sorted_idx]

    fig_sorted, ax_sorted = plt.subplots(figsize=(12, 5), dpi=150)
    colors = ['tab:red' if c < 0 else 'tab:blue' for c in sorted_corrs]
    ax_sorted.bar(range(len(sorted_corrs)), sorted_corrs, color=colors, width=1.0, edgecolor='none')
    ax_sorted.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax_sorted.set_xlabel('Neuron (sorted by rank correlation)')
    ax_sorted.set_ylabel('Spearman ρ')
    ax_sorted.set_title(f'LL: Final Layer Global Rank Correlations (Sorted)\n(mean={np.mean(final_corrs):.3f}, std={np.std(final_corrs):.3f})')
    ax_sorted.set_ylim(-1, 1)
    plt.tight_layout()
    figures["ll_rank_neurons/final_layer_sorted"] = fig_sorted
    plt.close(fig_sorted)

    # --- Plot 4: Scatter plot - |readout weight| vs |rank correlation| ---
    abs_rank_corr = np.abs(final_corrs)
    corr_readout_rank = np.corrcoef(abs_readout, abs_rank_corr)[0, 1]

    fig_scatter, ax_scatter = plt.subplots(figsize=(8, 8), dpi=150)
    ax_scatter.scatter(abs_rank_corr, abs_readout, alpha=0.6, s=20)
    ax_scatter.set_xlabel('|Rank Correlation| (Spearman ρ)')
    ax_scatter.set_ylabel('|Readout Weight|')
    ax_scatter.set_title(f'LL: Rank Neurons vs Readout-Important Neurons\n(Correlation: r={corr_readout_rank:.4f})')

    slope, intercept = np.polyfit(abs_rank_corr, abs_readout, 1)
    x_line = np.array([abs_rank_corr.min(), abs_rank_corr.max()])
    y_line = slope * x_line + intercept
    ax_scatter.plot(x_line, y_line, 'r-', linewidth=2, label=f'Best fit: y={slope:.3f}x+{intercept:.3f}')
    ax_scatter.legend()
    plt.tight_layout()
    figures["ll_rank_neurons/rank_vs_readout_scatter"] = fig_scatter
    plt.close(fig_scatter)

    # --- Plot 5: Overlap analysis ---
    k_values = [10, 20, 30, 50]
    overlaps = []

    top_rank_neurons = np.argsort(abs_rank_corr)[::-1]
    top_readout_neurons = np.argsort(abs_readout)[::-1]

    for k in k_values:
        top_k_rank = set(top_rank_neurons[:k])
        top_k_readout = set(top_readout_neurons[:k])
        overlap = len(top_k_rank & top_k_readout)
        overlaps.append(overlap / k)

    fig_overlap, ax_overlap = plt.subplots(figsize=(8, 6), dpi=150)
    ax_overlap.bar(range(len(k_values)), overlaps, color='purple', alpha=0.7)
    ax_overlap.set_xticks(range(len(k_values)))
    ax_overlap.set_xticklabels([f'Top {k}' for k in k_values])
    ax_overlap.set_ylabel('Fraction Overlap')
    ax_overlap.set_title('LL: Overlap Between Top Rank Neurons and Top Readout Neurons')
    ax_overlap.set_ylim(0, 1)
    ax_overlap.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random chance')
    for i, (k, ov) in enumerate(zip(k_values, overlaps)):
        ax_overlap.text(i, ov + 0.02, f'{ov:.2f}\n({int(ov*k)}/{k})', ha='center', fontsize=9)
    ax_overlap.legend()
    plt.tight_layout()
    figures["ll_rank_neurons/overlap_analysis"] = fig_overlap
    plt.close(fig_overlap)

    # --- Plot 6: Rank encoding strength across layers (global, list1, list2) ---
    encoding_global = [np.mean(np.abs(rank_correlations_global[layer])) for layer in layers_to_analyze]
    encoding_list1 = [np.mean(np.abs(rank_correlations_list1[layer])) for layer in layers_to_analyze]
    encoding_list2 = [np.mean(np.abs(rank_correlations_list2[layer])) for layer in layers_to_analyze]

    x = np.arange(len(layers_to_analyze))
    width = 0.25

    fig_strength, ax_strength = plt.subplots(figsize=(10, 6), dpi=150)
    bars1 = ax_strength.bar(x - width, encoding_global, width, label='Global', color='steelblue', alpha=0.7)
    bars2 = ax_strength.bar(x, encoding_list1, width, label='List 1', color='coral', alpha=0.7)
    bars3 = ax_strength.bar(x + width, encoding_list2, width, label='List 2', color='lightgreen', alpha=0.7)

    ax_strength.set_xticks(x)
    ax_strength.set_xticklabels([l.capitalize() for l in layers_to_analyze])
    ax_strength.set_ylabel('Mean |Spearman ρ|')
    ax_strength.set_title('LL: Rank Encoding Strength Across Layers')
    ax_strength.set_ylim(0, 1)
    ax_strength.legend()

    # Add values
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            ax_strength.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{bar.get_height():.2f}', ha='center', fontsize=8)
    plt.tight_layout()
    figures["ll_rank_neurons/encoding_strength_by_layer"] = fig_strength
    plt.close(fig_strength)

    model.train()
    return figures


def plot_adjacent_pair_heatmap_ti(args, model, num_networks=128):
    """
    After TI training, compute correlations between adjacent pair representations
    and single item representations at each layer.

    Produces heatmaps with adjacent pairs on x-axis (AB, BA, BC, CB, ...) and
    single items on y-axis (A, B, C, ...), colored red-white-blue.

    Returns a dict of figures for wandb logging.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    num_items = args.item_range[-1] - 1
    item_size = args.item_size
    item_labels = [chr(ord('A') + i) for i in range(num_items)]

    # Generate batch items
    batch_items = generate_batch_items(num_items, item_size, num_networks, change_items_throughout_batch=True)

    # Generate training trials
    trials, correct_choices, pair_indices, num_train_trials = generate_batch_trials_ti(
        batch_items, args.num_train_trials, 0, arbitrary=args.arbitrary
    )

    trials = torch.tensor(trials, dtype=torch.float32).to(device)
    correct_choices_t = torch.tensor(correct_choices, dtype=torch.float32).to(device)

    figures = {}

    # Layer info
    num_layers = 1 + args.extra_layers + 1
    layer_names = ['Embedding'] + [f'Hidden {i+1}' for i in range(args.extra_layers)] + ['Final']

    # Adjacent pairs for TI: all consecutive item pairs in both orders
    adjacent_pairs = []
    adjacent_pair_labels = []
    for i in range(num_items - 1):
        adjacent_pairs.append((i, i + 1))
        adjacent_pair_labels.append(f'{item_labels[i]}{item_labels[i+1]}')
        adjacent_pairs.append((i + 1, i))
        adjacent_pair_labels.append(f'{item_labels[i+1]}{item_labels[i]}')
    num_adjacent_pairs = len(adjacent_pairs)

    # Storage: [layer][pair_idx][item_idx] -> list of correlations across networks
    adj_pair_item_corr_pos1 = [[[[] for _ in range(num_items)] for _ in range(num_adjacent_pairs)] for _ in range(num_layers)]
    adj_pair_item_corr_pos2 = [[[[] for _ in range(num_items)] for _ in range(num_adjacent_pairs)] for _ in range(num_layers)]

    # Initialize plastic weights
    plastic_weights = torch.zeros(num_networks, args.hidden_size, args.hidden_size,
                                   dtype=torch.float32, requires_grad=False).to(device)
    extra_plastic_weights = [torch.zeros(num_networks, args.hidden_size, args.hidden_size,
                                          dtype=torch.float32, requires_grad=False).to(device)
                             for _ in range(args.extra_layers)]

    # Run training phase
    for trial_idx in range(num_train_trials):
        batch_trial = trials[:, trial_idx, :]
        batch_correct_choice = correct_choices_t[:, trial_idx]

        with torch.inference_mode():
            output = model(batch_trial, plastic_weights, batch_correct_choice,
                          extra_plastic_weights=extra_plastic_weights, store_embeddings=False)

        plastic_weights = output.plastic_weights
        extra_plastic_weights = output.extra_plastic_weights

    # Freeze plastic weights
    frozen_pw = plastic_weights.clone()
    frozen_epw = [epw.clone() for epw in extra_plastic_weights]

    # Process each network individually
    for net_idx in range(num_networks):
        items = batch_items[net_idx]  # (num_items, item_size)
        single_pw = frozen_pw[net_idx:net_idx+1]
        single_epw = [epw[net_idx:net_idx+1] for epw in frozen_epw]
        dummy_reward = torch.tensor([0.0], dtype=torch.float32).to(device)
        zeros = np.zeros(item_size)

        # Get single item representations
        all_pos1_embeddings = []  # [item_idx][layer_idx] -> embedding
        all_pos2_embeddings = []

        for item_idx in range(num_items):
            item = items[item_idx]

            # Position 1: [item, zeros]
            pos1_input = np.concatenate([item, zeros])
            pos1_tensor = torch.tensor(pos1_input, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.inference_mode():
                output_pos1 = model(pos1_tensor, single_pw, dummy_reward,
                                   extra_plastic_weights=single_epw, store_embeddings=True)
            pos1_embeddings = [emb.squeeze(0).cpu().numpy() for emb in output_pos1.embeddings]
            all_pos1_embeddings.append(pos1_embeddings)

            # Position 2: [zeros, item]
            pos2_input = np.concatenate([zeros, item])
            pos2_tensor = torch.tensor(pos2_input, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.inference_mode():
                output_pos2 = model(pos2_tensor, single_pw, dummy_reward,
                                   extra_plastic_weights=single_epw, store_embeddings=True)
            pos2_embeddings = [emb.squeeze(0).cpu().numpy() for emb in output_pos2.embeddings]
            all_pos2_embeddings.append(pos2_embeddings)

        # Get adjacent pair representations and compute correlations
        for pair_idx, (left_item_idx, right_item_idx) in enumerate(adjacent_pairs):
            pair_input = np.concatenate([items[left_item_idx], items[right_item_idx]])
            pair_tensor = torch.tensor(pair_input, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.inference_mode():
                output_pair = model(pair_tensor, single_pw, dummy_reward,
                                    extra_plastic_weights=single_epw, store_embeddings=True)
            pair_embeddings = [emb.squeeze(0).cpu().numpy() for emb in output_pair.embeddings]

            for item_idx in range(num_items):
                for layer_idx in range(num_layers):
                    pair_emb = pair_embeddings[layer_idx]
                    pos1_emb = all_pos1_embeddings[item_idx][layer_idx]
                    pos2_emb = all_pos2_embeddings[item_idx][layer_idx]

                    corr_pos1 = np.corrcoef(pair_emb, pos1_emb)[0, 1]
                    corr_pos2 = np.corrcoef(pair_emb, pos2_emb)[0, 1]

                    if np.isnan(corr_pos1):
                        corr_pos1 = 0.0
                    if np.isnan(corr_pos2):
                        corr_pos2 = 0.0

                    adj_pair_item_corr_pos1[layer_idx][pair_idx][item_idx].append(corr_pos1)
                    adj_pair_item_corr_pos2[layer_idx][pair_idx][item_idx].append(corr_pos2)

    # Create heatmaps
    for layer_idx in range(num_layers):
        # Position 1 heatmap
        heatmap_data_pos1 = np.zeros((num_items, num_adjacent_pairs))
        for pair_idx in range(num_adjacent_pairs):
            for item_idx in range(num_items):
                vals = adj_pair_item_corr_pos1[layer_idx][pair_idx][item_idx]
                heatmap_data_pos1[item_idx, pair_idx] = np.mean(vals) if vals else 0.0

        fig_pos1, ax_pos1 = plt.subplots(figsize=(14, 6), dpi=150)
        im = ax_pos1.imshow(heatmap_data_pos1, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax_pos1.set_xticks(np.arange(num_adjacent_pairs))
        ax_pos1.set_xticklabels(adjacent_pair_labels)
        ax_pos1.set_yticks(np.arange(num_items))
        ax_pos1.set_yticklabels(item_labels)
        ax_pos1.set_xlabel('Adjacent Pair')
        ax_pos1.set_ylabel('Single Item')
        ax_pos1.set_title(f'TI: Adjacent Pair vs Single Item (Position 1) Correlations\n{layer_names[layer_idx]} Layer (After Training)')
        plt.colorbar(im, ax=ax_pos1, label='Correlation')

        for i in range(num_items):
            for j in range(num_adjacent_pairs):
                val = heatmap_data_pos1[i, j]
                text_color = 'white' if abs(val) > 0.5 else 'black'
                ax_pos1.text(j, i, f'{val:.2f}', ha='center', va='center', color=text_color, fontsize=7)

        plt.tight_layout()
        figures[f"ti_adjacent_pair_heatmap/pos1_{layer_names[layer_idx].lower().replace(' ', '_')}"] = fig_pos1
        plt.close(fig_pos1)

        # Position 2 heatmap
        heatmap_data_pos2 = np.zeros((num_items, num_adjacent_pairs))
        for pair_idx in range(num_adjacent_pairs):
            for item_idx in range(num_items):
                vals = adj_pair_item_corr_pos2[layer_idx][pair_idx][item_idx]
                heatmap_data_pos2[item_idx, pair_idx] = np.mean(vals) if vals else 0.0

        fig_pos2, ax_pos2 = plt.subplots(figsize=(14, 6), dpi=150)
        im = ax_pos2.imshow(heatmap_data_pos2, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax_pos2.set_xticks(np.arange(num_adjacent_pairs))
        ax_pos2.set_xticklabels(adjacent_pair_labels)
        ax_pos2.set_yticks(np.arange(num_items))
        ax_pos2.set_yticklabels(item_labels)
        ax_pos2.set_xlabel('Adjacent Pair')
        ax_pos2.set_ylabel('Single Item')
        ax_pos2.set_title(f'TI: Adjacent Pair vs Single Item (Position 2) Correlations\n{layer_names[layer_idx]} Layer (After Training)')
        plt.colorbar(im, ax=ax_pos2, label='Correlation')

        for i in range(num_items):
            for j in range(num_adjacent_pairs):
                val = heatmap_data_pos2[i, j]
                text_color = 'white' if abs(val) > 0.5 else 'black'
                ax_pos2.text(j, i, f'{val:.2f}', ha='center', va='center', color=text_color, fontsize=7)

        plt.tight_layout()
        figures[f"ti_adjacent_pair_heatmap/pos2_{layer_names[layer_idx].lower().replace(' ', '_')}"] = fig_pos2
        plt.close(fig_pos2)

    model.train()
    return figures


def plot_linking_pair_item_correlations(args, model, num_networks=128):
    """
    Compute correlations between DE/ED linking pair vectors and single item vectors at each layer.

    This analysis is performed on frozen networks after training on list 1 and list 2 trials
    (but BEFORE the linking pair DE/ED is shown). We want to see if the DE/ED representations
    correlate with individual item representations not present in the linking trial.

    For each layer, we compute:
    - Correlation of DE with all items when presented in position 1 (left: [item, zeros])
    - Correlation of DE with all items when presented in position 2 (right: [zeros, item])
    - Same for ED

    Items not in the linking pair: A, B, C (list 1 except D) and F, G, H (list 2 except E)

    Returns a dict of figures for wandb logging.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    num_items = 8  # Fixed for list linking: 4 items per list
    item_size = args.item_size
    item_labels = [chr(ord('A') + i) for i in range(num_items)]

    # Generate batch items (different items per network for robust averaging)
    batch_items = generate_batch_items(num_items, item_size, num_networks, change_items_throughout_batch=True)

    # Set list linking parameters
    num_trials_list_1 = getattr(args, 'num_trials_list_1', 20)
    num_trials_list_2 = getattr(args, 'num_trials_list_2', 20)
    # NO linking pair trials - we train only on list 1 and list 2

    figures = {}

    # Storage for correlations across networks
    # Layers: embedding layer + extra_layers + final layer
    num_layers = 1 + args.extra_layers + 1
    layer_names = ['Embedding'] + [f'Hidden {i+1}' for i in range(args.extra_layers)] + ['Final']

    # Correlations BEFORE linking pair: [layer][item_idx] -> list of correlations across networks
    de_pos1_correlations = [[[] for _ in range(num_items)] for _ in range(num_layers)]
    de_pos2_correlations = [[[] for _ in range(num_items)] for _ in range(num_layers)]
    ed_pos1_correlations = [[[] for _ in range(num_items)] for _ in range(num_layers)]
    ed_pos2_correlations = [[[] for _ in range(num_items)] for _ in range(num_layers)]

    # Correlations AFTER linking pair shown once: [layer][item_idx] -> list of correlations
    # Half networks see DE first, half see ED first
    de_pos1_correlations_after = [[[] for _ in range(num_items)] for _ in range(num_layers)]
    de_pos2_correlations_after = [[[] for _ in range(num_items)] for _ in range(num_layers)]
    ed_pos1_correlations_after = [[[] for _ in range(num_items)] for _ in range(num_layers)]
    ed_pos2_correlations_after = [[[] for _ in range(num_items)] for _ in range(num_layers)]

    # Split by which linking pair was seen: correlations for networks that saw DE first
    de_pos1_correlations_after_sawDE = [[[] for _ in range(num_items)] for _ in range(num_layers)]
    de_pos2_correlations_after_sawDE = [[[] for _ in range(num_items)] for _ in range(num_layers)]
    ed_pos1_correlations_after_sawDE = [[[] for _ in range(num_items)] for _ in range(num_layers)]
    ed_pos2_correlations_after_sawDE = [[[] for _ in range(num_items)] for _ in range(num_layers)]

    # Split by which linking pair was seen: correlations for networks that saw ED first
    de_pos1_correlations_after_sawED = [[[] for _ in range(num_items)] for _ in range(num_layers)]
    de_pos2_correlations_after_sawED = [[[] for _ in range(num_items)] for _ in range(num_layers)]
    ed_pos1_correlations_after_sawED = [[[] for _ in range(num_items)] for _ in range(num_layers)]
    ed_pos2_correlations_after_sawED = [[[] for _ in range(num_items)] for _ in range(num_layers)]

    # Correlations AFTER both linking pairs shown: [layer][item_idx] -> list of correlations
    # Networks that saw DE first now see ED, and vice versa
    de_pos1_correlations_after_both = [[[] for _ in range(num_items)] for _ in range(num_layers)]
    de_pos2_correlations_after_both = [[[] for _ in range(num_items)] for _ in range(num_layers)]
    ed_pos1_correlations_after_both = [[[] for _ in range(num_items)] for _ in range(num_layers)]
    ed_pos2_correlations_after_both = [[[] for _ in range(num_items)] for _ in range(num_layers)]

    # Adjacent pair heatmap data: correlations between adjacent pairs and single items
    # After list training (before linking pair)
    # pair_labels: AB, BA, BC, CB, CD, DC, EF, FE, FG, GF, GH, HG
    # For each layer: [pair_idx][item_idx] -> list of correlations across networks
    adjacent_pairs = [
        (0, 1), (1, 0),  # AB, BA
        (1, 2), (2, 1),  # BC, CB
        (2, 3), (3, 2),  # CD, DC
        (4, 5), (5, 4),  # EF, FE
        (5, 6), (6, 5),  # FG, GF
        (6, 7), (7, 6),  # GH, HG
    ]
    adjacent_pair_labels = ['AB', 'BA', 'BC', 'CB', 'CD', 'DC', 'EF', 'FE', 'FG', 'GF', 'GH', 'HG']
    num_adjacent_pairs = len(adjacent_pairs)
    # [layer][pair_idx][item_idx] -> list of correlations
    adjacent_pair_item_corr_pos1 = [[[[] for _ in range(num_items)] for _ in range(num_adjacent_pairs)] for _ in range(num_layers)]
    adjacent_pair_item_corr_pos2 = [[[[] for _ in range(num_items)] for _ in range(num_adjacent_pairs)] for _ in range(num_layers)]

    for network_idx in range(num_networks):
        items = batch_items[network_idx]  # (num_items, item_size)

        # Initialize plastic weights for this network
        plastic_weights = torch.zeros(1, args.hidden_size, args.hidden_size,
                                       dtype=torch.float32, requires_grad=False).to(device)
        extra_plastic_weights = [torch.zeros(1, args.hidden_size, args.hidden_size,
                                              dtype=torch.float32, requires_grad=False).to(device)
                                 for _ in range(args.extra_layers)]

        # === Train on list 1 (items 0-3: A, B, C, D) ===
        list_1_items = items[:4]
        for _ in range(num_trials_list_1):
            # Random adjacent pair from list 1
            high_idx = np.random.randint(0, 3)  # 0, 1, or 2
            low_idx = high_idx + 1
            item_high = list_1_items[high_idx]
            item_low = list_1_items[low_idx]

            # Random presentation order
            if np.random.random() < 0.5:
                trial_input = np.concatenate([item_high, item_low])
                correct = 0.0  # left item is higher
            else:
                trial_input = np.concatenate([item_low, item_high])
                correct = 1.0  # right item is higher

            trial_tensor = torch.tensor(trial_input, dtype=torch.float32).unsqueeze(0).to(device)
            correct_tensor = torch.tensor([correct], dtype=torch.float32).to(device)

            with torch.inference_mode():
                output = model(trial_tensor, plastic_weights, correct_tensor,
                              extra_plastic_weights=extra_plastic_weights, store_embeddings=False)
            plastic_weights = output.plastic_weights
            extra_plastic_weights = output.extra_plastic_weights

        # === Train on list 2 (items 4-7: E, F, G, H) ===
        list_2_items = items[4:]
        for _ in range(num_trials_list_2):
            # Random adjacent pair from list 2
            high_idx = np.random.randint(0, 3)  # 0, 1, or 2 (relative to list 2)
            low_idx = high_idx + 1
            item_high = list_2_items[high_idx]
            item_low = list_2_items[low_idx]

            # Random presentation order
            if np.random.random() < 0.5:
                trial_input = np.concatenate([item_high, item_low])
                correct = 0.0
            else:
                trial_input = np.concatenate([item_low, item_high])
                correct = 1.0

            trial_tensor = torch.tensor(trial_input, dtype=torch.float32).unsqueeze(0).to(device)
            correct_tensor = torch.tensor([correct], dtype=torch.float32).to(device)

            with torch.inference_mode():
                output = model(trial_tensor, plastic_weights, correct_tensor,
                              extra_plastic_weights=extra_plastic_weights, store_embeddings=False)
            plastic_weights = output.plastic_weights
            extra_plastic_weights = output.extra_plastic_weights

        # === Freeze plastic weights ===
        frozen_pw = plastic_weights.clone()
        frozen_epw = [epw.clone() for epw in extra_plastic_weights]

        # === Get DE representation (D=item 3, E=item 4) ===
        item_d = items[3]
        item_e = items[4]
        de_input = np.concatenate([item_d, item_e])
        de_tensor = torch.tensor(de_input, dtype=torch.float32).unsqueeze(0).to(device)
        # Use dummy reward (won't affect embeddings in inference mode)
        dummy_reward = torch.tensor([0.0], dtype=torch.float32).to(device)

        with torch.inference_mode():
            output_de = model(de_tensor, frozen_pw, dummy_reward,
                             extra_plastic_weights=frozen_epw, store_embeddings=True)
        de_embeddings = [emb.squeeze(0).cpu().numpy() for emb in output_de.embeddings]

        # === Get ED representation ===
        ed_input = np.concatenate([item_e, item_d])
        ed_tensor = torch.tensor(ed_input, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.inference_mode():
            output_ed = model(ed_tensor, frozen_pw, dummy_reward,
                             extra_plastic_weights=frozen_epw, store_embeddings=True)
        ed_embeddings = [emb.squeeze(0).cpu().numpy() for emb in output_ed.embeddings]

        # === Get single item representations ===
        zeros = np.zeros(item_size)

        # Store all single item embeddings for adjacent pair correlations
        all_pos1_embeddings = []  # [item_idx][layer_idx] -> embedding
        all_pos2_embeddings = []

        for item_idx in range(num_items):
            item = items[item_idx]

            # Position 1: [item, zeros]
            pos1_input = np.concatenate([item, zeros])
            pos1_tensor = torch.tensor(pos1_input, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.inference_mode():
                output_pos1 = model(pos1_tensor, frozen_pw, dummy_reward,
                                   extra_plastic_weights=frozen_epw, store_embeddings=True)
            pos1_embeddings = [emb.squeeze(0).cpu().numpy() for emb in output_pos1.embeddings]
            all_pos1_embeddings.append(pos1_embeddings)

            # Position 2: [zeros, item]
            pos2_input = np.concatenate([zeros, item])
            pos2_tensor = torch.tensor(pos2_input, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.inference_mode():
                output_pos2 = model(pos2_tensor, frozen_pw, dummy_reward,
                                   extra_plastic_weights=frozen_epw, store_embeddings=True)
            pos2_embeddings = [emb.squeeze(0).cpu().numpy() for emb in output_pos2.embeddings]
            all_pos2_embeddings.append(pos2_embeddings)

            # Compute correlations at each layer
            for layer_idx in range(num_layers):
                # Correlation with DE
                de_emb = de_embeddings[layer_idx]
                pos1_emb = pos1_embeddings[layer_idx]
                pos2_emb = pos2_embeddings[layer_idx]
                ed_emb = ed_embeddings[layer_idx]

                # Pearson correlation
                corr_de_pos1 = np.corrcoef(de_emb, pos1_emb)[0, 1]
                corr_de_pos2 = np.corrcoef(de_emb, pos2_emb)[0, 1]
                corr_ed_pos1 = np.corrcoef(ed_emb, pos1_emb)[0, 1]
                corr_ed_pos2 = np.corrcoef(ed_emb, pos2_emb)[0, 1]

                # Handle NaN (can happen with zero variance)
                if np.isnan(corr_de_pos1):
                    corr_de_pos1 = 0.0
                if np.isnan(corr_de_pos2):
                    corr_de_pos2 = 0.0
                if np.isnan(corr_ed_pos1):
                    corr_ed_pos1 = 0.0
                if np.isnan(corr_ed_pos2):
                    corr_ed_pos2 = 0.0

                de_pos1_correlations[layer_idx][item_idx].append(corr_de_pos1)
                de_pos2_correlations[layer_idx][item_idx].append(corr_de_pos2)
                ed_pos1_correlations[layer_idx][item_idx].append(corr_ed_pos1)
                ed_pos2_correlations[layer_idx][item_idx].append(corr_ed_pos2)

        # === Compute adjacent pair vs single item correlations (heatmap data) ===
        for pair_idx, (left_item_idx, right_item_idx) in enumerate(adjacent_pairs):
            # Get the pair representation
            pair_input = np.concatenate([items[left_item_idx], items[right_item_idx]])
            pair_tensor = torch.tensor(pair_input, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.inference_mode():
                output_pair = model(pair_tensor, frozen_pw, dummy_reward,
                                    extra_plastic_weights=frozen_epw, store_embeddings=True)
            pair_embeddings = [emb.squeeze(0).cpu().numpy() for emb in output_pair.embeddings]

            # Compute correlations with each single item
            for item_idx in range(num_items):
                for layer_idx in range(num_layers):
                    pair_emb = pair_embeddings[layer_idx]
                    pos1_emb = all_pos1_embeddings[item_idx][layer_idx]
                    pos2_emb = all_pos2_embeddings[item_idx][layer_idx]

                    corr_pos1 = np.corrcoef(pair_emb, pos1_emb)[0, 1]
                    corr_pos2 = np.corrcoef(pair_emb, pos2_emb)[0, 1]

                    if np.isnan(corr_pos1):
                        corr_pos1 = 0.0
                    if np.isnan(corr_pos2):
                        corr_pos2 = 0.0

                    adjacent_pair_item_corr_pos1[layer_idx][pair_idx][item_idx].append(corr_pos1)
                    adjacent_pair_item_corr_pos2[layer_idx][pair_idx][item_idx].append(corr_pos2)

        # === AFTER: Show linking pair once and compute correlations ===
        # Half networks see DE, half see ED
        # D > E, so DE correct answer is 0 (left is higher), ED correct answer is 1 (right is higher)
        if network_idx < num_networks // 2:
            # Show DE
            linking_input = de_input
            linking_correct = 0.0  # D is higher, D is on left
        else:
            # Show ED
            linking_input = ed_input
            linking_correct = 1.0  # D is higher, D is on right

        linking_tensor = torch.tensor(linking_input, dtype=torch.float32).unsqueeze(0).to(device)
        linking_correct_tensor = torch.tensor([linking_correct], dtype=torch.float32).to(device)

        # Update plastic weights with linking pair
        with torch.inference_mode():
            output_linking = model(linking_tensor, frozen_pw, linking_correct_tensor,
                                   extra_plastic_weights=frozen_epw, store_embeddings=False)
        frozen_pw_after = output_linking.plastic_weights
        frozen_epw_after = output_linking.extra_plastic_weights

        # Get DE representation after linking pair shown
        with torch.inference_mode():
            output_de_after = model(de_tensor, frozen_pw_after, dummy_reward,
                                    extra_plastic_weights=frozen_epw_after, store_embeddings=True)
        de_embeddings_after = [emb.squeeze(0).cpu().numpy() for emb in output_de_after.embeddings]

        # Get ED representation after linking pair shown
        with torch.inference_mode():
            output_ed_after = model(ed_tensor, frozen_pw_after, dummy_reward,
                                    extra_plastic_weights=frozen_epw_after, store_embeddings=True)
        ed_embeddings_after = [emb.squeeze(0).cpu().numpy() for emb in output_ed_after.embeddings]

        # Get single item representations after linking pair (with updated weights)
        for item_idx in range(num_items):
            item = items[item_idx]

            # Position 1: [item, zeros]
            pos1_input = np.concatenate([item, zeros])
            pos1_tensor = torch.tensor(pos1_input, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.inference_mode():
                output_pos1_after = model(pos1_tensor, frozen_pw_after, dummy_reward,
                                          extra_plastic_weights=frozen_epw_after, store_embeddings=True)
            pos1_embeddings_after = [emb.squeeze(0).cpu().numpy() for emb in output_pos1_after.embeddings]

            # Position 2: [zeros, item]
            pos2_input = np.concatenate([zeros, item])
            pos2_tensor = torch.tensor(pos2_input, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.inference_mode():
                output_pos2_after = model(pos2_tensor, frozen_pw_after, dummy_reward,
                                          extra_plastic_weights=frozen_epw_after, store_embeddings=True)
            pos2_embeddings_after = [emb.squeeze(0).cpu().numpy() for emb in output_pos2_after.embeddings]

            # Compute correlations at each layer
            for layer_idx in range(num_layers):
                de_emb_after = de_embeddings_after[layer_idx]
                ed_emb_after = ed_embeddings_after[layer_idx]
                pos1_emb_after = pos1_embeddings_after[layer_idx]
                pos2_emb_after = pos2_embeddings_after[layer_idx]

                corr_de_pos1_after = np.corrcoef(de_emb_after, pos1_emb_after)[0, 1]
                corr_de_pos2_after = np.corrcoef(de_emb_after, pos2_emb_after)[0, 1]
                corr_ed_pos1_after = np.corrcoef(ed_emb_after, pos1_emb_after)[0, 1]
                corr_ed_pos2_after = np.corrcoef(ed_emb_after, pos2_emb_after)[0, 1]

                # Handle NaN
                if np.isnan(corr_de_pos1_after):
                    corr_de_pos1_after = 0.0
                if np.isnan(corr_de_pos2_after):
                    corr_de_pos2_after = 0.0
                if np.isnan(corr_ed_pos1_after):
                    corr_ed_pos1_after = 0.0
                if np.isnan(corr_ed_pos2_after):
                    corr_ed_pos2_after = 0.0

                de_pos1_correlations_after[layer_idx][item_idx].append(corr_de_pos1_after)
                de_pos2_correlations_after[layer_idx][item_idx].append(corr_de_pos2_after)
                ed_pos1_correlations_after[layer_idx][item_idx].append(corr_ed_pos1_after)
                ed_pos2_correlations_after[layer_idx][item_idx].append(corr_ed_pos2_after)

                # Store split by which linking pair was seen
                if network_idx < num_networks // 2:
                    # This network saw DE
                    de_pos1_correlations_after_sawDE[layer_idx][item_idx].append(corr_de_pos1_after)
                    de_pos2_correlations_after_sawDE[layer_idx][item_idx].append(corr_de_pos2_after)
                    ed_pos1_correlations_after_sawDE[layer_idx][item_idx].append(corr_ed_pos1_after)
                    ed_pos2_correlations_after_sawDE[layer_idx][item_idx].append(corr_ed_pos2_after)
                else:
                    # This network saw ED
                    de_pos1_correlations_after_sawED[layer_idx][item_idx].append(corr_de_pos1_after)
                    de_pos2_correlations_after_sawED[layer_idx][item_idx].append(corr_de_pos2_after)
                    ed_pos1_correlations_after_sawED[layer_idx][item_idx].append(corr_ed_pos1_after)
                    ed_pos2_correlations_after_sawED[layer_idx][item_idx].append(corr_ed_pos2_after)

        # === AFTER BOTH: Show the other linking pair ===
        # Networks that saw DE first now see ED, and vice versa
        if network_idx < num_networks // 2:
            # This network saw DE first, now show ED
            linking_input_2 = ed_input
            linking_correct_2 = 1.0  # D is higher, D is on right in ED
        else:
            # This network saw ED first, now show DE
            linking_input_2 = de_input
            linking_correct_2 = 0.0  # D is higher, D is on left in DE

        linking_tensor_2 = torch.tensor(linking_input_2, dtype=torch.float32).unsqueeze(0).to(device)
        linking_correct_tensor_2 = torch.tensor([linking_correct_2], dtype=torch.float32).to(device)

        # Update plastic weights with second linking pair
        with torch.inference_mode():
            output_linking_2 = model(linking_tensor_2, frozen_pw_after, linking_correct_tensor_2,
                                     extra_plastic_weights=frozen_epw_after, store_embeddings=False)
        frozen_pw_after_both = output_linking_2.plastic_weights
        frozen_epw_after_both = output_linking_2.extra_plastic_weights

        # Get DE representation after both linking pairs shown
        with torch.inference_mode():
            output_de_after_both = model(de_tensor, frozen_pw_after_both, dummy_reward,
                                         extra_plastic_weights=frozen_epw_after_both, store_embeddings=True)
        de_embeddings_after_both = [emb.squeeze(0).cpu().numpy() for emb in output_de_after_both.embeddings]

        # Get ED representation after both linking pairs shown
        with torch.inference_mode():
            output_ed_after_both = model(ed_tensor, frozen_pw_after_both, dummy_reward,
                                         extra_plastic_weights=frozen_epw_after_both, store_embeddings=True)
        ed_embeddings_after_both = [emb.squeeze(0).cpu().numpy() for emb in output_ed_after_both.embeddings]

        # Get single item representations after both linking pairs
        for item_idx in range(num_items):
            item = items[item_idx]

            # Position 1: [item, zeros]
            pos1_input = np.concatenate([item, zeros])
            pos1_tensor = torch.tensor(pos1_input, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.inference_mode():
                output_pos1_after_both = model(pos1_tensor, frozen_pw_after_both, dummy_reward,
                                               extra_plastic_weights=frozen_epw_after_both, store_embeddings=True)
            pos1_embeddings_after_both = [emb.squeeze(0).cpu().numpy() for emb in output_pos1_after_both.embeddings]

            # Position 2: [zeros, item]
            pos2_input = np.concatenate([zeros, item])
            pos2_tensor = torch.tensor(pos2_input, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.inference_mode():
                output_pos2_after_both = model(pos2_tensor, frozen_pw_after_both, dummy_reward,
                                               extra_plastic_weights=frozen_epw_after_both, store_embeddings=True)
            pos2_embeddings_after_both = [emb.squeeze(0).cpu().numpy() for emb in output_pos2_after_both.embeddings]

            # Compute correlations at each layer
            for layer_idx in range(num_layers):
                de_emb_both = de_embeddings_after_both[layer_idx]
                ed_emb_both = ed_embeddings_after_both[layer_idx]
                pos1_emb_both = pos1_embeddings_after_both[layer_idx]
                pos2_emb_both = pos2_embeddings_after_both[layer_idx]

                corr_de_pos1_both = np.corrcoef(de_emb_both, pos1_emb_both)[0, 1]
                corr_de_pos2_both = np.corrcoef(de_emb_both, pos2_emb_both)[0, 1]
                corr_ed_pos1_both = np.corrcoef(ed_emb_both, pos1_emb_both)[0, 1]
                corr_ed_pos2_both = np.corrcoef(ed_emb_both, pos2_emb_both)[0, 1]

                # Handle NaN
                if np.isnan(corr_de_pos1_both):
                    corr_de_pos1_both = 0.0
                if np.isnan(corr_de_pos2_both):
                    corr_de_pos2_both = 0.0
                if np.isnan(corr_ed_pos1_both):
                    corr_ed_pos1_both = 0.0
                if np.isnan(corr_ed_pos2_both):
                    corr_ed_pos2_both = 0.0

                de_pos1_correlations_after_both[layer_idx][item_idx].append(corr_de_pos1_both)
                de_pos2_correlations_after_both[layer_idx][item_idx].append(corr_de_pos2_both)
                ed_pos1_correlations_after_both[layer_idx][item_idx].append(corr_ed_pos1_both)
                ed_pos2_correlations_after_both[layer_idx][item_idx].append(corr_ed_pos2_both)

    # === Create bar charts ===
    # Include all items (A-H), with D and E being the linking pair items
    all_items = list(range(num_items))
    all_labels = item_labels

    # Compute means and standard errors
    def compute_stats(correlations_list, item_indices):
        means = []
        sems = []
        for item_idx in item_indices:
            vals = np.array(correlations_list[item_idx])
            means.append(np.mean(vals))
            sems.append(np.std(vals) / np.sqrt(len(vals)))
        return np.array(means), np.array(sems)

    # Create figure for each layer
    for layer_idx in range(num_layers):
        # DE correlations
        fig_de, axes_de = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

        # Position 1 correlations
        means_pos1, sems_pos1 = compute_stats(de_pos1_correlations[layer_idx], all_items)
        x = np.arange(len(all_items))
        bars1 = axes_de[0].bar(x, means_pos1, yerr=sems_pos1, capsize=5, color='steelblue', edgecolor='black')
        axes_de[0].set_xticks(x)
        axes_de[0].set_xticklabels(all_labels)
        axes_de[0].set_xlabel('Item')
        axes_de[0].set_ylabel('Correlation with DE')
        axes_de[0].set_title(f'DE vs Items in Position 1 (Left)\n{layer_names[layer_idx]} Layer')
        axes_de[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes_de[0].set_ylim(-1, 1)

        # Add value labels
        for bar, mean in zip(bars1, means_pos1):
            axes_de[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                           f'{mean:.2f}', ha='center', fontsize=8)

        # Position 2 correlations
        means_pos2, sems_pos2 = compute_stats(de_pos2_correlations[layer_idx], all_items)
        bars2 = axes_de[1].bar(x, means_pos2, yerr=sems_pos2, capsize=5, color='coral', edgecolor='black')
        axes_de[1].set_xticks(x)
        axes_de[1].set_xticklabels(all_labels)
        axes_de[1].set_xlabel('Item')
        axes_de[1].set_ylabel('Correlation with DE')
        axes_de[1].set_title(f'DE vs Items in Position 2 (Right)\n{layer_names[layer_idx]} Layer')
        axes_de[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes_de[1].set_ylim(-1, 1)

        for bar, mean in zip(bars2, means_pos2):
            axes_de[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                           f'{mean:.2f}', ha='center', fontsize=8)

        plt.tight_layout()
        figures[f"ll_linking_corr/de_{layer_names[layer_idx].lower().replace(' ', '_')}"] = fig_de
        plt.close(fig_de)

        # ED correlations
        fig_ed, axes_ed = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

        # Position 1 correlations
        means_pos1, sems_pos1 = compute_stats(ed_pos1_correlations[layer_idx], all_items)
        bars1 = axes_ed[0].bar(x, means_pos1, yerr=sems_pos1, capsize=5, color='steelblue', edgecolor='black')
        axes_ed[0].set_xticks(x)
        axes_ed[0].set_xticklabels(all_labels)
        axes_ed[0].set_xlabel('Item')
        axes_ed[0].set_ylabel('Correlation with ED')
        axes_ed[0].set_title(f'ED vs Items in Position 1 (Left)\n{layer_names[layer_idx]} Layer')
        axes_ed[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes_ed[0].set_ylim(-1, 1)

        for bar, mean in zip(bars1, means_pos1):
            axes_ed[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                           f'{mean:.2f}', ha='center', fontsize=8)

        # Position 2 correlations
        means_pos2, sems_pos2 = compute_stats(ed_pos2_correlations[layer_idx], all_items)
        bars2 = axes_ed[1].bar(x, means_pos2, yerr=sems_pos2, capsize=5, color='coral', edgecolor='black')
        axes_ed[1].set_xticks(x)
        axes_ed[1].set_xticklabels(all_labels)
        axes_ed[1].set_xlabel('Item')
        axes_ed[1].set_ylabel('Correlation with ED')
        axes_ed[1].set_title(f'ED vs Items in Position 2 (Right)\n{layer_names[layer_idx]} Layer')
        axes_ed[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes_ed[1].set_ylim(-1, 1)

        for bar, mean in zip(bars2, means_pos2):
            axes_ed[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                           f'{mean:.2f}', ha='center', fontsize=8)

        plt.tight_layout()
        figures[f"ll_linking_corr/ed_{layer_names[layer_idx].lower().replace(' ', '_')}"] = fig_ed
        plt.close(fig_ed)

    # === Create summary figure showing all layers ===
    # Compute mean across all non-linking items for each layer
    fig_summary, axes_summary = plt.subplots(2, 2, figsize=(14, 10), dpi=150)

    layer_x = np.arange(num_layers)

    # DE Position 1 - mean across items by layer
    de_pos1_layer_means = []
    de_pos1_layer_sems = []
    for layer_idx in range(num_layers):
        all_vals = []
        for item_idx in all_items:
            all_vals.extend(de_pos1_correlations[layer_idx][item_idx])
        de_pos1_layer_means.append(np.mean(all_vals))
        de_pos1_layer_sems.append(np.std(all_vals) / np.sqrt(len(all_vals)))

    axes_summary[0, 0].bar(layer_x, de_pos1_layer_means, yerr=de_pos1_layer_sems,
                           capsize=5, color='steelblue', edgecolor='black')
    axes_summary[0, 0].set_xticks(layer_x)
    axes_summary[0, 0].set_xticklabels(layer_names, rotation=45, ha='right')
    axes_summary[0, 0].set_ylabel('Mean Correlation')
    axes_summary[0, 0].set_title('DE vs Items in Position 1 (Left)')
    axes_summary[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes_summary[0, 0].set_ylim(-1, 1)

    # DE Position 2
    de_pos2_layer_means = []
    de_pos2_layer_sems = []
    for layer_idx in range(num_layers):
        all_vals = []
        for item_idx in all_items:
            all_vals.extend(de_pos2_correlations[layer_idx][item_idx])
        de_pos2_layer_means.append(np.mean(all_vals))
        de_pos2_layer_sems.append(np.std(all_vals) / np.sqrt(len(all_vals)))

    axes_summary[0, 1].bar(layer_x, de_pos2_layer_means, yerr=de_pos2_layer_sems,
                           capsize=5, color='coral', edgecolor='black')
    axes_summary[0, 1].set_xticks(layer_x)
    axes_summary[0, 1].set_xticklabels(layer_names, rotation=45, ha='right')
    axes_summary[0, 1].set_ylabel('Mean Correlation')
    axes_summary[0, 1].set_title('DE vs Items in Position 2 (Right)')
    axes_summary[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes_summary[0, 1].set_ylim(-1, 1)

    # ED Position 1
    ed_pos1_layer_means = []
    ed_pos1_layer_sems = []
    for layer_idx in range(num_layers):
        all_vals = []
        for item_idx in all_items:
            all_vals.extend(ed_pos1_correlations[layer_idx][item_idx])
        ed_pos1_layer_means.append(np.mean(all_vals))
        ed_pos1_layer_sems.append(np.std(all_vals) / np.sqrt(len(all_vals)))

    axes_summary[1, 0].bar(layer_x, ed_pos1_layer_means, yerr=ed_pos1_layer_sems,
                           capsize=5, color='steelblue', edgecolor='black')
    axes_summary[1, 0].set_xticks(layer_x)
    axes_summary[1, 0].set_xticklabels(layer_names, rotation=45, ha='right')
    axes_summary[1, 0].set_ylabel('Mean Correlation')
    axes_summary[1, 0].set_title('ED vs Items in Position 1 (Left)')
    axes_summary[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes_summary[1, 0].set_ylim(-1, 1)

    # ED Position 2
    ed_pos2_layer_means = []
    ed_pos2_layer_sems = []
    for layer_idx in range(num_layers):
        all_vals = []
        for item_idx in all_items:
            all_vals.extend(ed_pos2_correlations[layer_idx][item_idx])
        ed_pos2_layer_means.append(np.mean(all_vals))
        ed_pos2_layer_sems.append(np.std(all_vals) / np.sqrt(len(all_vals)))

    axes_summary[1, 1].bar(layer_x, ed_pos2_layer_means, yerr=ed_pos2_layer_sems,
                           capsize=5, color='coral', edgecolor='black')
    axes_summary[1, 1].set_xticks(layer_x)
    axes_summary[1, 1].set_xticklabels(layer_names, rotation=45, ha='right')
    axes_summary[1, 1].set_ylabel('Mean Correlation')
    axes_summary[1, 1].set_title('ED vs Items in Position 2 (Right)')
    axes_summary[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes_summary[1, 1].set_ylim(-1, 1)

    plt.suptitle('Linking Pair vs Single Item Correlations by Layer\n(Before Linking Pair Training)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    figures["ll_linking_corr/summary_by_layer"] = fig_summary
    plt.close(fig_summary)

    # === Create heatmaps for adjacent pair vs single item correlations ===
    for layer_idx in range(num_layers):
        # Position 1 heatmap
        heatmap_data_pos1 = np.zeros((num_items, num_adjacent_pairs))
        for pair_idx in range(num_adjacent_pairs):
            for item_idx in range(num_items):
                vals = adjacent_pair_item_corr_pos1[layer_idx][pair_idx][item_idx]
                heatmap_data_pos1[item_idx, pair_idx] = np.mean(vals) if vals else 0.0

        fig_hm_pos1, ax_hm_pos1 = plt.subplots(figsize=(14, 6), dpi=150)
        im = ax_hm_pos1.imshow(heatmap_data_pos1, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax_hm_pos1.set_xticks(np.arange(num_adjacent_pairs))
        ax_hm_pos1.set_xticklabels(adjacent_pair_labels)
        ax_hm_pos1.set_yticks(np.arange(num_items))
        ax_hm_pos1.set_yticklabels(item_labels)
        ax_hm_pos1.set_xlabel('Adjacent Pair')
        ax_hm_pos1.set_ylabel('Single Item')
        ax_hm_pos1.set_title(f'Adjacent Pair vs Single Item (Position 1) Correlations\n{layer_names[layer_idx]} Layer (After List Training)')
        plt.colorbar(im, ax=ax_hm_pos1, label='Correlation')

        # Add text annotations
        for i in range(num_items):
            for j in range(num_adjacent_pairs):
                val = heatmap_data_pos1[i, j]
                text_color = 'white' if abs(val) > 0.5 else 'black'
                ax_hm_pos1.text(j, i, f'{val:.2f}', ha='center', va='center', color=text_color, fontsize=7)

        plt.tight_layout()
        figures[f"ll_adjacent_pair_heatmap/pos1_{layer_names[layer_idx].lower().replace(' ', '_')}"] = fig_hm_pos1
        plt.close(fig_hm_pos1)

        # Position 2 heatmap
        heatmap_data_pos2 = np.zeros((num_items, num_adjacent_pairs))
        for pair_idx in range(num_adjacent_pairs):
            for item_idx in range(num_items):
                vals = adjacent_pair_item_corr_pos2[layer_idx][pair_idx][item_idx]
                heatmap_data_pos2[item_idx, pair_idx] = np.mean(vals) if vals else 0.0

        fig_hm_pos2, ax_hm_pos2 = plt.subplots(figsize=(14, 6), dpi=150)
        im = ax_hm_pos2.imshow(heatmap_data_pos2, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax_hm_pos2.set_xticks(np.arange(num_adjacent_pairs))
        ax_hm_pos2.set_xticklabels(adjacent_pair_labels)
        ax_hm_pos2.set_yticks(np.arange(num_items))
        ax_hm_pos2.set_yticklabels(item_labels)
        ax_hm_pos2.set_xlabel('Adjacent Pair')
        ax_hm_pos2.set_ylabel('Single Item')
        ax_hm_pos2.set_title(f'Adjacent Pair vs Single Item (Position 2) Correlations\n{layer_names[layer_idx]} Layer (After List Training)')
        plt.colorbar(im, ax=ax_hm_pos2, label='Correlation')

        # Add text annotations
        for i in range(num_items):
            for j in range(num_adjacent_pairs):
                val = heatmap_data_pos2[i, j]
                text_color = 'white' if abs(val) > 0.5 else 'black'
                ax_hm_pos2.text(j, i, f'{val:.2f}', ha='center', va='center', color=text_color, fontsize=7)

        plt.tight_layout()
        figures[f"ll_adjacent_pair_heatmap/pos2_{layer_names[layer_idx].lower().replace(' ', '_')}"] = fig_hm_pos2
        plt.close(fig_hm_pos2)

    # === Create bar charts for AFTER linking pair shown ===
    # Create figure for each layer
    for layer_idx in range(num_layers):
        # DE correlations after
        fig_de_after, axes_de_after = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

        # Position 1 correlations
        means_pos1, sems_pos1 = compute_stats(de_pos1_correlations_after[layer_idx], all_items)
        x = np.arange(len(all_items))
        bars1 = axes_de_after[0].bar(x, means_pos1, yerr=sems_pos1, capsize=5, color='steelblue', edgecolor='black')
        axes_de_after[0].set_xticks(x)
        axes_de_after[0].set_xticklabels(all_labels)
        axes_de_after[0].set_xlabel('Item')
        axes_de_after[0].set_ylabel('Correlation with DE')
        axes_de_after[0].set_title(f'DE vs Items in Position 1 (Left) - AFTER\n{layer_names[layer_idx]} Layer')
        axes_de_after[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes_de_after[0].set_ylim(-1, 1)

        for bar, mean in zip(bars1, means_pos1):
            axes_de_after[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                                  f'{mean:.2f}', ha='center', fontsize=8)

        # Position 2 correlations
        means_pos2, sems_pos2 = compute_stats(de_pos2_correlations_after[layer_idx], all_items)
        bars2 = axes_de_after[1].bar(x, means_pos2, yerr=sems_pos2, capsize=5, color='coral', edgecolor='black')
        axes_de_after[1].set_xticks(x)
        axes_de_after[1].set_xticklabels(all_labels)
        axes_de_after[1].set_xlabel('Item')
        axes_de_after[1].set_ylabel('Correlation with DE')
        axes_de_after[1].set_title(f'DE vs Items in Position 2 (Right) - AFTER\n{layer_names[layer_idx]} Layer')
        axes_de_after[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes_de_after[1].set_ylim(-1, 1)

        for bar, mean in zip(bars2, means_pos2):
            axes_de_after[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                                  f'{mean:.2f}', ha='center', fontsize=8)

        plt.tight_layout()
        figures[f"ll_linking_corr_after/de_{layer_names[layer_idx].lower().replace(' ', '_')}"] = fig_de_after
        plt.close(fig_de_after)

        # ED correlations after
        fig_ed_after, axes_ed_after = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

        # Position 1 correlations
        means_pos1, sems_pos1 = compute_stats(ed_pos1_correlations_after[layer_idx], all_items)
        bars1 = axes_ed_after[0].bar(x, means_pos1, yerr=sems_pos1, capsize=5, color='steelblue', edgecolor='black')
        axes_ed_after[0].set_xticks(x)
        axes_ed_after[0].set_xticklabels(all_labels)
        axes_ed_after[0].set_xlabel('Item')
        axes_ed_after[0].set_ylabel('Correlation with ED')
        axes_ed_after[0].set_title(f'ED vs Items in Position 1 (Left) - AFTER\n{layer_names[layer_idx]} Layer')
        axes_ed_after[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes_ed_after[0].set_ylim(-1, 1)

        for bar, mean in zip(bars1, means_pos1):
            axes_ed_after[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                                  f'{mean:.2f}', ha='center', fontsize=8)

        # Position 2 correlations
        means_pos2, sems_pos2 = compute_stats(ed_pos2_correlations_after[layer_idx], all_items)
        bars2 = axes_ed_after[1].bar(x, means_pos2, yerr=sems_pos2, capsize=5, color='coral', edgecolor='black')
        axes_ed_after[1].set_xticks(x)
        axes_ed_after[1].set_xticklabels(all_labels)
        axes_ed_after[1].set_xlabel('Item')
        axes_ed_after[1].set_ylabel('Correlation with ED')
        axes_ed_after[1].set_title(f'ED vs Items in Position 2 (Right) - AFTER\n{layer_names[layer_idx]} Layer')
        axes_ed_after[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes_ed_after[1].set_ylim(-1, 1)

        for bar, mean in zip(bars2, means_pos2):
            axes_ed_after[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                                  f'{mean:.2f}', ha='center', fontsize=8)

        plt.tight_layout()
        figures[f"ll_linking_corr_after/ed_{layer_names[layer_idx].lower().replace(' ', '_')}"] = fig_ed_after
        plt.close(fig_ed_after)

        # === Split plots: Networks that saw DE ===
        # DE correlations for networks that saw DE
        fig_de_sawDE, axes_de_sawDE = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

        means_pos1, sems_pos1 = compute_stats(de_pos1_correlations_after_sawDE[layer_idx], all_items)
        bars1 = axes_de_sawDE[0].bar(x, means_pos1, yerr=sems_pos1, capsize=5, color='steelblue', edgecolor='black')
        axes_de_sawDE[0].set_xticks(x)
        axes_de_sawDE[0].set_xticklabels(all_labels)
        axes_de_sawDE[0].set_xlabel('Item')
        axes_de_sawDE[0].set_ylabel('Correlation with DE')
        axes_de_sawDE[0].set_title(f'DE vs Items in Position 1 (Left)\n{layer_names[layer_idx]} Layer (Saw DE)')
        axes_de_sawDE[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes_de_sawDE[0].set_ylim(-1, 1)

        for bar, mean in zip(bars1, means_pos1):
            axes_de_sawDE[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                                  f'{mean:.2f}', ha='center', fontsize=8)

        means_pos2, sems_pos2 = compute_stats(de_pos2_correlations_after_sawDE[layer_idx], all_items)
        bars2 = axes_de_sawDE[1].bar(x, means_pos2, yerr=sems_pos2, capsize=5, color='coral', edgecolor='black')
        axes_de_sawDE[1].set_xticks(x)
        axes_de_sawDE[1].set_xticklabels(all_labels)
        axes_de_sawDE[1].set_xlabel('Item')
        axes_de_sawDE[1].set_ylabel('Correlation with DE')
        axes_de_sawDE[1].set_title(f'DE vs Items in Position 2 (Right)\n{layer_names[layer_idx]} Layer (Saw DE)')
        axes_de_sawDE[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes_de_sawDE[1].set_ylim(-1, 1)

        for bar, mean in zip(bars2, means_pos2):
            axes_de_sawDE[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                                  f'{mean:.2f}', ha='center', fontsize=8)

        plt.tight_layout()
        figures[f"ll_linking_corr_after_split/de_sawDE_{layer_names[layer_idx].lower().replace(' ', '_')}"] = fig_de_sawDE
        plt.close(fig_de_sawDE)

        # ED correlations for networks that saw DE
        fig_ed_sawDE, axes_ed_sawDE = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

        means_pos1, sems_pos1 = compute_stats(ed_pos1_correlations_after_sawDE[layer_idx], all_items)
        bars1 = axes_ed_sawDE[0].bar(x, means_pos1, yerr=sems_pos1, capsize=5, color='steelblue', edgecolor='black')
        axes_ed_sawDE[0].set_xticks(x)
        axes_ed_sawDE[0].set_xticklabels(all_labels)
        axes_ed_sawDE[0].set_xlabel('Item')
        axes_ed_sawDE[0].set_ylabel('Correlation with ED')
        axes_ed_sawDE[0].set_title(f'ED vs Items in Position 1 (Left)\n{layer_names[layer_idx]} Layer (Saw DE)')
        axes_ed_sawDE[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes_ed_sawDE[0].set_ylim(-1, 1)

        for bar, mean in zip(bars1, means_pos1):
            axes_ed_sawDE[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                                  f'{mean:.2f}', ha='center', fontsize=8)

        means_pos2, sems_pos2 = compute_stats(ed_pos2_correlations_after_sawDE[layer_idx], all_items)
        bars2 = axes_ed_sawDE[1].bar(x, means_pos2, yerr=sems_pos2, capsize=5, color='coral', edgecolor='black')
        axes_ed_sawDE[1].set_xticks(x)
        axes_ed_sawDE[1].set_xticklabels(all_labels)
        axes_ed_sawDE[1].set_xlabel('Item')
        axes_ed_sawDE[1].set_ylabel('Correlation with ED')
        axes_ed_sawDE[1].set_title(f'ED vs Items in Position 2 (Right)\n{layer_names[layer_idx]} Layer (Saw DE)')
        axes_ed_sawDE[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes_ed_sawDE[1].set_ylim(-1, 1)

        for bar, mean in zip(bars2, means_pos2):
            axes_ed_sawDE[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                                  f'{mean:.2f}', ha='center', fontsize=8)

        plt.tight_layout()
        figures[f"ll_linking_corr_after_split/ed_sawDE_{layer_names[layer_idx].lower().replace(' ', '_')}"] = fig_ed_sawDE
        plt.close(fig_ed_sawDE)

        # === Split plots: Networks that saw ED ===
        # DE correlations for networks that saw ED
        fig_de_sawED, axes_de_sawED = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

        means_pos1, sems_pos1 = compute_stats(de_pos1_correlations_after_sawED[layer_idx], all_items)
        bars1 = axes_de_sawED[0].bar(x, means_pos1, yerr=sems_pos1, capsize=5, color='steelblue', edgecolor='black')
        axes_de_sawED[0].set_xticks(x)
        axes_de_sawED[0].set_xticklabels(all_labels)
        axes_de_sawED[0].set_xlabel('Item')
        axes_de_sawED[0].set_ylabel('Correlation with DE')
        axes_de_sawED[0].set_title(f'DE vs Items in Position 1 (Left)\n{layer_names[layer_idx]} Layer (Saw ED)')
        axes_de_sawED[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes_de_sawED[0].set_ylim(-1, 1)

        for bar, mean in zip(bars1, means_pos1):
            axes_de_sawED[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                                  f'{mean:.2f}', ha='center', fontsize=8)

        means_pos2, sems_pos2 = compute_stats(de_pos2_correlations_after_sawED[layer_idx], all_items)
        bars2 = axes_de_sawED[1].bar(x, means_pos2, yerr=sems_pos2, capsize=5, color='coral', edgecolor='black')
        axes_de_sawED[1].set_xticks(x)
        axes_de_sawED[1].set_xticklabels(all_labels)
        axes_de_sawED[1].set_xlabel('Item')
        axes_de_sawED[1].set_ylabel('Correlation with DE')
        axes_de_sawED[1].set_title(f'DE vs Items in Position 2 (Right)\n{layer_names[layer_idx]} Layer (Saw ED)')
        axes_de_sawED[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes_de_sawED[1].set_ylim(-1, 1)

        for bar, mean in zip(bars2, means_pos2):
            axes_de_sawED[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                                  f'{mean:.2f}', ha='center', fontsize=8)

        plt.tight_layout()
        figures[f"ll_linking_corr_after_split/de_sawED_{layer_names[layer_idx].lower().replace(' ', '_')}"] = fig_de_sawED
        plt.close(fig_de_sawED)

        # ED correlations for networks that saw ED
        fig_ed_sawED, axes_ed_sawED = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

        means_pos1, sems_pos1 = compute_stats(ed_pos1_correlations_after_sawED[layer_idx], all_items)
        bars1 = axes_ed_sawED[0].bar(x, means_pos1, yerr=sems_pos1, capsize=5, color='steelblue', edgecolor='black')
        axes_ed_sawED[0].set_xticks(x)
        axes_ed_sawED[0].set_xticklabels(all_labels)
        axes_ed_sawED[0].set_xlabel('Item')
        axes_ed_sawED[0].set_ylabel('Correlation with ED')
        axes_ed_sawED[0].set_title(f'ED vs Items in Position 1 (Left)\n{layer_names[layer_idx]} Layer (Saw ED)')
        axes_ed_sawED[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes_ed_sawED[0].set_ylim(-1, 1)

        for bar, mean in zip(bars1, means_pos1):
            axes_ed_sawED[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                                  f'{mean:.2f}', ha='center', fontsize=8)

        means_pos2, sems_pos2 = compute_stats(ed_pos2_correlations_after_sawED[layer_idx], all_items)
        bars2 = axes_ed_sawED[1].bar(x, means_pos2, yerr=sems_pos2, capsize=5, color='coral', edgecolor='black')
        axes_ed_sawED[1].set_xticks(x)
        axes_ed_sawED[1].set_xticklabels(all_labels)
        axes_ed_sawED[1].set_xlabel('Item')
        axes_ed_sawED[1].set_ylabel('Correlation with ED')
        axes_ed_sawED[1].set_title(f'ED vs Items in Position 2 (Right)\n{layer_names[layer_idx]} Layer (Saw ED)')
        axes_ed_sawED[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes_ed_sawED[1].set_ylim(-1, 1)

        for bar, mean in zip(bars2, means_pos2):
            axes_ed_sawED[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                                  f'{mean:.2f}', ha='center', fontsize=8)

        plt.tight_layout()
        figures[f"ll_linking_corr_after_split/ed_sawED_{layer_names[layer_idx].lower().replace(' ', '_')}"] = fig_ed_sawED
        plt.close(fig_ed_sawED)

    # === Create summary figure for AFTER ===
    fig_summary_after, axes_summary_after = plt.subplots(2, 2, figsize=(14, 10), dpi=150)

    # DE Position 1 after
    de_pos1_layer_means_after = []
    de_pos1_layer_sems_after = []
    for layer_idx in range(num_layers):
        all_vals = []
        for item_idx in all_items:
            all_vals.extend(de_pos1_correlations_after[layer_idx][item_idx])
        de_pos1_layer_means_after.append(np.mean(all_vals))
        de_pos1_layer_sems_after.append(np.std(all_vals) / np.sqrt(len(all_vals)))

    axes_summary_after[0, 0].bar(layer_x, de_pos1_layer_means_after, yerr=de_pos1_layer_sems_after,
                                  capsize=5, color='steelblue', edgecolor='black')
    axes_summary_after[0, 0].set_xticks(layer_x)
    axes_summary_after[0, 0].set_xticklabels(layer_names, rotation=45, ha='right')
    axes_summary_after[0, 0].set_ylabel('Mean Correlation')
    axes_summary_after[0, 0].set_title('DE vs Items in Position 1 (Left)')
    axes_summary_after[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes_summary_after[0, 0].set_ylim(-1, 1)

    # DE Position 2 after
    de_pos2_layer_means_after = []
    de_pos2_layer_sems_after = []
    for layer_idx in range(num_layers):
        all_vals = []
        for item_idx in all_items:
            all_vals.extend(de_pos2_correlations_after[layer_idx][item_idx])
        de_pos2_layer_means_after.append(np.mean(all_vals))
        de_pos2_layer_sems_after.append(np.std(all_vals) / np.sqrt(len(all_vals)))

    axes_summary_after[0, 1].bar(layer_x, de_pos2_layer_means_after, yerr=de_pos2_layer_sems_after,
                                  capsize=5, color='coral', edgecolor='black')
    axes_summary_after[0, 1].set_xticks(layer_x)
    axes_summary_after[0, 1].set_xticklabels(layer_names, rotation=45, ha='right')
    axes_summary_after[0, 1].set_ylabel('Mean Correlation')
    axes_summary_after[0, 1].set_title('DE vs Items in Position 2 (Right)')
    axes_summary_after[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes_summary_after[0, 1].set_ylim(-1, 1)

    # ED Position 1 after
    ed_pos1_layer_means_after = []
    ed_pos1_layer_sems_after = []
    for layer_idx in range(num_layers):
        all_vals = []
        for item_idx in all_items:
            all_vals.extend(ed_pos1_correlations_after[layer_idx][item_idx])
        ed_pos1_layer_means_after.append(np.mean(all_vals))
        ed_pos1_layer_sems_after.append(np.std(all_vals) / np.sqrt(len(all_vals)))

    axes_summary_after[1, 0].bar(layer_x, ed_pos1_layer_means_after, yerr=ed_pos1_layer_sems_after,
                                  capsize=5, color='steelblue', edgecolor='black')
    axes_summary_after[1, 0].set_xticks(layer_x)
    axes_summary_after[1, 0].set_xticklabels(layer_names, rotation=45, ha='right')
    axes_summary_after[1, 0].set_ylabel('Mean Correlation')
    axes_summary_after[1, 0].set_title('ED vs Items in Position 1 (Left)')
    axes_summary_after[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes_summary_after[1, 0].set_ylim(-1, 1)

    # ED Position 2 after
    ed_pos2_layer_means_after = []
    ed_pos2_layer_sems_after = []
    for layer_idx in range(num_layers):
        all_vals = []
        for item_idx in all_items:
            all_vals.extend(ed_pos2_correlations_after[layer_idx][item_idx])
        ed_pos2_layer_means_after.append(np.mean(all_vals))
        ed_pos2_layer_sems_after.append(np.std(all_vals) / np.sqrt(len(all_vals)))

    axes_summary_after[1, 1].bar(layer_x, ed_pos2_layer_means_after, yerr=ed_pos2_layer_sems_after,
                                  capsize=5, color='coral', edgecolor='black')
    axes_summary_after[1, 1].set_xticks(layer_x)
    axes_summary_after[1, 1].set_xticklabels(layer_names, rotation=45, ha='right')
    axes_summary_after[1, 1].set_ylabel('Mean Correlation')
    axes_summary_after[1, 1].set_title('ED vs Items in Position 2 (Right)')
    axes_summary_after[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes_summary_after[1, 1].set_ylim(-1, 1)

    plt.suptitle('Linking Pair vs Single Item Correlations by Layer\n(After Linking Pair Shown Once)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    figures["ll_linking_corr_after/summary_by_layer"] = fig_summary_after
    plt.close(fig_summary_after)

    # === Create summary figures for SPLIT data ===
    # Summary for networks that saw DE
    fig_summary_sawDE, axes_summary_sawDE = plt.subplots(2, 2, figsize=(14, 10), dpi=150)

    # DE Position 1 (saw DE)
    de_pos1_layer_means_sawDE = []
    de_pos1_layer_sems_sawDE = []
    for layer_idx in range(num_layers):
        all_vals = []
        for item_idx in all_items:
            all_vals.extend(de_pos1_correlations_after_sawDE[layer_idx][item_idx])
        de_pos1_layer_means_sawDE.append(np.mean(all_vals))
        de_pos1_layer_sems_sawDE.append(np.std(all_vals) / np.sqrt(len(all_vals)))

    axes_summary_sawDE[0, 0].bar(layer_x, de_pos1_layer_means_sawDE, yerr=de_pos1_layer_sems_sawDE,
                                  capsize=5, color='steelblue', edgecolor='black')
    axes_summary_sawDE[0, 0].set_xticks(layer_x)
    axes_summary_sawDE[0, 0].set_xticklabels(layer_names, rotation=45, ha='right')
    axes_summary_sawDE[0, 0].set_ylabel('Mean Correlation')
    axes_summary_sawDE[0, 0].set_title('DE vs Items in Position 1 (Left)')
    axes_summary_sawDE[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes_summary_sawDE[0, 0].set_ylim(-1, 1)

    # DE Position 2 (saw DE)
    de_pos2_layer_means_sawDE = []
    de_pos2_layer_sems_sawDE = []
    for layer_idx in range(num_layers):
        all_vals = []
        for item_idx in all_items:
            all_vals.extend(de_pos2_correlations_after_sawDE[layer_idx][item_idx])
        de_pos2_layer_means_sawDE.append(np.mean(all_vals))
        de_pos2_layer_sems_sawDE.append(np.std(all_vals) / np.sqrt(len(all_vals)))

    axes_summary_sawDE[0, 1].bar(layer_x, de_pos2_layer_means_sawDE, yerr=de_pos2_layer_sems_sawDE,
                                  capsize=5, color='coral', edgecolor='black')
    axes_summary_sawDE[0, 1].set_xticks(layer_x)
    axes_summary_sawDE[0, 1].set_xticklabels(layer_names, rotation=45, ha='right')
    axes_summary_sawDE[0, 1].set_ylabel('Mean Correlation')
    axes_summary_sawDE[0, 1].set_title('DE vs Items in Position 2 (Right)')
    axes_summary_sawDE[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes_summary_sawDE[0, 1].set_ylim(-1, 1)

    # ED Position 1 (saw DE)
    ed_pos1_layer_means_sawDE = []
    ed_pos1_layer_sems_sawDE = []
    for layer_idx in range(num_layers):
        all_vals = []
        for item_idx in all_items:
            all_vals.extend(ed_pos1_correlations_after_sawDE[layer_idx][item_idx])
        ed_pos1_layer_means_sawDE.append(np.mean(all_vals))
        ed_pos1_layer_sems_sawDE.append(np.std(all_vals) / np.sqrt(len(all_vals)))

    axes_summary_sawDE[1, 0].bar(layer_x, ed_pos1_layer_means_sawDE, yerr=ed_pos1_layer_sems_sawDE,
                                  capsize=5, color='steelblue', edgecolor='black')
    axes_summary_sawDE[1, 0].set_xticks(layer_x)
    axes_summary_sawDE[1, 0].set_xticklabels(layer_names, rotation=45, ha='right')
    axes_summary_sawDE[1, 0].set_ylabel('Mean Correlation')
    axes_summary_sawDE[1, 0].set_title('ED vs Items in Position 1 (Left)')
    axes_summary_sawDE[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes_summary_sawDE[1, 0].set_ylim(-1, 1)

    # ED Position 2 (saw DE)
    ed_pos2_layer_means_sawDE = []
    ed_pos2_layer_sems_sawDE = []
    for layer_idx in range(num_layers):
        all_vals = []
        for item_idx in all_items:
            all_vals.extend(ed_pos2_correlations_after_sawDE[layer_idx][item_idx])
        ed_pos2_layer_means_sawDE.append(np.mean(all_vals))
        ed_pos2_layer_sems_sawDE.append(np.std(all_vals) / np.sqrt(len(all_vals)))

    axes_summary_sawDE[1, 1].bar(layer_x, ed_pos2_layer_means_sawDE, yerr=ed_pos2_layer_sems_sawDE,
                                  capsize=5, color='coral', edgecolor='black')
    axes_summary_sawDE[1, 1].set_xticks(layer_x)
    axes_summary_sawDE[1, 1].set_xticklabels(layer_names, rotation=45, ha='right')
    axes_summary_sawDE[1, 1].set_ylabel('Mean Correlation')
    axes_summary_sawDE[1, 1].set_title('ED vs Items in Position 2 (Right)')
    axes_summary_sawDE[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes_summary_sawDE[1, 1].set_ylim(-1, 1)

    plt.suptitle('Linking Pair vs Single Item Correlations by Layer\n(Networks that Saw DE)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    figures["ll_linking_corr_after_split/summary_sawDE"] = fig_summary_sawDE
    plt.close(fig_summary_sawDE)

    # Summary for networks that saw ED
    fig_summary_sawED, axes_summary_sawED = plt.subplots(2, 2, figsize=(14, 10), dpi=150)

    # DE Position 1 (saw ED)
    de_pos1_layer_means_sawED = []
    de_pos1_layer_sems_sawED = []
    for layer_idx in range(num_layers):
        all_vals = []
        for item_idx in all_items:
            all_vals.extend(de_pos1_correlations_after_sawED[layer_idx][item_idx])
        de_pos1_layer_means_sawED.append(np.mean(all_vals))
        de_pos1_layer_sems_sawED.append(np.std(all_vals) / np.sqrt(len(all_vals)))

    axes_summary_sawED[0, 0].bar(layer_x, de_pos1_layer_means_sawED, yerr=de_pos1_layer_sems_sawED,
                                  capsize=5, color='steelblue', edgecolor='black')
    axes_summary_sawED[0, 0].set_xticks(layer_x)
    axes_summary_sawED[0, 0].set_xticklabels(layer_names, rotation=45, ha='right')
    axes_summary_sawED[0, 0].set_ylabel('Mean Correlation')
    axes_summary_sawED[0, 0].set_title('DE vs Items in Position 1 (Left)')
    axes_summary_sawED[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes_summary_sawED[0, 0].set_ylim(-1, 1)

    # DE Position 2 (saw ED)
    de_pos2_layer_means_sawED = []
    de_pos2_layer_sems_sawED = []
    for layer_idx in range(num_layers):
        all_vals = []
        for item_idx in all_items:
            all_vals.extend(de_pos2_correlations_after_sawED[layer_idx][item_idx])
        de_pos2_layer_means_sawED.append(np.mean(all_vals))
        de_pos2_layer_sems_sawED.append(np.std(all_vals) / np.sqrt(len(all_vals)))

    axes_summary_sawED[0, 1].bar(layer_x, de_pos2_layer_means_sawED, yerr=de_pos2_layer_sems_sawED,
                                  capsize=5, color='coral', edgecolor='black')
    axes_summary_sawED[0, 1].set_xticks(layer_x)
    axes_summary_sawED[0, 1].set_xticklabels(layer_names, rotation=45, ha='right')
    axes_summary_sawED[0, 1].set_ylabel('Mean Correlation')
    axes_summary_sawED[0, 1].set_title('DE vs Items in Position 2 (Right)')
    axes_summary_sawED[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes_summary_sawED[0, 1].set_ylim(-1, 1)

    # ED Position 1 (saw ED)
    ed_pos1_layer_means_sawED = []
    ed_pos1_layer_sems_sawED = []
    for layer_idx in range(num_layers):
        all_vals = []
        for item_idx in all_items:
            all_vals.extend(ed_pos1_correlations_after_sawED[layer_idx][item_idx])
        ed_pos1_layer_means_sawED.append(np.mean(all_vals))
        ed_pos1_layer_sems_sawED.append(np.std(all_vals) / np.sqrt(len(all_vals)))

    axes_summary_sawED[1, 0].bar(layer_x, ed_pos1_layer_means_sawED, yerr=ed_pos1_layer_sems_sawED,
                                  capsize=5, color='steelblue', edgecolor='black')
    axes_summary_sawED[1, 0].set_xticks(layer_x)
    axes_summary_sawED[1, 0].set_xticklabels(layer_names, rotation=45, ha='right')
    axes_summary_sawED[1, 0].set_ylabel('Mean Correlation')
    axes_summary_sawED[1, 0].set_title('ED vs Items in Position 1 (Left)')
    axes_summary_sawED[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes_summary_sawED[1, 0].set_ylim(-1, 1)

    # ED Position 2 (saw ED)
    ed_pos2_layer_means_sawED = []
    ed_pos2_layer_sems_sawED = []
    for layer_idx in range(num_layers):
        all_vals = []
        for item_idx in all_items:
            all_vals.extend(ed_pos2_correlations_after_sawED[layer_idx][item_idx])
        ed_pos2_layer_means_sawED.append(np.mean(all_vals))
        ed_pos2_layer_sems_sawED.append(np.std(all_vals) / np.sqrt(len(all_vals)))

    axes_summary_sawED[1, 1].bar(layer_x, ed_pos2_layer_means_sawED, yerr=ed_pos2_layer_sems_sawED,
                                  capsize=5, color='coral', edgecolor='black')
    axes_summary_sawED[1, 1].set_xticks(layer_x)
    axes_summary_sawED[1, 1].set_xticklabels(layer_names, rotation=45, ha='right')
    axes_summary_sawED[1, 1].set_ylabel('Mean Correlation')
    axes_summary_sawED[1, 1].set_title('ED vs Items in Position 2 (Right)')
    axes_summary_sawED[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes_summary_sawED[1, 1].set_ylim(-1, 1)

    plt.suptitle('Linking Pair vs Single Item Correlations by Layer\n(Networks that Saw ED)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    figures["ll_linking_corr_after_split/summary_sawED"] = fig_summary_sawED
    plt.close(fig_summary_sawED)

    # === Create bar charts for AFTER BOTH linking pairs shown ===
    for layer_idx in range(num_layers):
        # DE correlations after both
        fig_de_both, axes_de_both = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

        # Position 1 correlations
        means_pos1, sems_pos1 = compute_stats(de_pos1_correlations_after_both[layer_idx], all_items)
        x = np.arange(len(all_items))
        bars1 = axes_de_both[0].bar(x, means_pos1, yerr=sems_pos1, capsize=5, color='steelblue', edgecolor='black')
        axes_de_both[0].set_xticks(x)
        axes_de_both[0].set_xticklabels(all_labels)
        axes_de_both[0].set_xlabel('Item')
        axes_de_both[0].set_ylabel('Correlation with DE')
        axes_de_both[0].set_title(f'DE vs Items in Position 1 (Left) - AFTER BOTH\n{layer_names[layer_idx]} Layer')
        axes_de_both[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes_de_both[0].set_ylim(-1, 1)

        for bar, mean in zip(bars1, means_pos1):
            axes_de_both[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                                 f'{mean:.2f}', ha='center', fontsize=8)

        # Position 2 correlations
        means_pos2, sems_pos2 = compute_stats(de_pos2_correlations_after_both[layer_idx], all_items)
        bars2 = axes_de_both[1].bar(x, means_pos2, yerr=sems_pos2, capsize=5, color='coral', edgecolor='black')
        axes_de_both[1].set_xticks(x)
        axes_de_both[1].set_xticklabels(all_labels)
        axes_de_both[1].set_xlabel('Item')
        axes_de_both[1].set_ylabel('Correlation with DE')
        axes_de_both[1].set_title(f'DE vs Items in Position 2 (Right) - AFTER BOTH\n{layer_names[layer_idx]} Layer')
        axes_de_both[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes_de_both[1].set_ylim(-1, 1)

        for bar, mean in zip(bars2, means_pos2):
            axes_de_both[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                                 f'{mean:.2f}', ha='center', fontsize=8)

        plt.tight_layout()
        figures[f"ll_linking_corr_after_both/de_{layer_names[layer_idx].lower().replace(' ', '_')}"] = fig_de_both
        plt.close(fig_de_both)

        # ED correlations after both
        fig_ed_both, axes_ed_both = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

        # Position 1 correlations
        means_pos1, sems_pos1 = compute_stats(ed_pos1_correlations_after_both[layer_idx], all_items)
        bars1 = axes_ed_both[0].bar(x, means_pos1, yerr=sems_pos1, capsize=5, color='steelblue', edgecolor='black')
        axes_ed_both[0].set_xticks(x)
        axes_ed_both[0].set_xticklabels(all_labels)
        axes_ed_both[0].set_xlabel('Item')
        axes_ed_both[0].set_ylabel('Correlation with ED')
        axes_ed_both[0].set_title(f'ED vs Items in Position 1 (Left) - AFTER BOTH\n{layer_names[layer_idx]} Layer')
        axes_ed_both[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes_ed_both[0].set_ylim(-1, 1)

        for bar, mean in zip(bars1, means_pos1):
            axes_ed_both[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                                 f'{mean:.2f}', ha='center', fontsize=8)

        # Position 2 correlations
        means_pos2, sems_pos2 = compute_stats(ed_pos2_correlations_after_both[layer_idx], all_items)
        bars2 = axes_ed_both[1].bar(x, means_pos2, yerr=sems_pos2, capsize=5, color='coral', edgecolor='black')
        axes_ed_both[1].set_xticks(x)
        axes_ed_both[1].set_xticklabels(all_labels)
        axes_ed_both[1].set_xlabel('Item')
        axes_ed_both[1].set_ylabel('Correlation with ED')
        axes_ed_both[1].set_title(f'ED vs Items in Position 2 (Right) - AFTER BOTH\n{layer_names[layer_idx]} Layer')
        axes_ed_both[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes_ed_both[1].set_ylim(-1, 1)

        for bar, mean in zip(bars2, means_pos2):
            axes_ed_both[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                                 f'{mean:.2f}', ha='center', fontsize=8)

        plt.tight_layout()
        figures[f"ll_linking_corr_after_both/ed_{layer_names[layer_idx].lower().replace(' ', '_')}"] = fig_ed_both
        plt.close(fig_ed_both)

    # === Create summary figure for AFTER BOTH ===
    fig_summary_both, axes_summary_both = plt.subplots(2, 2, figsize=(14, 10), dpi=150)

    # DE Position 1 after both
    de_pos1_layer_means_both = []
    de_pos1_layer_sems_both = []
    for layer_idx in range(num_layers):
        all_vals = []
        for item_idx in all_items:
            all_vals.extend(de_pos1_correlations_after_both[layer_idx][item_idx])
        de_pos1_layer_means_both.append(np.mean(all_vals))
        de_pos1_layer_sems_both.append(np.std(all_vals) / np.sqrt(len(all_vals)))

    axes_summary_both[0, 0].bar(layer_x, de_pos1_layer_means_both, yerr=de_pos1_layer_sems_both,
                                 capsize=5, color='steelblue', edgecolor='black')
    axes_summary_both[0, 0].set_xticks(layer_x)
    axes_summary_both[0, 0].set_xticklabels(layer_names, rotation=45, ha='right')
    axes_summary_both[0, 0].set_ylabel('Mean Correlation')
    axes_summary_both[0, 0].set_title('DE vs Items in Position 1 (Left)')
    axes_summary_both[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes_summary_both[0, 0].set_ylim(-1, 1)

    # DE Position 2 after both
    de_pos2_layer_means_both = []
    de_pos2_layer_sems_both = []
    for layer_idx in range(num_layers):
        all_vals = []
        for item_idx in all_items:
            all_vals.extend(de_pos2_correlations_after_both[layer_idx][item_idx])
        de_pos2_layer_means_both.append(np.mean(all_vals))
        de_pos2_layer_sems_both.append(np.std(all_vals) / np.sqrt(len(all_vals)))

    axes_summary_both[0, 1].bar(layer_x, de_pos2_layer_means_both, yerr=de_pos2_layer_sems_both,
                                 capsize=5, color='coral', edgecolor='black')
    axes_summary_both[0, 1].set_xticks(layer_x)
    axes_summary_both[0, 1].set_xticklabels(layer_names, rotation=45, ha='right')
    axes_summary_both[0, 1].set_ylabel('Mean Correlation')
    axes_summary_both[0, 1].set_title('DE vs Items in Position 2 (Right)')
    axes_summary_both[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes_summary_both[0, 1].set_ylim(-1, 1)

    # ED Position 1 after both
    ed_pos1_layer_means_both = []
    ed_pos1_layer_sems_both = []
    for layer_idx in range(num_layers):
        all_vals = []
        for item_idx in all_items:
            all_vals.extend(ed_pos1_correlations_after_both[layer_idx][item_idx])
        ed_pos1_layer_means_both.append(np.mean(all_vals))
        ed_pos1_layer_sems_both.append(np.std(all_vals) / np.sqrt(len(all_vals)))

    axes_summary_both[1, 0].bar(layer_x, ed_pos1_layer_means_both, yerr=ed_pos1_layer_sems_both,
                                 capsize=5, color='steelblue', edgecolor='black')
    axes_summary_both[1, 0].set_xticks(layer_x)
    axes_summary_both[1, 0].set_xticklabels(layer_names, rotation=45, ha='right')
    axes_summary_both[1, 0].set_ylabel('Mean Correlation')
    axes_summary_both[1, 0].set_title('ED vs Items in Position 1 (Left)')
    axes_summary_both[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes_summary_both[1, 0].set_ylim(-1, 1)

    # ED Position 2 after both
    ed_pos2_layer_means_both = []
    ed_pos2_layer_sems_both = []
    for layer_idx in range(num_layers):
        all_vals = []
        for item_idx in all_items:
            all_vals.extend(ed_pos2_correlations_after_both[layer_idx][item_idx])
        ed_pos2_layer_means_both.append(np.mean(all_vals))
        ed_pos2_layer_sems_both.append(np.std(all_vals) / np.sqrt(len(all_vals)))

    axes_summary_both[1, 1].bar(layer_x, ed_pos2_layer_means_both, yerr=ed_pos2_layer_sems_both,
                                 capsize=5, color='coral', edgecolor='black')
    axes_summary_both[1, 1].set_xticks(layer_x)
    axes_summary_both[1, 1].set_xticklabels(layer_names, rotation=45, ha='right')
    axes_summary_both[1, 1].set_ylabel('Mean Correlation')
    axes_summary_both[1, 1].set_title('ED vs Items in Position 2 (Right)')
    axes_summary_both[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes_summary_both[1, 1].set_ylim(-1, 1)

    plt.suptitle('Linking Pair vs Single Item Correlations by Layer\n(After Both Linking Pairs Shown)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    figures["ll_linking_corr_after_both/summary_by_layer"] = fig_summary_both
    plt.close(fig_summary_both)

    model.train()
    return figures
