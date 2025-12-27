import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

def plot_pca_inputs(batch_trials, model, episode):
    # batch_trials shape batch_size x num_trials x 2*item_size
    model.eval()
    batch_trial_embeddings = model.embedding_layer(batch_trials) # shape batch_size x num_trials x hidden_size
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
