import logging
import numpy as np
import torch

from matplotlib import pyplot as plt

from generate_data import generate_batch_items, generate_batch_trials_ti, generate_batch_trials_ll, generate_batch_items_ai, generate_batch_trials_ai
from mlp import create_plastic_weights, clone_plastic_weights, pw_batch_size, repeat_interleave_pw, zero_plastic_weights, pw_mask_set, pw_mask_set_scaled
from plots import zero_shot_symbolic_distance_plot

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def update_symbolic_distance_bookkeeping(symbolic_distance_bookkeeping, pair_indices, choice_sampled, correct_choices, episode):
    batch_size = pair_indices.shape[0]
    num_trials = pair_indices.shape[1]

    for batch_index in range(batch_size):
        # Build list of trial dicts for this episode
        episode_trials = []
        for trial_num in range(num_trials):
            episode_trials.append({
                "item_1": int(pair_indices[batch_index][trial_num][0]),
                "item_2": int(pair_indices[batch_index][trial_num][1]),
                "model_output": int(choice_sampled[batch_index][trial_num]),
                "correct_choice": int(correct_choices[batch_index][trial_num]),
            })
        # Append this episode's trials to the batch's bookkeeping
        symbolic_distance_bookkeeping[batch_index].append(episode_trials)

    return symbolic_distance_bookkeeping

def full_eval_ll(args, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    num_items = args.item_range[-1] - 1
    zero_shot_trials = {
        (i,j): [] for i in range(num_items) for j in range(i + 1, num_items)
    }

    all_batch_items = generate_batch_items(args.item_range[-1] - 1, args.item_size, args.full_eval_batch_size, change_items_throughout_batch=True)

    num_batchs = args.full_eval_batch_size // args.batch_size + (args.full_eval_batch_size % args.batch_size > 0)

    for batch_num in range(num_batchs):
        batch_start = batch_num * args.batch_size
        batch_end = min((batch_num + 1) * args.batch_size, args.full_eval_batch_size)
        batch_items = all_batch_items[batch_start:batch_end]
        batch_size = batch_items.shape[0]
        plastic_weights, extra_plastic_weights, embed_pw = create_plastic_weights(batch_size, args.hidden_size, args.extra_layers, getattr(args, 'multi_neuromodulator', 1), device, direct_readout=getattr(args, 'direct_readout', False), first_plastic_input_size=getattr(model, 'first_plastic_input_size', args.hidden_size), plastic_embedding=getattr(args, 'plastic_embedding', False), input_size=2*args.item_size)

        trials, correct_choices, pair_indices = generate_batch_trials_ll(batch_items, args.num_trials_list_1, args.num_trials_list_2, args.num_trials_linking_pair, 1, put_linking_trials_first=args.put_linking_trials_first, randomize_list_order=args.randomize_list_order)

        trials = torch.tensor(trials, dtype=torch.float32).to(device)
        correct_choices = torch.tensor(correct_choices, dtype=torch.float32).to(device)
        num_train_trials = args.num_trials_list_1 + args.num_trials_list_2 + args.num_trials_linking_pair

        for trial in range(num_train_trials + 1):
            batch_trial = trials[:, trial, :]
            batch_correct_choice = correct_choices[:, trial]

            trial_input = batch_trial

            freeze_plastic = getattr(args, 'freeze_plastic_during_test', False) and trial >= num_train_trials
            with torch.inference_mode():
                output = model(trial_input, plastic_weights, batch_correct_choice, extra_plastic_weights=extra_plastic_weights, embed_plastic_weights=embed_pw, freeze_plastic=freeze_plastic)
            plastic_weights = output.plastic_weights
            extra_plastic_weights = output.extra_plastic_weights
            embed_pw = output.embed_plastic_weights

            if torch.isnan(output.choice).any() or (output.choice < 0).any() or (output.choice > 1).any():
                logger.info(f"Trial {trial}: choice has invalid values - min={output.choice.min()}, max={output.choice.max()}, nan={torch.isnan(output.choice).sum()}")
                break

            # Only process the test trial (last trial)
            if trial == num_train_trials:
                choice_sampled = output.sampled_choices.squeeze(-1)
                batch_pair_indices = pair_indices[:, trial]

                for episode in range(batch_size):
                    correct_test_choice = int(choice_sampled[episode].item() == batch_correct_choice[episode].item())
                    left_item_index = batch_pair_indices[episode][0]
                    right_item_index = batch_pair_indices[episode][1]
                    if left_item_index > right_item_index:
                        left_item_index, right_item_index = right_item_index, left_item_index

                    zero_shot_trials[(left_item_index, right_item_index)].append(correct_test_choice)

    model.train()
    return zero_shot_trials

def full_eval_ti(args, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    num_items = args.item_range[-1] - 1
    zero_shot_trials = {
        (i,j): [] for i in range(num_items) for j in range(i + 1, num_items)
    }

    all_batch_items = generate_batch_items(args.item_range[-1] - 1, args.item_size, args.full_eval_batch_size, change_items_throughout_batch=True)

    num_batchs = args.full_eval_batch_size // args.batch_size + (args.full_eval_batch_size % args.batch_size > 0)

    for batch_num in range(num_batchs):
        batch_start = batch_num * args.batch_size
        batch_end = min((batch_num + 1) * args.batch_size, args.full_eval_batch_size)
        batch_items = all_batch_items[batch_start:batch_end]
        batch_size = batch_items.shape[0]
        plastic_weights, extra_plastic_weights, embed_pw = create_plastic_weights(batch_size, args.hidden_size, args.extra_layers, getattr(args, 'multi_neuromodulator', 1), device, direct_readout=getattr(args, 'direct_readout', False), first_plastic_input_size=getattr(model, 'first_plastic_input_size', args.hidden_size), plastic_embedding=getattr(args, 'plastic_embedding', False), input_size=2*args.item_size)

        # Generate trials: num_train_trials training trials + 1 test trial
        trials, correct_choices, pair_indices, num_train_trials = generate_batch_trials_ti(batch_items, args.num_train_trials, 1, arbitrary=args.arbitrary)

        trials = torch.tensor(trials, dtype=torch.float32).to(device)
        correct_choices = torch.tensor(correct_choices, dtype=torch.float32).to(device)

        for trial in range(num_train_trials + 1):
            batch_trial = trials[:, trial, :]
            batch_correct_choice = correct_choices[:, trial]

            trial_input = batch_trial

            freeze_plastic = getattr(args, 'freeze_plastic_during_test', False) and trial >= num_train_trials
            with torch.inference_mode():
                output = model(trial_input, plastic_weights, batch_correct_choice, extra_plastic_weights=extra_plastic_weights, embed_plastic_weights=embed_pw, freeze_plastic=freeze_plastic)
            plastic_weights = output.plastic_weights
            extra_plastic_weights = output.extra_plastic_weights
            embed_pw = output.embed_plastic_weights

            if torch.isnan(output.choice).any() or (output.choice < 0).any() or (output.choice > 1).any():
                logger.info(f"Trial {trial}: choice has invalid values - min={output.choice.min()}, max={output.choice.max()}, nan={torch.isnan(output.choice).sum()}")
                break

            # Only process the test trial (last trial)
            if trial == num_train_trials:
                choice_sampled = output.sampled_choices.squeeze(-1)
                batch_pair_indices = pair_indices[:, trial]

                for episode in range(batch_size):
                    correct_test_choice = int(choice_sampled[episode].item() == batch_correct_choice[episode].item())
                    left_item_index = batch_pair_indices[episode][0]
                    right_item_index = batch_pair_indices[episode][1]
                    if left_item_index > right_item_index:
                        left_item_index, right_item_index = right_item_index, left_item_index

                    zero_shot_trials[(left_item_index, right_item_index)].append(correct_test_choice)

    model.train()
    return zero_shot_trials

def full_eval_ai(args, model):
    """
    Full evaluation for associative inference task.

    Evaluates zero-shot performance on the first test trial after training.
    Returns data structured for a heatmap: accuracy for each (item1, item2) pair.

    Items are flattened from (group, index) to a single ID: item_id = group * num_items_per_group + index

    Returns:
        zero_shot_trials: dict mapping (item1_id, item2_id) -> list of 0/1 (incorrect/correct)
        metadata: dict with num_groups, num_items_per_group, total_items for plotting
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    num_groups = args.associative_inference_num_groups
    num_items_per_group = args.associative_inference_num_items_per_group
    total_items = num_groups * num_items_per_group

    # Initialize dict for all possible ordered pairs including diagonal (for heatmap)
    zero_shot_trials = {
        (i, j): [] for i in range(total_items) for j in range(total_items)
    }

    # Generate all batch items upfront
    all_batch_items = generate_batch_items_ai(
        num_groups,
        num_items_per_group,
        args.item_size,
        args.full_eval_batch_size,
        change_items_throughout_batch=True
    )

    num_batches = args.full_eval_batch_size // args.batch_size + (args.full_eval_batch_size % args.batch_size > 0)

    for batch_num in range(num_batches):
        batch_start = batch_num * args.batch_size
        batch_end = min((batch_num + 1) * args.batch_size, args.full_eval_batch_size)
        batch_items = all_batch_items[batch_start:batch_end]
        batch_size = batch_items.shape[0]
        plastic_weights, extra_plastic_weights, embed_pw = create_plastic_weights(batch_size, args.hidden_size, args.extra_layers, getattr(args, 'multi_neuromodulator', 1), device, direct_readout=getattr(args, 'direct_readout', False), first_plastic_input_size=getattr(model, 'first_plastic_input_size', args.hidden_size), plastic_embedding=getattr(args, 'plastic_embedding', False), input_size=2*args.item_size)

        # Generate trials with only 1 test trial
        trials, correct_choices, pair_indices, num_train_trials = generate_batch_trials_ai(
            batch_items,
            num_items_per_group,
            num_test_trials=1,
            exclude_same_item=args.ai_exclude_same_item
        )

        trials = torch.tensor(trials, dtype=torch.float32).to(device)
        correct_choices = torch.tensor(correct_choices, dtype=torch.float32).to(device)

        for trial in range(num_train_trials + 1):
            batch_trial = trials[:, trial, :]
            batch_correct_choice = correct_choices[:, trial]

            trial_input = batch_trial

            freeze_plastic = getattr(args, 'freeze_plastic_during_test', False) and trial >= num_train_trials
            with torch.inference_mode():
                output = model(trial_input, plastic_weights, batch_correct_choice, extra_plastic_weights=extra_plastic_weights, embed_plastic_weights=embed_pw, freeze_plastic=freeze_plastic)
            plastic_weights = output.plastic_weights
            extra_plastic_weights = output.extra_plastic_weights
            embed_pw = output.embed_plastic_weights

            if torch.isnan(output.choice).any() or (output.choice < 0).any() or (output.choice > 1).any():
                logger.info(f"Trial {trial}: choice has invalid values - min={output.choice.min()}, max={output.choice.max()}, nan={torch.isnan(output.choice).sum()}")
                break

            # Only process the test trial (last trial)
            if trial == num_train_trials:
                choice_sampled = output.sampled_choices.squeeze(-1)
                batch_pair_indices = pair_indices[:, trial]  # shape: (batch_size, 2, 2) -> [[g1, idx1], [g2, idx2]]

                for episode in range(batch_size):
                    correct_test_choice = int(choice_sampled[episode].item() == batch_correct_choice[episode].item())

                    # Extract group and index for each item
                    g1, idx1 = batch_pair_indices[episode][0]
                    g2, idx2 = batch_pair_indices[episode][1]

                    # Flatten to single item IDs
                    item1_id = int(g1 * num_items_per_group + idx1)
                    item2_id = int(g2 * num_items_per_group + idx2)

                    zero_shot_trials[(item1_id, item2_id)].append(correct_test_choice)

    model.train()

    metadata = {
        'num_groups': num_groups,
        'num_items_per_group': num_items_per_group,
        'total_items': total_items,
        'exclude_same_item': args.ai_exclude_same_item,
    }

    return zero_shot_trials, metadata


def more_items_generalization_test(args, model):
    """
    Length generalization test with zero-shot evaluation.

    For each length (number of items):
    1. Train a separate batch on that length
    2. Freeze plastic weights
    3. Do zero-shot evaluation on all pairs
    4. Record zero-shot accuracy
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    additional_items = args.additional_items
    max_training_items = args.item_range[-1] - 1
    batch_size = args.batch_size * 4  # Use 4x batch size like other evals
    item_size = args.item_size

    length_generalization_logging_dict = {}

    plot_x_values = []
    plot_zero_shot_accuracies = {}  # {num_items: {sd: accuracy}}

    # Base number of training trials, will increase with item count
    base_num_train_trials = args.num_train_trials

    for i in range(1, additional_items + 1):
        num_items = max_training_items + i
        plot_x_values.append(num_items)

        # Scale training trials with number of items
        extended_num_train_trials = base_num_train_trials + 2 * i

        # Initialize plastic weights
        plastic_weights, extra_plastic_weights, embed_pw = create_plastic_weights(batch_size, args.hidden_size, args.extra_layers, getattr(args, 'multi_neuromodulator', 1), device, direct_readout=getattr(args, 'direct_readout', False), first_plastic_input_size=getattr(model, 'first_plastic_input_size', args.hidden_size), plastic_embedding=getattr(args, 'plastic_embedding', False), input_size=2*args.item_size)

        # Generate batch items (different items per network)
        batch_items = generate_batch_items(num_items, item_size, batch_size, change_items_throughout_batch=True)

        # Generate training trials only (no test trials needed for zero-shot eval)
        trials, correct_choices, _, actual_num_train_trials = generate_batch_trials_ti(
            batch_items, extended_num_train_trials, 0, arbitrary=args.arbitrary
        )

        trials = torch.tensor(trials, dtype=torch.float32).to(device)
        correct_choices = torch.tensor(correct_choices, dtype=torch.float32).to(device)

        # Run training trials
        for trial in range(actual_num_train_trials):
            batch_trial = trials[:, trial, :]
            batch_correct_choice = correct_choices[:, trial]

            with torch.inference_mode():
                output = model(batch_trial, plastic_weights, batch_correct_choice,
                              extra_plastic_weights=extra_plastic_weights, embed_plastic_weights=embed_pw)
            plastic_weights = output.plastic_weights
            extra_plastic_weights = output.extra_plastic_weights
            embed_pw = output.embed_plastic_weights

        # Freeze weights and do zero-shot evaluation on all pairs
        frozen_pw, frozen_epw, frozen_embed_pw = clone_plastic_weights(plastic_weights, extra_plastic_weights, embed_pw=embed_pw)

        # Returns dict of {(i, j): [list of 0/1 correctness]}
        zero_shot_trials = _evaluate_all_pairs_zero_shot_variable_items(
            args, model, batch_items, frozen_pw, frozen_epw, device, num_items, frozen_embed_pw=frozen_embed_pw
        )
        plot_zero_shot_accuracies[num_items] = zero_shot_trials

        # Compute per-SD accuracies for logging
        acc_by_sd = {}
        for (i, j), correctness_list in zero_shot_trials.items():
            sd = j - i
            if sd not in acc_by_sd:
                acc_by_sd[sd] = []
            acc_by_sd[sd].extend(correctness_list)
        acc_by_sd = {sd: np.mean(vals) for sd, vals in acc_by_sd.items()}

        # Log per-SD accuracies
        for sd, acc in acc_by_sd.items():
            length_generalization_logging_dict[f"length_generalize/{num_items}_SD{sd}_accuracy"] = acc
        overall_acc = np.mean(list(acc_by_sd.values()))
        length_generalization_logging_dict[f"length_generalize/{num_items}_overall_accuracy"] = overall_acc
        logger.info(f"Length generalization {num_items} items: Overall accuracy = {overall_acc:.4f}, by SD: {acc_by_sd}")

    # --- Create separate symbolic distance plot for each length ---
    symbolic_distance_figs = {}
    for num_items in plot_x_values:
        zero_shot_trials = plot_zero_shot_accuracies[num_items]
        fig = zero_shot_symbolic_distance_plot(
            zero_shot_trials,
            num_items,
            title=f'Length Generalization: {num_items} Items'
        )
        symbolic_distance_figs[num_items] = fig

    model.train()

    return length_generalization_logging_dict, symbolic_distance_figs


def _evaluate_all_pairs_zero_shot_variable_items(args, model, batch_items, frozen_pw, frozen_epw, device, num_items, frozen_embed_pw=None):
    """
    Evaluate zero-shot accuracy on all possible pairs with frozen plastic weights.
    Similar to _evaluate_all_pairs_zero_shot but takes num_items as parameter.

    Returns:
        dict: {(i, j): [list of 0/1 correctness]} for each pair, compatible with zero_shot_symbolic_distance_plot
    """
    batch_size = pw_batch_size(frozen_pw)

    # Track per-pair correctness
    zero_shot_trials = {(i, j): [] for i in range(num_items) for j in range(i + 1, num_items)}

    for i in range(num_items):
        for j in range(i + 1, num_items):
            item_higher = batch_items[:, i, :]
            item_lower = batch_items[:, j, :]

            # Test order 1: higher item first (correct choice = 0)
            trial_input = np.concatenate([item_higher, item_lower], axis=1)
            trial_input = torch.tensor(trial_input, dtype=torch.float32).to(device)
            correct_choice = torch.zeros(batch_size, dtype=torch.float32).to(device)

            with torch.inference_mode():
                output = model(trial_input, frozen_pw, correct_choice,
                              extra_plastic_weights=frozen_epw, store_embeddings=False, embed_plastic_weights=frozen_embed_pw)

            choice_sampled = output.sampled_choices.squeeze(-1)
            correct_mask = (choice_sampled == correct_choice).cpu().numpy()
            zero_shot_trials[(i, j)].extend(correct_mask.astype(int).tolist())

            # Test order 2: lower item first (correct choice = 1)
            trial_input_rev = np.concatenate([item_lower, item_higher], axis=1)
            trial_input_rev = torch.tensor(trial_input_rev, dtype=torch.float32).to(device)
            correct_choice_rev = torch.ones(batch_size, dtype=torch.float32).to(device)

            with torch.inference_mode():
                output_rev = model(trial_input_rev, frozen_pw, correct_choice_rev,
                                  extra_plastic_weights=frozen_epw, store_embeddings=False)

            choice_sampled_rev = output_rev.sampled_choices.squeeze(-1)
            correct_mask_rev = (choice_sampled_rev == correct_choice_rev).cpu().numpy()
            zero_shot_trials[(i, j)].extend(correct_mask_rev.astype(int).tolist())

    return zero_shot_trials

def mass_presentation_test(args, model):
    """
    Mass presentation test with zero-shot evaluation at checkpoints.

    1. Run training trials
    2. Continuously run mass presentation trials
    3. At checkpoints (20, 100, 500): freeze weights, evaluate zero-shot on all pairs
    4. Unfreeze and continue after each checkpoint
    5. Track neuromodulator throughout training + mass presentation
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    num_items = args.item_range[-1] - 1
    num_train_trials = args.num_train_trials
    batch_size = args.batch_size * 4  # Use 4x batch size like other evals
    item_size = args.item_size

    mass_presentation_logging_dict = {}
    checkpoint_counts = [20, 100, 500]
    max_mass_presentations = checkpoint_counts[-1]

    # Storage for zero-shot accuracies at each checkpoint
    zero_shot_accuracies = {}

    # Initialize plastic weights once
    plastic_weights, extra_plastic_weights, embed_pw = create_plastic_weights(batch_size, args.hidden_size, args.extra_layers, getattr(args, 'multi_neuromodulator', 1), device, direct_readout=getattr(args, 'direct_readout', False), first_plastic_input_size=getattr(model, 'first_plastic_input_size', args.hidden_size), plastic_embedding=getattr(args, 'plastic_embedding', False), input_size=2*args.item_size)

    # Generate batch items (different items per network for robust averaging)
    batch_items = generate_batch_items(num_items, item_size, batch_size, change_items_throughout_batch=True)

    # Generate trials: training + max mass presentations (no test trials needed)
    trials, correct_choices, pair_indices, actual_num_train_trials = generate_batch_trials_ti(
        batch_items, num_train_trials, 0, arbitrary=args.arbitrary,
        mass_presentation=max_mass_presentations
    )

    trials = torch.tensor(trials, dtype=torch.float32).to(device)
    correct_choices = torch.tensor(correct_choices, dtype=torch.float32).to(device)

    total_trial_count = actual_num_train_trials + max_mass_presentations
    num_train_trials = actual_num_train_trials  # Use the actual count

    # Neuromodulator tracking
    neuromodulator_vals = []
    num_neuromodulators = None

    # --- Run training trials ---
    for trial in range(num_train_trials):
        batch_trial = trials[:, trial, :]
        batch_correct_choice = correct_choices[:, trial]

        with torch.inference_mode():
            output = model(batch_trial, plastic_weights, batch_correct_choice,
                          extra_plastic_weights=extra_plastic_weights, embed_plastic_weights=embed_pw)
        plastic_weights = output.plastic_weights
        extra_plastic_weights = output.extra_plastic_weights
        embed_pw = output.embed_plastic_weights

        # Track neuromodulator
        if args.extra_layers > 0:
            nm_data = output.neuromodulator.detach().cpu().numpy()
            if args.use_extra_neuromodulator:
                nm_avg_per_nm = np.mean(nm_data, axis=0).flatten()
                neuromodulator_vals.append(nm_avg_per_nm)
                if num_neuromodulators is None:
                    num_neuromodulators = len(nm_avg_per_nm)
            else:
                neuromodulator_vals.append(np.mean(nm_data))

    # --- Run mass presentation trials with checkpoints ---
    current_checkpoint_idx = 0

    for mass_trial in range(max_mass_presentations):
        trial_idx = num_train_trials + mass_trial
        batch_trial = trials[:, trial_idx, :]
        batch_correct_choice = correct_choices[:, trial_idx]

        with torch.inference_mode():
            output = model(batch_trial, plastic_weights, batch_correct_choice,
                          extra_plastic_weights=extra_plastic_weights, embed_plastic_weights=embed_pw)
        plastic_weights = output.plastic_weights
        extra_plastic_weights = output.extra_plastic_weights
        embed_pw = output.embed_plastic_weights

        # Track neuromodulator
        if args.extra_layers > 0:
            nm_data = output.neuromodulator.detach().cpu().numpy()
            if args.use_extra_neuromodulator:
                nm_avg_per_nm = np.mean(nm_data, axis=0).flatten()
                neuromodulator_vals.append(nm_avg_per_nm)
            else:
                neuromodulator_vals.append(np.mean(nm_data))

        # Check if we've reached a checkpoint
        mass_count = mass_trial + 1  # 1-indexed count
        if current_checkpoint_idx < len(checkpoint_counts) and mass_count == checkpoint_counts[current_checkpoint_idx]:
            # Freeze weights and do zero-shot evaluation on all pairs
            frozen_pw, frozen_epw, frozen_embed_pw = clone_plastic_weights(plastic_weights, extra_plastic_weights, embed_pw=embed_pw)

            # Returns dict of {(i, j): [list of 0/1 correctness]}
            zero_shot_trials = _evaluate_all_pairs_zero_shot(
                args, model, batch_items, frozen_pw, frozen_epw, device, frozen_embed_pw=frozen_embed_pw
            )
            zero_shot_accuracies[mass_count] = zero_shot_trials

            # Compute per-SD accuracies for logging
            acc_by_sd = {}
            for (i, j), correctness_list in zero_shot_trials.items():
                sd = j - i
                if sd not in acc_by_sd:
                    acc_by_sd[sd] = []
                acc_by_sd[sd].extend(correctness_list)
            acc_by_sd = {sd: np.mean(vals) for sd, vals in acc_by_sd.items()}

            # Log per-SD accuracies
            for sd, acc in acc_by_sd.items():
                mass_presentation_logging_dict[f"mass_presentation/{mass_count}_SD{sd}_accuracy"] = acc
            overall_acc = np.mean(list(acc_by_sd.values()))
            mass_presentation_logging_dict[f"mass_presentation/{mass_count}_overall_accuracy"] = overall_acc
            logger.info(f"Mass presentation {mass_count}: Overall accuracy = {overall_acc:.4f}, by SD: {acc_by_sd}")

            current_checkpoint_idx += 1

    # --- Create symbolic distance plots for each checkpoint ---
    num_items = args.item_range[-1] - 1
    symbolic_distance_figs = {}
    for checkpoint in checkpoint_counts:
        if checkpoint in zero_shot_accuracies:
            fig = zero_shot_symbolic_distance_plot(
                zero_shot_accuracies[checkpoint],
                num_items,
                title=f'Mass Presentation {checkpoint}: Zero-Shot Symbolic Distance'
            )
            symbolic_distance_figs[checkpoint] = fig

    # --- Create neuromodulator figure ---
    neuromodulator_fig = None
    if args.extra_layers > 0 and len(neuromodulator_vals) > 0:
        plt.figure(dpi=300)

        # Phase boundaries
        phase_boundaries = [
            (num_train_trials, "Train End"),
        ]
        # Add checkpoint markers
        for checkpoint in checkpoint_counts:
            phase_boundaries.append((num_train_trials + checkpoint, f"MP={checkpoint}"))

        # Add vertical lines and labels
        for x_pos, label in phase_boundaries:
            plt.axvline(x=x_pos, color='gray', linestyle=':', linewidth=1, zorder=0)
            plt.text(x_pos, plt.gca().get_ylim()[1] if plt.gca().get_ylim()[1] != 1.0 else 0.5,
                    label, ha='right', va='top', fontsize=8, color='gray', rotation=90)

        # Add horizontal dotted line at y=0
        plt.axhline(y=0.0, color='gray', linestyle=':', linewidth=1, zorder=0)

        if args.use_extra_neuromodulator and num_neuromodulators is not None:
            colors = plt.cm.tab10(np.linspace(0, 1, num_neuromodulators))
            neuromodulator_vals_array = np.array(neuromodulator_vals)
            for nm_idx in range(num_neuromodulators):
                nm_vals = neuromodulator_vals_array[:, nm_idx]
                label = f"Extra NM {nm_idx}" if nm_idx < num_neuromodulators - 1 else "Main NM"
                plt.plot(range(total_trial_count), nm_vals, alpha=0.2, color=colors[nm_idx])
                plt.plot(range(total_trial_count), ema_smooth(nm_vals), label=label, color=colors[nm_idx])
        else:
            plt.plot(range(total_trial_count), neuromodulator_vals, alpha=0.2, color='blue')
            plt.plot(range(total_trial_count), ema_smooth(neuromodulator_vals), label="Neuromodulator", color="blue")

        plt.xlabel('Trial')
        plt.ylabel('Neuromodulator Values')
        plt.title(f'Neuromodulator During Training + {max_mass_presentations} Mass Presentations')
        plt.legend()
        plt.tight_layout()
        neuromodulator_fig = plt.gcf()
        plt.close()

    model.train()

    return mass_presentation_logging_dict, symbolic_distance_figs, neuromodulator_fig


def _evaluate_all_pairs_zero_shot(args, model, batch_items, frozen_pw, frozen_epw, device, frozen_embed_pw=None):
    """
    Evaluate zero-shot accuracy on all possible pairs with frozen plastic weights.

    For TI with n items, evaluates all n*(n-1)/2 unique pairs where we test if model
    correctly identifies the higher-ranked item.

    Returns:
        dict: {(i, j): [list of 0/1 correctness]} for each pair, compatible with zero_shot_symbolic_distance_plot
    """
    num_items = args.item_range[-1] - 1
    batch_size = pw_batch_size(frozen_pw)

    # Track per-pair correctness
    zero_shot_trials = {(i, j): [] for i in range(num_items) for j in range(i + 1, num_items)}

    # Generate all unique pairs (higher_rank, lower_rank)
    for i in range(num_items):
        for j in range(i + 1, num_items):
            item_higher = batch_items[:, i, :]
            item_lower = batch_items[:, j, :]

            # Test order 1: higher item first (correct choice = 0)
            trial_input = np.concatenate([item_higher, item_lower], axis=1)
            trial_input = torch.tensor(trial_input, dtype=torch.float32).to(device)
            correct_choice = torch.zeros(batch_size, dtype=torch.float32).to(device)

            with torch.inference_mode():
                output = model(trial_input, frozen_pw, correct_choice,
                              extra_plastic_weights=frozen_epw, store_embeddings=False, embed_plastic_weights=frozen_embed_pw)

            choice_sampled = output.sampled_choices.squeeze(-1)
            correct_mask = (choice_sampled == correct_choice).cpu().numpy()
            zero_shot_trials[(i, j)].extend(correct_mask.astype(int).tolist())

            # Test order 2: lower item first (correct choice = 1)
            trial_input_rev = np.concatenate([item_lower, item_higher], axis=1)
            trial_input_rev = torch.tensor(trial_input_rev, dtype=torch.float32).to(device)
            correct_choice_rev = torch.ones(batch_size, dtype=torch.float32).to(device)

            with torch.inference_mode():
                output_rev = model(trial_input_rev, frozen_pw, correct_choice_rev,
                                  extra_plastic_weights=frozen_epw, store_embeddings=False)

            choice_sampled_rev = output_rev.sampled_choices.squeeze(-1)
            correct_mask_rev = (choice_sampled_rev == correct_choice_rev).cpu().numpy()
            zero_shot_trials[(i, j)].extend(correct_mask_rev.astype(int).tolist())

    return zero_shot_trials

def new_items_old_items_test(args, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    num_items = args.item_range[-1] - 1
    num_train_trials = args.num_train_trials
    num_test_trials = args.num_test_trials*10
    batch_size = args.batch_size
    item_size = args.item_size

    plastic_weights, extra_plastic_weights, embed_pw = create_plastic_weights(batch_size, args.hidden_size, args.extra_layers, getattr(args, 'multi_neuromodulator', 1), device, direct_readout=getattr(args, 'direct_readout', False), first_plastic_input_size=getattr(model, 'first_plastic_input_size', args.hidden_size), plastic_embedding=getattr(args, 'plastic_embedding', False), input_size=2*args.item_size)

    batch_items_old = generate_batch_items(num_items, item_size, batch_size, change_items_throughout_batch=args.change_items_throughout_batch)
    batch_items_new = generate_batch_items(num_items, item_size, batch_size, change_items_throughout_batch=args.change_items_throughout_batch)

    trials_old, correct_choices_old, pair_indices_old, actual_num_train_trials = generate_batch_trials_ti(batch_items_old, num_train_trials, num_test_trials, arbitrary=args.arbitrary, mass_presentation=0)
    trials_old_revisited, correct_choices_old_revisited, pair_indices_old_revisited, _ = generate_batch_trials_ti(batch_items_old, num_train_trials, num_test_trials, arbitrary=args.arbitrary, mass_presentation=0)
    trials_new, correct_choices_new, pair_indices_new, _ = generate_batch_trials_ti(batch_items_new, num_train_trials, num_test_trials, arbitrary=args.arbitrary, mass_presentation=0)
    num_train_trials = actual_num_train_trials

    trials_old = torch.tensor(trials_old, dtype=torch.float32).to(device)
    trials_new = torch.tensor(trials_new, dtype=torch.float32).to(device)
    trials_old_test = torch.tensor(trials_old_revisited[:, num_train_trials:, :], dtype=torch.float32).to(device)
    trials = torch.cat([trials_old, trials_new, trials_old_test], dim=1).to(device)

    correct_choices_old = torch.tensor(correct_choices_old, dtype=torch.float32).to(device)
    correct_choices_new = torch.tensor(correct_choices_new, dtype=torch.float32).to(device)
    correct_choices_old_test = torch.tensor(correct_choices_old_revisited[:, num_train_trials:], dtype=torch.float32).to(device)
    correct_choices = torch.cat([correct_choices_old, correct_choices_new, correct_choices_old_test], dim=-1)

    correct_train_choices_old = 0
    correct_test_choices_old = 0
    correct_train_choices_new = 0
    correct_test_choices_new = 0
    correct_test_choices_old_revisited = 0

    plot_accuracies = {
        "old_train": 0.0,
        "old_test": 0.0,
        "new_train": 0.0,
        "new_test": 0.0,
        "old_revisited_test": 0.0,
    }

    accuracies_dict = {}

    total_trials = 2*(num_train_trials + num_test_trials) + num_test_trials

    if args.extra_layers > 0:
        # Track neuromodulator values - will be list of arrays if use_extra_neuromodulator, else list of scalars
        neuromodulator_vals = []
        num_neuromodulators = None

    avg_accuracies_per_trial = []

    for trial in range(total_trials):
        batch_trial = trials[:, trial, :]
        batch_correct_choice = correct_choices[:, trial]

        trial_input = batch_trial

        with torch.inference_mode():
            output = model(trial_input, plastic_weights, batch_correct_choice, extra_plastic_weights=extra_plastic_weights, embed_plastic_weights=embed_pw)
        plastic_weights = output.plastic_weights
        extra_plastic_weights = output.extra_plastic_weights
        embed_pw = output.embed_plastic_weights

        if args.extra_layers > 0:
            nm_data = output.neuromodulator.detach().cpu().numpy()
            if args.use_extra_neuromodulator:
                # Average across batch, keep individual neuromodulators separate
                # nm_data shape: (batch, 1, num_neuromodulators) or (batch, 1, 1, num_neuromodulators)
                nm_avg_per_nm = np.mean(nm_data, axis=0).flatten()  # shape: (num_neuromodulators,)
                neuromodulator_vals.append(nm_avg_per_nm)
                if num_neuromodulators is None:
                    num_neuromodulators = len(nm_avg_per_nm)
            else:
                # Average everything together (original behavior)
                neuromodulator_vals.append(np.mean(nm_data))

        if torch.isnan(output.choice).any() or (output.choice < 0).any() or (output.choice > 1).any():
            print(f"Trial {trial}: choice has invalid values - min={output.choice.min()}, max={output.choice.max()}, nan={torch.isnan(output.choice).sum()}")
            break

        choice_sampled = output.sampled_choices.squeeze(-1)

        avg_accuracies_per_trial.append((choice_sampled == batch_correct_choice).sum().item() / batch_size)

        if trial < num_train_trials:
            correct_train_choices_old += (choice_sampled == batch_correct_choice).sum().item()
        elif trial < num_train_trials + num_test_trials:
            correct_test_choices_old += (choice_sampled == batch_correct_choice).sum().item()
        elif trial < 2*num_train_trials + num_test_trials:
            correct_train_choices_new += (choice_sampled == batch_correct_choice).sum().item()
        elif trial < 2*num_train_trials + 2*num_test_trials:
            correct_test_choices_new += (choice_sampled == batch_correct_choice).sum().item()
        else:
            correct_test_choices_old_revisited += (choice_sampled == batch_correct_choice).sum().item()
    
    train_accuracy_old = correct_train_choices_old / (num_train_trials * batch_size)
    test_accuracy_old = correct_test_choices_old / (num_test_trials * batch_size)
    train_accuracy_new = correct_train_choices_new / (num_train_trials * batch_size)
    test_accuracy_new = correct_test_choices_new / (num_test_trials * batch_size)
    test_accuracy_old_revisited = correct_test_choices_old_revisited / (num_test_trials * batch_size)
    plot_accuracies["old_train"] = train_accuracy_old
    plot_accuracies["old_test"] = test_accuracy_old
    plot_accuracies["new_train"] = train_accuracy_new
    plot_accuracies["new_test"] = test_accuracy_new
    plot_accuracies["old_revisited_test"] = test_accuracy_old_revisited

    accuracies_dict.update({
        "old_train": train_accuracy_old,
        "old_test": test_accuracy_old,
        "new_train": train_accuracy_new,
        "new_test": test_accuracy_new,
        "old_revisited_test": test_accuracy_old_revisited,
    })

    plt.figure(dpi=300)
    plt.bar(list(plot_accuracies.keys()), list(plot_accuracies.values()))
    plt.xlabel('Trial Type')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    accuracies_fig = plt.gcf()
    plt.close()


    plt.figure(dpi=300)
    phase_boundaries = [
      (num_train_trials, "Old Train"),
      (num_train_trials + num_test_trials, "Old Test"),
      (2*num_train_trials + num_test_trials, "New Train"),
      (2*num_train_trials + 2*num_test_trials, "New Test"),
      # Last phase (Old Revisited) ends at total_trials, no line needed
    ]

    # Add vertical lines and labels
    for x_pos, label in phase_boundaries:
        plt.axvline(x=x_pos, color='gray', linestyle=':', linewidth=1, zorder=0)
        plt.text(x_pos, plt.ylim()[1], label, ha='right', va='top', fontsize=8, color='gray', rotation=90)

    # Add horizontal dotted line at y=0
    plt.axhline(y=0.0, color='gray', linestyle=':', linewidth=1, zorder=0)

    if args.extra_layers > 0:
        if args.use_extra_neuromodulator and num_neuromodulators is not None:
            # Plot each neuromodulator separately
            colors = plt.cm.tab10(np.linspace(0, 1, num_neuromodulators))
            neuromodulator_vals_array = np.array(neuromodulator_vals)  # shape: (num_trials, num_neuromodulators)
            for nm_idx in range(num_neuromodulators):
                nm_vals = neuromodulator_vals_array[:, nm_idx]
                label = f"Extra NM {nm_idx}" if nm_idx < num_neuromodulators - 1 else "Main NM"
                # Raw data (translucent)
                plt.plot(range(total_trials), nm_vals, alpha=0.2, color=colors[nm_idx])
                # Smoothed data (solid)
                plt.plot(range(total_trials), ema_smooth(nm_vals), label=label, color=colors[nm_idx])
        else:
            # Original behavior - single line
            # Raw data (translucent)
            plt.plot(range(total_trials), neuromodulator_vals, alpha=0.2, color='blue')
            # Smoothed data (solid)
            plt.plot(range(total_trials), ema_smooth(neuromodulator_vals), label="Neuromodulator", color="blue")

        plt.xlabel('Trial')
        plt.ylabel('Neuromodulator Values')
        plt.legend()
        plt.tight_layout()
        neuromodulator_fig = plt.gcf()
        plt.close()

    plt.figure(dpi=300)

    # Add vertical lines and labels
    for x_pos, label in phase_boundaries:
        plt.axvline(x=x_pos, color='gray', linestyle=':', linewidth=1, zorder=0)
        plt.text(x_pos, plt.ylim()[1], label, ha='right', va='top', fontsize=8, color='gray', rotation=90)
    
    # Raw data (translucent)
    plt.plot(range(total_trials), avg_accuracies_per_trial, alpha=0.2, color='blue')
    
    # Smoothed data (solid)
    plt.plot(range(total_trials), ema_smooth(avg_accuracies_per_trial), color='blue')

    plt.xlabel('Trial')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    avg_accuracies_per_trial_fig = plt.gcf()
    plt.close()

    model.train()
    
    if args.extra_layers == 0:
        neuromodulator_fig = None
    return accuracies_dict, accuracies_fig, neuromodulator_fig, avg_accuracies_per_trial_fig

def ema_smooth(values, alpha=0.1):
    # Lower alpha = more smoothing
    smoothed = []
    ema = values[0]
    for v in values:
        ema = alpha * v + (1 - alpha) * ema
        smoothed.append(ema)
    return smoothed


def generate_controlled_order_trials_ll(batch_items, num_trials_per_pair, degenerate_order=True, test_pair=(1, 6)):
    """
    Generate list linking trials with controlled ordering to test the chain hypothesis.

    For 8 items A(0), B(1), C(2), D(3), E(4), F(5), G(6), H(7):
    - List 1 adjacent pairs: AB(0,1), BC(1,2), CD(2,3)
    - List 2 adjacent pairs: EF(4,5), FG(5,6), GH(6,7)
    - Linking pair: DE(3,4)

    Degenerate order (breaks chain for BG):
    - List 1: BC before CD (so h1_BC doesn't contain h1_CD)
    - List 2: FG before EF (so h1_FG doesn't contain h1_EF)

    Enriched order (preserves chain for BG):
    - List 1: CD before BC (so h1_BC contains h1_CD)
    - List 2: EF before FG (so h1_FG contains h1_EF)

    Args:
        batch_items: (batch_size, 8, item_size) array of item embeddings
        num_trials_per_pair: number of trials for each adjacent pair
        degenerate_order: if True, use degenerate ordering; if False, use enriched ordering
        test_pair: tuple (high_idx, low_idx) for the test pair, default (1, 6) for BG

    Returns:
        trials: (batch_size, num_train_trials + 1, 2*item_size) - training + 1 test trial
        correct_choices: (batch_size, num_train_trials + 1)
        pair_indices: (batch_size, num_train_trials + 1, 2)
    """
    batch_size = batch_items.shape[0]
    num_items = batch_items.shape[1]  # Should be 8

    all_trials = []
    all_correct_choices = []
    all_pair_indices = []

    # Define the pairs for each list
    # List 1: AB(0,1), BC(1,2), CD(2,3)
    # List 2: EF(4,5), FG(5,6), GH(6,7)

    if degenerate_order:
        # Degenerate: BC before CD, FG before EF
        # End pairs (AB, GH) can be anywhere - put them first
        list_1_order = [(0, 1), (1, 2), (2, 3)]  # AB, BC, CD
        list_2_order = [(6, 7), (5, 6), (4, 5)]  # GH, FG, EF
    else:
        # Enriched: CD before BC, EF before FG
        # End pairs (AB, GH) can be anywhere - put them last
        list_1_order = [(2, 3), (1, 2), (0, 1)]  # CD, BC, AB
        list_2_order = [(4, 5), (5, 6), (6, 7)]  # EF, FG, GH

    linking_pair = (3, 4)  # DE

    for batch_idx in range(batch_size):
        batch_trials = []
        batch_correct_choices = []
        batch_pair_indices = []

        # Generate List 1 trials in specified order
        for high_idx, low_idx in list_1_order:
            for _ in range(num_trials_per_pair):
                item_1 = batch_items[batch_idx, high_idx]
                item_2 = batch_items[batch_idx, low_idx]
                # Randomly swap presentation order
                swap = np.random.randint(0, 2)
                choice = swap  # 0 if item_1 (higher) is in position 1, 1 if item_2 (lower) is in position 1
                if swap:
                    item_pair = np.concatenate([item_2, item_1], axis=0)
                else:
                    item_pair = np.concatenate([item_1, item_2], axis=0)
                batch_trials.append(item_pair)
                batch_correct_choices.append(choice)
                batch_pair_indices.append([high_idx, low_idx])

        # Generate List 2 trials in specified order
        for high_idx, low_idx in list_2_order:
            for _ in range(num_trials_per_pair):
                item_1 = batch_items[batch_idx, high_idx]
                item_2 = batch_items[batch_idx, low_idx]
                swap = np.random.randint(0, 2)
                choice = swap
                if swap:
                    item_pair = np.concatenate([item_2, item_1], axis=0)
                else:
                    item_pair = np.concatenate([item_1, item_2], axis=0)
                batch_trials.append(item_pair)
                batch_correct_choices.append(choice)
                batch_pair_indices.append([high_idx, low_idx])

        # Generate linking pair trials (DE)
        high_idx, low_idx = linking_pair
        for _ in range(num_trials_per_pair):
            item_1 = batch_items[batch_idx, high_idx]
            item_2 = batch_items[batch_idx, low_idx]
            swap = np.random.randint(0, 2)
            choice = swap
            if swap:
                item_pair = np.concatenate([item_2, item_1], axis=0)
            else:
                item_pair = np.concatenate([item_1, item_2], axis=0)
            batch_trials.append(item_pair)
            batch_correct_choices.append(choice)
            batch_pair_indices.append([high_idx, low_idx])

        # Generate test trial (BG)
        high_idx, low_idx = test_pair
        item_1 = batch_items[batch_idx, high_idx]
        item_2 = batch_items[batch_idx, low_idx]
        swap = np.random.randint(0, 2)
        choice = swap
        if swap:
            item_pair = np.concatenate([item_2, item_1], axis=0)
        else:
            item_pair = np.concatenate([item_1, item_2], axis=0)
        batch_trials.append(item_pair)
        batch_correct_choices.append(choice)
        batch_pair_indices.append([high_idx, low_idx])

        all_trials.append(np.array(batch_trials))
        all_correct_choices.append(np.array(batch_correct_choices))
        all_pair_indices.append(np.array(batch_pair_indices))

    return np.array(all_trials), np.array(all_correct_choices), np.array(all_pair_indices)


def eval_controlled_order_ll(args, model, num_networks=128, num_trials_per_pair=3):
    """
    Evaluate list linking with controlled trial ordering to test the chain hypothesis.

    Splits networks into two groups:
    - Degenerate: BC before CD, FG before EF (chain broken for BG)
    - Enriched: CD before BC, EF before FG (chain preserved for BG)

    Tests on both BG (symbolic distance 5, requires chain) and CF (symbolic distance 3, direct overlap).

    Returns accuracy on test trials for each group.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Test pairs: BG (1,6) needs chain, CF (2,5) has direct overlap
    test_pairs = {
        'BG': (1, 6),  # Symbolic distance 5, requires chain
        'CF': (2, 5),  # Symbolic distance 3, direct overlap with CD and EF
    }

    all_results = {}

    for test_name, test_pair in test_pairs.items():
        # Generate items for all networks
        all_batch_items = generate_batch_items(8, args.item_size, num_networks, change_items_throughout_batch=True)

        # Split into two groups
        half = num_networks // 2
        degenerate_items = all_batch_items[:half]
        enriched_items = all_batch_items[half:]

        results = {
            'degenerate': {'correct': 0, 'total': 0},
            'enriched': {'correct': 0, 'total': 0}
        }

        for condition, batch_items, degenerate_order in [
            ('degenerate', degenerate_items, True),
            ('enriched', enriched_items, False)
        ]:
            batch_size = batch_items.shape[0]
            plastic_weights, extra_plastic_weights, embed_pw = create_plastic_weights(batch_size, args.hidden_size, args.extra_layers, getattr(args, 'multi_neuromodulator', 1), device, direct_readout=getattr(args, 'direct_readout', False), first_plastic_input_size=getattr(model, 'first_plastic_input_size', args.hidden_size), plastic_embedding=getattr(args, 'plastic_embedding', False), input_size=2*args.item_size)

            trials, correct_choices, _ = generate_controlled_order_trials_ll(
                batch_items, num_trials_per_pair, degenerate_order=degenerate_order,
                test_pair=test_pair
            )

            trials = torch.tensor(trials, dtype=torch.float32).to(device)
            correct_choices_t = torch.tensor(correct_choices, dtype=torch.float32).to(device)

            num_trials = trials.shape[1]

            # Run through all trials
            for trial_idx in range(num_trials):
                batch_trial = trials[:, trial_idx, :]
                batch_correct_choice = correct_choices_t[:, trial_idx]

                with torch.inference_mode():
                    output = model(batch_trial, plastic_weights, batch_correct_choice,
                                  extra_plastic_weights=extra_plastic_weights, embed_plastic_weights=embed_pw)

                plastic_weights = output.plastic_weights
                extra_plastic_weights = output.extra_plastic_weights
                embed_pw = output.embed_plastic_weights

                # Check test trial (last trial)
                if trial_idx == num_trials - 1:
                    choice_sampled = output.sampled_choices.squeeze(-1)
                    for batch_idx in range(batch_size):
                        is_correct = int(choice_sampled[batch_idx].item() == batch_correct_choice[batch_idx].item())
                        results[condition]['correct'] += is_correct
                        results[condition]['total'] += 1

        # Calculate accuracies for this test pair
        all_results[f'{test_name}_degenerate_accuracy'] = results['degenerate']['correct'] / results['degenerate']['total']
        all_results[f'{test_name}_enriched_accuracy'] = results['enriched']['correct'] / results['enriched']['total']

    model.train()

    return all_results


def plastic_weight_ablation_ll(args, model):
    """
    Evaluate list linking performance with plastic weight ablations.
    Ablated weights are zeroed from the start and never updated during training.

    1. innate_only: both plastic weights zeroed throughout training
    2. first_layer_only: extra_plastic_weights updated, final plastic_weights always zero
    3. second_layer_only: final plastic_weights updated, extra_plastic_weights always zero

    Returns dict of {ablation_name: zero_shot_trials} for plotting with zero_shot_symbolic_distance_plot
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    num_items = args.item_range[-1] - 1

    # Initialize result dicts for each ablation condition
    ablation_results = {
        'innate_only': {(i,j): [] for i in range(num_items) for j in range(i + 1, num_items)},
        'first_layer_only': {(i,j): [] for i in range(num_items) for j in range(i + 1, num_items)},
        'second_layer_only': {(i,j): [] for i in range(num_items) for j in range(i + 1, num_items)},
    }

    all_batch_items = generate_batch_items(num_items, args.item_size, args.full_eval_batch_size, change_items_throughout_batch=True)

    num_batchs = args.full_eval_batch_size // args.batch_size + (args.full_eval_batch_size % args.batch_size > 0)

    # Define ablation conditions: (name, update_final_pw, update_extra_pw)
    ablation_conditions = [
        ('innate_only', False, False),
        ('first_layer_only', False, True),
        ('second_layer_only', True, False),
    ]

    for ablation_name, update_final_pw, update_extra_pw in ablation_conditions:
        for batch_num in range(num_batchs):
            batch_start = batch_num * args.batch_size
            batch_end = min((batch_num + 1) * args.batch_size, args.full_eval_batch_size)
            batch_items = all_batch_items[batch_start:batch_end]
            batch_size = batch_items.shape[0]

            # Initialize plastic weights (always start at zero)
            plastic_weights, extra_plastic_weights, embed_pw = create_plastic_weights(batch_size, args.hidden_size, args.extra_layers, getattr(args, 'multi_neuromodulator', 1), device, direct_readout=getattr(args, 'direct_readout', False), first_plastic_input_size=getattr(model, 'first_plastic_input_size', args.hidden_size), plastic_embedding=getattr(args, 'plastic_embedding', False), input_size=2*args.item_size)

            trials, correct_choices, pair_indices = generate_batch_trials_ll(
                batch_items, args.num_trials_list_1, args.num_trials_list_2, args.num_trials_linking_pair, 1,
                put_linking_trials_first=args.put_linking_trials_first, randomize_list_order=args.randomize_list_order
            )

            trials = torch.tensor(trials, dtype=torch.float32).to(device)
            correct_choices_t = torch.tensor(correct_choices, dtype=torch.float32).to(device)
            num_train_trials = args.num_trials_list_1 + args.num_trials_list_2 + args.num_trials_linking_pair

            # Run through training trials, selectively updating plastic weights
            for trial in range(num_train_trials):
                batch_trial = trials[:, trial, :]
                batch_correct_choice = correct_choices_t[:, trial]

                with torch.inference_mode():
                    output = model(batch_trial, plastic_weights, batch_correct_choice, extra_plastic_weights=extra_plastic_weights, embed_plastic_weights=embed_pw)

                # Only update weights that are not ablated
                if update_final_pw:
                    plastic_weights = output.plastic_weights
                if update_extra_pw:
                    extra_plastic_weights = output.extra_plastic_weights
                embed_pw = output.embed_plastic_weights

            # Evaluate on test trial
            test_trial = trials[:, num_train_trials, :]
            test_correct_choice = correct_choices_t[:, num_train_trials]
            test_pair_indices = pair_indices[:, num_train_trials]

            with torch.inference_mode():
                output = model(test_trial, plastic_weights, test_correct_choice, extra_plastic_weights=extra_plastic_weights, embed_plastic_weights=embed_pw)

            choice_sampled = output.sampled_choices.squeeze(-1)
            for episode in range(batch_size):
                correct_test_choice = int(choice_sampled[episode].item() == test_correct_choice[episode].item())
                left_idx, right_idx = test_pair_indices[episode][0], test_pair_indices[episode][1]
                if left_idx > right_idx:
                    left_idx, right_idx = right_idx, left_idx
                ablation_results[ablation_name][(left_idx, right_idx)].append(correct_test_choice)

    model.train()
    return ablation_results


def plastic_weight_ablation_ti(args, model):
    """
    Evaluate transitive inference performance with plastic weight ablations.
    Ablated weights are zeroed from the start and never updated during training.

    1. innate_only: both plastic weights zeroed throughout training
    2. first_layer_only: extra_plastic_weights updated, final plastic_weights always zero
    3. second_layer_only: final plastic_weights updated, extra_plastic_weights always zero

    Returns dict of {ablation_name: zero_shot_trials} for plotting with zero_shot_symbolic_distance_plot
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    num_items = args.item_range[-1] - 1

    # Initialize result dicts for each ablation condition
    ablation_results = {
        'innate_only': {(i,j): [] for i in range(num_items) for j in range(i + 1, num_items)},
        'first_layer_only': {(i,j): [] for i in range(num_items) for j in range(i + 1, num_items)},
        'second_layer_only': {(i,j): [] for i in range(num_items) for j in range(i + 1, num_items)},
    }

    all_batch_items = generate_batch_items(num_items, args.item_size, args.full_eval_batch_size, change_items_throughout_batch=True)

    num_batchs = args.full_eval_batch_size // args.batch_size + (args.full_eval_batch_size % args.batch_size > 0)

    # Define ablation conditions: (name, update_final_pw, update_extra_pw)
    ablation_conditions = [
        ('innate_only', False, False),
        ('first_layer_only', False, True),
        ('second_layer_only', True, False),
    ]

    for ablation_name, update_final_pw, update_extra_pw in ablation_conditions:
        for batch_num in range(num_batchs):
            batch_start = batch_num * args.batch_size
            batch_end = min((batch_num + 1) * args.batch_size, args.full_eval_batch_size)
            batch_items = all_batch_items[batch_start:batch_end]
            batch_size = batch_items.shape[0]

            # Initialize plastic weights (always start at zero)
            plastic_weights, extra_plastic_weights, embed_pw = create_plastic_weights(batch_size, args.hidden_size, args.extra_layers, getattr(args, 'multi_neuromodulator', 1), device, direct_readout=getattr(args, 'direct_readout', False), first_plastic_input_size=getattr(model, 'first_plastic_input_size', args.hidden_size), plastic_embedding=getattr(args, 'plastic_embedding', False), input_size=2*args.item_size)

            # Generate TI trials - use num_test_trials=1 for a single zero-shot test
            trials, correct_choices, pair_indices, num_train_trials = generate_batch_trials_ti(
                batch_items, args.num_train_trials, num_test_trials=1, arbitrary=args.arbitrary
            )

            trials = torch.tensor(trials, dtype=torch.float32).to(device)
            correct_choices_t = torch.tensor(correct_choices, dtype=torch.float32).to(device)

            # Run through training trials, selectively updating plastic weights
            for trial in range(num_train_trials):
                batch_trial = trials[:, trial, :]
                batch_correct_choice = correct_choices_t[:, trial]

                with torch.inference_mode():
                    output = model(batch_trial, plastic_weights, batch_correct_choice, extra_plastic_weights=extra_plastic_weights, embed_plastic_weights=embed_pw)

                # Only update weights that are not ablated
                if update_final_pw:
                    plastic_weights = output.plastic_weights
                if update_extra_pw:
                    extra_plastic_weights = output.extra_plastic_weights
                embed_pw = output.embed_plastic_weights

            # Evaluate on test trial
            test_trial = trials[:, num_train_trials, :]
            test_correct_choice = correct_choices_t[:, num_train_trials]
            test_pair_indices = pair_indices[:, num_train_trials]

            with torch.inference_mode():
                output = model(test_trial, plastic_weights, test_correct_choice, extra_plastic_weights=extra_plastic_weights, embed_plastic_weights=embed_pw)

            choice_sampled = output.sampled_choices.squeeze(-1)
            for episode in range(batch_size):
                correct_test_choice = int(choice_sampled[episode].item() == test_correct_choice[episode].item())
                left_idx, right_idx = test_pair_indices[episode][0], test_pair_indices[episode][1]
                if left_idx > right_idx:
                    left_idx, right_idx = right_idx, left_idx
                ablation_results[ablation_name][(left_idx, right_idx)].append(correct_test_choice)

    model.train()
    return ablation_results


def top_alpha_ablation(args, model, k=10):
    """
    Ablation that zeroes out the top k alpha weights by magnitude in the final layer,
    then evaluates zero-shot performance on TI and LL tasks.

    Args:
        args: experiment arguments
        model: the trained model
        k: number of top alpha weights to zero out (default: 10)

    Returns:
        ti_zero_shot_trials: dict for TI task
        ll_zero_shot_trials: dict for LL task
        info_fig: figure showing ablated weight coordinates and magnitudes
    """
    logger.info(f"Running top-{k} alpha ablation...")

    # --- Save original alpha weights ---
    original_alpha = model.alpha.data.clone()
    original_alpha_flat = original_alpha.flatten()

    # --- Find top k weights by magnitude ---
    alpha_flat = model.alpha.data.flatten()
    top_indices_flat = torch.topk(alpha_flat.abs(), k=k).indices

    # Convert flat indices to 2D coordinates
    alpha_shape = model.alpha.data.shape
    top_coords = []
    top_values = []
    for flat_idx in top_indices_flat:
        row = flat_idx.item() // alpha_shape[1]
        col = flat_idx.item() % alpha_shape[1]
        val = alpha_flat[flat_idx].item()
        top_coords.append((row, col))
        top_values.append(val)

    # Compute statistics for context
    alpha_mean = alpha_flat.mean().item()
    alpha_std = alpha_flat.std().item()

    # Log the weights being ablated
    logger.info(f"Alpha mean: {alpha_mean:.6f}, std: {alpha_std:.6f}")
    logger.info(f"Top {k} alpha weights by magnitude:")
    for i, ((row, col), val) in enumerate(zip(top_coords, top_values)):
        std_from_mean = (val - alpha_mean) / alpha_std
        logger.info(f"  {i+1}. ({row}, {col}): value={val:.6f}, std_from_mean={std_from_mean:.2f}")

    # --- Zero out top k weights ---
    alpha_flat[top_indices_flat] = 0.0

    # --- Run zero-shot evaluations ---
    ti_zero_shot_trials = full_eval_ti(args, model)
    ll_zero_shot_trials = full_eval_ll(args, model)

    # --- Restore original alpha weights ---
    model.alpha.data.copy_(original_alpha)

    # --- Verify restoration ---
    restored_alpha_flat = model.alpha.data.flatten()
    max_diff = (restored_alpha_flat - original_alpha_flat).abs().max().item()
    assert max_diff < 1e-7, f"Alpha restoration failed! Max diff: {max_diff}"
    logger.info(f"Alpha weights restored successfully (max diff: {max_diff:.2e})")

    # --- Create info figure ---
    info_fig = _create_alpha_ablation_info_figure(
        top_coords, top_values, alpha_mean, alpha_std, alpha_shape, k
    )

    return ti_zero_shot_trials, ll_zero_shot_trials, info_fig


def _create_alpha_ablation_info_figure(top_coords, top_values, alpha_mean, alpha_std, alpha_shape, k):
    """
    Create an info figure showing the ablated alpha weight details.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    # --- Left plot: Bar chart of ablated weight values ---
    ax_bar = axes[0]
    x_labels = [f"({r},{c})" for r, c in top_coords]
    colors = ['tab:red' if v < 0 else 'tab:blue' for v in top_values]
    bars = ax_bar.bar(range(k), top_values, color=colors, edgecolor='black', alpha=0.7)

    ax_bar.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax_bar.axhline(y=alpha_mean, color='green', linestyle='--', alpha=0.7, label=f'mean={alpha_mean:.4f}')
    ax_bar.axhline(y=alpha_mean + 3*alpha_std, color='orange', linestyle=':', alpha=0.7, label=f'+3σ={alpha_mean + 3*alpha_std:.4f}')
    ax_bar.axhline(y=alpha_mean - 3*alpha_std, color='orange', linestyle=':', alpha=0.7, label=f'-3σ={alpha_mean - 3*alpha_std:.4f}')

    ax_bar.set_xticks(range(k))
    ax_bar.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
    ax_bar.set_xlabel('Coordinate (row, col)')
    ax_bar.set_ylabel('Alpha Value')
    ax_bar.set_title(f'Top {k} Alpha Weights by Magnitude (Ablated)')
    ax_bar.legend(loc='upper right', fontsize=8)

    # --- Right plot: Table with detailed info ---
    ax_table = axes[1]
    ax_table.axis('off')

    # Build table data
    table_data = [['Rank', 'Coord', 'Value', 'STD from Mean']]
    for i, ((row, col), val) in enumerate(zip(top_coords, top_values)):
        std_from_mean = (val - alpha_mean) / alpha_std
        table_data.append([
            f"{i+1}",
            f"({row}, {col})",
            f"{val:.6f}",
            f"{std_from_mean:+.2f}σ"
        ])

    # Add summary row
    table_data.append(['', '', '', ''])
    table_data.append(['', 'Alpha Shape:', f'{alpha_shape[0]}x{alpha_shape[1]}', ''])
    table_data.append(['', 'Mean:', f'{alpha_mean:.6f}', ''])
    table_data.append(['', 'Std:', f'{alpha_std:.6f}', ''])

    table = ax_table.table(
        cellText=table_data,
        loc='center',
        cellLoc='center',
        colWidths=[0.15, 0.25, 0.3, 0.3]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style header row
    for j in range(4):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    ax_table.set_title(f'Ablated Alpha Weight Details\n(Top {k} by |value|)', fontsize=12, pad=20)

    fig.suptitle(f'Top-{k} Alpha Ablation Info', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.close()

    return fig


def continual_learning_eval(args, model, num_networks=512):
    """
    Evaluate continual learning: train on one task, then another, then evaluate both.

    Tests:
    1. TI → AI: Train TI, then AI, evaluate both
    2. AI → TI: Train AI, then TI, evaluate both
    3. LL → AI: Train LL, then AI, evaluate both
    4. AI → LL: Train AI, then LL, evaluate both

    Args:
        args: experiment arguments
        model: the trained model
        num_networks: number of networks to average over (default: 512)

    Returns:
        results: dict with zero_shot_trials for each condition and task
        metadata: dict with AI metadata for plotting
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    results = {}

    # --- TI ↔ AI ---
    logger.info("Running continual learning eval: TI → AI")
    ti_then_ai_ti, ti_then_ai_ai, ai_metadata = _continual_eval_ti_ai(
        args, model, device, num_networks, ti_first=True
    )
    results['ti_then_ai_ti'] = ti_then_ai_ti
    results['ti_then_ai_ai'] = ti_then_ai_ai

    logger.info("Running continual learning eval: AI → TI")
    ai_then_ti_ti, ai_then_ti_ai, _ = _continual_eval_ti_ai(
        args, model, device, num_networks, ti_first=False
    )
    results['ai_then_ti_ti'] = ai_then_ti_ti
    results['ai_then_ti_ai'] = ai_then_ti_ai

    # --- LL ↔ AI ---
    logger.info("Running continual learning eval: LL → AI")
    ll_then_ai_ll, ll_then_ai_ai, _ = _continual_eval_ll_ai(
        args, model, device, num_networks, ll_first=True
    )
    results['ll_then_ai_ll'] = ll_then_ai_ll
    results['ll_then_ai_ai'] = ll_then_ai_ai

    logger.info("Running continual learning eval: AI → LL")
    ai_then_ll_ll, ai_then_ll_ai, _ = _continual_eval_ll_ai(
        args, model, device, num_networks, ll_first=False
    )
    results['ai_then_ll_ll'] = ai_then_ll_ll
    results['ai_then_ll_ai'] = ai_then_ll_ai

    model.train()

    return results, ai_metadata


def _continual_eval_ti_ai(args, model, device, num_networks, ti_first=True):
    """
    Continual learning evaluation for TI and AI.

    Args:
        ti_first: If True, train TI then AI. If False, train AI then TI.

    Returns:
        ti_zero_shot_trials: dict for TI zero-shot evaluation
        ai_zero_shot_trials: dict for AI zero-shot evaluation
        ai_metadata: metadata for AI plotting
    """
    num_items_ti = args.item_range[-1] - 1
    num_groups = args.associative_inference_num_groups
    num_items_per_group = args.associative_inference_num_items_per_group
    total_items_ai = num_groups * num_items_per_group

    # Initialize result dicts
    ti_zero_shot_trials = {(i, j): [] for i in range(num_items_ti) for j in range(i + 1, num_items_ti)}
    ai_zero_shot_trials = {(i, j): [] for i in range(total_items_ai) for j in range(total_items_ai)}

    # Process in batches
    num_batches = num_networks // args.batch_size + (num_networks % args.batch_size > 0)

    for batch_num in range(num_batches):
        batch_start = batch_num * args.batch_size
        batch_end = min((batch_num + 1) * args.batch_size, num_networks)
        batch_size = batch_end - batch_start

        # Initialize plastic weights
        plastic_weights, extra_plastic_weights, embed_pw = create_plastic_weights(batch_size, args.hidden_size, args.extra_layers, getattr(args, 'multi_neuromodulator', 1), device, direct_readout=getattr(args, 'direct_readout', False), first_plastic_input_size=getattr(model, 'first_plastic_input_size', args.hidden_size), plastic_embedding=getattr(args, 'plastic_embedding', False), input_size=2*args.item_size)

        # Generate items for both tasks
        batch_items_ti = generate_batch_items(num_items_ti, args.item_size, batch_size, change_items_throughout_batch=True)
        batch_items_ai = generate_batch_items_ai(num_groups, num_items_per_group, args.item_size, batch_size, change_items_throughout_batch=True)

        # Generate training trials only (no test trials)
        trials_ti, correct_choices_ti, _, num_train_trials_ti = generate_batch_trials_ti(
            batch_items_ti, args.num_train_trials, num_test_trials=0, arbitrary=args.arbitrary
        )
        trials_ai, correct_choices_ai, _, num_train_trials_ai = generate_batch_trials_ai(
            batch_items_ai, num_items_per_group, num_test_trials=0,
            exclude_same_item=args.ai_exclude_same_item
        )

        trials_ti = torch.tensor(trials_ti, dtype=torch.float32).to(device)
        correct_choices_ti = torch.tensor(correct_choices_ti, dtype=torch.float32).to(device)
        trials_ai = torch.tensor(trials_ai, dtype=torch.float32).to(device)
        correct_choices_ai = torch.tensor(correct_choices_ai, dtype=torch.float32).to(device)

        # Train on first task, then second task
        if ti_first:
            # Train TI
            plastic_weights, extra_plastic_weights = _run_training_trials(
                model, trials_ti, correct_choices_ti, plastic_weights, extra_plastic_weights, num_train_trials_ti
            )
            # Train AI
            plastic_weights, extra_plastic_weights = _run_training_trials(
                model, trials_ai, correct_choices_ai, plastic_weights, extra_plastic_weights, num_train_trials_ai
            )
        else:
            # Train AI
            plastic_weights, extra_plastic_weights = _run_training_trials(
                model, trials_ai, correct_choices_ai, plastic_weights, extra_plastic_weights, num_train_trials_ai
            )
            # Train TI
            plastic_weights, extra_plastic_weights = _run_training_trials(
                model, trials_ti, correct_choices_ti, plastic_weights, extra_plastic_weights, num_train_trials_ti
            )

        # Freeze weights and evaluate both tasks
        frozen_pw, frozen_epw, frozen_embed_pw = clone_plastic_weights(plastic_weights, extra_plastic_weights, embed_pw=embed_pw)

        # Evaluate TI zero-shot
        ti_results = _evaluate_all_pairs_zero_shot_with_items(
            model, batch_items_ti, frozen_pw, frozen_epw, device, num_items_ti, frozen_embed_pw=frozen_embed_pw
        )
        for key, values in ti_results.items():
            ti_zero_shot_trials[key].extend(values)

        # Evaluate AI zero-shot
        ai_results = _evaluate_all_pairs_zero_shot_ai(
            model, batch_items_ai, frozen_pw, frozen_epw, device, num_groups, num_items_per_group,
            exclude_same_item=args.ai_exclude_same_item, frozen_embed_pw=frozen_embed_pw
        )
        for key, values in ai_results.items():
            ai_zero_shot_trials[key].extend(values)

    ai_metadata = {
        'num_groups': num_groups,
        'num_items_per_group': num_items_per_group,
        'total_items': total_items_ai,
        'exclude_same_item': args.ai_exclude_same_item,
    }

    return ti_zero_shot_trials, ai_zero_shot_trials, ai_metadata


def _continual_eval_ll_ai(args, model, device, num_networks, ll_first=True):
    """
    Continual learning evaluation for LL and AI.

    Args:
        ll_first: If True, train LL then AI. If False, train AI then LL.

    Returns:
        ll_zero_shot_trials: dict for LL zero-shot evaluation
        ai_zero_shot_trials: dict for AI zero-shot evaluation
        ai_metadata: metadata for AI plotting
    """
    num_items_ll = 8  # LL uses 8 items
    num_groups = args.associative_inference_num_groups
    num_items_per_group = args.associative_inference_num_items_per_group
    total_items_ai = num_groups * num_items_per_group

    # Initialize result dicts
    ll_zero_shot_trials = {(i, j): [] for i in range(num_items_ll) for j in range(i + 1, num_items_ll)}
    ai_zero_shot_trials = {(i, j): [] for i in range(total_items_ai) for j in range(total_items_ai)}

    # Process in batches
    num_batches = num_networks // args.batch_size + (num_networks % args.batch_size > 0)

    for batch_num in range(num_batches):
        batch_start = batch_num * args.batch_size
        batch_end = min((batch_num + 1) * args.batch_size, num_networks)
        batch_size = batch_end - batch_start

        # Initialize plastic weights
        plastic_weights, extra_plastic_weights, embed_pw = create_plastic_weights(batch_size, args.hidden_size, args.extra_layers, getattr(args, 'multi_neuromodulator', 1), device, direct_readout=getattr(args, 'direct_readout', False), first_plastic_input_size=getattr(model, 'first_plastic_input_size', args.hidden_size), plastic_embedding=getattr(args, 'plastic_embedding', False), input_size=2*args.item_size)

        # Generate items for both tasks
        batch_items_ll = generate_batch_items(num_items_ll, args.item_size, batch_size, change_items_throughout_batch=True)
        batch_items_ai = generate_batch_items_ai(num_groups, num_items_per_group, args.item_size, batch_size, change_items_throughout_batch=True)

        # Generate training trials only (no test trials)
        num_train_trials_ll = args.num_trials_list_1 + args.num_trials_list_2 + args.num_trials_linking_pair
        trials_ll, correct_choices_ll, _ = generate_batch_trials_ll(
            batch_items_ll, args.num_trials_list_1, args.num_trials_list_2, args.num_trials_linking_pair,
            num_test_trials=0, put_linking_trials_first=args.put_linking_trials_first,
            randomize_list_order=args.randomize_list_order
        )
        trials_ai, correct_choices_ai, _, num_train_trials_ai = generate_batch_trials_ai(
            batch_items_ai, num_items_per_group, num_test_trials=0,
            exclude_same_item=args.ai_exclude_same_item
        )

        trials_ll = torch.tensor(trials_ll, dtype=torch.float32).to(device)
        correct_choices_ll = torch.tensor(correct_choices_ll, dtype=torch.float32).to(device)
        trials_ai = torch.tensor(trials_ai, dtype=torch.float32).to(device)
        correct_choices_ai = torch.tensor(correct_choices_ai, dtype=torch.float32).to(device)

        # Train on first task, then second task
        if ll_first:
            # Train LL
            plastic_weights, extra_plastic_weights = _run_training_trials(
                model, trials_ll, correct_choices_ll, plastic_weights, extra_plastic_weights, num_train_trials_ll
            )
            # Train AI
            plastic_weights, extra_plastic_weights = _run_training_trials(
                model, trials_ai, correct_choices_ai, plastic_weights, extra_plastic_weights, num_train_trials_ai
            )
        else:
            # Train AI
            plastic_weights, extra_plastic_weights = _run_training_trials(
                model, trials_ai, correct_choices_ai, plastic_weights, extra_plastic_weights, num_train_trials_ai
            )
            # Train LL
            plastic_weights, extra_plastic_weights = _run_training_trials(
                model, trials_ll, correct_choices_ll, plastic_weights, extra_plastic_weights, num_train_trials_ll
            )

        # Freeze weights and evaluate both tasks
        frozen_pw, frozen_epw, frozen_embed_pw = clone_plastic_weights(plastic_weights, extra_plastic_weights, embed_pw=embed_pw)

        # Evaluate LL zero-shot
        ll_results = _evaluate_all_pairs_zero_shot_with_items(
            model, batch_items_ll, frozen_pw, frozen_epw, device, num_items_ll, frozen_embed_pw=frozen_embed_pw
        )
        for key, values in ll_results.items():
            ll_zero_shot_trials[key].extend(values)

        # Evaluate AI zero-shot
        ai_results = _evaluate_all_pairs_zero_shot_ai(
            model, batch_items_ai, frozen_pw, frozen_epw, device, num_groups, num_items_per_group,
            exclude_same_item=args.ai_exclude_same_item, frozen_embed_pw=frozen_embed_pw
        )
        for key, values in ai_results.items():
            ai_zero_shot_trials[key].extend(values)

    ai_metadata = {
        'num_groups': num_groups,
        'num_items_per_group': num_items_per_group,
        'total_items': total_items_ai,
        'exclude_same_item': args.ai_exclude_same_item,
    }

    return ll_zero_shot_trials, ai_zero_shot_trials, ai_metadata


def _run_training_trials(model, trials, correct_choices, plastic_weights, extra_plastic_weights, num_trials):
    """
    Run training trials and update plastic weights.

    Returns:
        Updated plastic_weights and extra_plastic_weights
    """
    for trial in range(num_trials):
        batch_trial = trials[:, trial, :]
        batch_correct_choice = correct_choices[:, trial]

        with torch.inference_mode():
            output = model(batch_trial, plastic_weights, batch_correct_choice,
                          extra_plastic_weights=extra_plastic_weights, embed_plastic_weights=embed_pw)
        plastic_weights = output.plastic_weights
        extra_plastic_weights = output.extra_plastic_weights
        embed_pw = output.embed_plastic_weights

    return plastic_weights, extra_plastic_weights


def _evaluate_all_pairs_zero_shot_with_items(model, batch_items, frozen_pw, frozen_epw, device, num_items, frozen_embed_pw=None):
    """
    Evaluate zero-shot accuracy on all possible pairs with frozen plastic weights.
    Similar to _evaluate_all_pairs_zero_shot but takes batch_items and num_items as parameters.

    Returns:
        dict: {(i, j): [list of 0/1 correctness]} for each pair
    """
    batch_size = pw_batch_size(frozen_pw)

    # Track per-pair correctness
    zero_shot_trials = {(i, j): [] for i in range(num_items) for j in range(i + 1, num_items)}

    for i in range(num_items):
        for j in range(i + 1, num_items):
            item_higher = batch_items[:, i, :]
            item_lower = batch_items[:, j, :]

            # Test order 1: higher item first (correct choice = 0)
            trial_input = np.concatenate([item_higher, item_lower], axis=1)
            trial_input = torch.tensor(trial_input, dtype=torch.float32).to(device)
            correct_choice = torch.zeros(batch_size, dtype=torch.float32).to(device)

            with torch.inference_mode():
                output = model(trial_input, frozen_pw, correct_choice,
                              extra_plastic_weights=frozen_epw, store_embeddings=False, embed_plastic_weights=frozen_embed_pw)

            choice_sampled = output.sampled_choices.squeeze(-1)
            correct_mask = (choice_sampled == correct_choice).cpu().numpy()
            zero_shot_trials[(i, j)].extend(correct_mask.astype(int).tolist())

            # Test order 2: lower item first (correct choice = 1)
            trial_input_rev = np.concatenate([item_lower, item_higher], axis=1)
            trial_input_rev = torch.tensor(trial_input_rev, dtype=torch.float32).to(device)
            correct_choice_rev = torch.ones(batch_size, dtype=torch.float32).to(device)

            with torch.inference_mode():
                output_rev = model(trial_input_rev, frozen_pw, correct_choice_rev,
                                  extra_plastic_weights=frozen_epw, store_embeddings=False)

            choice_sampled_rev = output_rev.sampled_choices.squeeze(-1)
            correct_mask_rev = (choice_sampled_rev == correct_choice_rev).cpu().numpy()
            zero_shot_trials[(i, j)].extend(correct_mask_rev.astype(int).tolist())

    return zero_shot_trials


def _evaluate_all_pairs_zero_shot_ai(model, batch_items, frozen_pw, frozen_epw, device, num_groups, num_items_per_group, exclude_same_item=False, frozen_embed_pw=None):
    """
    Evaluate zero-shot accuracy on all possible AI pairs with frozen plastic weights.

    For AI, the correct answer is 1 if same group (including same item), 0 if different groups.

    Args:
        batch_items: shape (batch_size, num_groups, num_items_per_group, item_size)
        exclude_same_item: If True, skip same-item pairs [A,A]

    Returns:
        dict: {(item1_id, item2_id): [list of 0/1 correctness]} for each ordered pair (optionally excluding diagonal)
    """
    batch_size = pw_batch_size(frozen_pw)
    total_items = num_groups * num_items_per_group

    # Track per-pair correctness (all pairs, optionally excluding diagonal)
    zero_shot_trials = {(i, j): [] for i in range(total_items) for j in range(total_items)}

    for g1 in range(num_groups):
        for idx1 in range(num_items_per_group):
            for g2 in range(num_groups):
                for idx2 in range(num_items_per_group):
                    # Skip same-item pairs if exclude_same_item is True
                    if exclude_same_item and g1 == g2 and idx1 == idx2:
                        continue

                    item1 = batch_items[:, g1, idx1, :]  # shape: (batch_size, item_size)
                    item2 = batch_items[:, g2, idx2, :]

                    # Correct answer: 1 if same group (including same item), 0 if different groups
                    correct = 1.0 if g1 == g2 else 0.0

                    # Test this pair
                    trial_input = np.concatenate([item1, item2], axis=1)
                    trial_input = torch.tensor(trial_input, dtype=torch.float32).to(device)
                    correct_choice = torch.full((batch_size,), correct, dtype=torch.float32).to(device)

                    with torch.inference_mode():
                        output = model(trial_input, frozen_pw, correct_choice,
                                      extra_plastic_weights=frozen_epw, store_embeddings=False, embed_plastic_weights=frozen_embed_pw)

                    choice_sampled = output.sampled_choices.squeeze(-1)
                    correct_mask = (choice_sampled == correct_choice).cpu().numpy()

                    # Convert to flat item IDs
                    item1_id = g1 * num_items_per_group + idx1
                    item2_id = g2 * num_items_per_group + idx2

                    zero_shot_trials[(item1_id, item2_id)].extend(correct_mask.astype(int).tolist())

    return zero_shot_trials


def ai_generalization_test(args, model, additional_groups=3, additional_items_per_group=3):
    """
    Generalization test for associative inference task.

    Tests all combinations of:
    - num_groups: base to base + additional_groups
    - num_items_per_group: base to base + additional_items_per_group

    For each combination:
    1. Train on that configuration
    2. Freeze plastic weights
    3. Do zero-shot evaluation on all pairs
    4. Record accuracies (overall, adjacent, nonadjacent)

    Args:
        args: experiment arguments
        model: the trained model
        additional_groups: number of additional groups to test beyond base
        additional_items_per_group: number of additional items per group to test beyond base

    Returns:
        results: dict with accuracies for each (num_groups, num_items_per_group) combination
        heatmap_figs: dict of heatmap figures for each combination
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    base_num_groups = args.associative_inference_num_groups
    base_num_items_per_group = args.associative_inference_num_items_per_group
    batch_size = args.batch_size * 4  # Use 4x batch size like other evals
    item_size = args.item_size

    results = {}
    heatmap_figs = {}

    # Test all combinations
    for g_offset in range(additional_groups + 1):
        for i_offset in range(additional_items_per_group + 1):
            num_groups = base_num_groups + g_offset
            num_items_per_group = base_num_items_per_group + i_offset
            total_items = num_groups * num_items_per_group

            logger.info(f"AI generalization test: {num_groups} groups × {num_items_per_group} items/group")

            # Initialize plastic weights
            plastic_weights, extra_plastic_weights, embed_pw = create_plastic_weights(batch_size, args.hidden_size, args.extra_layers, getattr(args, 'multi_neuromodulator', 1), device, direct_readout=getattr(args, 'direct_readout', False), first_plastic_input_size=getattr(model, 'first_plastic_input_size', args.hidden_size), plastic_embedding=getattr(args, 'plastic_embedding', False), input_size=2*args.item_size)

            # Generate batch items (different items per network)
            batch_items = generate_batch_items_ai(
                num_groups, num_items_per_group, item_size, batch_size,
                change_items_throughout_batch=True
            )

            # Generate training trials only (no test trials)
            trials, correct_choices, _, num_train_trials = generate_batch_trials_ai(
                batch_items, num_items_per_group, num_test_trials=0,
                exclude_same_item=args.ai_exclude_same_item
            )

            trials = torch.tensor(trials, dtype=torch.float32).to(device)
            correct_choices = torch.tensor(correct_choices, dtype=torch.float32).to(device)

            # Run training trials
            for trial in range(num_train_trials):
                batch_trial = trials[:, trial, :]
                batch_correct_choice = correct_choices[:, trial]

                with torch.inference_mode():
                    output = model(batch_trial, plastic_weights, batch_correct_choice,
                                  extra_plastic_weights=extra_plastic_weights, embed_plastic_weights=embed_pw)
                plastic_weights = output.plastic_weights
                extra_plastic_weights = output.extra_plastic_weights
                embed_pw = output.embed_plastic_weights

            # Freeze weights and do zero-shot evaluation on all pairs
            frozen_pw, frozen_epw, frozen_embed_pw = clone_plastic_weights(plastic_weights, extra_plastic_weights, embed_pw=embed_pw)

            # Evaluate all pairs
            zero_shot_trials = _evaluate_all_pairs_zero_shot_ai(
                model, batch_items, frozen_pw, frozen_epw, device, num_groups, num_items_per_group,
                exclude_same_item=args.ai_exclude_same_item, frozen_embed_pw=frozen_embed_pw
            )

            # Compute accuracies
            all_results = [r for res in zero_shot_trials.values() for r in res]
            overall_acc = np.mean(all_results) if all_results else 0.0

            # Compute adjacent vs nonadjacent accuracies
            adj_results = []
            nonadj_results = []
            for (item1_id, item2_id), res in zero_shot_trials.items():
                idx1 = item1_id % num_items_per_group
                idx2 = item2_id % num_items_per_group
                if abs(idx1 - idx2) > 1:
                    nonadj_results.extend(res)
                else:
                    adj_results.extend(res)

            adj_acc = np.mean(adj_results) if adj_results else 0.0
            nonadj_acc = np.mean(nonadj_results) if nonadj_results else 0.0

            # Store results
            key = (num_groups, num_items_per_group)
            results[key] = {
                'overall': overall_acc,
                'adjacent': adj_acc,
                'nonadjacent': nonadj_acc,
                'zero_shot_trials': zero_shot_trials,
            }

            logger.info(f"  Overall: {overall_acc:.4f}, Adjacent: {adj_acc:.4f}, Nonadjacent: {nonadj_acc:.4f}")

            # Create heatmap
            metadata = {
                'num_groups': num_groups,
                'num_items_per_group': num_items_per_group,
                'total_items': total_items,
                'exclude_same_item': args.ai_exclude_same_item,
            }
            from plots import ai_heatmap_plot
            fig = ai_heatmap_plot(zero_shot_trials, metadata)
            # Update title
            fig.axes[0].set_title(f'AI Generalization: {num_groups}G × {num_items_per_group}I\n'
                                   f'Overall: {overall_acc:.2f}, Adj: {adj_acc:.2f}, Nonadj: {nonadj_acc:.2f}')
            heatmap_figs[key] = fig

    model.train()

    return results, heatmap_figs
