import numpy as np

def generate_items(num_items, item_size):
    # Generate a list of binary vectors for our items
    items =[]
    for _ in range(num_items):
        candidate = None
        too_similar = True
        counter = 0 # Counter to prevent infinite loop in case we can't find a different item. Occurs when small num_items and item_size are used.
        while too_similar:
            counter += 1
            if counter > 10000:
                raise ValueError("Could not generate a full list of different items")
            too_similar = False
            candidate = np.random.randint(2, size=item_size) * 2 - 1
            for item in items:
                if np.mean(item == candidate) > .66 :
                    too_similar = True

        assert candidate is not None
        items.append(candidate)
    return np.array(items) # Return a 2D array of shape (num_items, item_size)

def generate_batch_items(num_items, item_size, batch_size, change_items_throughout_batch=False):
    # Apply generate_items to each batch index. Each batch index corresponds to a different agent running in parallel.
    batch_items = []
    for _ in range(batch_size):
        if not change_items_throughout_batch:
            batch_items = [generate_items(num_items, item_size)] * batch_size
        else:
            batch_items.append(generate_items(num_items, item_size))
    return np.array(batch_items) # Return a 3D array of shape (batch_size, num_items, item_size)

def generate_trial(items, is_test=False):
    num_items = items.shape[0]
    if not is_test: # Adjacent pair
        high_item_index = np.random.randint(0, num_items-1)
        low_item_index = high_item_index + 1
    else: # Non-adjacent pair
        high_item_index = np.random.randint(0, num_items-2) # Max index is num_items-3 (top value excluded in np.random.randint) to leave room for low_item_index
        low_item_index = np.random.randint(high_item_index+2, num_items)
    item_1 = items[high_item_index]
    item_2 = items[low_item_index]
    item_pair, choice = generate_pair(item_1, item_2)
    return item_pair, choice, [high_item_index, low_item_index]

def generate_trial_arbitrary(items):
    num_items = items.shape[0]
    high_item_index = np.random.randint(0, num_items-1)
    low_item_index = np.random.randint(high_item_index+1, num_items)
    item_1 = items[high_item_index]
    item_2 = items[low_item_index]
    item_pair, choice = generate_pair(item_1, item_2)
    return item_pair, choice, [high_item_index, low_item_index]

def generate_cross_list_trial(list_1_items, list_2_items):
    num_items_1 = list_1_items.shape[0]
    num_items_2 = list_2_items.shape[0]

    high_item_index = np.random.randint(0, num_items_1)
    # Make sure we're not using the linking pair in test trials
    if high_item_index == num_items_1-1:
        start = 1
    else:
        start = 0
    low_item_index = np.random.randint(start, num_items_2)

    item_1 = list_1_items[high_item_index]
    item_2 = list_2_items[low_item_index]
    item_pair, choice = generate_pair(item_1, item_2)
    return item_pair, choice, [high_item_index, low_item_index]

def generate_batch_trials_ti(batch_items, num_train_trials, num_test_trials, arbitrary=False, mass_presentation=0, exhaustive_test=False):
    """
    Generate TI training and test trials.

    Training trials: First generates all adjacent pairs with both orderings,
    then fills remaining slots with random adjacent pairs.
    If num_train_trials < 2*(num_items-1), it's automatically increased.

    Returns:
        trials_np: array of shape (batch_size, actual_num_train_trials+num_test_trials, 2*item_size)
        correct_choices_np: array of shape (batch_size, actual_num_train_trials+num_test_trials)
        pair_indices_np: array of shape (batch_size, actual_num_train_trials+num_test_trials, 2)
        actual_num_train_trials: the actual number of training trials used
    """
    batch_size = batch_items.shape[0]
    num_items = batch_items.shape[1]

    # Minimum training trials = all adjacent pairs with both orderings
    min_train_trials = 2 * (num_items - 1)
    actual_num_train_trials = max(num_train_trials, min_train_trials)

    trials = []
    correct_choices = []
    pair_indices = []

    for batch_index in range(batch_size):
        items = batch_items[batch_index]
        batch_trials = []
        batch_correct_choices = []
        batch_pair_indices = []

        # === Generate all adjacent pairs with both orderings ===
        for high_idx in range(num_items - 1):
            low_idx = high_idx + 1
            item_high = items[high_idx]
            item_low = items[low_idx]

            # Ordering 1: high item on left -> choice = 0
            item_pair_1 = np.concatenate([item_high, item_low], axis=0)
            batch_trials.append(item_pair_1)
            batch_correct_choices.append(0)
            batch_pair_indices.append([high_idx, low_idx])

            # Ordering 2: high item on right -> choice = 1
            item_pair_2 = np.concatenate([item_low, item_high], axis=0)
            batch_trials.append(item_pair_2)
            batch_correct_choices.append(1)
            batch_pair_indices.append([high_idx, low_idx])

        # === Fill remaining training slots with random adjacent pairs ===
        num_remaining = actual_num_train_trials - min_train_trials
        for _ in range(num_remaining):
            item_pair, choice, trial_pair_indices = generate_trial(items, is_test=False)
            batch_trials.append(item_pair)
            batch_correct_choices.append(choice)
            batch_pair_indices.append(trial_pair_indices)

        # Shuffle training trials
        train_indices = list(range(actual_num_train_trials))
        np.random.shuffle(train_indices)
        batch_trials = [batch_trials[i] for i in train_indices]
        batch_correct_choices = [batch_correct_choices[i] for i in train_indices]
        batch_pair_indices = [batch_pair_indices[i] for i in train_indices]

        # === Generate mass presentation trials (if any) ===
        # These come before test trials and use a random non-adjacent pair
        if mass_presentation > 0:
            item_pair, choice, trial_pair_indices = generate_trial(items, is_test=True)
            for _ in range(mass_presentation):
                batch_trials.append(item_pair)
                batch_correct_choices.append(choice)
                batch_pair_indices.append(trial_pair_indices)

        # === Generate test trials ===
        if exhaustive_test:
            # Generate all N*(N-1) ordered pairs exactly once
            test_trials_buf = []
            for i in range(num_items):
                for j in range(num_items):
                    if i == j:
                        continue
                    item_pair = np.concatenate([items[i], items[j]], axis=0)
                    # i on left, j on right. Correct choice: 0 if i ranks higher (i < j), 1 if j ranks higher (i > j)
                    choice = 0 if i < j else 1
                    test_trials_buf.append((item_pair, choice, [min(i, j), max(i, j)]))
            # Shuffle test trials
            np.random.shuffle(test_trials_buf)
            for item_pair, choice, trial_pair_indices in test_trials_buf:
                batch_trials.append(item_pair)
                batch_correct_choices.append(choice)
                batch_pair_indices.append(trial_pair_indices)
        else:
            for _ in range(num_test_trials):
                if arbitrary:
                    item_pair, choice, trial_pair_indices = generate_trial_arbitrary(items)
                else:
                    item_pair, choice, trial_pair_indices = generate_trial(items, is_test=True)
                batch_trials.append(item_pair)
                batch_correct_choices.append(choice)
                batch_pair_indices.append(trial_pair_indices)

        trials.append(np.array(batch_trials))
        correct_choices.append(np.array(batch_correct_choices))
        pair_indices.append(np.array(batch_pair_indices))

    trials_np = np.array(trials)
    correct_choices_np = np.array(correct_choices)
    pair_indices_np = np.array(pair_indices)

    return trials_np, correct_choices_np, pair_indices_np, actual_num_train_trials

def generate_batch_trials_ll(batch_items, num_trials_list_1, num_trials_list_2, num_trials_linking_pair, num_test_trials, put_linking_trials_first=False, randomize_list_order=False):
    batch_size = batch_items.shape[0]
    num_items = batch_items.shape[1]
    trials = []
    correct_choices = []
    pair_indices = []
    batch_items_list_1 = batch_items[:, :num_items//2]
    batch_items_list_2 = batch_items[:, num_items//2:]
    batch_items_linking_pair = batch_items[:, (num_items//2 - 1):(num_items//2 + 1)]  # D and E (last of list 1, first of list 2)

    # Offsets to convert local indices to global indices
    list_2_offset = num_items // 2
    linking_pair_offset = num_items // 2 - 1

    for batch_index in range(batch_size):
        batch_trials = []
        batch_correct_choices = []
        batch_pair_indices = []
        is_test=False

        # Generate linking pair trials first if requested
        if put_linking_trials_first:
            for _ in range(num_trials_linking_pair):
                item_pair, choice, trial_pair_indices = generate_trial(batch_items_linking_pair[batch_index], is_test)
                global_pair_indices = [trial_pair_indices[0] + linking_pair_offset, trial_pair_indices[1] + linking_pair_offset]
                batch_trials.append(item_pair)
                batch_correct_choices.append(choice)
                batch_pair_indices.append(global_pair_indices)

        # Randomly decide list order if requested (50/50 chance)
        list_1_first = True
        if randomize_list_order:
            list_1_first = np.random.random() < 0.5

        def add_list_1_trials():
            for _ in range(num_trials_list_1):
                item_pair, choice, trial_pair_indices = generate_trial(batch_items_list_1[batch_index], is_test)
                # List 1 local indices are already global (0 to num_items//2 - 1)
                batch_trials.append(item_pair)
                batch_correct_choices.append(choice)
                batch_pair_indices.append(trial_pair_indices)

        def add_list_2_trials():
            for _ in range(num_trials_list_2):
                item_pair, choice, trial_pair_indices = generate_trial(batch_items_list_2[batch_index], is_test)
                # Convert list 2 local indices to global indices
                global_pair_indices = [trial_pair_indices[0] + list_2_offset, trial_pair_indices[1] + list_2_offset]
                batch_trials.append(item_pair)
                batch_correct_choices.append(choice)
                batch_pair_indices.append(global_pair_indices)

        if list_1_first:
            add_list_1_trials()
            add_list_2_trials()
        else:
            add_list_2_trials()
            add_list_1_trials()

        # Generate linking pair trials at end if not putting them first
        if not put_linking_trials_first:
            for _ in range(num_trials_linking_pair):
                item_pair, choice, trial_pair_indices = generate_trial(batch_items_linking_pair[batch_index], is_test)
                global_pair_indices = [trial_pair_indices[0] + linking_pair_offset, trial_pair_indices[1] + linking_pair_offset]
                batch_trials.append(item_pair)
                batch_correct_choices.append(choice)
                batch_pair_indices.append(global_pair_indices)

        for _ in range(num_test_trials):
            item_pair, choice, trial_pair_indices = generate_trial_arbitrary(batch_items[batch_index])
            # Test trials already use global indices
            batch_trials.append(item_pair)
            batch_correct_choices.append(choice)
            batch_pair_indices.append(trial_pair_indices)
        trials.append(np.array(batch_trials))
        correct_choices.append(np.array(batch_correct_choices))
        pair_indices.append(np.array(batch_pair_indices))
    return np.array(trials), np.array(correct_choices), np.array(pair_indices) # Return a 3D array of shape (batch_size, num_train_trials+num_test_trials, 2*item_size) and a 2D array of shape (batch_size, num_train_trials+num_test_trials)

def generate_pair(item_1, item_2):
    swap = np.random.randint(0, 2)
    choice = swap # 0 if item_1 is chosen, 1 if item_2 is chosen
    if swap:
        item_1, item_2 = item_2, item_1
    item_pair = np.concatenate([item_1, item_2], axis=0)
    return item_pair, choice

def generate_episode_ai(num_groups, num_items_per_group, item_size, num_test_trials, exclude_same_item=False):
    """
    Generate an episode for associative inference task.

    Items are organized into groups. Training trials pair items with adjacent indices
    (i, i+1) from any group combination. Test trials can pair any items.
    Correct response: 1 if same group (associated), 0 if different groups.

    Args:
        num_groups: Number of groups
        num_items_per_group: Number of items per group
        item_size: Dimensionality of each item
        num_test_trials: Number of test trials to include
        exclude_same_item: If True, exclude same-item pairs [A,A] from training and test

    Returns:
        trials: array of shape (num_trials, 2*item_size)
        correct_choices: array of shape (num_trials,) - 1 if same group, 0 if different
        pair_indices: array of shape (num_trials, 2, 2) - [[g1, idx1], [g2, idx2]] per trial
        items: array of shape (num_groups, num_items_per_group, item_size)
        num_train_trials: number of training trials
    """
    # Generate items organized into groups
    num_items = num_groups * num_items_per_group
    items_flat = generate_items(num_items, item_size)
    items = items_flat.reshape(num_groups, num_items_per_group, item_size)

    # Training trials:
    # 1. Adjacent index pairs (|idx1-idx2| == 1) for all group combinations
    # 2. Same index pairs (idx1 == idx2) for all group combinations (including same item)
    training_trials_info = []

    # Adjacent index pairs for all group combinations
    for idx in range(num_items_per_group - 1):
        idx1, idx2 = idx, idx + 1
        for g1 in range(num_groups):
            for g2 in range(num_groups):
                # Forward order: item at idx1 from g1, item at idx2 from g2
                training_trials_info.append((g1, idx1, g2, idx2))
                # Reverse order: item at idx2 from g2, item at idx1 from g1
                training_trials_info.append((g2, idx2, g1, idx1))

    # Same index pairs for all group combinations (optionally excluding same item [A A])
    for idx in range(num_items_per_group):
        for g1 in range(num_groups):
            for g2 in range(num_groups):
                # Skip same-item pairs if exclude_same_item is True
                if exclude_same_item and g1 == g2:
                    continue
                training_trials_info.append((g1, idx, g2, idx))

    # Shuffle training trials
    np.random.shuffle(training_trials_info)

    trials = []
    correct_choices = []
    pair_indices = []

    # Generate training trials
    for g1, idx1, g2, idx2 in training_trials_info:
        item1 = items[g1, idx1]
        item2 = items[g2, idx2]
        correct = 1.0 if g1 == g2 else 0.0
        item_pair = np.concatenate([item1, item2], axis=0)
        trials.append(item_pair)
        correct_choices.append(correct)
        pair_indices.append([[g1, idx1], [g2, idx2]])

    num_train_trials = len(trials)

    # Test trials: all possible pairs (optionally excluding same item for diagonal)
    test_trials_info = []
    for g1 in range(num_groups):
        for idx1 in range(num_items_per_group):
            for g2 in range(num_groups):
                for idx2 in range(num_items_per_group):
                    # Skip same-item pairs if exclude_same_item is True
                    if exclude_same_item and g1 == g2 and idx1 == idx2:
                        continue
                    test_trials_info.append((g1, idx1, g2, idx2))

    # Shuffle and take first num_test_trials
    np.random.shuffle(test_trials_info)
    test_trials_info = test_trials_info[:num_test_trials]

    # Generate test trials
    for g1, idx1, g2, idx2 in test_trials_info:
        item1 = items[g1, idx1]
        item2 = items[g2, idx2]
        correct = 1.0 if g1 == g2 else 0.0
        item_pair = np.concatenate([item1, item2], axis=0)
        trials.append(item_pair)
        correct_choices.append(correct)
        pair_indices.append([[g1, idx1], [g2, idx2]])

    return (np.array(trials),
            np.array(correct_choices),
            np.array(pair_indices),
            items,
            num_train_trials)


def generate_batch_items_ai(num_groups, num_items_per_group, item_size, batch_size, change_items_throughout_batch=False):
    """
    Generate batched items for associative inference task.

    Args:
        num_groups: Number of groups
        num_items_per_group: Number of items per group
        item_size: Dimensionality of each item
        batch_size: Number of parallel batches
        change_items_throughout_batch: If True, each batch has different items

    Returns:
        batch_items: array of shape (batch_size, num_groups, num_items_per_group, item_size)
    """
    num_items = num_groups * num_items_per_group
    batch_items = []

    if not change_items_throughout_batch:
        items_flat = generate_items(num_items, item_size)
        items = items_flat.reshape(num_groups, num_items_per_group, item_size)
        batch_items = [items] * batch_size
    else:
        for _ in range(batch_size):
            items_flat = generate_items(num_items, item_size)
            items = items_flat.reshape(num_groups, num_items_per_group, item_size)
            batch_items.append(items)

    return np.array(batch_items)


def generate_batch_trials_ai(batch_items, num_items_per_group, num_test_trials, nonadj_ratio=-1.0, exclude_same_item=False):
    """
    Generate batched trials for associative inference task.

    Args:
        batch_items: array of shape (batch_size, num_groups, num_items_per_group, item_size)
        num_items_per_group: Number of items per group
        num_test_trials: Number of test trials per batch
        nonadj_ratio: Ratio of nonadjacent test trials (0.0-1.0). -1 disables weighting.
                      Adjacent = same group AND |idx1-idx2| == 1
                      Nonadjacent = different groups OR same group with |idx1-idx2| > 1
        exclude_same_item: If True, exclude same-item pairs [A,A] from training and test

    Returns:
        trials: array of shape (batch_size, num_trials, 2*item_size)
        correct_choices: array of shape (batch_size, num_trials)
        pair_indices: array of shape (batch_size, num_trials, 2, 2)
        num_train_trials: number of training trials (same for all batches)
    """
    batch_size = batch_items.shape[0]
    num_groups = batch_items.shape[1]

    all_trials = []
    all_correct_choices = []
    all_pair_indices = []

    for batch_idx in range(batch_size):
        items = batch_items[batch_idx]  # shape: (num_groups, num_items_per_group, item_size)

        # Training trials:
        # 1. Adjacent index pairs (|idx1-idx2| == 1) for all group combinations
        # 2. Same index pairs (idx1 == idx2) for all group combinations (including same item)
        training_trials_info = []

        # Adjacent index pairs for all group combinations
        for idx in range(num_items_per_group - 1):
            idx1, idx2 = idx, idx + 1
            for g1 in range(num_groups):
                for g2 in range(num_groups):
                    # Forward order
                    training_trials_info.append((g1, idx1, g2, idx2))
                    # Reverse order
                    training_trials_info.append((g2, idx2, g1, idx1))

        # Same index pairs for all group combinations (optionally excluding same item [A A])
        for idx in range(num_items_per_group):
            for g1 in range(num_groups):
                for g2 in range(num_groups):
                    # Skip same-item pairs if exclude_same_item is True
                    if exclude_same_item and g1 == g2:
                        continue
                    training_trials_info.append((g1, idx, g2, idx))

        np.random.shuffle(training_trials_info)

        train_trials, train_correct_choices, train_pair_indices = generate_batch_trials_ai_helper(training_trials_info, items)
        num_train_trials = len(train_trials)

        # Test trials: all possible pairs (optionally excluding same item for diagonal)
        test_trials_info = []
        for g1 in range(num_groups):
            for idx1 in range(num_items_per_group):
                for g2 in range(num_groups):
                    for idx2 in range(num_items_per_group):
                        # Skip same-item pairs if exclude_same_item is True
                        if exclude_same_item and g1 == g2 and idx1 == idx2:
                            continue
                        test_trials_info.append((g1, idx1, g2, idx2))

        # Apply nonadjacent ratio weighting if enabled
        if nonadj_ratio >= 0.0:
            # Separate into adjacent and nonadjacent pools
            # Adjacent: same group AND |idx1-idx2| == 1
            adjacent_pool = []
            nonadjacent_pool = []
            for trial_info in test_trials_info:
                g1, idx1, g2, idx2 = trial_info
                is_adjacent = (g1 == g2) and (abs(idx1 - idx2) == 1)
                if is_adjacent:
                    adjacent_pool.append(trial_info)
                else:
                    nonadjacent_pool.append(trial_info)

            # Calculate how many of each to sample
            num_nonadj = int(round(num_test_trials * nonadj_ratio))
            num_adj = num_test_trials - num_nonadj

            # Sample with replacement from each pool
            sampled_test_trials_info = []
            if num_nonadj > 0 and len(nonadjacent_pool) > 0:
                nonadj_indices = np.random.choice(len(nonadjacent_pool), size=num_nonadj, replace=True)
                sampled_test_trials_info.extend([nonadjacent_pool[i] for i in nonadj_indices])
            if num_adj > 0 and len(adjacent_pool) > 0:
                adj_indices = np.random.choice(len(adjacent_pool), size=num_adj, replace=True)
                sampled_test_trials_info.extend([adjacent_pool[i] for i in adj_indices])

            # Shuffle the combined sampled trials
            np.random.shuffle(sampled_test_trials_info)
            test_trials_info = sampled_test_trials_info
        else:
            # Original behavior: random shuffle and take first num_test_trials
            np.random.shuffle(test_trials_info)
            test_trials_info = test_trials_info[:num_test_trials]

        # Generate test trials
        test_trials, test_correct_choices, test_pair_indices = generate_batch_trials_ai_helper(test_trials_info, items)

        # Combine train and test trials for this batch
        all_trials.append(train_trials + test_trials)
        all_correct_choices.append(train_correct_choices + test_correct_choices)
        all_pair_indices.append(train_pair_indices + test_pair_indices)

    return (np.array(all_trials),
            np.array(all_correct_choices),
            np.array(all_pair_indices),
            num_train_trials)

def generate_batch_trials_ai_helper(trials_info, items):
    trials = []
    correct_choices = []
    pair_indices = []

    # Generate training trials
    for g1, idx1, g2, idx2 in trials_info:
        item1 = items[g1, idx1]
        item2 = items[g2, idx2]
        correct = 1.0 if g1 == g2 else 0.0
        item_pair = np.concatenate([item1, item2], axis=0)
        trials.append(item_pair)
        correct_choices.append(correct)
        pair_indices.append([[g1, idx1], [g2, idx2]])

    return trials, correct_choices, pair_indices


def generate_interleaved_ti_ai_batch(
    num_items_ti, item_size, batch_size,
    num_train_trials_ti, num_test_trials_ti,
    num_groups, num_items_per_group,
    ai_num_test_trials,
    change_items_throughout_batch=False,
    arbitrary_ti=False,
    ai_test_nonadj_ratio=-1.0,
    ai_exclude_same_item=False
):
    """
    Generate an interleaved batch where each episode mixes TI and AI trials.

    Training phase: TI adjacent pairs + AI training trials (randomly shuffled together)
    Test phase: TI non-adjacent pairs + AI test trials (randomly shuffled together)

    Args:
        num_items_ti: Number of items for TI task
        item_size: Dimensionality of each item
        batch_size: Number of parallel batches
        num_train_trials_ti: Number of TI training trials
        num_test_trials_ti: Number of TI test trials
        num_groups: Number of groups for AI
        num_items_per_group: Number of items per group for AI
        ai_num_test_trials: Number of AI test trials
        change_items_throughout_batch: If True, each batch has different items
        arbitrary_ti: If True, TI test trials can include adjacent pairs
        ai_test_nonadj_ratio: Ratio of nonadjacent test trials for AI (-1 disables)
        ai_exclude_same_item: If True, exclude same-item pairs [A,A] from AI

    Returns:
        trials: array of shape (batch_size, total_trials, 2*item_size)
        correct_choices: array of shape (batch_size, total_trials)
        task_labels: array of shape (batch_size, total_trials) - 0 for TI, 1 for AI
        ti_pair_indices: array of shape (batch_size, total_trials, 2) - TI indices or [-1,-1] for AI
        ai_pair_indices: array of shape (batch_size, total_trials, 2, 2) - AI indices or [[-1,-1],[-1,-1]] for TI
        num_train_trials: total number of training trials
        num_test_trials: total number of test trials
        num_ti_train: number of TI training trials
        num_ai_train: number of AI training trials
    """
    all_trials = []
    all_correct_choices = []
    all_task_labels = []
    all_ti_pair_indices = []
    all_ai_pair_indices = []

    # For consistent counts across batches, compute AI train trial count from first batch
    num_ai_train = None

    for batch_idx in range(batch_size):
        # Generate TI items
        if batch_idx == 0 or change_items_throughout_batch:
            ti_items = generate_items(num_items_ti, item_size)

        # Generate AI items (separate from TI items)
        num_ai_items = num_groups * num_items_per_group
        if batch_idx == 0 or change_items_throughout_batch:
            ai_items_flat = generate_items(num_ai_items, item_size)
            ai_items = ai_items_flat.reshape(num_groups, num_items_per_group, item_size)

        # === Generate TI training trials (all adjacent pairs with both orderings, then fill remaining) ===
        ti_train_trials = []
        ti_train_correct = []
        ti_train_indices = []

        # Minimum training trials = all adjacent pairs with both orderings
        min_ti_train_trials = 2 * (num_items_ti - 1)
        actual_num_train_trials_ti = max(num_train_trials_ti, min_ti_train_trials)

        # First, generate all adjacent pairs with both orderings
        for high_idx in range(num_items_ti - 1):
            low_idx = high_idx + 1
            item_high = ti_items[high_idx]
            item_low = ti_items[low_idx]

            # Ordering 1: high item on left -> choice = 0
            item_pair_1 = np.concatenate([item_high, item_low], axis=0)
            ti_train_trials.append(item_pair_1)
            ti_train_correct.append(0)
            ti_train_indices.append([high_idx, low_idx])

            # Ordering 2: high item on right -> choice = 1
            item_pair_2 = np.concatenate([item_low, item_high], axis=0)
            ti_train_trials.append(item_pair_2)
            ti_train_correct.append(1)
            ti_train_indices.append([high_idx, low_idx])

        # Fill remaining training slots with random adjacent pairs
        num_remaining = actual_num_train_trials_ti - min_ti_train_trials
        for _ in range(num_remaining):
            item_pair, choice, pair_idx = generate_trial(ti_items, is_test=False)
            ti_train_trials.append(item_pair)
            ti_train_correct.append(choice)
            ti_train_indices.append(pair_idx)

        # Shuffle TI training trials
        ti_train_order = list(range(len(ti_train_trials)))
        np.random.shuffle(ti_train_order)
        ti_train_trials = [ti_train_trials[i] for i in ti_train_order]
        ti_train_correct = [ti_train_correct[i] for i in ti_train_order]
        ti_train_indices = [ti_train_indices[i] for i in ti_train_order]

        # === Generate AI training trials ===
        ai_training_info = []
        # Adjacent index pairs for all group combinations
        for idx in range(num_items_per_group - 1):
            idx1, idx2 = idx, idx + 1
            for g1 in range(num_groups):
                for g2 in range(num_groups):
                    ai_training_info.append((g1, idx1, g2, idx2))
                    ai_training_info.append((g2, idx2, g1, idx1))

        # Same index pairs for all group combinations
        for idx in range(num_items_per_group):
            for g1 in range(num_groups):
                for g2 in range(num_groups):
                    if ai_exclude_same_item and g1 == g2:
                        continue
                    ai_training_info.append((g1, idx, g2, idx))

        np.random.shuffle(ai_training_info)

        ai_train_trials = []
        ai_train_correct = []
        ai_train_indices = []
        for g1, idx1, g2, idx2 in ai_training_info:
            item1 = ai_items[g1, idx1]
            item2 = ai_items[g2, idx2]
            correct = 1.0 if g1 == g2 else 0.0
            item_pair = np.concatenate([item1, item2], axis=0)
            ai_train_trials.append(item_pair)
            ai_train_correct.append(correct)
            ai_train_indices.append([[g1, idx1], [g2, idx2]])

        if num_ai_train is None:
            num_ai_train = len(ai_train_trials)

        # === Generate TI test trials (non-adjacent or arbitrary) ===
        ti_test_trials = []
        ti_test_correct = []
        ti_test_indices = []
        for _ in range(num_test_trials_ti):
            if arbitrary_ti:
                item_pair, choice, pair_idx = generate_trial_arbitrary(ti_items)
            else:
                item_pair, choice, pair_idx = generate_trial(ti_items, is_test=True)
            ti_test_trials.append(item_pair)
            ti_test_correct.append(choice)
            ti_test_indices.append(pair_idx)

        # === Generate AI test trials ===
        ai_test_info = []
        for g1 in range(num_groups):
            for idx1 in range(num_items_per_group):
                for g2 in range(num_groups):
                    for idx2 in range(num_items_per_group):
                        if ai_exclude_same_item and g1 == g2 and idx1 == idx2:
                            continue
                        ai_test_info.append((g1, idx1, g2, idx2))

        # Apply nonadjacent ratio if enabled
        if ai_test_nonadj_ratio >= 0.0:
            adjacent_pool = []
            nonadjacent_pool = []
            for trial_info in ai_test_info:
                g1, idx1, g2, idx2 = trial_info
                is_adjacent = (g1 == g2) and (abs(idx1 - idx2) == 1)
                if is_adjacent:
                    adjacent_pool.append(trial_info)
                else:
                    nonadjacent_pool.append(trial_info)

            num_nonadj = int(round(ai_num_test_trials * ai_test_nonadj_ratio))
            num_adj = ai_num_test_trials - num_nonadj

            sampled_test_info = []
            if num_nonadj > 0 and len(nonadjacent_pool) > 0:
                nonadj_indices = np.random.choice(len(nonadjacent_pool), size=num_nonadj, replace=True)
                sampled_test_info.extend([nonadjacent_pool[i] for i in nonadj_indices])
            if num_adj > 0 and len(adjacent_pool) > 0:
                adj_indices = np.random.choice(len(adjacent_pool), size=num_adj, replace=True)
                sampled_test_info.extend([adjacent_pool[i] for i in adj_indices])
            np.random.shuffle(sampled_test_info)
            ai_test_info = sampled_test_info
        else:
            np.random.shuffle(ai_test_info)
            ai_test_info = ai_test_info[:ai_num_test_trials]

        ai_test_trials = []
        ai_test_correct = []
        ai_test_indices = []
        for g1, idx1, g2, idx2 in ai_test_info:
            item1 = ai_items[g1, idx1]
            item2 = ai_items[g2, idx2]
            correct = 1.0 if g1 == g2 else 0.0
            item_pair = np.concatenate([item1, item2], axis=0)
            ai_test_trials.append(item_pair)
            ai_test_correct.append(correct)
            ai_test_indices.append([[g1, idx1], [g2, idx2]])

        # === Interleave training trials ===
        num_ti_train = len(ti_train_trials)
        num_ai_train_this_batch = len(ai_train_trials)
        total_train = num_ti_train + num_ai_train_this_batch

        # Create combined training arrays with task labels
        train_indices = list(range(total_train))
        np.random.shuffle(train_indices)

        combined_train_trials = ti_train_trials + ai_train_trials
        combined_train_correct = ti_train_correct + ai_train_correct
        combined_train_task_labels = [0] * num_ti_train + [1] * num_ai_train_this_batch  # 0=TI, 1=AI
        combined_ti_indices = ti_train_indices + [[[-1, -1], [-1, -1]]] * num_ai_train_this_batch  # Placeholder for AI
        combined_ai_indices = [[[-1, -1], [-1, -1]]] * num_ti_train + ai_train_indices  # Placeholder for TI

        # Shuffle training
        shuffled_train_trials = [combined_train_trials[i] for i in train_indices]
        shuffled_train_correct = [combined_train_correct[i] for i in train_indices]
        shuffled_train_task = [combined_train_task_labels[i] for i in train_indices]
        shuffled_train_ti_idx = [combined_ti_indices[i] if combined_train_task_labels[i] == 0 else [-1, -1] for i in train_indices]
        shuffled_train_ai_idx = [combined_ai_indices[i] for i in train_indices]

        # === Interleave test trials ===
        num_ti_test = len(ti_test_trials)
        num_ai_test = len(ai_test_trials)
        total_test = num_ti_test + num_ai_test

        test_indices = list(range(total_test))
        np.random.shuffle(test_indices)

        combined_test_trials = ti_test_trials + ai_test_trials
        combined_test_correct = ti_test_correct + ai_test_correct
        combined_test_task_labels = [0] * num_ti_test + [1] * num_ai_test
        combined_test_ti_indices = ti_test_indices + [[[-1, -1], [-1, -1]]] * num_ai_test
        combined_test_ai_indices = [[[-1, -1], [-1, -1]]] * num_ti_test + ai_test_indices

        shuffled_test_trials = [combined_test_trials[i] for i in test_indices]
        shuffled_test_correct = [combined_test_correct[i] for i in test_indices]
        shuffled_test_task = [combined_test_task_labels[i] for i in test_indices]
        shuffled_test_ti_idx = [combined_test_ti_indices[i] if combined_test_task_labels[i] == 0 else [-1, -1] for i in test_indices]
        shuffled_test_ai_idx = [combined_test_ai_indices[i] for i in test_indices]

        # Combine train and test
        batch_trials = shuffled_train_trials + shuffled_test_trials
        batch_correct = shuffled_train_correct + shuffled_test_correct
        batch_task_labels = shuffled_train_task + shuffled_test_task
        batch_ti_indices = shuffled_train_ti_idx + shuffled_test_ti_idx
        batch_ai_indices = shuffled_train_ai_idx + shuffled_test_ai_idx

        all_trials.append(batch_trials)
        all_correct_choices.append(batch_correct)
        all_task_labels.append(batch_task_labels)
        all_ti_pair_indices.append(batch_ti_indices)
        all_ai_pair_indices.append(batch_ai_indices)

    # Use actual counts (num_ti_train comes from len(ti_train_trials) which includes auto-adjusted count)
    num_train_trials = num_ti_train + num_ai_train
    num_test_trials = num_test_trials_ti + len(ai_test_trials)

    return (
        np.array(all_trials),
        np.array(all_correct_choices),
        np.array(all_task_labels),  # 0 for TI, 1 for AI
        np.array(all_ti_pair_indices),  # TI pair indices (or [-1,-1] for AI trials)
        np.array(all_ai_pair_indices),  # AI pair indices (or [[-1,-1],[-1,-1]] for TI trials)
        num_train_trials,
        num_test_trials,
        num_ti_train,  # Actual TI training trials (may be auto-adjusted)
        num_ai_train,
    )


def generate_grouped_ti_ai_batch(
    num_items_ti, item_size, batch_size,
    num_train_trials_ti, num_test_trials_ti,
    num_groups, num_items_per_group,
    ai_num_test_trials,
    change_items_throughout_batch=False,
    arbitrary_ti=False,
    ai_test_nonadj_ratio=-1.0,
    ai_exclude_same_item=False
):
    """
    Generate a grouped batch where TI and AI trials are contiguous blocks (not interleaved).

    Train phase: [TI block, AI block] or [AI block, TI block] (random order)
    Test phase: [TI block, AI block] or [AI block, TI block] (random order, independent of train)

    Same signature and return types as generate_interleaved_ti_ai_batch.
    """
    all_trials = []
    all_correct_choices = []
    all_task_labels = []
    all_ti_pair_indices = []
    all_ai_pair_indices = []

    num_ai_train = None

    for batch_idx in range(batch_size):
        # Generate TI items
        if batch_idx == 0 or change_items_throughout_batch:
            ti_items = generate_items(num_items_ti, item_size)

        # Generate AI items (separate from TI items)
        num_ai_items = num_groups * num_items_per_group
        if batch_idx == 0 or change_items_throughout_batch:
            ai_items_flat = generate_items(num_ai_items, item_size)
            ai_items = ai_items_flat.reshape(num_groups, num_items_per_group, item_size)

        # === Generate TI training trials (all adjacent pairs with both orderings, then fill remaining) ===
        ti_train_trials = []
        ti_train_correct = []
        ti_train_indices = []

        min_ti_train_trials = 2 * (num_items_ti - 1)
        actual_num_train_trials_ti = max(num_train_trials_ti, min_ti_train_trials)

        for high_idx in range(num_items_ti - 1):
            low_idx = high_idx + 1
            item_high = ti_items[high_idx]
            item_low = ti_items[low_idx]

            item_pair_1 = np.concatenate([item_high, item_low], axis=0)
            ti_train_trials.append(item_pair_1)
            ti_train_correct.append(0)
            ti_train_indices.append([high_idx, low_idx])

            item_pair_2 = np.concatenate([item_low, item_high], axis=0)
            ti_train_trials.append(item_pair_2)
            ti_train_correct.append(1)
            ti_train_indices.append([high_idx, low_idx])

        num_remaining = actual_num_train_trials_ti - min_ti_train_trials
        for _ in range(num_remaining):
            item_pair, choice, pair_idx = generate_trial(ti_items, is_test=False)
            ti_train_trials.append(item_pair)
            ti_train_correct.append(choice)
            ti_train_indices.append(pair_idx)

        # Shuffle TI training trials within block
        ti_train_order = list(range(len(ti_train_trials)))
        np.random.shuffle(ti_train_order)
        ti_train_trials = [ti_train_trials[i] for i in ti_train_order]
        ti_train_correct = [ti_train_correct[i] for i in ti_train_order]
        ti_train_indices = [ti_train_indices[i] for i in ti_train_order]

        # === Generate AI training trials ===
        ai_training_info = []
        for idx in range(num_items_per_group - 1):
            idx1, idx2 = idx, idx + 1
            for g1 in range(num_groups):
                for g2 in range(num_groups):
                    ai_training_info.append((g1, idx1, g2, idx2))
                    ai_training_info.append((g2, idx2, g1, idx1))

        for idx in range(num_items_per_group):
            for g1 in range(num_groups):
                for g2 in range(num_groups):
                    if ai_exclude_same_item and g1 == g2:
                        continue
                    ai_training_info.append((g1, idx, g2, idx))

        np.random.shuffle(ai_training_info)

        ai_train_trials = []
        ai_train_correct = []
        ai_train_indices = []
        for g1, idx1, g2, idx2 in ai_training_info:
            item1 = ai_items[g1, idx1]
            item2 = ai_items[g2, idx2]
            correct = 1.0 if g1 == g2 else 0.0
            item_pair = np.concatenate([item1, item2], axis=0)
            ai_train_trials.append(item_pair)
            ai_train_correct.append(correct)
            ai_train_indices.append([[g1, idx1], [g2, idx2]])

        if num_ai_train is None:
            num_ai_train = len(ai_train_trials)

        # === Generate TI test trials ===
        ti_test_trials = []
        ti_test_correct = []
        ti_test_indices = []
        for _ in range(num_test_trials_ti):
            if arbitrary_ti:
                item_pair, choice, pair_idx = generate_trial_arbitrary(ti_items)
            else:
                item_pair, choice, pair_idx = generate_trial(ti_items, is_test=True)
            ti_test_trials.append(item_pair)
            ti_test_correct.append(choice)
            ti_test_indices.append(pair_idx)

        # === Generate AI test trials ===
        ai_test_info = []
        for g1 in range(num_groups):
            for idx1 in range(num_items_per_group):
                for g2 in range(num_groups):
                    for idx2 in range(num_items_per_group):
                        if ai_exclude_same_item and g1 == g2 and idx1 == idx2:
                            continue
                        ai_test_info.append((g1, idx1, g2, idx2))

        if ai_test_nonadj_ratio >= 0.0:
            adjacent_pool = []
            nonadjacent_pool = []
            for trial_info in ai_test_info:
                g1, idx1, g2, idx2 = trial_info
                is_adjacent = (g1 == g2) and (abs(idx1 - idx2) == 1)
                if is_adjacent:
                    adjacent_pool.append(trial_info)
                else:
                    nonadjacent_pool.append(trial_info)

            num_nonadj = int(round(ai_num_test_trials * ai_test_nonadj_ratio))
            num_adj = ai_num_test_trials - num_nonadj

            sampled_test_info = []
            if num_nonadj > 0 and len(nonadjacent_pool) > 0:
                nonadj_indices = np.random.choice(len(nonadjacent_pool), size=num_nonadj, replace=True)
                sampled_test_info.extend([nonadjacent_pool[i] for i in nonadj_indices])
            if num_adj > 0 and len(adjacent_pool) > 0:
                adj_indices = np.random.choice(len(adjacent_pool), size=num_adj, replace=True)
                sampled_test_info.extend([adjacent_pool[i] for i in adj_indices])
            np.random.shuffle(sampled_test_info)
            ai_test_info = sampled_test_info
        else:
            np.random.shuffle(ai_test_info)
            ai_test_info = ai_test_info[:ai_num_test_trials]

        ai_test_trials = []
        ai_test_correct = []
        ai_test_indices = []
        for g1, idx1, g2, idx2 in ai_test_info:
            item1 = ai_items[g1, idx1]
            item2 = ai_items[g2, idx2]
            correct = 1.0 if g1 == g2 else 0.0
            item_pair = np.concatenate([item1, item2], axis=0)
            ai_test_trials.append(item_pair)
            ai_test_correct.append(correct)
            ai_test_indices.append([[g1, idx1], [g2, idx2]])

        # === Group training trials as contiguous blocks (random order) ===
        num_ti_train = len(ti_train_trials)
        num_ai_train_this_batch = len(ai_train_trials)

        train_ti_first = np.random.random() < 0.5  # Random train block order

        if train_ti_first:
            grouped_train_trials = ti_train_trials + ai_train_trials
            grouped_train_correct = ti_train_correct + ai_train_correct
            grouped_train_task_labels = [0] * num_ti_train + [1] * num_ai_train_this_batch
            grouped_train_ti_idx = ti_train_indices + [[-1, -1]] * num_ai_train_this_batch
            grouped_train_ai_idx = [[[-1, -1], [-1, -1]]] * num_ti_train + ai_train_indices
        else:
            grouped_train_trials = ai_train_trials + ti_train_trials
            grouped_train_correct = ai_train_correct + ti_train_correct
            grouped_train_task_labels = [1] * num_ai_train_this_batch + [0] * num_ti_train
            grouped_train_ti_idx = [[-1, -1]] * num_ai_train_this_batch + ti_train_indices
            grouped_train_ai_idx = ai_train_indices + [[[-1, -1], [-1, -1]]] * num_ti_train

        # === Group test trials as contiguous blocks (random order, independent of train) ===
        num_ti_test = len(ti_test_trials)
        num_ai_test = len(ai_test_trials)

        test_ti_first = np.random.random() < 0.5  # Random test block order (independent)

        if test_ti_first:
            grouped_test_trials = ti_test_trials + ai_test_trials
            grouped_test_correct = ti_test_correct + ai_test_correct
            grouped_test_task_labels = [0] * num_ti_test + [1] * num_ai_test
            grouped_test_ti_idx = ti_test_indices + [[-1, -1]] * num_ai_test
            grouped_test_ai_idx = [[[-1, -1], [-1, -1]]] * num_ti_test + ai_test_indices
        else:
            grouped_test_trials = ai_test_trials + ti_test_trials
            grouped_test_correct = ai_test_correct + ti_test_correct
            grouped_test_task_labels = [1] * num_ai_test + [0] * num_ti_test
            grouped_test_ti_idx = [[-1, -1]] * num_ai_test + ti_test_indices
            grouped_test_ai_idx = ai_test_indices + [[[-1, -1], [-1, -1]]] * num_ti_test

        # Combine train and test
        batch_trials = grouped_train_trials + grouped_test_trials
        batch_correct = grouped_train_correct + grouped_test_correct
        batch_task_labels = grouped_train_task_labels + grouped_test_task_labels
        batch_ti_indices = grouped_train_ti_idx + grouped_test_ti_idx
        batch_ai_indices = grouped_train_ai_idx + grouped_test_ai_idx

        all_trials.append(batch_trials)
        all_correct_choices.append(batch_correct)
        all_task_labels.append(batch_task_labels)
        all_ti_pair_indices.append(batch_ti_indices)
        all_ai_pair_indices.append(batch_ai_indices)

    num_train_trials = num_ti_train + num_ai_train
    num_test_trials = num_test_trials_ti + len(ai_test_trials)

    return (
        np.array(all_trials),
        np.array(all_correct_choices),
        np.array(all_task_labels),
        np.array(all_ti_pair_indices),
        np.array(all_ai_pair_indices),
        num_train_trials,
        num_test_trials,
        num_ti_train,
        num_ai_train,
    )

