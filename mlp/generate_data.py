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
    return item_pair, choice

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
    return item_pair, choice

def generate_batch_trials_ti(batch_items, num_train_trials, num_test_trials):
    batch_size = batch_items.shape[0]
    trials = []
    correct_choices = []
    for batch_index in range(batch_size):
        batch_trials = []
        batch_correct_choices = []
        for i in range(num_train_trials+num_test_trials):
            if i < num_train_trials:
                is_test = False
            else:
                is_test = True
            item_pair, choice = generate_trial(batch_items[batch_index], is_test)
            batch_trials.append(item_pair)
            batch_correct_choices.append(choice)
        trials.append(np.array(batch_trials))
        correct_choices.append(np.array(batch_correct_choices))
    return np.array(trials), np.array(correct_choices) # Return a 3D array of shape (batch_size, num_train_trials+num_test_trials, 2*item_size) and a 2D array of shape (batch_size, num_train_trials+num_test_trials)

def generate_batch_trials_ll(batch_items, num_trials_list_1, num_trials_list_2, num_trials_linking_pair, num_test_trials):
    batch_size = batch_items.shape[0]
    num_items = batch_items.shape[1]
    trials = []
    correct_choices = []
    batch_items_list_1 = batch_items[:, :num_items//2]
    batch_items_list_2 = batch_items[:, num_items//2:]
    batch_items_linking_pair = batch_items[:, num_items//2:(num_items//2)+2]
    for batch_index in range(batch_size):
        batch_trials = []
        batch_correct_choices = []
        is_test=False
        for i in range(num_trials_list_1):
            item_pair, choice = generate_trial(batch_items_list_1[batch_index], is_test)
            batch_trials.append(item_pair)
            batch_correct_choices.append(choice)
        for i in range(num_trials_list_2):
            item_pair, choice = generate_trial(batch_items_list_2[batch_index], is_test)
            batch_trials.append(item_pair)
            batch_correct_choices.append(choice)
        for i in range(num_trials_linking_pair):
            item_pair, choice = generate_trial(batch_items_linking_pair[batch_index], is_test)
            batch_trials.append(item_pair)
            batch_correct_choices.append(choice)
        for i in range(num_test_trials):
            item_pair, choice = generate_cross_list_trial(batch_items_list_1[batch_index], batch_items_list_2[batch_index])
            batch_trials.append(item_pair)
            batch_correct_choices.append(choice)
        trials.append(np.array(batch_trials))
        correct_choices.append(np.array(batch_correct_choices))
    return np.array(trials), np.array(correct_choices) # Return a 3D array of shape (batch_size, num_train_trials+num_test_trials, 2*item_size) and a 2D array of shape (batch_size, num_train_trials+num_test_trials)

def generate_pair(item_1, item_2):
    swap = np.random.randint(0, 2)
    choice = swap # 0 if item_1 is chosen, 1 if item_2 is chosen
    if swap:
        item_1, item_2 = item_2, item_1
    item_pair = np.concatenate([item_1, item_2], axis=0)
    return item_pair, choice