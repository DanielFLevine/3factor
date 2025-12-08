"""
Configuration parameters for the MetaTI reinforcement learning model.
Based on Miconi et al. ICLR 2019 - Stimulus-response task.
"""

# Default parameters dictionary
params = {
    # Random seed
    'rngseed': -1,  # RNG seed, or -1 for no seed
    
    # Reward/penalty parameters
    'rew': 1.0,  # reward amount
    'wp': 0.0,  # penalty for taking action 1 (not used here)
    
    # Loss coefficients
    'bent': 0.1,  # entropy incentive (actually sum-of-squares)
    'blossv': 0.1,  # value prediction loss coefficient
    'gr': 0.9,  # Gamma for temporal reward discounting
    'testlmult': 3.0,  # multiplier for the loss during the test trials
    
    # Network architecture
    'batch_size': 32,  # Batch size
    'input_plastic': False,  # If True, use plastic weights for input connections
    'fixed_random_plastic_init': False,  # If True, plastic weights reset to same random values each episode
    'reset_random_recurrent_init': False,  # If True, reset the recurrent weights to random values each episode
    
    # Training parameters
    'lr': 1e-4,  # learning rate
    'gc': 2.0,  # Gradient clipping
    'eps': 1e-6,  # A parameter for Adam
    'number_of_training_iterations': 100100,  # Number of training iterations
    'save_every': 1000,  # Save model every N iterations
    'pe': 100,  # "print every" - logging frequency
    
    # Regularization
    'l2': 0,  # L2 penalty
    'lpw': 1e-4,  # plastic weight loss
    'lda': 0,  # DA output penalty
    'lhl1': 0,  # Hidden layer L1 penalty
    
    # Task structure
    'number_of_cues_range': range(4, 9),  # Range of number of cues to randomly sample from
    'number_of_cues': None,  # Actual number of cues used in the current episode (set in main.py)
    'cue_size': 15,  # Cue size - number of binary elements in each cue vector
    'triallen': 4,  # Each trial has: stimulus presentation, 'go' cue, then 2 empty trials
    'nbtraintrials': 20,  # Number of training trials per episode
    'nbtesttrials': 10,  # Number of test trials per episode
    'number_of_episodes_between_resets': 3,  # Number of episodes between resets
    
    # Note: The following are computed from the above and set in main.py:
    # 'nbtrials': computed as nbtraintrials + NBMASSEDTRIALS + nbtesttrials
    # 'eplen': computed as nbtrials * triallen (episode length)
    # 'outputsize': set to 2 (response and no response)
    # 'inputsize': computed based on NBSTIMBITS + ADDINPUT + outputsize
}


def get_params_with_seed(seed=-1):
    """
    Get a copy of params with a specific random seed.
    
    Args:
        seed (int): Random seed to use, or -1 for no seed
        
    Returns:
        dict: Copy of params with updated seed
    """
    params_copy = params.copy()
    params_copy['rngseed'] = seed
    return params_copy

