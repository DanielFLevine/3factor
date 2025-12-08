"""
Constants for the MetaTI reinforcement learning model.
These values typically do not change during experiments.
"""

# Device configuration (computed lazily to avoid import errors)
def get_device():
    """Get the appropriate device for PyTorch computations."""
    import torch
    return 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

# Set DEVICE at module level when torch is available
try:
    import torch
    DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
except ImportError:
    DEVICE = 'cpu'  # Fallback if torch is not installed

# Input/output structure constants
ADDINPUT = 4  # 1 input for previous reward, 1 for numstep, 1 unused, 1 "Bias" input
NUMRESPONSESTEP = 1  # Which step in a trial is the response step

# Data probability
PROBAOLDDATA = 0.25  # Probability of using old cue data in test trials

# Alpha (plasticity coefficient) configuration
POSALPHA = False  # Whether to clip alpha to positive values only

SECONDMODULATOR = False # Whether to use a second neuromodulator

# Experimental mode flags
EVAL = False  # Are we running in evaluation mode?

# Trial reset flags
RESETHIDDENEVERYTRIAL = True  # Reset hidden state at the start of each trial
RESETETEVERYTRIAL = True  # Reset eligibility trace at the start of each trial
RESETPWEVERYTRIAL = False  # Reset plastic weights at the start of each trial

# Advanced experimental configurations
ONLYTWOLASTADJ = False  # Special constraint on test trials (requires nbcues == 7)
LINKEDLISTSEVAL = False  # Special linked-list evaluation mode
LINKINGISSHAM = False  # Sham linking for control experiments (requires LINKEDLISTSEVAL)
FIXEDCUES = False  # Use fixed cues across all episodes (debugging only)
HALFNOBARREDPAIRUNTILT18 = False  # Special barred pair constraint (eval only)
BARREDPAIR = [3, 4]  # The "barred" pair for special experiments
MIXNETWORKS = False  # Mix components from different saved networks (eval only)
FREEZE_INPUT_WEIGHTS = False  # Freeze the input weights
FREEZE_OUTPUT_WEIGHTS = False  # Freeze the output weights
TURN_OFF_RECURRENT_WEIGHTS = False  # Turn off the recurrent weights
ZERO_RECURRENT_WEIGHTS = False  # Zero the recurrent weights
FREEZE_VALUE_WEIGHTS = False  # Turn off the value weights
FREEZE_NEUROMODULATOR_WEIGHTS = False  # Turn off the neuromodulator weights

# Evaluation-specific settings
# These settings modify params when EVAL is True:
# - nbiter becomes 1
# - bs becomes 2000
# - nbcues becomes 8
# - nbepsbwresets becomes 1 (unless LINKEDLISTSEVAL)
# - gradient computation is disabled


def get_nbstimbits(cue_size):
    """
    Calculate the number of stimulus bits.
    
    Args:
        cue_size (int): Size of each cue vector
        
    Returns:
        int: Total number of stimulus bits including response cue
    """
    return 2 * cue_size + 1  # The additional bit is for the response cue (i.e. the "Go" cue)


def get_input_size(cue_size, output_size):
    """
    Calculate the total input size.
    
    Args:
        cue_size (int): Size of each cue vector
        output_size (int): Number of output actions
        
    Returns:
        int: Total input size
    """
    nbstimbits = get_nbstimbits(cue_size)
    return nbstimbits + ADDINPUT + output_size

