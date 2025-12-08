import random
import numpy as np
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and hasattr(torch, "mps"):
        try:
            state["torch_mps"] = torch.mps.get_rng_state()
        except AttributeError:
            logger.debug("torch.mps.get_rng_state unavailable; skipping MPS RNG state")
    return state


def restore_rng_state(state):
    if not state:
        return
    python_state = state.get("python")
    if python_state is not None:
        random.setstate(python_state)
    numpy_state = state.get("numpy")
    if numpy_state is not None:
        np.random.set_state(numpy_state)
    torch_state = state.get("torch")
    if torch_state is not None:
        torch.set_rng_state(torch_state)
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)
    mps_state = state.get("torch_mps")
    if mps_state is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and hasattr(torch, "mps"):
        try:
            torch.mps.set_rng_state(mps_state)
        except AttributeError:
            logger.debug("torch.mps.set_rng_state unavailable; skipping MPS RNG restore")