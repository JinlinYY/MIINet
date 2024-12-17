import numpy as np
import random
import torch


def set_seed(seed):
    random.seed(seed)  # Python random seed
    np.random.seed(seed)  # NumPy random seed
    torch.manual_seed(seed)  # PyTorch random seed

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # CUDA seed for current GPU
        torch.cuda.manual_seed_all(seed)  # CUDA seed for all GPUs
        torch.backends.cudnn.deterministic = True  # Ensure deterministic CuDNN operations
        torch.backends.cudnn.benchmark = False  # Disable CuDNN benchmark mode for reproducibility