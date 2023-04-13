import random

import numpy as np
import torch
from torch.utils.data import Dataset


class sleepStagingDataset(Dataset):
    """Dataset for train and test"""

    def __init__(self, filepaths, channels):
        super(sleepStagingDataset, self).__init__()
        self.filepaths = filepaths
        self.channels = channels

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index):
        data = np.load(self.filepaths[index])
        ch_data = [data[ch] for ch in self.channels]
        ch_data = np.concatenate(ch_data, axis=0)
        target = data['target']
        return ch_data, target
    
    
def set_random_seeds(seed, cudnn_benchmark=None, cuda_deterministic=None):
    """Set seeds for python random module numpy.random and torch.

    For more details about reproducibility in pytorch see
    https://pytorch.org/docs/stable/notes/randomness.html

    Parameters
    ----------
    seed: int
        Random seed.
    cuda: bool
        Whether to set cuda seed with torch.
    cudnn_benchmark: bool (default=None)
        Whether pytorch will use cudnn benchmark. When set to `None` it will not modify
        torch.backends.cudnn.benchmark (displays warning in the case of possible lack of
        reproducibility). When set to True, results may not be reproducible (no warning displayed).
        When set to False it may slow down computations.

    Notes
    -----
    In some cases setting environment variable `PYTHONHASHSEED` may be needed before running a
    script to ensure full reproducibility. See
    https://forums.fast.ai/t/solved-reproducibility-where-is-the-randomness-coming-in/31628/14

    Using this function may not ensure full reproducibility of the results as we do not set
    `torch.use_deterministic_algorithms(True)`.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if torch.cuda.is_available():
        if isinstance(cudnn_benchmark, bool):
            torch.backends.cudnn.benchmark = cudnn_benchmark
        elif cudnn_benchmark is None:
            if torch.backends.cudnn.benchmark:
                warn(
                    "torch.backends.cudnn.benchmark was set to True which may results in lack of "
                    "reproducibility. In some cases to ensure reproducibility you may need to "
                    "set torch.backends.cudnn.benchmark to False.", UserWarning)
        else:
            raise ValueError(
                f"cudnn_benchmark expected to be bool or None, got '{cudnn_benchmark}'"
            )
        
        if isinstance(cuda_deterministic, bool):
            torch.backends.cudnn.deterministic = cuda_deterministic
        elif cuda_deterministic is None:
            torch.backends.cudnn.deterministic = False
        else:
            raise ValueError(
                f"cuda_deterministic expected to be bool or None, got '{cuda_deterministic}'"
            )
              
        torch.cuda.manual_seed_all(seed)