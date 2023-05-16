import random
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset


class StagingDataset(Dataset):
    """Dataset for train and test"""

    def __init__(self, filepaths, channels, desc, lazy_load=False):
        super(StagingDataset, self).__init__()
        self.filepaths = filepaths
        self.channels = channels
        self.lazy_load = lazy_load
        
        if not self.lazy_load:
            self.lazy_data = []
            self.lazy_targets = []
            
            for f in tqdm(self.filepaths, desc=f'Loading {desc} data'):
                lazy_temp = np.load(f)
                ch_temp = [lazy_temp[ch] for ch in self.channels]
                ch_temp = np.concatenate(ch_temp, axis=0)
                self.lazy_data.append(ch_temp)
                self.lazy_targets.append(lazy_temp['target'])
                
            self.lazy_data = np.stack(self.lazy_data, axis=0)
            self.lazy_targets = np.stack(self.lazy_targets, axis=0)

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, index):
        if not self.lazy_load:
            ch_data = self.lazy_data[index]
            target = self.lazy_targets[index]
        else:
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