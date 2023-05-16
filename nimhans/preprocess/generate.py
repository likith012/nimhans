import glob, os, warnings
from typing import List, Union

from tqdm import tqdm
import mne
import numpy as np
from joblib import Parallel, delayed, cpu_count, parallel_backend

from .datautil import StagingPreprocess
from ..utils import CHANNEL_MAPPING, LABEL_MAPPING

warnings.simplefilter(action='ignore', category=FutureWarning) 
warnings.simplefilter(action='ignore', category=UserWarning)
mne.set_log_level(verbose='WARNING')
    

def _get_channels(raw_path, ann_path, window_size, sfreq, preprocessed_path, modality, label_mapping):
    save_data = dict()
    ds = StagingPreprocess(raw_path, ann_path, CHANNEL_MAPPING, modality, window_size, sfreq, preload=True)
    # epochs_data = StagingPreprocess.create_windows(ds.raw, ds.description, window_size=window_size, 
    #                                           window_stride=window_size, label_mapping=label_mapping, drop_last=True,
    #                                           drop_bad=True)

    # func = lambda x: x * 1e6
    # epochs_data.apply_function(func)
    
    # cnt = 1
    # sub_id = int(epochs_data.metadata.subject_id[0])
    
    # for ep_idx in range(len(epochs_data)):
    #     for ch, pick in zip(epochs_data.ch_names, epochs_data.picks):
    #             data = epochs_data[ep_idx].get_data()
    #             save_data[ch] = data[:, pick, :] # Shape of (1, 3000)
    #     save_data['target'] = int(epochs_data[ep_idx].metadata.target)
    #     temp_save_path = os.path.join(preprocessed_path, f"{sub_id}_{cnt}.npz")
    #     np.savez(temp_save_path, **save_data)
    #     cnt += 1
        
            
def _preprocess_dataset(raw_paths, ann_paths, k, N, preprocessed_path, window_size, sfreq, modality, label_mapping):
    raw_paths_core = [f for i, f in enumerate(raw_paths) if i%N==k]
    ann_paths_core = [f for i, f in enumerate(ann_paths) if i%N==k]
    
    for raw_path, ann_path in tqdm(zip(raw_paths_core, ann_paths_core), desc="Dataset preprocessing...", total=len(raw_paths_core)):
        _get_channels(raw_path, ann_path, window_size, sfreq, preprocessed_path, modality, label_mapping)  
    
    
def generate(modality: Union[List[str], None] = None, data_path: Union[str, None] = None, preprocessed_path: Union[str, None] = None, n_jobs: int = -1, label_mapping: Union[dict[str, int], None] = None):
    
    assert isinstance(data_path, (str, type(None))), f"{data_path} should be of type str"
    assert modality is None or (isinstance(modality, list) and all(isinstance(item, str) for item in modality)), "Provided modality is not a list of str"
    
    if data_path is None:
        data_path = os.path.expanduser("~/.nimhans/edf_data")
        
    if preprocessed_path is None:
        preprocessed_path = os.path.expanduser("~/.nimhans/preprocessed")
    
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)

    if not os.path.exists(preprocessed_path):
        os.makedirs(preprocessed_path, exist_ok=True)

    if modality is None:
        modality = ['eeg']
        
    if label_mapping is None:
        label_mapping = LABEL_MAPPING
        
    if n_jobs == -1:
        n_jobs = cpu_count()

    edf_files = glob.glob(f'{data_path}/*.edf')
    ann_files = glob.glob(f'{data_path}/*.xlsx')
    
    window_size = 30.
    sfreq = 100
    
    with parallel_backend(backend='loky'):     
        Parallel(n_jobs=n_jobs, verbose=10)([delayed(_preprocess_dataset)(edf_files, ann_files, k, n_jobs, preprocessed_path, window_size, sfreq, modality, label_mapping) for k in range(n_jobs)]) 
        