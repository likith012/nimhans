import glob, os, warnings
from tqdm import tqdm
from dataclasses import dataclass
from typing import List

import mne
import numpy as np
from joblib import Parallel, delayed, cpu_count, parallel_backend

from .datautil import SleepStaging
from ..utils import CHANNEL_MAPPING, LABEL_MAPPING

warnings.simplefilter(action='ignore', category=FutureWarning) 
warnings.simplefilter(action='ignore', category=UserWarning)
mne.set_log_level(verbose='WARNING')


# @dataclass
# class channel_mapping:
#     eeg: List = ['C4:M1', # EEG with M1, M2 as reference electrodes
#                 'C3:M2',
#                 'O2:M1',
#                 'O1:M2']
#     ecg: List = ['ECG 2'] # ECG
#     eog: List =  ['E1:M2', # EOG with M2 as reference electrodes
#                   'E2:M2']
#     emg: List = ['EMG'] # EMG
#     spo2: List = ['SPO2'] # Pulse oximeter, measures oxygen saturation levels
#     limb: List = ['PLMr'] # Periodic limb movements
#     snore: List = ['Snore', # Snore signals
#                    'Pressure Snore'] # Percentage of snore signals
#     bpm: List = ['Pulse'] # Beats per minute
#     pg: List = ['Pleth'] # Plethosmography
#     position: List = ['Pos.'] # Sleep position of the person (4 types)
#     light: List = ['Light'] # Light levels, 0 during sleep
#     resp: List = ['Sum Effort', # Respiratory signals
#                   'Abdomen']
#     nasal: List = ['Pressure Flow'] # Nasal pressure flow
#     temp: List = ['Flow Th'] # Flow thremister: Thermistors utilizing change in temperature of exhaled air to assess the airflow 
#                              # have been traditionally used to detect apneas and hypopneas during PSG recording
#     thorax: List = ['Thorax']
    


def _get_channels(raw_path, ann_path, window_size_samples, SAVE_PATH, MODALITY):
    save_data = dict()
    ds = SleepStaging(raw_path=raw_path, ann_path=ann_path, channels=CHANNEL_MAPPING, modality=MODALITY, preload=True)
    epochs_data = SleepStaging.create_windows(ds.raw, window_size_samples=window_size_samples, 
                                              window_stride_samples=window_size_samples, mapping=LABEL_MAPPING, drop_last=True,
                                              drop_bad=True, description=ds.description)

    func = lambda x: x * 1e6
    epochs_data.apply_function(func)
    # preprocess(windows_subject_dataset, [Preprocessor(zscore)])
    
    cnt = 1
    sub_id = int(epochs_data.metadata.subject_id[0])
    
    for ep_idx in range(len(epochs_data)):
        for ch, pick in zip(epochs_data.ch_names, epochs_data.picks):
                data = epochs_data[ep_idx].get_data()
                save_data[ch] = data[:, pick, :] # Shape of (1, 3000)
        save_data['target'] = int(epochs_data[ep_idx].metadata.target)
        temp_save_path = os.path.join(SAVE_PATH, f"{sub_id}_{cnt}.npz")
        np.savez(temp_save_path, **save_data)
        cnt += 1
        
            
def _preprocess_dataset(raw_paths, ann_paths, k, N, DATA_SAVE_PATH, windows_size_samples, MODALITY):
    raw_paths_core = [f for i, f in enumerate(raw_paths) if i%N==k]
    ann_paths_core = [f for i, f in enumerate(ann_paths) if i%N==k]
    
    for raw_path, ann_path in tqdm(zip(raw_paths_core, ann_paths_core), desc="Dataset preprocessing...", total=len(raw_paths_core)):
        _get_channels(raw_path, ann_path, windows_size_samples, DATA_SAVE_PATH, MODALITY)  
    
    
def generate(data_path=None, modality=None, n_jobs=-1):
    assert isinstance(data_path, (str, type(None))), f"{data_path} should be of type str"
    assert modality is None or (isinstance(modality, list) and all(isinstance(item, str) for item in modality)), "modality is not a list of strings and/or None"
    
    if data_path is None:
        data_path = os.path.expanduser("~/.nimhans/edf_data")
    
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)

    if modality is None:
        modality = ['eeg']

    edf_files = glob.glob(f'{data_path}/*.edf')
    ann_files = glob.glob(f'{data_path}/*.xlsx')
    data_save_path = os.path.expanduser("~/.nimhans/subjects_data")
    
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path, exist_ok=True)

    window_size = 30
    sfreq = 100
    window_size_samples = window_size*sfreq
    
    with parallel_backend(backend='loky'):     
        Parallel(n_jobs=n_jobs, verbose=10)([delayed(_preprocess_dataset)(edf_files, ann_files, k, n_jobs, data_save_path, window_size_samples, modality) for k in range(n_jobs)]) 
        