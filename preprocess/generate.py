import glob, os, warnings
from tqdm import tqdm
from dataclasses import dataclass
from typing import List

import mne
import numpy as np
from joblib import Parallel, delayed, cpu_count, parallel_backend
from braindecode.preprocessing.preprocess import preprocess, Preprocessor
from braindecode.preprocessing.windowers import create_windows_from_events

from .utils import SleepStaging


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
    
    
# channel_mapping = {
#     'eeg': ['C4:M1', # EEG with M1, M2 as reference electrodes
#             'C3:M2',
#             'O2:M1',
#             'O1:M2',],
#     'ecg': ['ECG 2'], # ECG
#     'eog': ['E1:M2', # EOG with M2 as reference electrodes
#             'E2:M2',],
#     'emg': ['EMG'], # EMG
#     'spo2': ['SPO2'], # Pulse oximeter, measures oxygen saturation levels
#     'limb': ['PLMr'], # Periodic limb movements
#     'snore': ['Snore', # Snore signals
#             'Pressure Snore',], # Percentage of snore signals
#     'bpm': ['Pulse'], # Beats per minute
#     'pg': ['Pleth'], # Plethosmography
#     'position': ['Pos.'], # Sleep position of the person (4 types)
#     'light': ['Light'], # Light levels, 0 during sleep
#     'resp': ['Sum Effort', # Respiratory signals
#              'Abdomen'],
#     'nasal': ['Pressure Flow'], # Nasal pressure flow
#     'temp': ['Flow Th'], # Flow thremister: Thermistors utilizing change in temperature of exhaled air to assess the airflow 
#                         # have been traditionally used to detect apneas and hypopneas during PSG recording
#     'thorax': ['Thorax'],
# }

CHANNEL_MAPPING = {'C4:M1': 'eeg',
                'C3:M2': 'eeg',
                'O2:M1': 'eeg',
                'O1:M2': 'eeg',
                'ECG 2': 'ecg',
                'E1:M2': 'eog',
                'E2:M2': 'eog',
                'EMG': 'emg',
                'SPO2': 'misc',
                'PLMr': 'misc',
                'Snore': 'misc',
                'Pressure Snore': 'misc',
                'Pulse': 'misc',
                'Pleth': 'misc',
                'Pos.': 'misc',
                'Light': 'misc',
                'Sum Effort': 'resp',
                'Abdomen': 'resp',
                'Pressure Flow': 'misc',
                'Flow Th': 'temperature',      
                'Thorax': 'misc',
                'Battery': 'misc',
    }
LABEL_MAPPING = {  
    "W": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "R": 4,
}


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
    
    
def generate(DATA_PATH, MODALITY):
    EDF_FILES = glob.glob(f'{DATA_PATH}/*.edf')
    ANN_FILES = glob.glob(f'{DATA_PATH}/*.xlsx')
    DATA_SAVE_PATH = os.path.join(os.getcwd(), '.nimhans/subjects_data')
    
    if not os.path.exists(DATA_SAVE_PATH):
        os.makedirs(DATA_SAVE_PATH, exist_ok=True)

    window_size = 30
    sfreq = 100
    window_size_samples = window_size*sfreq
    n_jobs = cpu_count()
    
    with parallel_backend(backend='loky'):     
        Parallel(n_jobs=n_jobs, verbose=10)([delayed(_preprocess_dataset)(EDF_FILES, ANN_FILES, k, n_jobs, DATA_SAVE_PATH, window_size_samples, MODALITY) for k in range(n_jobs)]) # type: ignore

    # _preprocess_dataset(EDF_FILES, ANN_FILES, 0, n_jobs, DATA_SAVE_PATH, window_size_samples, MODALITY)
