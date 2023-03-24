import glob, os, warnings
from tqdm import tqdm
import multiprocessing

import mne
from braindecode.preprocessing.preprocess import preprocess, Preprocessor, zscore
from braindecode.preprocessing.windowers import create_windows_from_events

from .utils import SleepStaging


warnings.simplefilter(action='ignore', category=FutureWarning) 
warnings.simplefilter(action='ignore', category=UserWarning)
mne.set_log_level(verbose='WARNING')


DATA_PATH = r'C:\Users\likit\OneDrive\Desktop\edf_data'
EDF_FILES = glob.glob(f'{DATA_PATH}/*.edf')
ANN_FILES = glob.glob(f'{DATA_PATH}/*.xlsx')
DATA_SAVE_PATH = os.path.join(os.getcwd(), '.nimhans/subjects_data')
NUM_CORES = multiprocessing.cpu_count()

if not os.path.exists(DATA_SAVE_PATH):
    os.makedirs(DATA_SAVE_PATH, exist_ok=True)


window_size = 30
sfreq = 100
window_size_samples = window_size*sfreq


label_mapping = {  
    "Wake": 0,
    "N1": 1,
    "N2": 2,
    "N3": 3,
    "REM": 4,
    "A": 5
}
channel_mapping = {
    'eeg': ['E1:M2',
            'E2:M2',
            'C4:M1',
            'C3:M2',
            'O2:M1',
            'O1:M2',],
    'ecg': ['ECG 2'],
    'eog': ['EOG(L)', 'EOG(R)'],
    'emg': ['EMG'],
}

    
def _get_epochs(windows_subject):
    epochs_data = []
    for epoch in windows_subject.windows:
        epochs_data.append(epoch)
    epochs_data = np.stack(epochs_data, axis=0) # Shape of (num_epochs, num_channels, num_sample_points)
    return epochs_data

def _get_channels(raw, ann):
    channels_data = dict()
    for ch in channel_mapping.keys():
        subject_dataset = SleepStaging(raw_path=raw, ann_path=ann, channels=channel_mapping[ch], preload=True)
        preprocess(subject_dataset, [Preprocessor(lambda x: x * 1e6)])

        windows_subject_dataset = create_windows_from_events(
                                subject_dataset,
                                window_size_samples=window_size_samples,
                                window_stride_samples=window_size_samples,
                                preload=True,
                                mapping=label_mapping,
                            )
        preprocess(windows_subject_dataset, [Preprocessor(zscore)])

        channels_data[ch] = _get_epochs(*windows_subject_dataset.datasets)
    channels_data['y'] = windows_subject_dataset.datasets[0].y
    channels_data['subject_id'] = windows_subject_dataset.datasets[0].description['subject_id']
    channels_data['epoch_length'] = len(windows_subject_dataset.datasets[0])
    return channels_data, windows_subject_dataset.datasets[0].description['subject_id']


def preprocess_dataset(raw_paths, ann_paths, k, N):
    raw_paths_core = [f for i, f in enumerate(raw_paths) if i%N==k]
    ann_paths_core = [f for i, f in enumerate(ann_paths) if i%N==k]
    for raw, ann in tqdm(zip(raw_paths_core, ann_paths_core), desc="SHHS dataset preprocessing ...", total=len(raw_paths_core)):
        channels_data, subject_num = _get_channels(raw, ann)    
        subjects_save_path = os.path.join(DATA_SAVE_PATH, f"{subject_num}.npz")
        np.savez(subjects_save_path, **channels_data)

p_list = []
for k in range(NUM_CORES):
    process = multiprocessing.Process(target=preprocess_dataset, args=(EDF_FILES, ANN_FILES, k, NUM_CORES))
    process.start()
    p_list.append(process)
for i in p_list:
    i.join()
    