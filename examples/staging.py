import os
import glob
import inspect

import wandb
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from nimhans.utils import natural_keys, sleepStagingDataset, set_random_seeds, CHANNEL_MAPPING
from nimhans.models import SleepStagerChambon2018
from nimhans.trainer import classifierModel

set_random_seeds(42)


def map_subject_files(dataset_path):
    epoch_path = glob.glob(os.path.join(dataset_path, '*'))
    epoch_path.sort(key=natural_keys)

    subject_files = {}
    for path in epoch_path:
        subject_number = os.path.basename(path).split('_')[0]
        if subject_number not in subject_files:
            subject_files[subject_number] = []
        subject_files[subject_number].append(path)

    return subject_files

def _get_experiment_id():
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    basename = os.path.basename(filename)
    experiment_id = os.path.splitext(basename)[0]
    return experiment_id

def do_k_fold(experiment_id, subject_files, k, modality, weights_path, wandb, model_kwargs, n_jobs=-1, *args, **kwargs):
    subject_ids = list(subject_files.keys())
    channels = [key for key, value in CHANNEL_MAPPING.items() if value in modality]
    kf = KFold(n_splits=k, shuffle=True)
    for fold, (train_indices, test_indices) in enumerate(kf.split(subject_ids)):
        train_files = []
        test_files = []
        train_subject_ids = [subject_ids[i] for i in train_indices]
        test_subject_ids = [subject_ids[i] for i in test_indices]
        for subject_id in train_subject_ids:
            train_files += subject_files[subject_id]
        for subject_id in test_subject_ids:
            test_files += subject_files[subject_id]

        # create train and test dataloaders
        train_dataset = sleepStagingDataset(train_files, channels)
        test_dataset = sleepStagingDataset(test_files, channels)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=n_jobs,

        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=n_jobs,
        )

        assert isinstance(experiment_id, str), f"experiment_id should be of type str, got {type(experiment_id)}"
        experiment_id = experiment_id + '_' + str(fold)
        model = SleepStagerChambon2018(**model_kwargs)
        trainer = classifierModel(model, experiment_id, weights_path, train_loader, test_loader, wandb, *args, **kwargs)
        wandb.watch([trainer], log="all", log_freq=500)
        
        trainer.fit()


dataset_path = os.path.expanduser("~/.nimhans/subjects_data")
subject_files = map_subject_files(dataset_path)
weights_path = os.path.expanduser("~/.nimhans/weights")
if not os.path.exists(weights_path): os.makedirs(weights_path, exist_ok=True)

wandb = wandb.init(
    project="nimhans",
    name="sleep-staging",
    save_code=False,
    entity="sleep-staging",
)

wandb.save("./sleepStaging/helper_train.py")
wandb.save("./sleepStaging/train.py")

experiment_id = _get_experiment_id()
modality = ['eeg']
k = 5
model_kwargs = {'n_channels': 4, 'n_classes': 5, 'sfreq': 100}
batch_size = 256
n_classes = 5

do_k_fold(experiment_id, subject_files, k, modality, weights_path, wandb, model_kwargs, n_jobs=8, batch_size=batch_size, n_classes=n_classes)

