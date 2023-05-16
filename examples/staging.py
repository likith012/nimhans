import os, sys

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.split(current_path)[0])


import glob
import inspect
import argparse

import wandb
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from nimhans.utils import (
    natural_keys,
    StagingDataset,
    set_random_seeds,
    CHANNEL_MAPPING,
)
from nimhans.trainer import ClassifierTrainer
from nimhans import models


def _map_subject_files(dataset_path):
    epoch_path = glob.glob(os.path.join(dataset_path, "*"))
    epoch_path.sort(key=natural_keys)

    subject_files = {}
    for path in epoch_path:
        subject_number = os.path.basename(path).split("_")[0]
        if subject_number not in subject_files:
            subject_files[subject_number] = []
        subject_files[subject_number].append(path)

    return subject_files


def _get_experiment_id():
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    basename = os.path.basename(filename)
    experiment_id = os.path.splitext(basename)[0]
    return experiment_id


def _get_num_channels(modality, mapping):
        num_channels = 0
        for _, value in mapping.items():
            if value in modality:
                num_channels += 1
        return num_channels


def do_k_fold(
    experiment_name,
    model,
    subject_files,
    n_folds,
    modality,
    weights_path,
    loggr,
    n_jobs,
    batch_size,
    *args,
    **kwargs,
):
    subject_ids = list(subject_files.keys())
    channels = [key for key, value in CHANNEL_MAPPING.items() if value in modality]
    kf = KFold(n_splits=n_folds, shuffle=True)
    
    metrics = {"Accuracy": [], "F1": [], "Kappa": []}
    
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
        train_dataset = StagingDataset(train_files, channels, "train", lazy_load=True)
        test_dataset = StagingDataset(test_files, channels, "test", lazy_load=True)

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

        assert isinstance(
            experiment_name, str
        ), f"experiment_id should be of type str, got {type(experiment_name)}"
        trainer_name = experiment_name + "_fold_" + str(fold+1)
        
        trainer = ClassifierTrainer(
            model,
            trainer_name,
            weights_path,
            train_loader,
            test_loader,
            loggr,
            kfold=fold+1,
            *args,
            **kwargs,
        )
        if loggr is not None:
            loggr.watch([trainer], log="all", log_freq=500)

        trainer.fit()
        
        metrics["Accuracy"].append(trainer.best_accuracy)
        metrics["F1"].append(trainer.best_f1)
        metrics["Kappa"].append(trainer.best_kappa)
        
    np.save(os.path.join(weights_path, experiment_name + "_metrics.npy"), metrics)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, choices=["resnet", "chambon", "deepsleep", "blanco", "eldele", "usleep"], default="resnet", help="Model to train")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of folds for cross validation")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs to run in parallel")
    parser.add_argument("--n_classes", type=int, default=5, help="Number of classes")
    parser.add_argument("-d", "--dataset_path", type=str, default="~/.nimhans/preprocessed", help="Path to dataset")
    parser.add_argument("-w", "--weights_path", type=str, default="~/.nimhans/weights", help="Path to save weights")
    parser.add_argument("--modality", nargs='+', type=str, default=["eeg"], help="Modality of data")
    parser.add_argument("--loggr", type=str, default=None, help="Logger to use")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducibility")
    args = parser.parse_args() 

    if args.seed is not None:
        set_random_seeds(args.seed)
    
    if args.loggr is None:
        loggr = None
    elif args.loggr == "wandb":
        loggr = wandb.init(
        name="sleep-staging",
        project="nimhans",
        save_code=False,
        entity="sleep-staging",
        )
    
    subject_files = _map_subject_files(os.path.expanduser(args.dataset_path))
    experiment_id = _get_experiment_id() 
    n_channels = _get_num_channels(args.modality, CHANNEL_MAPPING)
    weights_path = os.path.join(os.path.expanduser(args.weights_path), experiment_id)
    experiment_name = args.model
            
    if not os.path.exists(weights_path):
        os.makedirs(weights_path, exist_ok=True)
    
    if args.model == "resnet":
        model_kwargs = {"input_channels": n_channels, "num_class": args.n_classes, "do_context": False}
        model = models.resnet50(**model_kwargs)
        
    elif args.model == "chambon":
        model_kwargs = {'n_channels': n_channels, 'sfreq': 100, 'n_classes': args.n_classes}
        model = models.SleepStagerChambon2018(**model_kwargs)
        
    elif args.model == "deepsleep":
        model_kwargs = {'n_channels': n_channels, 'n_classes': args.n_classes}
        model = models.DeepSleepNet(**model_kwargs)
        
    elif args.model == "blanco":
        model_kwargs = {'n_channels': n_channels, 'sfreq': 100, 'n_classes': args.n_classes}
        model = models.SleepStagerBlanco2020(**model_kwargs)
        
    elif args.model == "eldele":  
        model_kwargs = {'in_channels': n_channels, 'sfreq': 100, 'n_classes': args.n_classes}
        model = models.SleepStagerEldele2021(**model_kwargs)
        
    elif args.model == "usleep":        
        model_kwargs = {'in_chans': n_channels, 'sfreq': 100, 'n_classes': args.n_classes}
        model = models.USleep(**model_kwargs)
        
    
    do_k_fold(
        experiment_name,
        model,
        subject_files,
        args.n_folds,
        args.modality,
        weights_path,
        loggr,
        args.n_jobs,
        args.batch_size,
    )