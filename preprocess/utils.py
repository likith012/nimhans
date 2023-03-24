import numpy as np
import pandas as pd

import mne
from braindecode.datasets import BaseDataset, BaseConcatDataset


class SleepStaging(BaseConcatDataset):
    def __init__(
        self,
        window_size=30,
        sfreq=100,
        raw_path=None,
        ann_path=None,
        channels=None,
        preload=False,
        crop_wake_mins=0,
        crop=None,
    ):
        self.window_size = window_size
        self.sfreq = sfreq
        if (raw_path is None) or (ann_path is None) or (channels is None):
            raise Exception("Please provide paths for raw and annotations file!")

        raw, desc = self._load_raw(
            raw_path,
            ann_path,
            channels,
            preload=preload,
            crop_wake_mins=crop_wake_mins,
            crop=crop,
        )
        base_ds = BaseDataset(raw, desc)
        super().__init__([base_ds])

    def read_annotations(ann_fname):
        labels = []
        ann = pd.read_excel(ann_fname, sheet_name="Sleep profile")[7:]
        ann.reset_index(inplace=True, drop=True)
        ann.columns = ["timestamp", "stage"]
        ann_list = ann["stage"].tolist()
        timestamps = ann["timestamp"].tolist()

        for lbl in ann_list:
            if lbl == "Wake":
                labels.append(0)
            elif lbl == "N1":
                labels.append(1)
            elif lbl == "N2":
                labels.append(2)
            elif lbl == "N3":
                labels.append(3)
            elif lbl == "REM":
                labels.append(4)
            elif lbl == "A":
                labels.append(5)
            else:
                print(
                    "============================== Faulty file ============================="
                )

        labels = np.asarray(labels)
        onsets = [self.window_size * i for i in range(len(labels))]
        onsets = np.asarray(onsets)
        durations = np.repeat(self.window_size, len(labels))
        annots = mne.Annotations(onsets, durations, labels)
        return annots

    def _load_raw(
        self,
        raw_fname,
        ann_fname,
        channels,
        preload,
        crop_wake_mins,
        crop,
    ):
        raw = mne.io.read_raw_edf(raw_fname, preload=preload, include=channels)
        annots = self.read_annotations(ann_fname)
        raw.set_annotations(annots, emit_warning=False)
        raw.resample(self.sfreq, npad="auto")

        if crop_wake_mins > 0:
            # Find first and last sleep stages
            mask = [x[-1] in ["1", "2", "3", "R"] for x in annots.description]
            sleep_event_inds = np.where(mask)[0]

            # Crop raw
            tmin = annots[int(sleep_event_inds[0])]["onset"] - crop_wake_mins * 60
            tmax = annots[int(sleep_event_inds[-1])]["onset"] + crop_wake_mins * 60
            raw.crop(tmin=max(tmin, raw.times[0]), tmax=min(tmax, raw.times[-1]))

        if crop is not None:
            raw.crop(*crop)

        raw_basename = os.path.basename(raw_fname)
        subj_nb = int(raw_basename[-10:-4])
        desc = pd.Series(
            {
                "subject_id": subj_nb,
            },
            name="",
        )
        return raw, desc
