import os
from typing import List, Any, Dict, Union

import mne
import numpy as np
import pandas as pd


class StagingPreprocess:
    def __init__(
        self,
        raw_path: str,
        ann_path: str,
        channels: Dict[str, str],
        modality: List[str],
        window_size: float,
        sfreq: int,
        preload: bool = False,
        crop_wake_mins: int = 0,
        crop: Union[Any, None] = None
    ):
        if not isinstance(raw_path, str) or not isinstance(ann_path, str):
            raise Exception(f"raw_path and ann_path must be strings, found raw_path: {type(raw_path)} ann_path: {type(ann_path)}")
        
        self.window_size = window_size
        self.sfreq = sfreq
        raw, desc = self._load_raw(
            raw_path,
            ann_path,
            channels,
            modality,
            preload=preload,
            crop_wake_mins=crop_wake_mins,
            crop=crop,
        )
        
        self._raw = raw
        self._description = desc
    
    @property
    def raw(self):
        return self._raw
    
    @property
    def description(self):
        return self._description
        
    def read_annotations(self, ann_fname):
        labels = []
        ann = pd.read_excel(ann_fname, sheet_name="Sleep profile")[7:]
        ann.reset_index(inplace=True, drop=True)
        ann.columns = ["timestamp", "stage"]
        ann_list = ann["stage"].tolist()
        timestamps = ann["timestamp"].tolist() # to be used

        for lbl in ann_list:
            if lbl == "Wake":
                labels.append('W')
            elif lbl == "N1":
                labels.append('N1')
            elif lbl == "N2":
                labels.append('N2')
            elif lbl == "N3":
                labels.append('N3')
            elif lbl == "REM":
                labels.append('R')
            elif lbl == "A":
                labels.append('BAD_A')
            else:
                print(
                    "============================== Faulty file ============================="
                )

        labels = np.asarray(labels)
        onsets = [self.window_size * i for i in range(len(labels))]
        onsets = np.asarray(onsets)
        durations = np.repeat(float(self.window_size), len(labels))
        annots = mne.Annotations(onsets, durations, labels)
        return annots

    def _load_raw(
        self,
        raw_fname,
        ann_fname,
        channels,
        modality,
        preload,
        crop_wake_mins,
        crop,
    ):  
        raw = mne.io.read_raw_edf(raw_fname, preload=preload)
        # try:
        #     raw.set_channel_types(channels)
        # except:
        #     print(f'Please check the channels in {raw_fname}')
        raw.set_channel_types(channels)
        raw.pick(modality)
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
        subj_nb = int(raw_basename.split('.')[0][2:])
        desc = pd.DataFrame(
            {
                "subject_id": [subj_nb],
            }
        )
        return raw, desc

    @staticmethod
    def create_windows(raw, description, window_size: float = 30., window_stride: float = 30., label_mapping=None, drop_last: bool = False, drop_bad: bool = False):
        assert isinstance(window_size, (int, np.integer, float, np.floating)), "window_size has to be an integer or float"
        assert isinstance(window_stride, (int, np.integer, float, np.floating)), "window_stride has to be an integer or float"
                
        window_size_samples = int(window_size*raw.info['sfreq'])
        window_stride_samples = int(window_stride*raw.info['sfreq'])

        assert len(raw.annotations.description) > 0, "No annotations found in raw"
        assert window_size_samples > 0, "window size has to be larger than 0"
        assert window_stride_samples > 0, "window stride has to be larger than 0"
        
        if label_mapping is None:
            label_mapping = dict()
            unique_events = np.unique(raw.annotations.description)
            filtered_unique_events = [event for event in unique_events if not event.startswith("BAD_")]
            label_mapping.update(
                {v: k for k, v in enumerate(filtered_unique_events)}
            )
            
        events, events_id = mne.events_from_annotations(raw, event_id=label_mapping) # type: ignore
        onsets = events[:, 0] # starts compared to original start of recording
        targets = events[:, -1]
        filtered_durations = np.array(
            [ann['duration'] for ann in raw.annotations
                if ann['description'] in events_id]
        )
        stops = onsets + (filtered_durations * raw.info['sfreq']).astype(int)
        
        if window_size_samples is None:
            window_size_samples = stops[0] - onsets[0]
            if window_stride_samples is None:
                window_stride_samples = window_size_samples   
        
        if drop_last:
            if (stops-onsets)[-1] != window_size_samples:
                stops = stops[:-1]
                onsets = onsets[:-1]
                targets = targets[:-1]
                events = events[:-1]
                
        events = [[start, window_size_samples, targets[i_start]] for i_start, start in enumerate(onsets)] # new events
        metadata = pd.DataFrame({
            'start': onsets,
            'stop': stops,
            'size': window_size_samples,
            'stride': window_stride_samples,
            'target': targets})    
        
        desc_columns = list(description.columns)
        for col in desc_columns:
            metadata[col] = description[col].values[0]

        mne_epochs = mne.Epochs(
                raw, events, events_id, baseline=None, tmin=0,
                tmax=(window_size_samples - 1) / raw.info['sfreq'],
                metadata=metadata, preload=True, verbose=False)
        
        if drop_bad:
            mne_epochs.drop_bad()
        
        return mne_epochs