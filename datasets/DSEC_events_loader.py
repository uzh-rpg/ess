import os
import torch
import random
import numpy as np
from PIL import Image

import torchvision.transforms as transforms
import datasets.data_util as data_util
from pathlib import Path

from DSEC.dataset.provider import DatasetProvider


def DSECEvents(dsec_dir, nr_events_data=1, delta_t_per_data=50, nr_events_window=-1,
               augmentation=False, mode='train', task='segmentation', event_representation='voxel_grid',
               nr_bins_per_data=5, require_paired_data=False, separate_pol=False, normalize_event=False,
               semseg_num_classes=11, fixed_duration=False, resize=False):
    """
    Creates an iterator over the EventScape dataset.

    :param root: path to dataset root
    :param height: height of dataset image
    :param width: width of dataset image
    :param nr_events_window: number of events summed in the sliding histogram
    :param augmentation: flip, shift and random window start for training
    :param mode: 'train', 'test' or 'val'
    """
    dsec_dir = Path(dsec_dir)
    assert dsec_dir.is_dir()

    dataset_provider = DatasetProvider(dsec_dir, mode, event_representation=event_representation,
                                       nr_events_data=nr_events_data, delta_t_per_data=delta_t_per_data,
                                       nr_events_window=nr_events_window, nr_bins_per_data=nr_bins_per_data,
                                       require_paired_data=require_paired_data, normalize_event=normalize_event,
                                       separate_pol=separate_pol, semseg_num_classes=semseg_num_classes,
                                       augmentation=augmentation, fixed_duration=fixed_duration, resize=resize)
    if mode == 'train':
        train_dataset = dataset_provider.get_train_dataset()
        return train_dataset
    else:
        val_dataset = dataset_provider.get_val_dataset()
        return val_dataset
