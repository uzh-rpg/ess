from pathlib import Path

import torch

from DSEC.dataset.sequence import Sequence


class DatasetProvider:
    def __init__(self, dataset_path: Path, mode: str = 'train', event_representation: str = 'voxel_grid',
                 nr_events_data: int = 5, delta_t_per_data: int = 20,
                 nr_events_window=-1, nr_bins_per_data=5, require_paired_data=False, normalize_event=False,
                 separate_pol=False, semseg_num_classes=11, augmentation=False,
                 fixed_duration=False, resize=False):
        train_path = dataset_path / 'train'
        val_path = dataset_path / 'test'
        assert dataset_path.is_dir(), str(dataset_path)
        assert train_path.is_dir(), str(train_path)
        assert val_path.is_dir(), str(val_path)

        if mode == 'train':
            train_sequences = list()
            train_sequences_namelist = ['zurich_city_00_a', 'zurich_city_01_a', 'zurich_city_02_a',
                                        'zurich_city_04_a', 'zurich_city_05_a', 'zurich_city_06_a',
                                        'zurich_city_07_a', 'zurich_city_08_a']
            for child in train_path.iterdir():
                if any(k in str(child) for k in train_sequences_namelist):
                    train_sequences.append(Sequence(child, 'train', event_representation, nr_events_data, delta_t_per_data,
                                                    nr_events_window, nr_bins_per_data, require_paired_data, normalize_event
                                                    , separate_pol, semseg_num_classes, augmentation, fixed_duration
                                                    , resize=resize))
                else:
                    continue

            self.train_dataset = torch.utils.data.ConcatDataset(train_sequences)
            self.train_dataset.require_paired_data = require_paired_data

        elif mode == 'val':
            val_sequences = list()
            val_sequences_namelist = ['zurich_city_13_a', 'zurich_city_14_c', 'zurich_city_15_a']
            for child in val_path.iterdir():
                if any(k in str(child) for k in val_sequences_namelist):
                    val_sequences.append(Sequence(child, 'val', event_representation, nr_events_data, delta_t_per_data,
                                                  nr_events_window, nr_bins_per_data, require_paired_data, normalize_event
                                                  , separate_pol, semseg_num_classes, augmentation, fixed_duration
                                                  , resize=resize))
                else:
                    continue

            self.val_dataset = torch.utils.data.ConcatDataset(val_sequences)
            self.val_dataset.require_paired_data = require_paired_data


    def get_train_dataset(self):
        return self.train_dataset

    def get_val_dataset(self):
        # Implement this according to your needs.
        return self.val_dataset

    def get_test_dataset(self):
        # Implement this according to your needs.
        raise NotImplementedError
