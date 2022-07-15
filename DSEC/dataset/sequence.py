"""
Adapted from https://github.com/uzh-rpg/DSEC/blob/main/scripts/dataset/sequence.py
"""
from pathlib import Path
import weakref

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as f
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from joblib import Parallel, delayed

from DSEC.dataset.representations import VoxelGrid
from DSEC.utils.eventslicer import EventSlicer
import albumentations as A
import datasets.data_util as data_util
import random

class Sequence(Dataset):
    # This class assumes the following structure in a sequence directory:
    #
    # seq_name (e.g. zurich_city_00_a)
    # ├── semantic
    # │   ├── left
    # │   │   ├── 11classes
    # │   │   │   └──data
    # │   │   │       ├── 000000.png
    # │   │   │       └── ...
    # │   │   └── 19classes
    # │   │       └──data
    # │   │           ├── 000000.png
    # │   │           └── ...
    # │   └── timestamps.txt
    # └── events
    #     └── left
    #         ├── events.h5
    #         └── rectify_map.h5

    def __init__(self, seq_path: Path, mode: str='train', event_representation: str = 'voxel_grid',
                 nr_events_data: int = 5, delta_t_per_data: int = 20, nr_events_per_data: int = 100000,
                 nr_bins_per_data: int = 5, require_paired_data=False, normalize_event=False, separate_pol=False,
                 semseg_num_classes: int = 11, augmentation=False, fixed_duration=False, remove_time_window: int = 250,
                 resize=False):
        assert nr_bins_per_data >= 1
        assert seq_path.is_dir()
        self.sequence_name = seq_path.name
        self.mode = mode

        # Save output dimensions
        self.height = 480
        self.width = 640
        self.resize = resize
        self.shape_resize = None
        if self.resize:
            self.shape_resize = [448, 640]

        # Set event representation
        self.nr_events_data = nr_events_data
        self.num_bins = nr_bins_per_data
        assert nr_events_per_data > 0
        self.nr_events_per_data = nr_events_per_data
        self.event_representation = event_representation
        self.separate_pol = separate_pol
        self.normalize_event = normalize_event
        self.voxel_grid = VoxelGrid(self.num_bins, self.height, self.width, normalize=self.normalize_event)

        self.locations = ['left']
        self.semseg_num_classes = semseg_num_classes
        self.augmentation = augmentation

        # Save delta timestamp
        self.fixed_duration = fixed_duration
        if self.fixed_duration:
            delta_t_ms = nr_events_data * delta_t_per_data
            self.delta_t_us = delta_t_ms * 1000
        self.remove_time_window = remove_time_window

        self.require_paired_data = require_paired_data
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load timestamps
        self.timestamps = np.loadtxt(str(seq_path / 'semantic' / 'timestamps.txt'), dtype='int64')

        # load label paths
        if self.semseg_num_classes == 11:
            label_dir = seq_path / 'semantic' / '11classes' / 'data'
        elif self.semseg_num_classes == 19:
            label_dir = seq_path / 'semantic' / '19classes' / 'data'
        else:
            raise ValueError
        assert label_dir.is_dir()
        label_pathstrings = list()
        for entry in label_dir.iterdir():
            assert str(entry.name).endswith('.png')
            label_pathstrings.append(str(entry))
        label_pathstrings.sort()
        self.label_pathstrings = label_pathstrings

        assert len(self.label_pathstrings) == self.timestamps.size

        # load images paths
        if self.require_paired_data:
            img_dir = seq_path / 'images'
            img_left_dir = img_dir / 'left' / 'ev_inf'
            assert img_left_dir.is_dir()
            img_left_pathstrings = list()
            for entry in img_left_dir.iterdir():
                assert str(entry.name).endswith('.png')
                img_left_pathstrings.append(str(entry))
            img_left_pathstrings.sort()
            self.img_left_pathstrings = img_left_pathstrings

            assert len(self.img_left_pathstrings) == self.timestamps.size

        # Remove several label paths and corresponding timestamps in the remove_time_window.
        # This is necessary because we do not have enough events before the first label.
        self.timestamps = self.timestamps[(self.remove_time_window // 100 + 1) * 2:]
        del self.label_pathstrings[:(self.remove_time_window // 100 + 1) * 2]
        assert len(self.label_pathstrings) == self.timestamps.size
        if self.require_paired_data:
            del self.img_left_pathstrings[:(self.remove_time_window // 100 + 1) * 2]
            assert len(self.img_left_pathstrings) == self.timestamps.size

        self.h5f = dict()
        self.rectify_ev_maps = dict()
        self.event_slicers = dict()

        ev_dir = seq_path / 'events'
        for location in self.locations:
            ev_dir_location = ev_dir / location
            ev_data_file = ev_dir_location / 'events.h5'
            ev_rect_file = ev_dir_location / 'rectify_map.h5'

            h5f_location = h5py.File(str(ev_data_file), 'r')
            self.h5f[location] = h5f_location
            self.event_slicers[location] = EventSlicer(h5f_location)
            with h5py.File(str(ev_rect_file), 'r') as h5_rect:
                self.rectify_ev_maps[location] = h5_rect['rectify_map'][()]

    def events_to_voxel_grid(self, x, y, p, t):
        t = (t - t[0]).astype('float32')
        t = (t/t[-1])
        x = x.astype('float32')
        y = y.astype('float32')
        pol = p.astype('float32')
        return self.voxel_grid.convert(
                torch.from_numpy(x),
                torch.from_numpy(y),
                torch.from_numpy(pol),
                torch.from_numpy(t))

    def getHeightAndWidth(self):
        return self.height, self.width

    @staticmethod
    def get_disparity_map(filepath: Path):
        assert filepath.is_file()
        disp_16bit = cv2.imread(str(filepath), cv2.IMREAD_ANYDEPTH)
        return disp_16bit.astype('float32')/256

    @staticmethod
    def get_img(filepath: Path, shape_resize=None):
        assert filepath.is_file()
        img = Image.open(str(filepath))
        if shape_resize is not None:
            img = img.resize((shape_resize[1], shape_resize[0]))
        img_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
        img_tensor = img_transform(img)
        return img_tensor

    @staticmethod
    def get_label(filepath: Path):
        assert filepath.is_file()
        label = Image.open(str(filepath))
        label = np.array(label)
        return label

    @staticmethod
    def close_callback(h5f_dict):
        for k, h5f in h5f_dict.items():
            h5f.close()

    def __len__(self):
        return (self.timestamps.size + 1) // 2

    def rectify_events(self, x: np.ndarray, y: np.ndarray, location: str):
        assert location in self.locations
        # From distorted to undistorted
        rectify_map = self.rectify_ev_maps[location]
        assert rectify_map.shape == (self.height, self.width, 2), rectify_map.shape
        assert x.max() < self.width
        assert y.max() < self.height
        return rectify_map[y, x]

    def generate_event_tensor(self, job_id, events, event_tensor, nr_events_per_data):
        id_start = job_id * nr_events_per_data
        id_end = (job_id + 1) * nr_events_per_data
        events_temp = events[id_start:id_end]
        event_representation = self.events_to_voxel_grid(events_temp[:, 0], events_temp[:, 1], events_temp[:, 3],
                                                         events_temp[:, 2])
        event_tensor[(job_id * self.num_bins):((job_id+1) * self.num_bins), :, :] = event_representation

    def __getitem__(self, index):
        label_path = Path(self.label_pathstrings[index * 2])
        if self.resize:
            segmentation_mask = cv2.imread(str(label_path), 0)
            segmentation_mask = cv2.resize(segmentation_mask, (self.shape_resize[1], self.shape_resize[0]),
                                           interpolation=cv2.INTER_NEAREST)
            label = np.array(segmentation_mask)
        else:
            label = self.get_label(label_path)

        ts_end = self.timestamps[index * 2]

        output = {}
        for location in self.locations:
            if self.fixed_duration:
                ts_start = ts_end - self.delta_t_us
                event_tensor = None
                self.delta_t_per_data_us = self.delta_t_us / self.nr_events_data
                for i in range(self.nr_events_data):
                    t_s = ts_start + i * self.delta_t_per_data_us
                    t_end = ts_start + (i+1) * self.delta_t_per_data_us
                    event_data = self.event_slicers[location].get_events(t_s, t_end)

                    p = event_data['p']
                    t = event_data['t']
                    x = event_data['x']
                    y = event_data['y']

                    xy_rect = self.rectify_events(x, y, location)
                    x_rect = xy_rect[:, 0]
                    y_rect = xy_rect[:, 1]

                    if self.event_representation == 'voxel_grid':
                        event_representation = self.events_to_voxel_grid(x_rect, y_rect, p, t)
                    else:
                        events = np.stack([x_rect, y_rect, t, p], axis=1)
                        event_representation = data_util.generate_input_representation(events, self.event_representation,
                                                                  (self.height, self.width))
                        event_representation = torch.from_numpy(event_representation).type(torch.FloatTensor)

                    if event_tensor is None:
                        event_tensor = event_representation
                    else:
                        event_tensor = torch.cat([event_tensor, event_representation], dim=0)

            else:
                num_bins_total = self.nr_events_data * self.num_bins
                event_tensor = torch.zeros((num_bins_total, self.height, self.width))
                self.nr_events = self.nr_events_data * self.nr_events_per_data
                event_data = self.event_slicers[location].get_events_fixed_num(ts_end, self.nr_events)

                if self.nr_events >= event_data['t'].size:
                    start_index = 0
                else:
                    start_index = -self.nr_events

                p = event_data['p'][start_index:]
                t = event_data['t'][start_index:]
                x = event_data['x'][start_index:]
                y = event_data['y'][start_index:]
                nr_events_loaded = t.size

                xy_rect = self.rectify_events(x, y, location)
                x_rect = xy_rect[:, 0]
                y_rect = xy_rect[:, 1]

                nr_events_temp = nr_events_loaded // self.nr_events_data
                events = np.stack([x_rect, y_rect, t, p], axis=-1)
                Parallel(n_jobs=8, backend="threading")(
                    delayed(self.generate_event_tensor)(i, events, event_tensor, nr_events_temp) for i in range(self.nr_events_data))

            # remove 40 bottom rows
            event_tensor = event_tensor[:, :-40, :]

            if self.resize:
                event_tensor = f.interpolate(event_tensor.unsqueeze(0),
                                             size=(self.shape_resize[0], self.shape_resize[1]),
                                             mode='bilinear', align_corners=True).squeeze(0)

            label_tensor = torch.from_numpy(label).long()

            if self.augmentation:
                value_flip = round(random.random())
                if value_flip > 0.5:
                    event_tensor = torch.flip(event_tensor, [2])
                    label_tensor = torch.flip(label_tensor, [1])

        if 'representation' not in output:
            output['representation'] = dict()
        output['representation'][location] = event_tensor

        if self.require_paired_data:
            img_left_path = Path(self.img_left_pathstrings[index * 2])
            output['img_left'] = self.get_img(img_left_path, self.shape_resize)
            return output['representation']['left'], output['img_left'], label_tensor
        return output['representation']['left'], label_tensor


