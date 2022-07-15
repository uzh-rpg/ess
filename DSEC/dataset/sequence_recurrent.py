from pathlib import Path
import weakref

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image

from DSEC.dataset.representations import VoxelGrid
from DSEC.utils.eventslicer import EventSlicer
import albumentations as A


class SequenceRecurrent(Dataset):
    # NOTE: This is just an EXAMPLE class for convenience. Adapt it to your case.
    # In this example, we use the voxel grid representation.
    #
    # This class assumes the following structure in a sequence directory:
    #
    # seq_name (e.g. zurich_city_11_a)
    # ├── disparity
    # │   ├── event
    # │   │   ├── 000000.png
    # │   │   └── ...
    # │   └── timestamps.txt
    # └── events
    #     ├── left
    #     │   ├── events.h5
    #     │   └── rectify_map.h5
    #     └── right
    #         ├── events.h5
    #         └── rectify_map.h5

    def __init__(self, seq_path: Path, mode: str='train', event_representation: str = 'voxel_grid',
                 nr_events_data: int = 5, delta_t_per_data: int = 20, nr_events_per_data: int = 100000,
                 nr_bins_per_data: int = 5, require_paired_data=False, normalize_event=False, separate_pol=False,
                 semseg_num_classes: int = 11, augmentation=True, fixed_duration=False, loading_time_window: int = 250):
        assert nr_bins_per_data >= 1
        # assert delta_t_ms <= 200, 'adapt this code, if duration is higher than 100 ms'
        assert seq_path.is_dir()

        # NOTE: Adapt this code according to the present mode (e.g. train, val or test).
        self.mode = mode
        self.augmentation = augmentation

        # Save output dimensions
        self.height = 480
        self.width = 640

        # Set event representation
        self.nr_events_data = nr_events_data
        self.num_bins = nr_bins_per_data
        assert nr_events_per_data > 0
        self.nr_events_per_data = nr_events_per_data
        self.event_representation = event_representation
        self.separate_pol = separate_pol
        self.normalize_event = normalize_event
        self.voxel_grid = VoxelGrid(self.num_bins, self.height, self.width, normalize=self.normalize_event)

        self.locations = ['left']  # 'right'
        self.semseg_num_classes = semseg_num_classes

        # Save delta timestamp in ms
        self.fixed_duration = fixed_duration
        if self.fixed_duration:
            delta_t_ms = nr_events_data * delta_t_per_data
        else:
            delta_t_ms = loading_time_window
        self.delta_t_us = delta_t_ms * 1000

        self.require_paired_data = require_paired_data
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load disparity timestamps
        # disp_dir = seq_path / 'disparity'
        img_dir = seq_path / 'images'
        assert img_dir.is_dir()

        self.timestamps = np.loadtxt(img_dir / 'left' / 'exposure_timestamps.txt', comments='#', delimiter=',', dtype='int64')[:, 1]

        # load images paths
        if self.require_paired_data:
            img_left_dir = img_dir / 'left' / 'ev_inf'
            assert img_left_dir.is_dir()
            img_left_pathstrings = list()
            for entry in img_left_dir.iterdir():
                assert str(entry.name).endswith('.png')
                img_left_pathstrings.append(str(entry))
            img_left_pathstrings.sort()
            self.img_left_pathstrings = img_left_pathstrings

            assert len(self.img_left_pathstrings) == self.timestamps.size

        if self.mode == 'val':
            if self.semseg_num_classes == 11:
                label_dir = seq_path / 'semantic' / '11classes' / 'data'
            elif self.semseg_num_classes == 19:
                label_dir = seq_path / 'semantic' / '19classes' / 'data'
            elif self.semseg_num_classes == 6:
                label_dir = seq_path / 'semantic' / '6classes' / 'data'
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

        # Remove first disparity path and corresponding timestamp.
        # This is necessary because we do not have events before the first disparity map.
        # assert int(Path(self.disp_gt_pathstrings[0]).stem) == 0
        # del self.disp_gt_pathstrings[:(delta_t_ms // 100 + 1)]
        self.timestamps = self.timestamps[(delta_t_ms // 50 + 1):]
        if self.require_paired_data:
            del self.img_left_pathstrings[:(delta_t_ms // 50 + 1)]
            assert len(self.img_left_pathstrings) == self.timestamps.size
            if self.mode == 'val':
                del self.label_pathstrings[:(delta_t_ms // 50 + 1)]
                assert len(self.img_left_pathstrings) == len(self.label_pathstrings)

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

    def events_to_voxel_grid(self, x, y, p, t, device: str='cpu'):
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
    def get_img(filepath: Path):
        assert filepath.is_file()
        img = Image.open(str(filepath))
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
        # label_tensor = torch.from_numpy(label).long()
        return label

    @staticmethod
    def close_callback(h5f_dict):
        for k, h5f in h5f_dict.items():
            h5f.close()

    def __len__(self):
        if self.fixed_duration:
            return self.timestamps.size
        else:
            return self.event_slicers['left'].events['t'].size // self.nr_events_per_data

    def rectify_events(self, x: np.ndarray, y: np.ndarray, location: str):
        assert location in self.locations
        # From distorted to undistorted
        rectify_map = self.rectify_ev_maps[location]
        assert rectify_map.shape == (self.height, self.width, 2), rectify_map.shape
        assert x.max() < self.width
        assert y.max() < self.height
        return rectify_map[y, x]

    def __getitem__(self, index):
        if self.augmentation:
            transform_a = A.ReplayCompose([
                A.HorizontalFlip(p=0.5),
                # A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
                # A.PadIfNeeded(min_height=self.height-40, min_width=self.width, always_apply=True, border_mode=0),
                # A.RandomCrop(height=self.height-40, width=self.width, p=1)
            ])
            A_data = None

        label = np.zeros((self.height-40, self.width))

        output = {}
        for location in self.locations:
            if self.fixed_duration:
                if self.mode == 'val':
                    label_path = Path(self.label_pathstrings[index])
                    label = self.get_label(label_path)

                ts_end = self.timestamps[index]
                # ts_start should be fine (within the window as we removed the first disparity map)
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

                    event_representation = self.events_to_voxel_grid(x_rect, y_rect, p, t)

                    if event_tensor is None:
                        event_tensor = event_representation
                    else:
                        event_tensor = torch.cat([event_tensor, event_representation], dim=0)
            else:
                num_bins_total = self.nr_events_data * self.num_bins
                self.nr_events = self.nr_events_data * self.nr_events_per_data
                t_start_us_idx = index * self.nr_events
                t_end_us_idx = t_start_us_idx + self.nr_events
                event_data = self.event_slicers[location].get_events_fixed_num_recurrent(t_start_us_idx, t_end_us_idx)

                p = event_data['p']
                t = event_data['t']
                x = event_data['x']
                y = event_data['y']

                xy_rect = self.rectify_events(x, y, location)
                x_rect = xy_rect[:, 0]
                y_rect = xy_rect[:, 1]

                event_representation = self.events_to_voxel_grid(x_rect, y_rect, p, t)

                event_tensor = event_representation


            event_tensor = event_tensor[:, :-40, :]  # remove 40 bottom rows

            if self.augmentation:
                if A_data is None:
                    A_data = transform_a(image=event_tensor[0, :, :].numpy(), mask=label)
                    label = A_data['mask']
                for k in range(event_tensor.shape[0]):
                    event_tensor[k, :, :] = torch.from_numpy(
                        A.ReplayCompose.replay(A_data['replay'], image=event_tensor[k, :, :].numpy())['image'])

            if 'representation' not in output:
                output['representation'] = dict()
            output['representation'][location] = event_tensor

        label_tensor = torch.from_numpy(label).long()

        if self.require_paired_data:
            img_left_path = Path(self.img_left_pathstrings[index])
            # output['img_left'] = self.get_img(img_left_path)[:, :-40, :]  # remove 40 bottom rows
            output['img_left'] = self.get_img(img_left_path)
            return output['representation']['left'], output['img_left'], label_tensor
        return output['representation']['left'], label_tensor
