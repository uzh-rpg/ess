import glob
from os.path import join, exists, dirname, basename
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as f
import torchvision.transforms as transforms

from datasets.extract_data_tools.example_loader_ddd17 import load_files_in_directory, extract_events_from_memmap
import datasets.data_util as data_util
import albumentations as A
from PIL import Image
from utils.labels import shiftUpId, shiftDownId


def get_split(dirs, split):
    return {
        "train": [dirs[0], dirs[2], dirs[3], dirs[5], dirs[6]],
        "test": [dirs[4]],
        "valid": [dirs[1]]
    }[split]


def unzip_segmentation_masks(dirs):
    for d in dirs:
        assert exists(join(d, "segmentation_masks.zip"))
        if not exists(join(d, "segmentation_masks")):
            print("Unzipping segmentation mask in %s" % d)
            os.system("unzip %s -d %s" % (join(d, "segmentation_masks"), d))


class DDD17Events(Dataset):
    def __init__(self, root, split="train", event_representation='voxel_grid',
                 nr_events_data=5, delta_t_per_data=50, nr_bins_per_data=5, require_paired_data=False,
                 separate_pol=False, normalize_event=False, augmentation=False, fixed_duration=False,
                 nr_events_per_data=32000, resize=True, random_crop=False):
        data_dirs = sorted(glob.glob(join(root, "dir*")))
        assert len(data_dirs) > 0
        assert split in ["train", "valid", "test"]

        self.split = split
        self.augmentation = augmentation
        self.fixed_duration = fixed_duration
        self.nr_events_per_data = nr_events_per_data

        self.nr_events_data = nr_events_data
        self.delta_t_per_data = delta_t_per_data
        if self.fixed_duration:
            self.t_interval = nr_events_data * delta_t_per_data
        else:
            self.t_interval = -1
            self.nr_events = self.nr_events_data * self.nr_events_per_data
        assert self.t_interval in [10, 50, 250, -1]
        self.nr_temporal_bins = nr_bins_per_data
        self.require_paired_data = require_paired_data
        self.event_representation = event_representation
        self.shape = [260, 346]
        self.resize = resize
        self.shape_resize = [260, 352]
        self.random_crop = random_crop
        self.shape_crop = [120, 216]
        self.separate_pol = separate_pol
        self.normalize_event = normalize_event
        self.dirs = get_split(data_dirs, split)
        # unzip_segmentation_masks(self.dirs)

        self.files = []
        for d in self.dirs:
            self.files += glob.glob(join(d, "segmentation_masks", "*.png"))

        print("[DDD17Segmentation]: Found %s segmentation masks for split %s" % (len(self.files), split))

        # load events and image_idx -> event index mapping
        self.img_timestamp_event_idx = {}
        self.event_data = {}

        print("[DDD17Segmentation]: Loading real events.")
        self.event_dirs = self.dirs

        for d in self.event_dirs:
            img_timestamp_event_idx, t_events, xyp_events, _ = load_files_in_directory(d, self.t_interval)
            self.img_timestamp_event_idx[d] = img_timestamp_event_idx
            self.event_data[d] = [t_events, xyp_events]

        if self.augmentation:
            self.transform_a = A.ReplayCompose([
                A.HorizontalFlip(p=0.5)
            ])
            self.transform_a_random_crop = A.ReplayCompose([
                A.HorizontalFlip(p=0.5),
                A.RandomCrop(height=self.shape_crop[0], width=self.shape_crop[1], always_apply=True)])
        self.transform_a_center_crop = A.ReplayCompose([
            A.CenterCrop(height=self.shape_crop[0], width=self.shape_crop[1], always_apply=True),
        ])

    def __len__(self):
        return len(self.files)

    def apply_augmentation(self, transform_a, events, label):
        label = shiftUpId(label)
        A_data = transform_a(image=events[0, :, :].numpy(), mask=label)
        label = A_data['mask']
        label = shiftDownId(label)
        if self.random_crop and self.split == 'train':
            events_tensor = torch.zeros((events.shape[0], self.shape_crop[0], self.shape_crop[1]))
        else:
            events_tensor = events
        for k in range(events.shape[0]):
            events_tensor[k, :, :] = torch.from_numpy(
                A.ReplayCompose.replay(A_data['replay'], image=events[k, :, :].numpy())['image'])
        return events_tensor, label

    def __getitem__(self, idx):
        segmentation_mask_file = self.files[idx]
        segmentation_mask = cv2.imread(segmentation_mask_file, 0)
        label_original = np.array(segmentation_mask)
        if self.resize:
            segmentation_mask = cv2.resize(segmentation_mask, (self.shape_resize[1], self.shape_resize[0] - 60),
                                           interpolation=cv2.INTER_NEAREST)
        label = np.array(segmentation_mask)

        directory = dirname(dirname(segmentation_mask_file))

        img_idx = int(basename(segmentation_mask_file).split("_")[-1].split(".")[0]) - 1

        img_timestamp_event_idx = self.img_timestamp_event_idx[directory]
        t_events, xyp_events = self.event_data[directory]

        # events has form x, y, t_ns, p (in [0,1])
        events = extract_events_from_memmap(t_events, xyp_events, img_idx, img_timestamp_event_idx, self.fixed_duration,
                                            self.nr_events)
        t_ns = events[:, 2]
        delta_t_ns = int((t_ns[-1] - t_ns[0]) / self.nr_events_data)
        nr_events_loaded = events.shape[0]
        nr_events_temp = nr_events_loaded // self.nr_events_data

        id_end = 0
        event_tensor = None
        for i in range(self.nr_events_data):
            id_start = id_end
            if self.fixed_duration:
                id_end = np.searchsorted(t_ns, t_ns[0] + (i + 1) * delta_t_ns)
            else:
                id_end += nr_events_temp

            if id_end > nr_events_loaded:
                id_end = nr_events_loaded

            event_representation = data_util.generate_input_representation(events[id_start:id_end],
                                                                           self.event_representation,
                                                                           self.shape,
                                                                           nr_temporal_bins=self.nr_temporal_bins,
                                                                           separate_pol=self.separate_pol)

            event_representation = torch.from_numpy(event_representation)

            if self.normalize_event:
                event_representation = data_util.normalize_voxel_grid(event_representation)

            if self.resize:
                event_representation_resize = f.interpolate(event_representation.unsqueeze(0),
                                                            size=(self.shape_resize[0], self.shape_resize[1]),
                                                            mode='bilinear', align_corners=True)
                event_representation = event_representation_resize.squeeze(0)

            if event_tensor is None:
                event_tensor = event_representation
            else:
                event_tensor = torch.cat([event_tensor, event_representation], dim=0)

        event_tensor = event_tensor[:, :-60, :]  # remove 60 bottom rows

        if self.random_crop and self.split == 'train':
            event_tensor = event_tensor[:, -self.shape_crop[0]:, :]
            label = label[-self.shape_crop[0]:, :]
            if self.augmentation:
                event_tensor, label = self.apply_augmentation(self.transform_a_random_crop, event_tensor, label)

        else:
            if self.augmentation:
                event_tensor, label = self.apply_augmentation(self.transform_a, event_tensor, label)

        label_tensor = torch.from_numpy(label).long()

        if self.split == 'valid' and self.require_paired_data:
            segmentation_mask_filepath_list = str(segmentation_mask_file).split('/')
            segmentation_mask_filename = segmentation_mask_filepath_list[-1]
            dir_name = segmentation_mask_filepath_list[-3]
            filename_id = segmentation_mask_filename.split('_')[-1]
            img_filename = '_'.join(['img', filename_id])
            img_filepath_list = segmentation_mask_filepath_list
            img_filepath_list[-2] = 'imgs'
            img_filepath_list[-1] = img_filename
            img_file = '/'.join(img_filepath_list)
            if not os.path.exists(img_file):
                img_filename = filename_id.zfill(14)
                img_filepath_list[-1] = img_filename
                img_file = '/'.join(img_filepath_list)
            img = Image.open(img_file)

            if self.resize:
                img = img.resize((self.shape_resize[1], self.shape_resize[0]))
            img_transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor()
            ])
            img_tensor = img_transform(img)
            img_tensor = img_tensor[:, :-60, :]

            label_original_tensor = torch.from_numpy(label_original).long()
            return event_tensor, img_tensor, label_tensor, label_original_tensor
        return event_tensor, label_tensor

