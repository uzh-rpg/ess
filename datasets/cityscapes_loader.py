import numpy as np
import torch
from PIL import Image
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import albumentations as A
from utils.labels import Id2label_6_Cityscapes, Id2label_11_Cityscapes, fromIdToTrainId


class CityscapesGray(Dataset):
    def __init__(self, root, height=None, width=None, augmentation=False, split='train', target_type='semantic',
                 semseg_num_classes=6, standardization=False, random_crop=True):

        self.root = root
        self.split = split
        self.height = height
        self.width = width
        self.random_crop = random_crop
        if self.random_crop:
            self.height_resize = 256  # 154
            self.width_resize = 512  # 308
        else:
            self.height_resize = height
            self.width_resize = width
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize([self.height_resize, self.width_resize])
        ])
        self.cityscapes_dataset = datasets.Cityscapes(self.root, split=self.split, mode='fine', target_type=target_type,
                                                      transform=self.transform, target_transform=None)
        self.augmentation = augmentation
        self.standardization = standardization
        if self.standardization:
            mean = 0.3091
            std = 0.1852
            self.standardization_a = A.Normalize(mean=mean, std=std)

        if self.augmentation:
            self.transform_a = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(scale_limit=(0, 0.5), rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
                A.PadIfNeeded(min_height=self.height, min_width=self.width, always_apply=True, border_mode=0),
                A.RandomCrop(height=self.height, width=self.width, always_apply=True),
                A.GaussNoise(p=0.2),
                A.Perspective(p=0.2),
                A.RandomBrightnessContrast(p=0.5),
                A.OneOf(
                    [
                        A.Sharpen(p=1),
                        A.Blur(blur_limit=3, p=1),
                        A.MotionBlur(blur_limit=3, p=1),
                    ],
                    p=0.5,
                )
            ])

            self.transform_a_random_crop = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(scale_limit=(0, 0.5), rotate_limit=0, shift_limit=0, p=0.5, border_mode=0),
                A.PadIfNeeded(min_height=self.height, min_width=self.width, always_apply=True, border_mode=0),
                A.RandomCrop(height=self.height, width=self.width, always_apply=True),
                A.GaussNoise(p=0.2),
                A.Perspective(p=0.2),
                A.RandomBrightnessContrast(p=0.5),
                A.OneOf(
                    [
                        A.Sharpen(p=1),
                        A.Blur(blur_limit=3, p=1),
                        A.MotionBlur(blur_limit=3, p=1),
                    ],
                    p=0.5,
                )
            ])

        self.transform_a_center_crop = A.Compose([
            A.CenterCrop(height=self.height, width=self.width, always_apply=True),
        ])

        self.semseg_num_classes = semseg_num_classes
        self.require_paired_data = False

    def __len__(self):
        return len(self.cityscapes_dataset)

    def __getitem__(self, idx):
        img, label = self.cityscapes_dataset[idx]
        img = np.array(img)
        label = label.resize((self.width_resize, self.height_resize), Image.NEAREST)
        label = np.array(label)

        if self.standardization:
            Imin = np.min(img)
            Imax = np.max(img)
            img = 255.0 * (img - Imin) / (Imax - Imin)
            img = img.astype('uint8')

        if self.random_crop:
            img = img[:self.height, :]
            label = label[:self.height, :]

            if self.augmentation:
                sample = self.transform_a_random_crop(image=img, mask=label)
            else:
                sample = self.transform_a_center_crop(image=img, mask=label)
            img, label = sample["image"], sample['mask']

        else:
            if self.augmentation:
                sample = self.transform_a(image=img, mask=label)
                img, label = sample["image"], sample['mask']

        img = Image.fromarray(img.astype('uint8'))

        if self.semseg_num_classes == 6:
            label = fromIdToTrainId(label, Id2label_6_Cityscapes)
        elif self.semseg_num_classes == 11:
            label = fromIdToTrainId(label, Id2label_11_Cityscapes)

        label_tensor = torch.from_numpy(label).long()
        img_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        img_tensor = img_transform(img)

        return img_tensor, label_tensor

