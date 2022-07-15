import torch
import cv2
import numpy as np
from e2vid.model.model import *
from e2vid.utils.inference_utils import CropParameters, EventPreprocessor, IntensityRescaler, ImageFilter, ImageDisplay, \
    ImageWriter, UnsharpMaskFilter
from e2vid.utils.inference_utils import upsample_color_image, \
    merge_channels_into_color_image  # for color reconstruction
from e2vid.utils.util import robust_min, robust_max
from e2vid.utils.timers import CudaTimer, cuda_timers
from os.path import join
from collections import deque
import torchvision.transforms as transforms
import albumentations as A
from PIL import Image


class ImageReconstructor:
    def __init__(self, model, height, width, num_bins, device, options, augmentation=False, standardization=False):

        self.model = model
        self.use_gpu = options.use_gpu
        self.device = device
        self.height = height
        self.width = width
        self.num_bins = num_bins

        self.standardization = standardization

        self.augmentation = augmentation
        if self.augmentation:
            self.transform_a = A.Compose([
                A.GaussNoise(p=0.2),
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
            self.img_transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor()
            ])

        self.initialize(self.height, self.width, options)

    def initialize(self, height, width, options):
        # print('== Image reconstruction == ')
        # print('Image size: {}x{}'.format(self.height, self.width))

        self.no_recurrent = options.no_recurrent
        if self.no_recurrent:
            print('!!Recurrent connection disabled!!')

        self.perform_color_reconstruction = options.color  # whether to perform color reconstruction (only use this with the DAVIS346color)
        if self.perform_color_reconstruction:
            if options.auto_hdr:
                print('!!Warning: disabling auto HDR for color reconstruction!!')
            options.auto_hdr = False  # disable auto_hdr for color reconstruction (otherwise, each channel will be normalized independently)

        self.crop = CropParameters(self.width, self.height, self.model.num_encoders)

        self.last_states_for_each_channel = {'grayscale': None}

        if self.perform_color_reconstruction:
            self.crop_halfres = CropParameters(int(width / 2), int(height / 2),
                                               self.model.num_encoders)
            for channel in ['R', 'G', 'B', 'W']:
                self.last_states_for_each_channel[channel] = None

        self.event_preprocessor = EventPreprocessor(options)
        self.intensity_rescaler = IntensityRescaler(options)
        self.image_filter = ImageFilter(options)
        self.unsharp_mask_filter = UnsharpMaskFilter(options, device=self.device)
        # self.image_writer = ImageWriter(options)
        # self.image_display = ImageDisplay(options)

    def update_reconstruction(self, event_tensor, event_tensor_id=None, stamp=None):
        with torch.no_grad():

            with CudaTimer('Reconstruction'):

                with CudaTimer('NumPy (CPU) -> Tensor (GPU)'):
                    events = event_tensor
                    events = events.to(self.device)

                events = self.event_preprocessor(events)

                # Resize tensor to [1 x C x crop_size x crop_size] by applying zero padding
                events_for_each_channel = {'grayscale': self.crop.pad(events)}
                reconstructions_for_each_channel = {}
                # if self.perform_color_reconstruction:
                #     events_for_each_channel['R'] = self.crop_halfres.pad(events[:, :, 0::2, 0::2])
                #     events_for_each_channel['G'] = self.crop_halfres.pad(events[:, :, 0::2, 1::2])
                #     events_for_each_channel['W'] = self.crop_halfres.pad(events[:, :, 1::2, 0::2])
                #     events_for_each_channel['B'] = self.crop_halfres.pad(events[:, :, 1::2, 1::2])

                # Reconstruct new intensity image for each channel (grayscale + RGBW if color reconstruction is enabled)
                for channel in events_for_each_channel.keys():
                    with CudaTimer('Inference'):
                        new_predicted_frame, states, latent = self.model(events_for_each_channel[channel],
                                                                         self.last_states_for_each_channel[channel])

                    if self.no_recurrent:
                        self.last_states_for_each_channel[channel] = None
                    else:
                        self.last_states_for_each_channel[channel] = states

                    # Output reconstructed image
                    # crop = self.crop if channel == 'grayscale' else self.crop_halfres

                    # Unsharp mask (on GPU)
                    # new_predicted_frame = self.unsharp_mask_filter(new_predicted_frame)

                    # Intensity rescaler (on GPU)
                    # new_predicted_frame = self.intensity_rescaler(new_predicted_frame)

                    with CudaTimer('Tensor (GPU) -> NumPy (CPU)'):
                        reconstructions_for_each_channel[channel] = new_predicted_frame
                        # reconstructions_for_each_channel[channel] = new_predicted_frame.cpu().numpy()

                # if self.perform_color_reconstruction:
                #     out = merge_channels_into_color_image(reconstructions_for_each_channel)
                # else:
                out = reconstructions_for_each_channel['grayscale']

                if self.standardization:
                    batch_size, height, width = out.size(0), out.size(2), out.size(3)
                    out = out.view(out.size(0), -1)
                    out -= out.min(1, keepdim=True)[0]
                    out /= out.max(1, keepdim=True)[0]
                    out = out.view(batch_size, 1, height, width)

                    # Imin = torch.min(out).item()
                    # Imax = torch.max(out).item()
                    # out = 255.0 * (out - Imin) / (Imax - Imin)
                    # out.clamp_(0.0, 255.0)
                    # out = out.byte()  # convert to 8-bit tensor
                    # out = out.float().div(255)

                    # mean = [0.5371 for i in range(out.shape[0])]
                    # std = [0.1540 for i in range(out.shape[0])]
                    # stand_transform = transforms.Normalize(mean=mean, std=std)
                    # out = stand_transform(out.squeeze(1)).unsqueeze(1)
                    # out = torch.clamp(out, min=-1.0, max=1.0)
                    # out = (out + 1.0) / 2.0

                if self.augmentation:
                    for i in range(out.shape[0]):
                        img_aug = out[i].cpu()
                        img_aug = transforms.ToPILImage()(img_aug)
                        img_aug = np.array(img_aug)
                        img_aug = self.transform_a(image=img_aug)["image"]
                        img_aug = Image.fromarray(img_aug.astype('uint8')).convert('RGB')
                        out[i] = self.img_transform(img_aug).to(self.device)
                        # Post-processing, e.g bilateral filter (on CPU)
                # out = torch.from_numpy(self.image_filter(out)).to(self.device)

        return out, states, latent


class PostProcessor:
    def __init__(self, device, options):
        self.device = device
        self.unsharp_mask_filter = UnsharpMaskFilter(options, device=self.device)
        self.intensity_rescaler = IntensityRescaler(options)
        self.image_filter = ImageFilter(options)

    def process(self, new_predicted_frame):
        with torch.no_grad():
            # Unsharp mask (on GPU)
            new_predicted_frame = self.unsharp_mask_filter(new_predicted_frame)

            # Intensity rescaler (on GPU)
            new_predicted_frame = self.intensity_rescaler(new_predicted_frame)

            out = new_predicted_frame.cpu().numpy()

            # Post-processing, e.g bilateral filter (on CPU)
            out = torch.from_numpy(self.image_filter(out)).to(self.device)
        return out
