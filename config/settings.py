import os
import time
import yaml
import torch
import shutil
import numpy as np
import argparse
from e2vid.options.inference_options import set_inference_options


class Settings:
    def __init__(self, settings_yaml, generate_log=True):
        assert os.path.isfile(settings_yaml), settings_yaml

        with open(settings_yaml, 'r') as stream:
            settings = yaml.load(stream, yaml.Loader)

            # --- hardware ---
            hardware = settings['hardware']
            gpu_device = hardware['gpu_device']

            self.gpu_device = torch.device("cpu") if gpu_device == "cpu" else torch.device("cuda:" + str(gpu_device))

            self.num_cpu_workers = hardware['num_cpu_workers']
            if self.num_cpu_workers < 0:
                self.num_cpu_workers = os.cpu_count()

            self.path_to_model = 'e2vid/pretrained/E2VID_lightweight.pth.tar'

            # --- Model ---
            model = settings['model']
            self.model_name = model['model_name']
            self.skip_connect_encoder = model['skip_connect_encoder']
            self.skip_connect_task = model['skip_connect_task']
            self.skip_connect_task_type = model['skip_connect_task_type']
            self.data_augmentation_train = model['data_augmentation_train']
            self.train_on_event_labels = model['train_on_event_labels']

            # --- E2VID Config ---
            parser = argparse.ArgumentParser(description='E2VID.')
            parser.add_argument('-c', '--path_to_model', default=self.path_to_model, type=str,
                                help='path to model weights')
            set_inference_options(parser)
            args, unknown = parser.parse_known_args()
            self.e2vid_config = args

            # --- dataset sensor a ---
            dataset = settings['dataset']
            self.dataset_name_a = dataset['name_a']
            self.sensor_a_name = self.dataset_name_a.split('_')[-1]
            self.split_train_a = 'train'
            self.event_representation_a = None
            self.nr_events_window_a = None
            self.nr_temporal_bins_a = None
            self.require_paired_data_train_a = False
            self.require_paired_data_val_a = False
            self.input_channels_a_paired = None
            self.read_two_imgs_a = None
            self.extension_dataset_path_a = None

            if self.dataset_name_a in ['EventScape_rgb', 'EventScape_gray', 'EventScape_recurrent_gray',
                                       'EventScape_recurrent_rgb','Cityscapes_gray', 'DDD17_gray', 'DDD17_Cityscapes_gray']:
                self.input_channels_a = 1
                if self.dataset_name_a in ['Cityscapes_gray', 'DDD17_Cityscapes_gray']:
                    dataset_specs = dataset['cityscapes_img']
                    self.random_crop_a = dataset_specs['random_crop']
                elif self.dataset_name_a == 'DDD17_gray':
                    dataset_specs = dataset['DDD17_img']
                    self.split_train_a = dataset_specs['split_train']
                else:
                    dataset_specs = dataset['eventscape_img']

                if 'EventScape' in self.dataset_name_a:
                    self.towns_a = dataset_specs['towns']
                    self.read_two_imgs_a = dataset_specs['read_two_imgs']
                    self.require_paired_data_train_a = dataset_specs['require_paired_data_train']
                    self.require_paired_data_val_a = dataset_specs['require_paired_data_val']
                    if self.dataset_name_a == 'EventScape_rgb':
                        if self.read_two_imgs_a:
                            self.input_channels_a = 6
                        else:
                            self.input_channels_a = 3
                    else:
                        if self.read_two_imgs_a:
                            self.input_channels_a = 2
                        else:
                            self.input_channels_a = 1
                    self.nr_events_data_a = dataset_specs['nr_events_data']
                    self.nr_events_files_a = dataset_specs['nr_events_files_per_data']
                    self.event_representation_a = dataset_specs['event_representation']
                    self.nr_events_window_a = dataset_specs['nr_events_window']
                    self.nr_temporal_bins_a = dataset_specs['nr_temporal_bins']
                    if self.event_representation_a == 'voxel_grid':
                        self.separate_pol_a = dataset_specs['separate_pol']
                        self.input_channels_a_paired = dataset_specs['nr_temporal_bins']
                        if self.separate_pol_a:
                            self.input_channels_a_paired = dataset_specs['nr_temporal_bins'] * 2
                        self.normalize_event_a = dataset_specs['normalize_event']
                    else:
                        self.input_channels_a_paired = 2
            else:
                raise ValueError("Specified Dataset Sensor A: %s is not implemented" % self.dataset_name_a)

            self.img_size_a = dataset_specs['shape']
            self.dataset_path_a = dataset_specs['dataset_path']
            if self.dataset_name_a == 'DDD17_Cityscapes_gray':
                self.dataset_path_a_add = dataset['DDD17_img']['dataset_path']
                assert os.path.isdir(self.dataset_path_a_add)
            assert os.path.isdir(self.dataset_path_a)

            # --- dataset sensor b ---
            dataset = settings['dataset']
            self.dataset_name_b = dataset['name_b']
            self.sensor_b_name = self.dataset_name_b.split('_')[-1]
            self.split_train_b = 'train'
            self.event_representation_b = None
            self.nr_events_window_b = None
            self.nr_temporal_bins_b = None
            self.separate_pol_b = False
            self.normalize_event_b = False
            self.require_paired_data_train_b = False
            self.require_paired_data_val_b = False
            self.input_channels_b_paired = None
            self.read_two_imgs_b = None
            self.extension_dataset_path_b = None

            if self.dataset_name_b in ['EventScape_recurrent_events', 'DSEC_events', 'DDD17_events', 'E2VIDDriving_events']:
                if self.dataset_name_b == 'DSEC_events':
                    dataset_specs = dataset['DSEC_events']
                    self.delta_t_per_data_b = dataset_specs['delta_t_per_data']
                    self.semseg_label_train_b = False
                    self.semseg_label_val_b = True
                elif self.dataset_name_b == 'E2VIDDriving_events':
                    dataset_specs = dataset['E2VIDDriving_events']
                    self.semseg_label_train_b = False
                    self.semseg_label_val_b = False
                else:
                    if self.dataset_name_b == 'DDD17_events':
                        dataset_specs = dataset['DDD17_events']
                        self.split_train_b = dataset_specs['split_train']
                        self.delta_t_per_data_b = dataset_specs['delta_t_per_data']
                    else:
                        dataset_specs = dataset['eventscape_events']
                        self.nr_events_files_b = dataset_specs['nr_events_files_per_data']
                    self.semseg_label_train_b = True
                    self.semseg_label_val_b = True
                self.fixed_duration_b = dataset_specs['fixed_duration']
                self.nr_events_data_b = dataset_specs['nr_events_data']
                self.event_representation_b = dataset_specs['event_representation']
                self.nr_events_window_b = dataset_specs['nr_events_window']
                self.nr_temporal_bins_b = dataset_specs['nr_temporal_bins']
                if self.event_representation_b == 'voxel_grid':
                    self.separate_pol_b = dataset_specs['separate_pol']
                    self.input_channels_b = dataset_specs['nr_temporal_bins']
                    if self.separate_pol_b:
                        self.input_channels_b = dataset_specs['nr_temporal_bins'] * 2
                elif self.event_representation_b == 'ev_segnet':
                    self.input_channels_b = 6
                else:
                    self.input_channels_b = 2
                self.normalize_event_b = dataset_specs['normalize_event']
                self.require_paired_data_train_b = dataset_specs['require_paired_data_train']
                self.require_paired_data_val_b = dataset_specs['require_paired_data_val']
                if self.require_paired_data_train_b or self.require_paired_data_val_b:
                    self.input_channels_b_paired = 3
            else:
                raise ValueError("Specified Dataset Sensor B: %s is not implemented" % self.dataset_name_b)

            if 'EventScape' in self.dataset_name_b:
                self.towns_b = dataset_specs['towns']
            self.img_size_b = dataset_specs['shape']
            self.dataset_path_b = dataset_specs['dataset_path']
            assert os.path.isdir(self.dataset_path_b)


            # --- Task ---
            task = settings['task']
            self.semseg_num_classes = task['semseg_num_classes']
            if self.semseg_num_classes == 6:
                self.semseg_ignore_label = 255
                self.semseg_class_names = ['flat', 'background', 'object', 'vegetation', 'human', 'vehicle']
                self.semseg_color_map = np.zeros((self.semseg_num_classes, 3), dtype=np.uint8)
                self.semseg_color_map[0] = [128, 64,128]
                self.semseg_color_map[1] = [70, 70, 70]
                self.semseg_color_map[2] = [220,220,  0]
                self.semseg_color_map[3] = [107,142, 35]
                self.semseg_color_map[4] = [220, 20, 60]
                self.semseg_color_map[5] = [  0,  0,142]

            elif self.semseg_num_classes == 11:
                self.semseg_ignore_label = 255
                self.semseg_class_names = ['background', 'building','fence','person','pole','road',
                                           'sidewalk','vegetation','car','wall','traffic sign']
                self.semseg_color_map = np.zeros((self.semseg_num_classes, 3), dtype=np.uint8)
                self.semseg_color_map[0] = [0, 0, 0]
                self.semseg_color_map[1] = [70, 70, 70]
                self.semseg_color_map[2] = [190, 153, 153]
                self.semseg_color_map[3] = [220, 20, 60]
                self.semseg_color_map[4] = [153, 153, 153]
                self.semseg_color_map[5] = [128, 64, 128]
                self.semseg_color_map[6] = [244, 35, 232]
                self.semseg_color_map[7] = [107, 142, 35]
                self.semseg_color_map[8] = [0, 0, 142]
                self.semseg_color_map[9] = [102, 102, 156]
                self.semseg_color_map[10] = [220, 220, 0]

            # --- checkpoint ---
            checkpoint = settings['checkpoint']
            self.save_checkpoint = checkpoint['save_checkpoint']
            self.resume_training = checkpoint['resume_training']
            assert isinstance(self.resume_training, bool)
            self.load_pretrained_weights = checkpoint['load_pretrained_weights']
            self.resume_ckpt_file = checkpoint['resume_file']
            self.pretrained_file = checkpoint['pretrained_file']

            # --- directories ---
            directories = settings['dir']
            log_dir = directories['log']

            # --- logs ---
            if generate_log:
                timestr = time.strftime("%Y%m%d-%H%M%S")
                self.timestr = timestr
                log_dir = os.path.join(log_dir, timestr)
                os.makedirs(log_dir)
                settings_copy_filepath = os.path.join(log_dir, os.path.split(settings_yaml)[-1])
                shutil.copyfile(settings_yaml, settings_copy_filepath)
                self.ckpt_dir = os.path.join(log_dir, 'checkpoints')
                os.mkdir(self.ckpt_dir)
                self.vis_dir = os.path.join(log_dir, 'visualization')
                os.mkdir(self.vis_dir)
            else:
                self.ckpt_dir = os.path.join(log_dir, 'checkpoints')
                self.vis_dir = os.path.join(log_dir, 'visualization')

            # --- optimization ---
            optimization = settings['optim']
            self.batch_size_a = int(optimization['batch_size_a'])
            self.batch_size_b = int(optimization['batch_size_b'])
            self.lr_front = float(optimization['lr_front'])
            self.lr_back = float(optimization['lr_back'])
            self.lr_decay = float(optimization['lr_decay'])
            self.num_epochs = int(optimization['num_epochs'])
            self.val_epoch_step = int(optimization['val_epoch_step'])
            self.weight_task_loss = float(optimization['weight_task_loss'])
            self.weight_KL_loss = float(optimization['weight_cycle_pred_loss'])
            self.weight_cycle_loss = float(optimization['weight_cycle_emb_loss'])
            self.weight_cycle_task_loss = float(optimization['weight_cycle_task_loss'])
            self.task_loss = optimization['task_loss']
