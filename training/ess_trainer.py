import torch
import torchvision
import torch.nn.functional as f

import math
from tqdm import tqdm

from utils import radam
import utils.viz_utils as viz_utils
from models.style_networks import StyleEncoderE2VID, SemSegE2VID
import training.base_trainer
from evaluation.metrics import MetricsSemseg
from utils.loss_functions import TaskLoss, symJSDivLoss
from utils.viz_utils import plot_confusion_matrix

from e2vid.utils.loading_utils import load_model
from e2vid.image_reconstructor import ImageReconstructor

class ESSModel(training.base_trainer.BaseTrainer):
    def __init__(self, settings, train=True):
        self.is_training = train
        super(ESSModel, self).__init__(settings)
        self.do_val_training_epoch = False

    def init_fn(self):
        self.buildModels()
        self.createOptimizerDict()

        self.cycle_content_loss = torch.nn.L1Loss()
        self.cycle_pred_loss = symJSDivLoss()

        # Task Loss
        self.task_loss = TaskLoss(losses=self.settings.task_loss, gamma=2.0, num_classes=self.settings.semseg_num_classes,
                                  ignore_index=self.settings.semseg_ignore_label, reduction='mean')
        self.train_statistics = {}

        self.metrics_semseg_a = MetricsSemseg(self.settings.semseg_num_classes, self.settings.semseg_ignore_label,
                                              self.settings.semseg_class_names)
        if self.settings.semseg_label_val_b:
            self.metrics_semseg_b = MetricsSemseg(self.settings.semseg_num_classes, self.settings.semseg_ignore_label,
                                                  self.settings.semseg_class_names)
            self.metrics_semseg_cycle = MetricsSemseg(self.settings.semseg_num_classes, self.settings.semseg_ignore_label,
                                                      self.settings.semseg_class_names)

    def buildModels(self):
        # Front End Sensor A
        self.front_end_sensor_a = StyleEncoderE2VID(self.settings.input_channels_a,
                                                    skip_connect=self.settings.skip_connect_encoder)

        # Front End Sensor B
        self.front_end_sensor_b, self.e2vid_decoder = load_model(self.settings.path_to_model)
        for param in self.front_end_sensor_b.parameters():
            param.requires_grad = False
        self.front_end_sensor_b.eval()

        self.input_height = math.ceil(self.settings.img_size_b[0] / 8.0) * 8
        self.input_width = math.ceil(self.settings.img_size_b[1] / 8.0) * 8
        if self.settings.dataset_name_b == 'DDD17_events':
            self.input_height = 120
            self.input_width = 216
        self.input_height_valid = self.input_height
        self.input_width_valid = self.input_width
        self.reconstructor = ImageReconstructor(self.front_end_sensor_b, self.input_height, self.input_width,
                                                self.settings.nr_temporal_bins_b, self.settings.gpu_device,
                                                self.settings.e2vid_config)
        self.reconstructor_valid = self.reconstructor
        if self.settings.dataset_name_b == 'DDD17_events':
            self.input_height_valid = 200
            self.input_width_valid = 352
            self.reconstructor_valid = ImageReconstructor(self.front_end_sensor_b, self.input_height_valid, self.input_width_valid,
                                                    self.settings.nr_temporal_bins_b, self.settings.gpu_device,
                                                    self.settings.e2vid_config)

        self.models_dict = {"front_sensor_a": self.front_end_sensor_a,
                            "front_sensor_b": self.front_end_sensor_b}

        # Task Backend
        self.task_backend = SemSegE2VID(input_c=256, output_c=self.settings.semseg_num_classes,
                                        skip_connect=self.settings.skip_connect_task,
                                        skip_type=self.settings.skip_connect_task_type)
        self.models_dict["back_end"] = self.task_backend

    def createOptimizerDict(self):
        """Creates the dictionary containing the optimizer for the the specified subnetworks"""
        if not self.is_training:
            self.optimizers_dict = {}
            return
        front_sensor_a_params = filter(lambda p: p.requires_grad, self.front_end_sensor_a.parameters())
        optimizer_front_sensor_a = radam.RAdam(front_sensor_a_params,
                                               lr=self.settings.lr_front,
                                               weight_decay=0.,
                                               betas=(0., 0.999))
        self.optimizers_dict = {"optimizer_front_sensor_a": optimizer_front_sensor_a}

        # Task
        back_params = filter(lambda p: p.requires_grad, self.task_backend.parameters())
        optimizer_back = radam.RAdam(back_params,
                                     lr=self.settings.lr_back,
                                     weight_decay=0.,
                                     betas=(0., 0.999))
        self.optimizers_dict["optimizer_back"] = optimizer_back

    def train_step(self, input_batch):
        final_loss = 0.
        losses = {}
        outputs = {}

        # Task Step
        optimizers_list = ['optimizer_back']
        optimizers_list.append('optimizer_front_sensor_a')

        for key_word in optimizers_list:
            optimizer_key_word = self.optimizers_dict[key_word]
            optimizer_key_word.zero_grad()

        # training on images
        t_final_loss, t_losses, t_outputs = self.img_train_step(input_batch)
        if self.settings.dataset_name_b == 'DDD17_events':
            t_final_loss.backward()
        elif self.settings.dataset_name_b == 'DSEC_events':
            for p in self.models_dict['front_sensor_a'].parameters():
                p.requires_grad = False
            t_final_loss.backward()
            for p in self.models_dict['front_sensor_a'].parameters():
                p.requires_grad = True

        final_loss += t_final_loss
        losses.update(t_losses)
        outputs.update(t_outputs)

        # training on events
        e_event_final_loss, t_event_final_loss, event_losses, event_outputs = self.event_train_step(input_batch)
        for p in self.models_dict['back_end'].parameters():
            p.requires_grad = False
        e_event_final_loss.backward()
        for p in self.models_dict['back_end'].parameters():
            p.requires_grad = True
        t_event_final_loss.backward()
        final_loss += e_event_final_loss
        final_loss += t_event_final_loss
        losses.update(event_losses)
        outputs.update(event_outputs)

        for key_word in optimizers_list:
            optimizer_key_word = self.optimizers_dict[key_word]
            optimizer_key_word.step()

        return losses, outputs, final_loss

    def img_train_step(self, batch):
        data_a = batch[0][0]

        if self.settings.require_paired_data_train_a:
            labels_a = batch[0][2]
        else:
            labels_a = batch[0][1]

        # Set BatchNorm Statistics
        for model in self.models_dict:
            self.models_dict[model].train()
            if model in ['front_sensor_b', 'e2vid_decoder']:
                self.models_dict[model].eval()
        for p in self.models_dict['back_end'].parameters():
            p.requires_grad = True

        gen_model_sensor_a = self.models_dict['front_sensor_a']

        losses = {}
        out = {}
        t = 0.

        latent_fake = gen_model_sensor_a(data_a)

        t_loss, pred_a = self.trainTaskStep('sensor_a', latent_fake, labels_a, losses)
        t += t_loss

        if self.visualize_epoch():
            self.visTaskStep(data_a, pred_a, labels_a)

        return t, losses, out

    def trainTaskStep(self, sensor_name, latent_fake, labels, losses):
        content_features = {}
        for key in latent_fake.keys():
            if self.settings.dataset_name_b == 'DDD17_events':
                content_features[key] = latent_fake[key]
            elif self.settings.dataset_name_b == 'DSEC_events':
                content_features[key] = latent_fake[key].detach()
        task_backend = self.models_dict["back_end"]
        pred = task_backend(content_features)
        loss_pred = self.task_loss(pred[1], labels) * self.settings.weight_task_loss
        losses['semseg_' + sensor_name + '_loss'] = loss_pred.detach()

        return loss_pred, pred

    def visTaskStep(self, img, pred_img, labels):
        pred_img = pred_img[1]
        pred_img_lbl = pred_img.argmax(dim=1)

        semseg_img = viz_utils.prepare_semseg(pred_img_lbl, self.settings.semseg_color_map,
                                              self.settings.semseg_ignore_label)
        semseg_gt = viz_utils.prepare_semseg(labels, self.settings.semseg_color_map, self.settings.semseg_ignore_label)

        nrow = 4
        viz_tensors = torch.cat((viz_utils.createRGBImage(img[:nrow]),
                                 viz_utils.createRGBImage(semseg_img[:nrow].to(self.device)),
                                 viz_utils.createRGBImage(semseg_gt[:nrow].to(self.device))), dim=0)
        rgb_grid = torchvision.utils.make_grid(viz_tensors, nrow=nrow)
        self.img_summaries('train/semseg_img', rgb_grid, self.step_count)

    def trainCycleStep(self, first_sensor_name, second_sensor_name, content_first_sensor, content_second_sensor,
                       losses):
        g_loss = 0.
        cycle_name = first_sensor_name + '_to_' + second_sensor_name
        # latent_feature
        if self.settings.skip_connect_encoder:
            cycle_latent_loss_2x = self.cycle_content_loss(content_second_sensor[2], content_first_sensor[2]) * \
                                   self.settings.weight_cycle_loss
            g_loss += cycle_latent_loss_2x
            losses['cycle_latent_2x_' + cycle_name + '_loss'] = cycle_latent_loss_2x.cpu().detach()
            cycle_latent_loss_4x = self.cycle_content_loss(content_second_sensor[4], content_first_sensor[4]) * \
                                   self.settings.weight_cycle_loss
            g_loss += cycle_latent_loss_4x
            losses['cycle_latent_4x_' + cycle_name + '_loss'] = cycle_latent_loss_4x.cpu().detach()

        cycle_latent_loss_8x = self.cycle_content_loss(content_second_sensor[8], content_first_sensor[8]) * \
                               self.settings.weight_cycle_loss
        g_loss += cycle_latent_loss_8x
        losses['cycle_latent_8x_' + cycle_name + '_loss'] = cycle_latent_loss_8x.cpu().detach()

        task_backend = self.models_dict["back_end"]

        pred_second_sensor = task_backend(content_second_sensor)
        with torch.no_grad():
            pred_first_sensor_no_grad = task_backend(content_first_sensor)

        cycle_pred_loss_1x_events = self.cycle_pred_loss(pred_second_sensor[1], pred_first_sensor_no_grad[1])
        losses['cycle_pred_1x_' + cycle_name + '_loss'] = cycle_pred_loss_1x_events.cpu().detach()
        cycle_pred_loss_1x = cycle_pred_loss_1x_events
        if self.settings.dataset_name_b == 'DSEC_events':
            g_loss += cycle_pred_loss_1x

        cycle_pred_loss_2x_events = self.cycle_content_loss(pred_second_sensor[2], pred_first_sensor_no_grad[2]) * \
                                    self.settings.weight_cycle_task_loss
        cycle_pred_loss_2x = cycle_pred_loss_2x_events
        g_loss += cycle_pred_loss_2x
        losses['cycle_pred_2x_' + cycle_name + '_loss'] = cycle_pred_loss_2x.cpu().detach()

        cycle_pred_loss_4x_events = self.cycle_content_loss(pred_second_sensor[4], pred_first_sensor_no_grad[4]) * \
                                    self.settings.weight_cycle_task_loss
        cycle_pred_loss_4x = cycle_pred_loss_4x_events
        g_loss += cycle_pred_loss_4x
        losses['cycle_pred_4x_' + cycle_name + '_loss'] = cycle_pred_loss_4x.cpu().detach()

        return g_loss, pred_first_sensor_no_grad, pred_second_sensor

    def event_train_step(self, batch):
        data_b = batch[1][0]
        if self.settings.require_paired_data_train_b:
            labels_b = batch[1][2]
        else:
            labels_b = batch[1][1]

        # Set BatchNorm Statistics
        for model in self.models_dict:
            self.models_dict[model].train()
            if model in ['front_sensor_b', 'e2vid_decoder', "back_end"]:
                self.models_dict[model].eval()

        gen_model_sensor_a = self.models_dict['front_sensor_a']
        self.reconstructor.last_states_for_each_channel = {'grayscale': None}
        e_loss = 0.
        losses = {}
        out = {}

        # Train img encoder.
        with torch.no_grad():
            for i in range(self.settings.nr_events_data_b):
                event_tensor = data_b[:, i * self.settings.input_channels_b:(i + 1) * self.settings.input_channels_b, :, :]
                img_fake, states_real, latent_real = self.reconstructor.update_reconstruction(event_tensor)

        latent_fake = gen_model_sensor_a(img_fake.detach())

        for key in latent_real.keys():
            latent_real[key] = latent_real[key].detach()

        cycle_loss, pred_b, pred_a = self.trainCycleStep('sensor_b', 'sensor_a', latent_real, latent_fake, losses)
        e_loss += cycle_loss

        if self.visualize_epoch():
            self.visCycleStep(data_b, img_fake, pred_b, pred_a, labels_b)

        # Train task network.
        self.models_dict['back_end'].train()
        t_loss = 0.
        t_loss += self.TasktrainCycleStep('sensor_b', 'sensor_a', latent_real, latent_fake, losses)
        if self.settings.train_on_event_labels:
            t_loss_b, _ = self.trainTaskStep('sensor_b', latent_real, labels_b, losses)
            t_loss += t_loss_b

        return e_loss, t_loss, losses, out

    def TasktrainCycleStep(self, first_sensor_name, second_sensor_name, content_first_sensor, content_second_sensor,
                           losses):
        t_loss = 0.
        cycle_name = first_sensor_name + '_to_' + second_sensor_name

        task_backend = self.models_dict["back_end"]
        pred_first_sensor = task_backend(content_first_sensor)

        with torch.no_grad():
            pred_second_sensor_no_grad = task_backend(content_second_sensor)

        cycle_pred_loss_1x_events = self.cycle_pred_loss(pred_first_sensor[1], pred_second_sensor_no_grad[1]) * \
                                    self.settings.weight_KL_loss

        cycle_pred_loss_1x = cycle_pred_loss_1x_events
        t_loss += cycle_pred_loss_1x

        cycle_pred_loss_2x_events = self.cycle_content_loss(pred_first_sensor[2], pred_second_sensor_no_grad[2]) * \
                                    self.settings.weight_cycle_task_loss
        cycle_pred_loss_2x = cycle_pred_loss_2x_events
        t_loss += cycle_pred_loss_2x

        cycle_pred_loss_4x_events = self.cycle_content_loss(pred_first_sensor[4], pred_second_sensor_no_grad[4]) * \
                                    self.settings.weight_cycle_task_loss
        cycle_pred_loss_4x = cycle_pred_loss_4x_events
        t_loss += cycle_pred_loss_4x

        return t_loss

    def visCycleStep(self, events_real, img_fake, pred_events, pred_img, labels):
        pred_events = pred_events[1]
        pred_img = pred_img[1]
        pred_events_lbl = pred_events.argmax(dim=1)
        pred_img_lbl = pred_img.argmax(dim=1)

        semseg_events = viz_utils.prepare_semseg(pred_events_lbl, self.settings.semseg_color_map,
                                                 self.settings.semseg_ignore_label)
        semseg_img = viz_utils.prepare_semseg(pred_img_lbl, self.settings.semseg_color_map,
                                              self.settings.semseg_ignore_label)
        if self.settings.semseg_label_train_b:
            semseg_gt = viz_utils.prepare_semseg(labels, self.settings.semseg_color_map, self.settings.semseg_ignore_label)

            nrow = 4
            viz_tensors = torch.cat((viz_utils.createRGBImage(events_real[:nrow],
                                                              separate_pol=self.settings.separate_pol_b).clamp(0, 1).to(
                self.device),
                                     viz_utils.createRGBImage(img_fake[:nrow]),
                                     viz_utils.createRGBImage(semseg_events[:nrow].to(self.device)),
                                     viz_utils.createRGBImage(semseg_img[:nrow].to(self.device)),
                                     viz_utils.createRGBImage(semseg_gt[:nrow].to(self.device))), dim=0)
        else:
            nrow = 4
            viz_tensors = torch.cat((viz_utils.createRGBImage(events_real[:nrow],
                                                              separate_pol=self.settings.separate_pol_b).clamp(0, 1).to(
                self.device),
                                     viz_utils.createRGBImage(img_fake[:nrow]),
                                     viz_utils.createRGBImage(semseg_events[:nrow].to(self.device)),
                                     viz_utils.createRGBImage(semseg_img[:nrow].to(self.device))), dim=0)
        rgb_grid = torchvision.utils.make_grid(viz_tensors, nrow=nrow)
        self.img_summaries('train/semseg_cycle', rgb_grid, self.step_count)

    def validationEpoch(self, data_loader, sensor_name):
        val_dataset_length = data_loader.__len__()
        self.pbar = tqdm(total=val_dataset_length, unit='Batch', unit_scale=True)
        tqdm.write("Validation on " + sensor_name)
        cumulative_losses = {}
        total_nr_steps = None

        for i_batch, sample_batched in enumerate(data_loader):
            for i in range(len(sample_batched)):
                sample_batched[i] = sample_batched[i].to(self.device)
            self.validationBatchStep(sample_batched, sensor_name, i_batch, cumulative_losses, val_dataset_length)
            self.pbar.update(1)
            total_nr_steps = i_batch

        if sensor_name == 'sensor_a':
            metrics_semseg_a = self.metrics_semseg_a.get_metrics_summary()
            metric_semseg_a_mean_iou = metrics_semseg_a['mean_iou']
            cumulative_losses['semseg_sensor_a_mean_iou'] = metric_semseg_a_mean_iou
            metric_semseg_a_acc = metrics_semseg_a['acc']
            cumulative_losses['semseg_sensor_a_acc'] = metric_semseg_a_acc
            metrics_semseg_a_cm = metrics_semseg_a['cm']
            figure_semseg_a_cm = plot_confusion_matrix(metrics_semseg_a_cm, classes=self.settings.semseg_class_names,
                                                       normalize=True,
                                                       title='Normalized confusion matrix')
            self.summary_writer.add_figure('val_gray/semseg_cm',
                                           figure_semseg_a_cm, self.epoch_count)
        else:
            if self.settings.semseg_label_val_b:
                metrics_semseg_b = self.metrics_semseg_b.get_metrics_summary()
                metric_semseg_b_mean_iou = metrics_semseg_b['mean_iou']
                cumulative_losses['semseg_sensor_b_mean_iou'] = metric_semseg_b_mean_iou
                metric_semseg_b_acc = metrics_semseg_b['acc']
                cumulative_losses['semseg_sensor_b_acc'] = metric_semseg_b_acc
                metrics_semseg_b_cm = metrics_semseg_b['cm']
                figure_semseg_b_cm = plot_confusion_matrix(metrics_semseg_b_cm, classes=self.settings.semseg_class_names,
                                                           normalize=True,
                                                           title='Normalized confusion matrix')
                self.summary_writer.add_figure('val_events/semseg_cm',
                                               figure_semseg_b_cm, self.epoch_count)

                metrics_semseg_cycle = self.metrics_semseg_cycle.get_metrics_summary()
                metric_semseg_cycle_mean_iou = metrics_semseg_cycle['mean_iou']
                cumulative_losses['semseg_sensor_cycle_mean_iou'] = metric_semseg_cycle_mean_iou
                metric_semseg_cycle_acc = metrics_semseg_cycle['acc']
                cumulative_losses['semseg_sensor_cycle_acc'] = metric_semseg_cycle_acc
                metrics_semseg_cycle_cm = metrics_semseg_cycle['cm']
                figure_semseg_cycle_cm = plot_confusion_matrix(metrics_semseg_cycle_cm,
                                                               classes=self.settings.semseg_class_names,
                                                               normalize=True,
                                                               title='Normalized confusion matrix')
                self.summary_writer.add_figure('val_events/cycle_semseg_cm',
                                               figure_semseg_cycle_cm, self.epoch_count)

        self.val_summaries(cumulative_losses, total_nr_steps + 1)
        self.pbar.close()
        if self.val_confusion_matrix.sum() != 0:
            self.addValidationMatrix(sensor_name)

        self.saveValStatistics('val', sensor_name)

    def val_step(self, input_batch, sensor, i_batch, vis_reconstr_idx):
        """Calculates the performance measurements based on the input"""
        data = input_batch[0]
        paired_data = None
        if sensor == 'sensor_a':
            if self.settings.require_paired_data_val_a:
                paired_data = input_batch[1]
                labels = input_batch[2]
            else:
                labels = input_batch[1]
        else:
            if self.settings.require_paired_data_val_b:
                paired_data = input_batch[1]
                if self.settings.dataset_name_b == 'DDD17_events':
                    labels = input_batch[3]
                else:
                    labels = input_batch[2]
            else:
                labels = input_batch[1]

        gen_model = self.models_dict['front_' + sensor]

        losses = {}
        second_sensor = 'sensor_b'

        if sensor == 'sensor_a':
            content_first_sensor = gen_model(data)

        else:
            second_sensor = 'sensor_a'
            self.reconstructor_valid.last_states_for_each_channel = {'grayscale': None}
            for i in range(self.settings.nr_events_data_b):
                event_tensor = data[:, i * self.settings.input_channels_b:(i + 1) * self.settings.input_channels_b, :,
                               :]
                img_fake, _, content_first_sensor = self.reconstructor_valid.update_reconstruction(event_tensor)

        preds_first_sensor = self.valTaskStep(content_first_sensor, labels, losses, sensor)

        if sensor == 'sensor_b':
            preds_second_sensor = self.valCycleStep(content_first_sensor, img_fake, labels, losses, sensor,
                                                    second_sensor, preds_first_sensor)

        if vis_reconstr_idx != -1:
            if sensor == 'sensor_a':
                self.visualizeSensorA(data, labels, preds_first_sensor, vis_reconstr_idx, sensor)
            else:
                self.visualizeSensorB(data[:, -self.settings.input_channels_b:, :, :], preds_first_sensor,
                                              preds_second_sensor,
                                              labels, img_fake, paired_data,
                                              vis_reconstr_idx, sensor)
        return losses, None

    def valTaskStep(self, content_first_sensor, labels, losses, sensor):
        """Computes the task loss and updates metrics"""
        task_backend = self.models_dict["back_end"]
        preds = task_backend(content_first_sensor)

        if sensor == 'sensor_a' or self.settings.semseg_label_val_b:
            pred = preds[1]
            if sensor == 'sensor_b':
                pred = f.interpolate(pred, size=(self.settings.img_size_b), mode='nearest')
            pred_lbl = pred.argmax(dim=1)

            loss_pred = self.task_loss(pred, target=labels) * self.settings.weight_task_loss
            losses['semseg_' + sensor + '_loss'] = loss_pred.detach()
            if sensor == 'sensor_a':
                self.metrics_semseg_a.update_batch(pred_lbl, labels)
            else:
                self.metrics_semseg_b.update_batch(pred_lbl, labels)
        return preds

    def valCycleStep(self, content_first_sensor, img_fake, labels, losses, sensor, second_sensor,
                     preds_first_sensor):
        """Computes the cycle loss"""
        gen_second_sensor_model = self.models_dict['front_' + second_sensor]
        content_second_sensor = gen_second_sensor_model(img_fake)

        # latent_feature
        cycle_name = sensor + '_to_' + second_sensor
        if self.settings.skip_connect_encoder:
            cycle_latent_loss_2x = self.cycle_content_loss(content_first_sensor[2], content_second_sensor[2]) * \
                                   self.settings.weight_cycle_loss
            losses['cycle_latent_2x_' + cycle_name + '_loss'] = cycle_latent_loss_2x.cpu().detach()
            cycle_latent_loss_4x = self.cycle_content_loss(content_first_sensor[4], content_second_sensor[4]) * \
                                   self.settings.weight_cycle_loss
            losses['cycle_latent_4x_' + cycle_name + '_loss'] = cycle_latent_loss_4x.cpu().detach()
        cycle_latent_loss_8x = self.cycle_content_loss(content_first_sensor[8], content_second_sensor[8]) * \
                               self.settings.weight_cycle_loss
        losses['cycle_latent_8x_' + cycle_name + '_loss'] = cycle_latent_loss_8x.cpu().detach()

        preds_second_sensor = self.valCycleTask(content_second_sensor, labels, losses, cycle_name,
                                                    preds_first_sensor)

        return preds_second_sensor

    def valCycleTask(self, cycle_content_first_second, labels, losses, cycle_name, preds_first_sensor):
        """Computes the task performance of the E2VID reconstruction"""
        task_backend = self.models_dict["back_end"]
        preds_second_sensor = task_backend(cycle_content_first_second)
        if self.settings.semseg_label_val_b:
            pred_second_sensor = preds_second_sensor[1]
            pred_second_sensor = f.interpolate(pred_second_sensor, size=(self.settings.img_size_b), mode='nearest')
            pred_second_sensor_lbl = pred_second_sensor.argmax(dim=1)
            loss_pred = self.task_loss(pred_second_sensor, target=labels) * self.settings.weight_task_loss

            losses['semseg_' + cycle_name + '_loss'] = loss_pred.detach()
            self.metrics_semseg_cycle.update_batch(pred_second_sensor_lbl, labels)

        cycle_pred_loss_1x = self.cycle_pred_loss(preds_second_sensor[1], preds_first_sensor[1]) * \
                             self.settings.weight_KL_loss
        losses['cycle_pred_1x_' + cycle_name + '_loss'] = cycle_pred_loss_1x.cpu().detach()

        cycle_pred_loss_2x = self.cycle_content_loss(preds_first_sensor[2], preds_second_sensor[2]) * \
                             self.settings.weight_cycle_task_loss
        losses['cycle_pred_2x_' + cycle_name + '_loss'] = cycle_pred_loss_2x.cpu().detach()

        cycle_pred_loss_4x = self.cycle_content_loss(preds_first_sensor[4], preds_second_sensor[4]) * \
                             self.settings.weight_cycle_task_loss
        losses['cycle_pred_4x_' + cycle_name + '_loss'] = cycle_pred_loss_4x.cpu().detach()

        return preds_second_sensor

    def visualizeSensorA(self, data, labels, preds_first_sensor, vis_reconstr_idx, sensor):
        nrow = 4
        vis_tensors = [viz_utils.createRGBImage(data[:nrow])]

        pred = preds_first_sensor
        pred = pred[1]
        pred_lbl = pred.argmax(dim=1)

        semseg = viz_utils.prepare_semseg(pred_lbl, self.settings.semseg_color_map, self.settings.semseg_ignore_label)
        vis_tensors.append(viz_utils.createRGBImage(semseg[:nrow].to(self.device)))
        semseg_gt = viz_utils.prepare_semseg(labels, self.settings.semseg_color_map, self.settings.semseg_ignore_label)
        vis_tensors.append(viz_utils.createRGBImage(semseg_gt[:nrow]).to(self.device))

        rgb_grid = torchvision.utils.make_grid(torch.cat(vis_tensors, dim=0), nrow=nrow)
        self.img_summaries('val_' + sensor + '/reconst_input_' + sensor + '_' + str(vis_reconstr_idx),
                           rgb_grid, self.epoch_count)

    def visualizeSensorB(self, data, preds_first_sensor, preds_second_sensor, labels, img_fake, paired_data,
                                 vis_reconstr_idx, sensor):
        pred = preds_first_sensor[1]
        pred_lbl = pred.argmax(dim=1)
        pred_cycle = preds_second_sensor[1]
        pred_cycle_lbl = pred_cycle.argmax(dim=1)

        if self.settings.semseg_label_val_b:
            labels = f.interpolate(labels.float().unsqueeze(1), size=(self.input_height_valid, self.input_width_valid),
                                   mode='nearest').squeeze(1).long()

        semseg = viz_utils.prepare_semseg(pred_lbl, self.settings.semseg_color_map, self.settings.semseg_ignore_label)
        semseg_cycle = viz_utils.prepare_semseg(pred_cycle_lbl, self.settings.semseg_color_map,
                                                self.settings.semseg_ignore_label)
        if self.settings.semseg_label_val_b:
            semseg_gt = viz_utils.prepare_semseg(labels, self.settings.semseg_color_map, self.settings.semseg_ignore_label)

            nrow = 4
            vis_tensors = [
                viz_utils.createRGBImage(data[:nrow], separate_pol=self.settings.separate_pol_b).clamp(0, 1).to(
                    self.device),
                viz_utils.createRGBImage(img_fake[:nrow]).to(self.device),
                viz_utils.createRGBImage(semseg[:nrow].to(self.device)),
                viz_utils.createRGBImage(semseg_cycle[:nrow]).to(self.device),
                viz_utils.createRGBImage(semseg_gt[:nrow]).to(self.device)]

        else:
            nrow = 4
            vis_tensors = [
                viz_utils.createRGBImage(data[:nrow], separate_pol=self.settings.separate_pol_b).clamp(0, 1).to(
                    self.device),
                viz_utils.createRGBImage(img_fake[:nrow]).to(self.device),
                viz_utils.createRGBImage(semseg[:nrow].to(self.device)),
                viz_utils.createRGBImage(semseg_cycle[:nrow]).to(self.device)]

        if paired_data is not None:
            vis_tensors.append(viz_utils.createRGBImage(paired_data[:nrow]).to(self.device))

        rgb_grid = torchvision.utils.make_grid(torch.cat(vis_tensors, dim=0), nrow=nrow)
        self.img_summaries('val_' + sensor + '/reconst_input_' + sensor + '_' + str(vis_reconstr_idx),
                           rgb_grid, self.epoch_count)

    def resetValidationStatistics(self):
        self.metrics_semseg_a.reset()
        if self.settings.semseg_label_val_b:
            self.metrics_semseg_b.reset()
            self.metrics_semseg_cycle.reset()
