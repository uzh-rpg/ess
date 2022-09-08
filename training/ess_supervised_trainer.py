import torch
import torchvision
import torch.nn.functional as f

import math
from tqdm import tqdm

from utils import radam
import utils.viz_utils as viz_utils

from models.style_networks import SemSegE2VID
import training.base_trainer
from evaluation.metrics import MetricsSemseg
from utils.loss_functions import TaskLoss
from utils.viz_utils import plot_confusion_matrix

from e2vid.utils.loading_utils import load_model
from e2vid.image_reconstructor import ImageReconstructor


class ESSSupervisedModel(training.base_trainer.BaseTrainer):
    def __init__(self, settings, train=True):
        self.is_training = train
        super(ESSSupervisedModel, self).__init__(settings)
        self.do_val_training_epoch = False

    def init_fn(self):
        self.buildModels()
        self.createOptimizerDict()

        # Decoder Loss
        self.cycle_content_loss = torch.nn.L1Loss()
        self.cycle_attribute_loss = torch.nn.L1Loss()

        # Task Loss
        self.task_loss = TaskLoss(losses=self.settings.task_loss, gamma=2.0, num_classes=self.settings.semseg_num_classes,
                                  ignore_index=self.settings.semseg_ignore_label, reduction='mean')
        self.train_statistics = {}

        self.metrics_semseg_b = MetricsSemseg(self.settings.semseg_num_classes, self.settings.semseg_ignore_label,
                                              self.settings.semseg_class_names)
    def buildModels(self):
        # Front End Sensor B
        self.front_end_sensor_b, _ = load_model(self.settings.path_to_model)
        for param in self.front_end_sensor_b.parameters():
            param.requires_grad = False
        self.front_end_sensor_b.eval()

        self.input_height = math.ceil(self.settings.img_size_b[0] / 8.0) * 8
        self.input_width = math.ceil(self.settings.img_size_b[1] / 8.0) * 8
        self.reconstructor = ImageReconstructor(self.front_end_sensor_b, self.input_height, self.input_width,
                                                self.settings.nr_temporal_bins_b, self.settings.gpu_device,
                                                self.settings.e2vid_config)

        self.models_dict = {"front_sensor_b": self.front_end_sensor_b}

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

        # Task
        back_params = filter(lambda p: p.requires_grad, self.task_backend.parameters())
        optimizer_back = radam.RAdam(back_params,
                                     lr=self.settings.lr_back,
                                     weight_decay=0.,
                                     betas=(0., 0.999))
        self.optimizers_dict = {"optimizer_back": optimizer_back}

    def trainEpoch(self):
        self.pbar = tqdm(total=self.train_loader_sensor_b.__len__(), unit='Batch', unit_scale=True)
        for model in self.models_dict:
            self.models_dict[model].train()

        for i_batch, sample_batched in enumerate(self.train_loader_sensor_b):
            out = self.train_step(sample_batched)

            self.train_summaries(out[0])

            self.step_count += 1
            self.pbar.set_postfix(TrainLoss='{:.2f}'.format(out[-1].data.cpu().numpy()))
            self.pbar.update(1)
        self.pbar.close()

    def train_step(self, input_batch):
        # Task Step
        optimizers_list = []
        optimizers_list.append('optimizer_back')

        for key_word in optimizers_list:
            optimizer_key_word = self.optimizers_dict[key_word]
            optimizer_key_word.zero_grad()

        d_final_loss, d_losses, d_outputs = self.task_train_step(input_batch)

        d_final_loss.backward()
        for key_word in optimizers_list:
            optimizer_key_word = self.optimizers_dict[key_word]
            optimizer_key_word.step()

        return d_losses, d_outputs, d_final_loss


    def task_train_step(self, batch):

        data_b = batch[0].to(self.device)

        if self.settings.require_paired_data_train_b:
            labels_b = batch[2].to(self.device)
        else:
            labels_b = batch[1].to(self.device)

        # Set BatchNorm Statistics to Train
        for model in self.models_dict:
            self.models_dict[model].train()
            if model in ['front_sensor_b']:
                self.models_dict[model].eval()

        self.reconstructor.last_states_for_each_channel = {'grayscale': None}

        for i in range(self.settings.nr_events_data_b):
            event_tensor = data_b[:, i * self.settings.input_channels_b:(i + 1) * self.settings.input_channels_b, :, :]
            img_fake, states_real, latent_real = self.reconstructor.update_reconstruction(event_tensor)

        losses = {}
        outputs = {}
        t_loss = 0.

        loss, pred_b = self.trainTaskStep('sensor_b', latent_real, labels_b, losses)
        t_loss += loss

        if self.visualize_epoch():
            self.visTaskStep(data_b, pred_b, labels_b, img_fake)

        return t_loss, losses, outputs

    def trainTaskStep(self, sensor_name, content_features, labels, losses):
        for key in content_features.keys():
            content_features[key] = content_features[key].detach()
        task_backend = self.models_dict["back_end"]
        pred = task_backend(content_features)
        loss_pred = self.task_loss(pred[1], labels) * self.settings.weight_task_loss
        losses['semseg_' + sensor_name + '_loss'] = loss_pred.detach()

        return loss_pred, pred

    def visTaskStep(self, data, pred, labels, img_fake):
        pred = pred[1]
        pred_lbl = pred.argmax(dim=1)

        semseg = viz_utils.prepare_semseg(pred_lbl, self.settings.semseg_color_map,
                                              self.settings.semseg_ignore_label)
        semseg_gt = viz_utils.prepare_semseg(labels, self.settings.semseg_color_map, self.settings.semseg_ignore_label)

        nrow = 4
        viz_tensors = torch.cat(
            (viz_utils.createRGBImage(data[:nrow], separate_pol=self.settings.separate_pol_b).clamp(0, 1).to(
                self.device),
             viz_utils.createRGBImage(semseg[:nrow].to(self.device)),
             viz_utils.createRGBImage(semseg_gt[:nrow].to(self.device)),
             viz_utils.createRGBImage(img_fake[:nrow])), dim=0)
        rgb_grid = torchvision.utils.make_grid(viz_tensors, nrow=nrow)
        self.img_summaries('train/semseg_event', rgb_grid, self.step_count)

    def validationEpochs(self):
        self.resetValidationStatistics()

        with torch.no_grad():
            for model in self.models_dict:
                self.models_dict[model].eval()

            self.validationEpoch(self.val_loader_sensor_b, 'sensor_b')

            if len(self.validation_embeddings) != 0:
                self.saveEmbeddingSpace()

            if self.do_val_training_epoch:
                self.trainDatasetStatisticsEpoch('sensor_b', self.train_loader_sensor_b)

            self.resetValidationStatistics()

        self.pbar.close()

    def validationEpoch(self, data_loader, sensor_name):
        val_dataset_length = data_loader.__len__()
        self.pbar = tqdm(total=val_dataset_length, unit='Batch', unit_scale=True)
        tqdm.write("Validation on " + sensor_name)
        cumulative_losses = {}
        total_nr_steps = None

        for i_batch, sample_batched in enumerate(data_loader):
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

        if sensor == 'sensor_a':
            content_first_sensor = gen_model(data)

        else:
            self.reconstructor.last_states_for_each_channel = {'grayscale': None}
            for i in range(self.settings.nr_events_data_b):
                event_tensor = data[:, i * self.settings.input_channels_b:(i + 1) * self.settings.input_channels_b, :,
                               :]
                img_fake, _, content_first_sensor = self.reconstructor.update_reconstruction(event_tensor)

        self.valTaskStep(content_first_sensor, labels, losses, sensor)

        if vis_reconstr_idx != -1:
            if sensor == 'sensor_a':
                self.visualizeSensorA(data, labels, content_first_sensor, vis_reconstr_idx, sensor)
            else:
                self.visualizeSensorB(data[:, -self.settings.input_channels_b:, :, :], content_first_sensor,
                                              labels, img_fake, paired_data, vis_reconstr_idx, sensor)
        return losses, None

    def valTaskStep(self, content_first_sensor, labels, losses, sensor):
        """Computes the task loss and updates metrics"""
        task_backend = self.models_dict["back_end"]
        pred = task_backend(content_first_sensor)
        pred = pred[1]
        pred = f.interpolate(pred, size=(self.settings.img_size_b), mode='nearest')
        pred_lbl = pred.argmax(dim=1)

        loss_pred = self.task_loss(pred, target=labels)
        losses['semseg_' + sensor + '_loss'] = loss_pred.detach()
        if sensor == 'sensor_a':
            self.metrics_semseg_a.update_batch(pred_lbl, labels)
        else:
            self.metrics_semseg_b.update_batch(pred_lbl, labels)

    def visualizeSensorA(self, data, labels, content_first_sensor, vis_reconstr_idx, sensor):
        nrow = 4
        vis_tensors = [viz_utils.createRGBImage(data[:nrow])]

        task_backend = self.models_dict["back_end"]
        pred = task_backend(content_first_sensor)
        pred = pred[1]
        pred_lbl = pred.argmax(dim=1)

        semseg = viz_utils.prepare_semseg(pred_lbl, self.settings.semseg_color_map, self.settings.semseg_ignore_label)
        vis_tensors.append(viz_utils.createRGBImage(semseg[:nrow].to(self.device)))
        semseg_gt = viz_utils.prepare_semseg(labels, self.settings.semseg_color_map, self.settings.semseg_ignore_label)
        vis_tensors.append(viz_utils.createRGBImage(semseg_gt[:nrow]).to(self.device))

        rgb_grid = torchvision.utils.make_grid(torch.cat(vis_tensors, dim=0), nrow=nrow)
        self.img_summaries('val_' + sensor + '/reconst_input_' + sensor + '_' + str(vis_reconstr_idx),
                           rgb_grid, self.epoch_count)

    def visualizeSensorB(self, data, content_first_sensor, labels, img_fake, paired_data,
                                 vis_reconstr_idx, sensor):
        nrow = 4
        vis_tensors = [viz_utils.createRGBImage(data[:nrow], separate_pol=self.settings.separate_pol_b).clamp(0, 1).to(
            self.device)]

        task_backend = self.models_dict["back_end"]
        pred = task_backend(content_first_sensor)
        pred = pred[1]
        pred_lbl = pred.argmax(dim=1)

        labels = f.interpolate(labels.float().unsqueeze(1), size=(self.input_height, self.input_width), mode='nearest').squeeze(1).long()

        semseg = viz_utils.prepare_semseg(pred_lbl, self.settings.semseg_color_map, self.settings.semseg_ignore_label)
        vis_tensors.append(viz_utils.createRGBImage(semseg[:nrow].to(self.device)))
        semseg_gt = viz_utils.prepare_semseg(labels, self.settings.semseg_color_map, self.settings.semseg_ignore_label)
        vis_tensors.append(viz_utils.createRGBImage(semseg_gt[:nrow]).to(self.device))
        vis_tensors.append(viz_utils.createRGBImage(img_fake[:nrow]).to(self.device))
        vis_tensors.append(viz_utils.createRGBImage(paired_data[:nrow]).to(self.device))
        rgb_grid = torchvision.utils.make_grid(torch.cat(vis_tensors, dim=0), nrow=nrow)
        self.img_summaries('val_' + sensor + '/reconst_input_' + sensor + '_' + str(vis_reconstr_idx),
                           rgb_grid, self.epoch_count)

    def resetValidationStatistics(self):
        self.metrics_semseg_b.reset()



