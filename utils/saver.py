from __future__ import division
import os
import datetime

import torch


class CheckpointSaver(object):
    def __init__(self, save_dir):
        if save_dir is not None:
            self.save_dir = os.path.abspath(save_dir)
        return

    # save checkpoint
    def save_checkpoint(self, models, optimizers, epoch, step_count, batch_size_a, batch_size_b):
        timestamp = datetime.datetime.now()
        checkpoint_filename = os.path.abspath(os.path.join(self.save_dir, 'Epoch_' + str(epoch) + '.pt'))
        checkpoint = {}
        for model in models:
            checkpoint[model] = models[model].state_dict()
        for optimizer in optimizers:
            checkpoint[optimizer] = optimizers[optimizer].state_dict()
        checkpoint['epoch'] = epoch
        checkpoint['step_count'] = step_count
        checkpoint['batch_size_a'] = batch_size_a
        checkpoint['batch_size_b'] = batch_size_b
        print()
        print(timestamp, 'Epoch:', epoch, 'Iteration:', step_count)
        print('Saving checkpoint file [' + checkpoint_filename + ']')
        torch.save(checkpoint, checkpoint_filename) 
        return

    # load a checkpoint
    def load_checkpoint(self, models, optimizers, checkpoint_file=None, load_optimizer=True):
        checkpoint = torch.load(checkpoint_file)
        for model in models:
            if model in checkpoint:
                models[model].load_state_dict(checkpoint[model])
        if load_optimizer:
            for optimizer in optimizers:
                if optimizer in checkpoint:
                    optimizers[optimizer].load_state_dict(checkpoint[optimizer])
        print("Loading checkpoint with epoch {}, step {}"
              .format(checkpoint['epoch'], checkpoint['step_count']))
        return {'epoch': checkpoint['epoch'],
                'step_count': checkpoint['step_count'],
                'batch_size_a': checkpoint['batch_size_a'],
                'batch_size_b': checkpoint['batch_size_b']}

    def load_pretrained_weights(self, models, model_list, checkpoint_file=None):
        checkpoint = torch.load(checkpoint_file)
        load_model_list = []
        for model_name in model_list:
            if model_name in ['front_sensor_b', 'e2vid_decoder']:
                continue
            if model_name in checkpoint:
                load_model_list.append(model_name)
                models[model_name].load_state_dict(checkpoint[model_name])

        print("Loading pretrained checkpoints for {}".format(load_model_list))
