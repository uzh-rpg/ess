import cv2
import torch
import numpy as np
import torchvision.utils
import torch.nn.functional as F
import matplotlib.pyplot as plt
import itertools


def createRGBGrid(tensor_list, nrow):
    """Creates a grid of rgb values based on the tensor stored in tensor_list"""
    vis_tensor_list = []
    for tensor in tensor_list:
        vis_tensor_list.append(visualizeTensors(tensor))

    return torchvision.utils.make_grid(torch.cat(vis_tensor_list, dim=0), nrow=nrow)


def createRGBImage(tensor, separate_pol=True):
    """Creates a grid of rgb values based on the tensor stored in tensor_list"""
    if tensor.shape[1] == 3:
        return tensor
    elif tensor.shape[1] == 1:
        return tensor.expand(-1, 3, -1, -1)
    elif tensor.shape[1] == 2:
        return visualizeHistogram(tensor)
    elif tensor.shape[1] > 3:
        return visualizeVoxelGrid(tensor, separate_pol)


def visualizeTensors(tensor):
    """Creates a rgb image of the given tensor. Can be event histogram, event voxel grid, grayscale and rgb."""
    if tensor.shape[1] == 3:
        return tensor
    elif tensor.shape[1] == 1:
        return tensor.expand(-1, 3, -1, -1)
    elif tensor.shape[1] == 2:
        return visualizeHistogram(tensor)
    elif tensor.shape[1] > 3:
        return visualizeVoxelGrid(tensor)


def visualizeHistogram(histogram):
    """Visualizes the input histogram"""
    batch, _, height, width = histogram.shape
    torch_image = torch.zeros([batch, 1, height, width], device=histogram.device)

    return torch.cat([histogram.clamp(0, 1), torch_image], dim=1)


def visualizeVoxelGrid(voxel_grid, separate_pol=True):
    """Visualizes the input histogram"""
    batch, nr_channels, height, width = voxel_grid.shape
    if separate_pol:
        pos_events_idx = nr_channels // 2
        temporal_scaling = torch.arange(start=1, end=pos_events_idx+1, dtype=voxel_grid.dtype,
                                        device=voxel_grid.device)[None, :, None, None] / pos_events_idx
        pos_voxel_grid = voxel_grid[:, :pos_events_idx] * temporal_scaling
        neg_voxel_grid = voxel_grid[:, pos_events_idx:] * temporal_scaling

        torch_image = torch.zeros([batch, 1, height, width], device=voxel_grid.device)
        pos_image = torch.sum(pos_voxel_grid, dim=1, keepdim=True)
        neg_image = torch.sum(neg_voxel_grid, dim=1, keepdim=True)

        return torch.cat([neg_image.clamp(0, 1), pos_image.clamp(0, 1), torch_image], dim=1)

    sum_events = torch.sum(voxel_grid, dim=1).detach()
    event_preview = torch.zeros((batch, 3, height, width))
    b = event_preview[:, 0, :, :]
    r = event_preview[:, 2, :, :]
    b[sum_events > 0] = 255
    r[sum_events < 0] = 255
    return event_preview


def visualizeConfusionMatrix(confusion_matrix, path_name=None):
    """
    Visualizes the confustion matrix using matplotlib.

    :param confusion_matrix: NxN numpy array
    :param path_name: if no path name is given, just an image is returned
    """
    import matplotlib.pyplot as plt
    nr_classes = confusion_matrix.shape[0]
    fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    ax.matshow(confusion_matrix)
    ax.plot([-0.5, nr_classes - 0.5], [-0.5, nr_classes - 0.5], '-', color='grey')
    ax.set_xlabel('Labels')
    ax.set_ylabel('Predicted')

    if path_name is None:
        fig.tight_layout(pad=0)
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

        return data

    else:
        fig.savefig(path_name)
        plt.close()


def create_checkerboard(N, C, H, W):
    cell_sz = max(min(H, W) // 32, 1)
    mH = (H + cell_sz - 1) // cell_sz
    mW = (W + cell_sz - 1) // cell_sz
    checkerboard = torch.full((mH, mW), 0.25, dtype=torch.float32)
    checkerboard[0::2, 0::2] = 0.75
    checkerboard[1::2, 1::2] = 0.75
    checkerboard = checkerboard.float().view(1, 1, mH, mW)
    checkerboard = F.interpolate(checkerboard, scale_factor=cell_sz, mode='nearest')
    checkerboard = checkerboard[:, :, :H, :W].repeat(N, C, 1, 1)
    return checkerboard


def prepare_semseg(img, semseg_color_map, semseg_ignore_label):
    assert (img.dim() == 3 or img.dim() == 4 and img.shape[1] == 1) and img.dtype in (torch.int, torch.long), \
        f'Expecting 4D tensor with semseg classes, got {img.shape}'
    if img.dim() == 4:
        img = img.squeeze(1)
    colors = torch.tensor(semseg_color_map, dtype=torch.float32)
    assert colors.dim() == 2 and colors.shape[1] == 3
    if torch.max(colors) > 128:
        colors /= 255
    img = img.cpu().clone()  # N x H x W
    N, H, W = img.shape
    img_color_ids = torch.unique(img)
    assert all(c_id == semseg_ignore_label or 0 <= c_id < len(semseg_color_map) for c_id in img_color_ids)
    checkerboard, mask_ignore = None, None
    if semseg_ignore_label in img_color_ids:
        checkerboard = create_checkerboard(N, 3, H, W)
        # blackboard = create_blackboard(N, 3, H, W)
        mask_ignore = img == semseg_ignore_label
        img[mask_ignore] = 0
    img = colors[img]  # N x H x W x 3
    img = img.permute(0, 3, 1, 2)

    # checkerboard
    if semseg_ignore_label in img_color_ids:
        mask_ignore = mask_ignore.unsqueeze(1).repeat(1, 3, 1, 1)
        img[mask_ignore] = checkerboard[mask_ignore]
        # img[mask_ignore] = blackboard[mask_ignore]
    return img


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = cm.numpy()
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig = plt.figure()

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig
