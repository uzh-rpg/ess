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


def drawBoundingBoxes(np_image, bounding_boxes, class_name=None, ground_truth=True, rescale_image=False):
    """
    Draws the bounding boxes in the image

    :param np_image: [H, W, C]
    :param bounding_boxes: list of bounding boxes with shape [x, y, width, height].
    :param class_name: string
    """
    np_image = np_image.astype(np.float)
    resize_scale = 1.5
    if rescale_image:
        bounding_boxes[:, :4] = (bounding_boxes.astype(np.float)[:, :4] * resize_scale)
        new_dim = np.array(np_image.shape[:2], dtype=np.float) * resize_scale
        np_image = cv2.resize(np_image, tuple(new_dim.astype(int)[::-1]), interpolation=cv2.INTER_NEAREST)

    for i, bounding_box in enumerate(bounding_boxes):
        if bounding_box.sum() == 0:
            break
        if class_name is None:
            np_image = drawBoundingBox(np_image, bounding_box, ground_truth=ground_truth)
        else:
            np_image = drawBoundingBox(np_image, bounding_box, class_name[i], ground_truth)

    return np_image


def drawBoundingBox(np_image, bounding_box, class_name=None, ground_truth=False):
    """
    Draws a bounding box in the image.

    :param np_image: [H, W, C]
    :param bounding_box: [x, y, width, height].
    :param class_name: string
    """
    if ground_truth:
        bbox_color = np.array([0, 1, 1])
    else:
        bbox_color = np.array([1, 0, 1])
    height, width = bounding_box[2:4]

    np_image[bounding_box[0], bounding_box[1]:(bounding_box[1] + width)] = bbox_color
    np_image[bounding_box[0]:(bounding_box[0] + height), (bounding_box[1] + width)] = bbox_color
    np_image[(bounding_box[0] + height), bounding_box[1]:(bounding_box[1] + width)] = bbox_color
    np_image[bounding_box[0]:(bounding_box[0] + height), bounding_box[1]] = bbox_color

    if class_name is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_color = (0, 0, 0)
        font_scale = 0.5
        thickness = 1
        bottom_left = tuple(((bounding_box[[1, 0]] + np.array([+1, height - 2]))).astype(int))

        # Draw Box
        (text_width, text_height) = cv2.getTextSize(class_name, font, fontScale=font_scale, thickness=thickness)[0]
        box_coords = ((bottom_left[0], bottom_left[1] + 2),
                      (bottom_left[0] + text_width + 2, bottom_left[1] - text_height - 2 + 2))
        color_format = (int(bbox_color[0]), int(bbox_color[1]), int(bbox_color[2]))
        # np_image = cv2.UMat(np_image)
        np_image = cv2.UMat(np_image).get()
        cv2.rectangle(np_image, box_coords[0], box_coords[1], color_format, cv2.FILLED)

        cv2.putText(np_image, class_name, bottom_left, font, font_scale, font_color, thickness, cv2.LINE_AA)

    return np_image


def visualizeFlow(tensor_flow_map):
    """
    Visualizes the direction flow based on the HSV model
    """
    np_flow_map = tensor_flow_map.cpu().detach().numpy()
    batch_s, channel, height, width = np_flow_map.shape
    viz_array = np.zeros([batch_s, height, width, 3], dtype=np.uint8)
    hsv = np.zeros([height, width, 3], dtype=np.uint8)

    for i, sample_flow_map in enumerate(np_flow_map):
        hsv[..., 1] = 255
        mag, ang = cv2.cartToPolar(sample_flow_map[0, :, :], sample_flow_map[1, :, :])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        viz_array[i] = bgr

    return torch.from_numpy(viz_array.transpose([0, 3, 1, 2]) / 255.).to(tensor_flow_map.device)

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
        mask_ignore = img == semseg_ignore_label
        img[mask_ignore] = 0
    img = colors[img]  # N x H x W x 3
    img = img.permute(0, 3, 1, 2)

    # checkerboard
    # if semseg_ignore_label in img_color_ids:
    #     mask_ignore = mask_ignore.unsqueeze(1).repeat(1, 3, 1, 1)
    #     img[mask_ignore] = checkerboard[mask_ignore]
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

    # print(cm)

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
