import torch


def semseg_compute_confusion(y_hat_lbl, y_lbl, num_classes, ignore_label):
    assert torch.is_tensor(y_hat_lbl) and torch.is_tensor(y_lbl), 'Inputs must be torch tensors'
    assert y_lbl.device == y_hat_lbl.device, 'Input tensors have different device placement'

    assert y_hat_lbl.dim() == 3 or y_hat_lbl.dim() == 4 and y_hat_lbl.shape[1] == 1
    assert y_lbl.dim() == 3 or y_lbl.dim() == 4 and y_lbl.shape[1] == 1
    if y_hat_lbl.dim() == 4:
        y_hat_lbl = y_hat_lbl.squeeze(1)
    if y_lbl.dim() == 4:
        y_lbl = y_lbl.squeeze(1)

    mask = y_lbl != ignore_label
    y_hat_lbl = y_hat_lbl[mask]
    y_lbl = y_lbl[mask]

    # hack for bincounting 2 arrays together
    x = y_hat_lbl + num_classes * y_lbl
    bincount_2d = torch.bincount(x.long(), minlength=num_classes ** 2)
    assert bincount_2d.numel() == num_classes ** 2, 'Internal error'
    conf = bincount_2d.view((num_classes, num_classes)).long()
    return conf


def semseg_accum_confusion_to_iou(confusion_accum):
    conf = confusion_accum.double()
    diag = conf.diag()
    iou_per_class = 100 * diag / (conf.sum(dim=1) + conf.sum(dim=0) - diag).clamp(min=1e-12)
    iou_mean = iou_per_class.mean()
    return iou_mean, iou_per_class

def semseg_accum_confusion_to_acc(confusion_accum):
    conf = confusion_accum.double()
    diag = conf.diag()
    acc = 100 * diag.sum() / (conf.sum(dim=1).sum()).clamp(min=1e-12)
    return acc

class MetricsSemseg:
    def __init__(self, num_classes, ignore_label, class_names):
        self.num_classes = num_classes
        self.ignore_label = ignore_label
        self.class_names = class_names
        self.metrics_acc = None

    def reset(self):
        self.metrics_acc = None

    def update_batch(self, y_hat_lbl, y_lbl):
        with torch.no_grad():
            metrics_batch = semseg_compute_confusion(y_hat_lbl, y_lbl, self.num_classes, self.ignore_label).cpu()
            if self.metrics_acc is None:
                self.metrics_acc = metrics_batch
            else:
                self.metrics_acc += metrics_batch

    def get_metrics_summary(self):
        iou_mean, iou_per_class = semseg_accum_confusion_to_iou(self.metrics_acc)
        out = {self.class_names[i]: iou for i, iou in enumerate(iou_per_class)}
        out['mean_iou'] = iou_mean
        acc = semseg_accum_confusion_to_acc((self.metrics_acc))
        out['acc'] = acc
        out['cm'] = self.metrics_acc
        return out


