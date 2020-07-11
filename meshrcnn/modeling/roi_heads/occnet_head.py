import fvcore.nn.weight_init as weight_init
import torch
from detectron2.layers import Conv2d, ConvTranspose2d, cat, get_norm
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from torch import nn
from meshrcnn.modeling.roi_heads.occnet import OccupancyNetwork
from torch.nn import functional as F

from meshrcnn.modeling.roi_heads.occnet.decoder import DecoderCBatchNorm
from meshrcnn.modeling.roi_heads.occnet.encoder import MyEncoder

ROI_OCCNET_HEAD_REGISTRY = Registry("ROI_OCCNET_HEAD")


def occnet_rcnn_loss(logits, occupancies, loss_weight=1.0):
    # TODO KL divergence removed from original code
    loss_bin_cross_entropy = F.binary_cross_entropy_with_logits(logits, occupancies, reduction='none')
    loss = loss_bin_cross_entropy.sum(-1).mean()
    return loss * loss_weight / float(logits.shape[1])


def occnet_rcnn_inference(pred_occnet_logits, pred_instances):
    cls_agnostic_occupancy = pred_occnet_logits, pred_instances

    if cls_agnostic_occupancy:
        occnet_probs_pred = pred_occnet_logits.sigmoid()
    else:
        # TODO probably leave it out by now
        # Select occupancies corresponding to the predicted classes
        num_occupancies = pred_occnet_logits.shape[0]
        class_pred = cat([i.pred_classes for i in pred_instances])
        indices = torch.arange(num_occupancies, device=class_pred.device)
        occnet_probs_pred = pred_occnet_logits[indices, class_pred][:, None].sigmoid()
        # occnet_probs_pred.shape: (B, 1, D, H, W)

    num_boxes_per_image = [len(i) for i in pred_instances]
    occnet_probs_pred = occnet_probs_pred.split(num_boxes_per_image, dim=0)

    for prob, instances in zip(occnet_probs_pred, pred_instances):
        instances.pred_occupancies = prob  # (1, D, H, W)


is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

@ROI_OCCNET_HEAD_REGISTRY.register()
class OccNetRCNNHead(nn.Module):
    def __init__(self, cfg, input_shape, latent_dim=512):
        super(OccNetRCNNHead, self).__init__()
        decoder = DecoderCBatchNorm(3, 0, 256)
        encoder = MyEncoder(256)
        self.network = OccupancyNetwork(decoder, encoder, device=device)

    # Input 14x14xC feature map + set of points.
    #   Output of size
    # TODO is sample argument needed
    def forward(self, occnet_points, occnet_features, sample=True):
        return self.network.forward(occnet_points, occnet_features, sample=sample)


def build_occ_head(cfg, input_shape):
    name = cfg.MODEL.ROI_OCCNET_HEAD.NAME
    # TODO pass latent dim (?)
    return ROI_OCCNET_HEAD_REGISTRY.get(name)(cfg, input_shape)
