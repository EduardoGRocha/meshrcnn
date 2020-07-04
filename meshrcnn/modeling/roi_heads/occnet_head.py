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
from meshrcnn.structures.voxel import batch_crop_voxels_within_box

ROI_OCCNET_HEAD_REGISTRY = Registry("ROI_OCCNET_HEAD")


def occnet_rcnn_loss(logits, occupancies, loss_weight=1.0):
    # cls_agnostic_occupancy = pred_occnet_logits.size(1) == 1
    # total_num_occupancies = pred_occnet_logits.size(0)
    #
    # gt_classes = []
    # gt_occupancy_logits = []
    # for instances_per_image in instances:
    #     if len(instances_per_image) == 0:
    #         continue
    #     if not cls_agnostic_occupancy:
    #         gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
    #         gt_classes.append(gt_classes_per_image)
    #
    #     gt_voxels = instances_per_image.gt_voxels
    #     gt_K = instances_per_image.gt_K
    #     gt_occupancy_logits_per_image = batch_crop_voxels_within_box(
    #         gt_voxels, instances_per_image.proposal_boxes.tensor, gt_K, voxel_side_len
    #     ).to(device=pred_occnet_logits.device)
    #     gt_occupancy_logits.append(gt_occupancy_logits_per_image)
    #
    # if len(gt_occupancy_logits) == 0:
    #     return pred_occnet_logits.sum() * 0, gt_occupancy_logits
    #
    # gt_occupancy_logits = cat(gt_occupancy_logits, dim=0)
    # assert gt_occupancy_logits.numel() > 0, gt_occupancy_logits.shape
    #
    # if cls_agnostic_occupancy:
    #     pred_voxel_logits = pred_occnet_logits[:, 0]
    # else:
    #     indices = torch.arange(total_num_occupancies)
    #     gt_classes = cat(gt_classes, dim=0)
    #     pred_voxel_logits = pred_occnet_logits[indices, gt_classes]
    #
    # # Log the training accuracy (using gt classes and 0.5 threshold)
    # # Note that here we allow gt_occupancy_logits to be float as well
    # # (depend on the implementation of rasterize())
    # voxel_accurate = (pred_voxel_logits > 0.5) == (gt_occupancy_logits > 0.5)
    # voxel_accuracy = voxel_accurate.nonzero().size(0) / voxel_accurate.numel()
    # get_event_storage().put_scalar("voxel_rcnn/accuracy", voxel_accuracy)
    #
    # voxel_loss = F.binary_cross_entropy_with_logits(
    #     pred_voxel_logits, gt_occupancy_logits, reduction="mean"
    # )
    # voxel_loss = voxel_loss * loss_weight
    # return voxel_loss, gt_occupancy_logits
    # TODO KL divergence removed from original paper
    loss_bin_cross_entropy = F.binary_cross_entropy_with_logits(logits, occupancies, reduction='none')
    loss = loss_bin_cross_entropy.sum(-1).mean()
    return loss * loss_weight / 2048.0


def occnet_rcnn_inference(pred_occnet_logits, pred_instances):
    pass


is_cuda = torch.cuda.is_available()
device = torch.device("cuda" if is_cuda else "cpu")

@ROI_OCCNET_HEAD_REGISTRY.register()
class OccNetRCNNHead(nn.Module):
    def __init__(self, cfg, input_shape, latent_dim=512):
        super(OccNetRCNNHead, self).__init__()
        decoder = DecoderCBatchNorm(3, 0, 256)
        encoder = MyEncoder(256)
        self.network = OccupancyNetwork(decoder, encoder, device=device)

    # Input 12x12xC feature map + set of points.
    #   Output of size
    # TODO is sample argument needed
    # TODO change to 14x14 input
    def forward(self, occnet_points, occnet_features, sample=True):
    # def forward(self, occnet_features, sample=True):
        # return self.network.forward(torch.tensor([[[0., 0., 0.]], [[0., 0., 0.]]], device=device), occnet_features, sample=sample)
        return self.network.forward(occnet_points, occnet_features, sample=sample)


def build_occ_head(cfg, input_shape):
    name = cfg.MODEL.ROI_OCCNET_HEAD.NAME
    # TODO pass latent dim (?)
    return ROI_OCCNET_HEAD_REGISTRY.get(name)(cfg, input_shape)
