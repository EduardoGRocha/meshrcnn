# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict
import torch
import numpy as np

from detectron2.layers import ShapeSpec, cat
from detectron2.modeling import ROI_HEADS_REGISTRY
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.roi_heads import StandardROIHeads, select_foreground_proposals
from pytorch3d.ops import cubify
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere

from meshrcnn.modeling.roi_heads.mask_head import mask_rcnn_loss
from meshrcnn.modeling.roi_heads.mesh_head import (
    build_mesh_head,
    mesh_rcnn_inference,
    mesh_rcnn_loss,
)
from meshrcnn.modeling.roi_heads.voxel_head import (
    build_voxel_head,
    voxel_rcnn_inference,
    voxel_rcnn_loss,
)

from meshrcnn.modeling.roi_heads.occnet_head import (
    occnet_rcnn_loss,
    occnet_rcnn_inference,
    occnet_mesh_rcnn_inference,
    build_occ_head,
)

from meshrcnn.modeling.roi_heads.z_head import build_z_head, z_rcnn_inference, z_rcnn_loss
from meshrcnn.utils import vis as vis_utils
from meshrcnn.generation.generator import Generator3D


@ROI_HEADS_REGISTRY.register()
class MeshRCNNROIHeads(StandardROIHeads):
    """
    The ROI specific heads for Mesh R-CNN
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__(cfg, input_shape)
        self._init_z_head(cfg, input_shape)
        self._init_voxel_head(cfg, input_shape)
        self._init_mesh_head(cfg, input_shape)
        self._init_occ_head(cfg, input_shape)
        # If MODEL.VIS_MINIBATCH is True we store minibatch targets
        # for visualization purposes
        self._vis = cfg.MODEL.VIS_MINIBATCH
        self._misc = {}
        self._vis_dir = cfg.OUTPUT_DIR

    def _init_z_head(self, cfg, input_shape):
        # fmt: off
        self.zpred_on = cfg.MODEL.ZPRED_ON
        if not self.zpred_on:
            return
        z_pooler_resolution = cfg.MODEL.ROI_Z_HEAD.POOLER_RESOLUTION
        z_pooler_scales = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        z_sampling_ratio = cfg.MODEL.ROI_Z_HEAD.POOLER_SAMPLING_RATIO
        z_pooler_type = cfg.MODEL.ROI_Z_HEAD.POOLER_TYPE
        # fmt: on

        self.z_loss_weight = cfg.MODEL.ROI_Z_HEAD.Z_REG_WEIGHT
        self.z_smooth_l1_beta = cfg.MODEL.ROI_Z_HEAD.SMOOTH_L1_BETA

        in_channels = [input_shape[f].channels for f in self.in_features][0]

        self.z_pooler = ROIPooler(
            output_size=z_pooler_resolution,
            scales=z_pooler_scales,
            sampling_ratio=z_sampling_ratio,
            pooler_type=z_pooler_type,
        )
        shape = ShapeSpec(
            channels=in_channels, width=z_pooler_resolution, height=z_pooler_resolution
        )
        self.z_head = build_z_head(cfg, shape)

    def _init_voxel_head(self, cfg, input_shape):
        # fmt: off
        self.voxel_on = cfg.MODEL.VOXEL_ON
        if not self.voxel_on:
            return
        voxel_pooler_resolution = cfg.MODEL.ROI_VOXEL_HEAD.POOLER_RESOLUTION
        voxel_pooler_scales = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        voxel_sampling_ratio = cfg.MODEL.ROI_VOXEL_HEAD.POOLER_SAMPLING_RATIO
        voxel_pooler_type = cfg.MODEL.ROI_VOXEL_HEAD.POOLER_TYPE
        # fmt: on

        self.voxel_loss_weight = cfg.MODEL.ROI_VOXEL_HEAD.LOSS_WEIGHT
        self.cls_agnostic_voxel = cfg.MODEL.ROI_VOXEL_HEAD.CLS_AGNOSTIC_VOXEL
        self.cubify_thresh = cfg.MODEL.ROI_VOXEL_HEAD.CUBIFY_THRESH

        in_channels = [input_shape[f].channels for f in self.in_features][0]

        self.voxel_pooler = ROIPooler(
            output_size=voxel_pooler_resolution,
            scales=voxel_pooler_scales,
            sampling_ratio=voxel_sampling_ratio,
            pooler_type=voxel_pooler_type,
        )
        shape = ShapeSpec(
            channels=in_channels, width=voxel_pooler_resolution, height=voxel_pooler_resolution
        )
        self.voxel_head = build_voxel_head(cfg, shape)

    def _init_mesh_head(self, cfg, input_shape):
        # fmt: off
        self.mesh_on = cfg.MODEL.MESH_ON
        if not self.mesh_on:
            return
        mesh_pooler_resolution = cfg.MODEL.ROI_MESH_HEAD.POOLER_RESOLUTION
        mesh_pooler_scales = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        mesh_sampling_ratio = cfg.MODEL.ROI_MESH_HEAD.POOLER_SAMPLING_RATIO
        mesh_pooler_type = cfg.MODEL.ROI_MESH_HEAD.POOLER_TYPE
        # fmt: on

        self.chamfer_loss_weight = cfg.MODEL.ROI_MESH_HEAD.CHAMFER_LOSS_WEIGHT
        self.normals_loss_weight = cfg.MODEL.ROI_MESH_HEAD.NORMALS_LOSS_WEIGHT
        self.edge_loss_weight = cfg.MODEL.ROI_MESH_HEAD.EDGE_LOSS_WEIGHT
        self.gt_num_samples = cfg.MODEL.ROI_MESH_HEAD.GT_NUM_SAMPLES
        self.pred_num_samples = cfg.MODEL.ROI_MESH_HEAD.PRED_NUM_SAMPLES
        self.gt_coord_thresh = cfg.MODEL.ROI_MESH_HEAD.GT_COORD_THRESH
        self.ico_sphere_level = cfg.MODEL.ROI_MESH_HEAD.ICO_SPHERE_LEVEL

        in_channels = [input_shape[f].channels for f in self.in_features][0]

        self.mesh_pooler = ROIPooler(
            output_size=mesh_pooler_resolution,
            scales=mesh_pooler_scales,
            sampling_ratio=mesh_sampling_ratio,
            pooler_type=mesh_pooler_type,
        )
        self.mesh_head = build_mesh_head(
            cfg,
            ShapeSpec(
                channels=in_channels, height=mesh_pooler_resolution, width=mesh_pooler_resolution
            ),
        )

    def _init_occ_head(self, cfg, input_shape):
        # fmt: off
        self.occ_on = cfg.MODEL.OCC_ON
        if not self.occ_on:
            return
        occ_pooler_resolution = cfg.MODEL.ROI_OCCNET_HEAD.POOLER_RESOLUTION
        occ_pooler_scales = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        occ_sampling_ratio = cfg.MODEL.ROI_OCCNET_HEAD.POOLER_SAMPLING_RATIO
        occ_pooler_type = cfg.MODEL.ROI_OCCNET_HEAD.POOLER_TYPE
        occ_points_per_iteration = cfg.MODEL.ROI_OCCNET_HEAD.SAMPLED_POINTS_PER_ITERATION
        # fmt: on

        self.occ_loss_weight = cfg.MODEL.ROI_OCCNET_HEAD.LOSS_WEIGHT
        self.cls_agnostic_occupancy = cfg.MODEL.ROI_OCCNET_HEAD.CLS_AGNOSTIC_OCCUPANCY
        self.occ_points_per_iteration = occ_points_per_iteration

        in_channels = [input_shape[f].channels for f in self.in_features][0]

        self.occ_pooler = ROIPooler(
            output_size=occ_pooler_resolution,
            scales=occ_pooler_scales,
            sampling_ratio=occ_sampling_ratio,
            pooler_type=occ_pooler_type,
        )

        self.occ_head = build_occ_head(
            cfg,
            ShapeSpec(
                channels=in_channels, height=occ_pooler_resolution, width=occ_pooler_resolution
            ),
        )

    def forward(self, images, features, proposals, targets=None):
        """
        See :class:`ROIHeads.forward`.
        """
        if self._vis:
            self._misc["images"] = images
        # del images

        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        # elif targets is not None:
        #     proposals_inference = self.label_and_sample_proposals(proposals, targets)
        # del targets

        if self._vis:
            self._misc["proposals"] = proposals

        if self.training:
            losses = self._forward_box(features, proposals)
            # During training the proposals used by the box head are
            # used by the z, mask, voxel & mesh head.
            losses.update(self._forward_z(features, proposals))
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_shape(features, proposals))
            losses.update(self._forward_occupancy(features, proposals))
            # print minibatch examples
            if self._vis:
                vis_utils.visualize_minibatch(self._misc["images"], self._misc, self._vis_dir, True)

            return [], losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self._forward_with_given_boxes(features, pred_instances, targets)
            return pred_instances, {}

    def _forward_with_given_boxes(self, features, instances, targets):
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_z(features, instances)
        instances = self._forward_mask(features, instances)
        instances = self._forward_shape(features, instances)
        instances = self._forward_occupancy(features, instances, targets)
        return instances

    def forward_with_given_boxes(self, features, instances):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (Instances): the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_voxels`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_z(features, instances)
        instances = self._forward_mask(features, instances)
        instances = self._forward_shape(features, instances)
        instances = self._forward_occupancy(features, instances)
        return instances

    def _forward_z(self, features, instances):
        """
        Forward logic of the z prediction branch.
        """
        if not self.zpred_on:
            return {} if self.training else instances
        features = [features[f] for f in self.in_features]

        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            z_features = self.z_pooler(features, proposal_boxes)
            z_pred = self.z_head(z_features)
            src_boxes = cat([p.tensor for p in proposal_boxes])
            loss_z_reg = z_rcnn_loss(
                z_pred,
                proposals,
                src_boxes,
                loss_weight=self.z_loss_weight,
                smooth_l1_beta=self.z_smooth_l1_beta,
            )
            return {"loss_z_reg": loss_z_reg}
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            z_features = self.z_pooler(features, pred_boxes)
            z_pred = self.z_head(z_features)
            z_rcnn_inference(z_pred, instances)
            return instances

    def _forward_mask(self, features, instances):
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str,Tensor]): mapping from names to backbone features
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        features = [features[f] for f in self.in_features]

        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features = self.mask_pooler(features, proposal_boxes)
            mask_logits = self.mask_head.layers(mask_features)
            loss_mask, target_masks = mask_rcnn_loss(mask_logits, proposals)
            if self._vis:
                self._misc["target_masks"] = target_masks
                self._misc["fg_proposals"] = proposals
            return {"loss_mask": loss_mask}
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(features, pred_boxes)
            return self.mask_head(mask_features, instances)

    def _forward_shape(self, features, instances):
        """
        Forward logic for the voxel and mesh refinement branch.

        Args:
            features (list[Tensor]): #level input features for voxel prediction
            instances (list[Instances]): the per-image instances to train/predict meshes.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.
        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_voxels" & "pred_meshes" and return it.
        """
        if not self.voxel_on and not self.mesh_on:
            return {} if self.training else instances

        features = [features[f] for f in self.in_features]
        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]

            losses = {}
            if self.voxel_on:
                voxel_features = self.voxel_pooler(features, proposal_boxes)
                voxel_logits = self.voxel_head(voxel_features)
                loss_voxel, target_voxels = voxel_rcnn_loss(
                    voxel_logits, proposals, loss_weight=self.voxel_loss_weight
                )
                losses.update({"loss_voxel": loss_voxel})
                if self._vis:
                    self._misc["target_voxels"] = target_voxels
                if self.cls_agnostic_voxel:
                    with torch.no_grad():
                        vox_in = voxel_logits.sigmoid().squeeze(1)  # (N, V, V, V)
                        init_mesh = cubify(vox_in, self.cubify_thresh)  # 1
                else:
                    raise ValueError("No support for class specific predictions")

            if self.mesh_on:
                mesh_features = self.mesh_pooler(features, proposal_boxes)
                if not self.voxel_on:
                    if mesh_features.shape[0] > 0:
                        init_mesh = ico_sphere(self.ico_sphere_level, mesh_features.device)
                        init_mesh = init_mesh.extend(mesh_features.shape[0])
                    else:
                        init_mesh = Meshes(verts=[], faces=[])
                pred_meshes = self.mesh_head(mesh_features, init_mesh)

                # loss weights
                loss_weights = {
                    "chamfer": self.chamfer_loss_weight,
                    "normals": self.normals_loss_weight,
                    "edge": self.edge_loss_weight,
                }

                if not pred_meshes[0].isempty():
                    loss_chamfer, loss_normals, loss_edge, target_meshes = mesh_rcnn_loss(
                        pred_meshes,
                        proposals,
                        loss_weights=loss_weights,
                        gt_num_samples=self.gt_num_samples,
                        pred_num_samples=self.pred_num_samples,
                        gt_coord_thresh=self.gt_coord_thresh,
                    )
                    if self._vis:
                        self._misc["init_meshes"] = init_mesh
                        self._misc["target_meshes"] = target_meshes
                else:
                    loss_chamfer = sum(k.sum() for k in self.mesh_head.parameters()) * 0.0
                    loss_normals = sum(k.sum() for k in self.mesh_head.parameters()) * 0.0
                    loss_edge = sum(k.sum() for k in self.mesh_head.parameters()) * 0.0

                losses.update(
                    {
                        "loss_chamfer": loss_chamfer,
                        "loss_normals": loss_normals,
                        "loss_edge": loss_edge,
                    }
                )

            return losses
        else:
            pred_boxes = [x.pred_boxes for x in instances]

            if self.voxel_on:
                voxel_features = self.voxel_pooler(features, pred_boxes)
                voxel_logits = self.voxel_head(voxel_features)
                voxel_rcnn_inference(voxel_logits, instances)
                if self.cls_agnostic_voxel:
                    with torch.no_grad():
                        vox_in = voxel_logits.sigmoid().squeeze(1)  # (N, V, V, V)
                        init_mesh = cubify(vox_in, self.cubify_thresh)  # 1
                else:
                    raise ValueError("No support for class specific predictions")

            if self.mesh_on:
                mesh_features = self.mesh_pooler(features, pred_boxes)
                if not self.voxel_on:
                    if mesh_features.shape[0] > 0:
                        init_mesh = ico_sphere(self.ico_sphere_level, mesh_features.device)
                        init_mesh = init_mesh.extend(mesh_features.shape[0])
                    else:
                        init_mesh = Meshes(verts=[], faces=[])
                pred_meshes = self.mesh_head(mesh_features, init_mesh)
                mesh_rcnn_inference(pred_meshes[-1], instances)
            else:
                assert self.voxel_on
                mesh_rcnn_inference(init_mesh, instances)

            return instances

    def _forward_occupancy(self, features, instances, targets=None):
        if not self.occ_on:
            return {} if self.training else instances

        features = [features[f] for f in self.in_features]

        if self.training:
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]

            losses = {}
            occ_features = self.occ_pooler(features, proposal_boxes)
            # TODO hardcoded 100000
            idx = np.random.randint(100000, size=self.occ_points_per_iteration)
            sampled_points = [proposal.gt_points[:, :, idx, :] for proposal in proposals]
            concatenated_points = torch.squeeze(torch.cat(sampled_points))
            sampled_occupancies = [proposal.gt_occupancies[:, :, idx] for proposal in proposals]
            concatenated_occupancies = torch.squeeze(torch.cat(sampled_occupancies))
            occ_logits = self.occ_head(concatenated_points, occ_features).logits
            loss_occ = occnet_rcnn_loss(
                occ_logits, concatenated_occupancies, loss_weight=self.occ_loss_weight
            )
            losses.update({"loss_occnet": loss_occ})
            return losses
        else:
            '''
            pred_boxes = [x.pred_boxes for x in instances]
            points = []
            occupancies = []
            # pred_boxes = [pred_boxes[1]]
            # targets = [targets[1]]
            # features = [features[3]]
            # for pred_box, target in zip(pred_boxes, targets):
            #     points += [target.gt_points]*len(pred_box)
            #     occupancies += [target.gt_occupancies]*len(pred_box)
            points = [targets[0].gt_points]
            occupancies = [targets[0].gt_occupancies]
            # points = torch.squeeze(torch.cat(points))
            # occupancies = torch.squeeze(torch.cat(occupancies))
            points = torch.cat(points)[0]
            occupancies = torch.cat(occupancies)[0]

            # pred_boxes[0].tensor[1] = pred_boxes[0].tensor[0]
            occ_features = self.occ_pooler(features, pred_boxes)
            # occ_features[1] = occ_features[0]
            batch_size = occ_features.size(0)
            occ_latent = self.occ_head.network.encode_inputs(occ_features)
            # TODO setting sample to True
            z = self.occ_head.network.get_z_from_prior((batch_size,), sample=True)

            logits = self.occ_head(points, occ_features[0:1,:,:,:]).logits
            loss_occ = occnet_rcnn_loss(
                logits, occupancies, loss_weight=self.occ_loss_weight
            )
            log_sig = logits.sigmoid()
            log_sig[log_sig < 0.5] = 0
            log_sig[log_sig >= 0.5] = 1
            occ_example = (logits.sigmoid() >= 0.5)
            occ_gt = (occupancies >= 0.5)
            union = (occ_example | occ_gt)
            intersect = (occ_gt & occ_example)
            iou = intersect.sum().double() / union.sum().double()
            occnet_rcnn_inference(pred_boxes, instances)
            return instances
            '''

            # TODO: change to all boxes (not possible due to gpu constraints)
            # pred_boxes = [x.pred_boxes for x in instances]
            pred_boxes = [x.pred_boxes for x in instances]
            occ_features = self.occ_pooler(features, pred_boxes)


            #First approach: run separate generation for every feature vector
            # TODO: Can Generator3D handle multiple feature vectors at once?
            if targets is None:
                occupancy_network = self.occ_head.network
                meshes = []
                my_generator = Generator3D(
                            occupancy_network,
                            device=occ_features.device,
                            threshold=0.4,
                            resolution0=32,
                            upsampling_steps=2,
                            sample=False,
                            refinement_step=0,
                            simplify_nfaces=5000,
                            preprocessor=None,
                )
                #i = 0
                for feature in occ_features:
                    #i += 1
                    mesh, time = my_generator.generate_mesh(torch.unsqueeze(feature, 0))

                    # shortcut: save mesh
                    #mesh.export('/home/daniel/ADL4CV/meshes/mesh' + str(i) + '.obj')
                    meshes.append(mesh)
                meshes_pyt3d = Meshes(verts=[torch.tensor(mesh.vertices) for mesh in meshes],
                                   faces=[torch.tensor(mesh.faces) for mesh in meshes])


                # TODO: append generated meshes to instances
                mesh_rcnn_inference(meshes_pyt3d, instances)



            if not targets is None:
                # read gt_points  and occupancies for each image
                points = [x._fields['gt_points'] for x in targets]
                points_tensor = torch.squeeze(torch.cat(points, dim=0), 1)
                occupancies = [x._fields['gt_occupancies'] for x in targets]
                occupancies_tensor = torch.squeeze(torch.cat(occupancies, dim=0), 1)

                # feed gt_points and features through occ_head
                # TODO: run for all points; check if this logic is correct for multiple batches
                # something like:
                x = []
                n_batch = len(instances)
                n_instance = [len(x) for x in instances]
                for i in range(n_batch):
                    x.extend([torch.squeeze(points[i], 1) for j in range(n_instance[i])])
                points_tensor = torch.cat(x, dim=0)
                assert points_tensor.shape[0] == occ_features.shape[0]

                logits = self.occ_head(points_tensor[:, 0:100, :], occ_features).logits

                # occ net inference;
                # TODO: don't filter instances
                # occnet_rcnn_inference(logits, instances)
                filtered_instances = [instance[0] for instance in instances]
                occnet_rcnn_inference(logits, instances)

            return instances
