# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import json
import logging
import numpy as np
import os
import torch
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import Boxes, BoxMode, Instances
from pytorch3d.io import load_obj

from meshrcnn.structures import MeshInstances, VoxelInstances
from meshrcnn.utils import shape as shape_utils

from PIL import Image

__all__ = ["OccDatasetMapper"]

logger = logging.getLogger(__name__)


def annotations_to_instances(annotations, image_size):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annotations (list[dict]): a list of annotations, one per instance.
        image_size (tuple): height, width

    Returns:
        Instances: It will contains fields "gt_boxes", "gt_classes",
            "gt_masks", "gt_keypoints", if they can be obtained from `annos`.
    """
    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annotations]
    target = Instances(image_size)
    boxes = target.gt_boxes = Boxes(boxes)
    boxes.clip(image_size)

    classes = [obj["category_id"] for obj in annotations]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    if len(annotations) and "segmentation" in annotations[0]:
        masks = [obj["segmentation"] for obj in annotations]
        target.gt_masks = torch.stack(masks, dim=0)

    # camera
    if len(annotations) and "K" in annotations[0]:
        K = [torch.tensor(obj["K"]) for obj in annotations]
        target.gt_K = torch.stack(K, dim=0)

    # if len(annos) and "voxel" in annos[0]:
    #     voxels = [obj["voxel"] for obj in annos]
    #     target.gt_voxels = VoxelInstances(voxels)
    #
    # if len(annos) and "mesh" in annos[0]:
    #     meshes = [obj["mesh"] for obj in annos]
    #     target.gt_meshes = MeshInstances(meshes)

    if len(annotations) and "dz" in annotations[0]:
        dz = [obj["dz"] for obj in annotations]
        target.gt_dz = torch.tensor(dz)

    if len(annotations) and "pointcloud" in annotations[0]:
        pointcloud = [obj["pointcloud"] for obj in annotations]
        target.gt_pointcloud = torch.tensor(pointcloud)

    if len(annotations) and "points" in annotations[0]:
        points = [obj["points"] for obj in annotations]
        target.gt_points = torch.tensor(points)

    if len(annotations) and "occupancies" in annotations[0]:
        occupancies = [obj["occupancies"] for obj in annotations]
        target.gt_occupancies = torch.tensor(occupancies)

    return target


class OccDatasetMapper:
    """
    A callable which takes a dict produced by the detection dataset, and applies transformations,
    including image resizing and flipping. The transformation parameters are parsed from cfg file
    and depending on the is_train condition.

    Note that for our existing models, mean/std normalization is done by the model instead of here.
    """

    def __init__(self, cfg, is_train=True, dataset_names=None):
        self.tfm_gens = utils.build_transform_gen(cfg, is_train)

        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT
        self.mask_on        = cfg.MODEL.MASK_ON
        self.voxel_on       = cfg.MODEL.VOXEL_ON
        self.mesh_on        = cfg.MODEL.MESH_ON
        self.zpred_on       = cfg.MODEL.ZPRED_ON
        self.occ_on         = cfg.MODEL.OCC_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        # fmt: on

        if self.load_proposals:
            raise ValueError("Loading proposals not yet supported")

        if cfg.MODEL.LOAD_PROPOSALS:
            self.min_box_side_len = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )

        self.is_train = is_train

        assert dataset_names is not None
        # load unique occupancies all into memory
        all_pointcloud_models = {}
        all_points_models = {}
        for dataset_name in dataset_names:
            json_file = MetadataCatalog.get(dataset_name).json_file
            model_root = MetadataCatalog.get(dataset_name).image_root
            logger.info("Loading models from {}...".format(dataset_name))
            dataset_pointcloud_models = load_unique_pointclouds(json_file, model_root)
            dataset_points_models = load_unique_points(json_file, model_root)
            all_pointcloud_models.update(dataset_pointcloud_models)
            all_points_models.update(dataset_points_models)
            logger.info("Unique objects loaded: {}".format(len(dataset_pointcloud_models)))

        self._all_pointcloud_models = all_pointcloud_models
        self._all_points_models = all_points_models

    def __call__(self, dataset_dict):
        """
        Transform the dataset_dict according to the configured transformations.

        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a new dict that's going to be processed by the model.
                It currently does the following:
                1. Read the image from "file_name"
                2. Transform the image and annotations
                3. Prepare the annotations to :class:`Instances`
        """
        # get 3D models for each annotations and remove 3D mesh models from image dict
        pointcloud_models = []
        points_models = []
        occupancies_models = []
        if "annotations" in dataset_dict:
            for annotation in dataset_dict["annotations"]:
                pointcloud_dict = self._all_pointcloud_models[annotation["pointcloud"]][None]
                pointcloud_models.append([(pointcloud_dict).copy()])
                points_dict = self._all_points_models[annotation["points"]][None]
                points_models.append([(points_dict).copy()])
                occupancies_dict = self._all_points_models[annotation["points"]]["occupancies"]
                occupancies_models.append([(occupancies_dict).copy()])

        dataset_dict = {key: value for key, value in dataset_dict.items() if key != "mesh_models"}
        # TODO WTF?
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        if "annotations" in dataset_dict:
            for i, annotation in enumerate(dataset_dict["annotations"]):
                annotation["pointcloud"] = pointcloud_models[i]
                annotation["points"] = points_models[i]
                annotation["occupancies"] = occupancies_models[i]

        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
        # Can use uint8 if it turns out to be slow some day

        # USER: Remove if you don't use pre-computed proposals.
        if self.load_proposals:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, self.min_box_side_len, self.proposal_topk
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            annotations = [
                self.transform_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            # Should not be empty during training
            instances = annotations_to_instances(annotations, image_shape)
            dataset_dict["instances"] = instances[instances.gt_boxes.nonempty()]

        return dataset_dict

    def transform_annotations(self, annotation, transforms, image_size):
        """
        Apply image transformations to the annotations.

        After this method, the box mode will be set to XYXY_ABS.
        """
        bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
        # Note that bbox is 1d (per-instance bounding box)
        annotation["bbox"] = transforms.apply_box([bbox])[0]
        annotation["bbox_mode"] = BoxMode.XYXY_ABS

        # each instance contains 1 mask
        if self.mask_on and "segmentation" in annotation:
            annotation["segmentation"] = self._process_mask(annotation["segmentation"], transforms)
        else:
            annotation.pop("segmentation", None)

        # camera
        h, w = image_size
        annotation["K"] = [annotation["K"][0], w / 2.0, h / 2.0]
        annotation["R"] = torch.tensor(annotation["R"])
        annotation["t"] = torch.tensor(annotation["t"])

        if self.zpred_on and "mesh" in annotation:
            annotation["dz"] = self._process_dz(
                annotation["mesh"],
                transforms,
                focal_length=annotation["K"][0],
                R=annotation["R"],
                t=annotation["t"],
            )
        else:
            annotation.pop("dz", None)

        # each instance contains 1 voxel
        if self.voxel_on and "voxel" in annotation:
            annotation["voxel"] = self._process_voxel(
                annotation["voxel"], transforms, R=annotation["R"], t=annotation["t"]
            )
        else:
            annotation.pop("voxel", None)

        # each instance contains 1 mesh
        if self.mesh_on and "mesh" in annotation:
            annotation["mesh"] = self._process_mesh(
                annotation["mesh"], transforms, R=annotation["R"], t=annotation["t"]
            )
        else:
            annotation.pop("mesh", None)

        if self.occ_on and "pointcloud" in annotation:
            annotation["pointcloud"] = self._process_pointcloud(
                annotation["pointcloud"], transforms, R=annotation["R"], t=annotation["t"]
            )
        else:
            annotation.pop("pointcloud", None)

        if self.occ_on and "points" in annotation:
            annotation["points"] = self._process_points(
                annotation["points"], transforms, R=annotation["R"], t=annotation["t"]
            )
        else:
            annotation.pop("points", None)

        if self.occ_on and "occupancies" in annotation:
            annotation["occupancies"] = self._process_occupancies(
                annotation["occupancies"], transforms, R=annotation["R"], t=annotation["t"]
            )
        else:
            annotation.pop("occupancies", None)

        return annotation

    def _process_dz(self, mesh, transforms, focal_length=1.0, R=None, t=None):
        # clone mesh
        verts, faces = mesh
        # transform vertices to camera coordinate system
        verts = shape_utils.transform_verts(verts, R, t)
        assert all(
            isinstance(t, (T.HFlipTransform, T.NoOpTransform, T.ResizeTransform))
            for t in transforms.transforms
        )
        dz = verts[:, 2].max() - verts[:, 2].min()
        z_center = (verts[:, 2].max() + verts[:, 2].min()) / 2.0
        dz = dz / z_center
        dz = dz * focal_length
        for t in transforms.transforms:
            # NOTE normalize the dz by the height scaling of the image.
            # This is necessary s.t. z-regression targets log(dz/roi_h)
            # are invariant to the scaling of the roi_h
            if isinstance(t, T.ResizeTransform):
                dz = dz * (t.new_h * 1.0 / t.h)
        return dz

    def _process_mask(self, mask, transforms):
        # applies image transformations to mask
        mask = np.asarray(Image.open(mask))
        mask = transforms.apply_image(mask)
        mask = torch.as_tensor(np.ascontiguousarray(mask), dtype=torch.float32) / 255.0
        return mask

    def _process_voxel(self, voxel, transforms, R=None, t=None):
        # read voxel
        verts = shape_utils.read_voxel(voxel)
        # transform vertices to camera coordinate system
        verts = shape_utils.transform_verts(verts, R, t)

        # applies image transformations to voxels (represented as verts)
        # NOTE this function does not support generic transforms in T
        # the apply_coords functionality works for voxels for the following
        # transforms (HFlipTransform, NoOpTransform, ResizeTransform)
        assert all(
            isinstance(t, (T.HFlipTransform, T.NoOpTransform, T.ResizeTransform))
            for t in transforms.transforms
        )
        for t in transforms.transforms:
            if isinstance(t, T.HFlipTransform):
                verts[:, 0] = -verts[:, 0]
            elif isinstance(t, T.ResizeTransform):
                verts = t.apply_coords(verts)
            elif isinstance(t, T.NoOpTransform):
                pass
            else:
                raise ValueError("Transform {} not recognized".format(t))
        return verts

    def _process_mesh(self, mesh, transforms, R=None, t=None):
        # clone mesh
        verts, faces = mesh
        # transform vertices to camera coordinate system
        verts = shape_utils.transform_verts(verts, R, t)

        assert all(
            isinstance(t, (T.HFlipTransform, T.NoOpTransform, T.ResizeTransform))
            for t in transforms.transforms
        )
        for t in transforms.transforms:
            if isinstance(t, T.HFlipTransform):
                verts[:, 0] = -verts[:, 0]
            elif isinstance(t, T.ResizeTransform):
                verts = t.apply_coords(verts)
            elif isinstance(t, T.NoOpTransform):
                pass
            else:
                raise ValueError("Transform {} not recognized".format(t))
        return verts, faces

    def _process_pointcloud(self, pointcloud, transforms, R=None, t=None):
        # TODO pointclouds probably don't need any extra transformation
        return pointcloud

    def _process_points(self, points, transforms, R=None, t=None):
        # TODO points probably don't need any extra transformation
        return points

    def _process_occupancies(self, occupancies, transforms, R=None, t=None):
        # TODO occupancies probably don't need any extra transformation
        return occupancies


def load_unique_pointclouds(json_file, model_root):
    with open(json_file, "r") as f:
        annotations = json.load(f)["annotations"]
    # find unique models
    unique_occupancies = []
    for obj in annotations:
        model_type = obj["pointcloud"]
        if model_type not in unique_occupancies:
            unique_occupancies.append(model_type)

    # read unique occupancies
    object_pointclouds = {}
    for model in unique_occupancies:
        pointcloud_dict = np.load(os.path.join(model_root, model))
        occupancy = {
            None: pointcloud_dict['points'].astype(np.float32),
            'normals': pointcloud_dict['normals'].astype(np.float32)
        }
        object_pointclouds[model] = occupancy

    return object_pointclouds


def load_unique_points(json_file, model_root):
    with open(json_file, "r") as f:
        annotations = json.load(f)["annotations"]
    # find unique models
    unique_occupancies = []
    for obj in annotations:
        model_type = obj["points"]
        if model_type not in unique_occupancies:
            unique_occupancies.append(model_type)

    # read unique occupancies
    model_to_points = {}
    for model in unique_occupancies:
        points_dict = np.load(os.path.join(model_root, model))
        points = points_dict['points'].astype(np.float32)
        packed_occupancies = points_dict['occupancies']
        # unpack bits
        unpacked_occupancies = np.unpackbits(packed_occupancies)[:points.shape[0]]
        occupancies = unpacked_occupancies.astype(np.float32)

        model_points = {
            None: points,
            'occupancies': occupancies
        }
        model_to_points[model] = model_points

    return model_to_points
