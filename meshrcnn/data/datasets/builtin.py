# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".
"""
import os
from detectron2.data import DatasetCatalog, MetadataCatalog

from meshrcnn.data.datasets import load_pix3d_json, load_shapenet_json


def get_pix3d_metadata():
    meta = [
        {"name": "bed", "color": [255, 255, 25], "id": 1},  # noqa
        {"name": "bookcase", "color": [230, 25, 75], "id": 2},  # noqa
        {"name": "chair", "color": [250, 190, 190], "id": 3},  # noqa
        {"name": "desk", "color": [60, 180, 75], "id": 4},  # noqa
        {"name": "misc", "color": [230, 190, 255], "id": 5},  # noqa
        {"name": "sofa", "color": [0, 130, 200], "id": 6},  # noqa
        {"name": "table", "color": [245, 130, 48], "id": 7},  # noqa
        {"name": "tool", "color": [70, 240, 240], "id": 8},  # noqa
        {"name": "wardrobe", "color": [210, 245, 60], "id": 9},  # noqa
    ]
    return meta


def get_shapenet_metadata():
    meta = [
        {"name": "telephone", "color": [255, 255, 25], "id": 1},
    ]

    return meta


PIX3D_SPLITS = {
    "pix3d_s1_train": ("pix3d", "pix3d/pix3d_s1_train.json"),
    "pix3d_s1_test": ("pix3d", "pix3d/pix3d_s1_test.json"),
    "pix3d_s2_train": ("pix3d", "pix3d/pix3d_s2_train.json"),
    "pix3d_s2_test": ("pix3d", "pix3d/pix3d_s2_test.json"),
    "pix3d_s1_train_bookcase": ("pix3d", "pix3d/pix3d_s1_train_bookcase.json"),
    "pix3d_s1_test_bookcase": ("pix3d", "pix3d/pix3d_s1_test_bookcase.json"),
    "pix3d_s1_train_bookcase_tool": ("pix3d", "pix3d/pix3d_s1_train_bookcase_tool.json"),
    "pix3d_s1_test_bookcase_tool": ("pix3d", "pix3d/pix3d_s1_test_bookcase_tool.json"),
}

SHAPENET_SPLITS = {
    "shapenet_phones": ("ShapeNetPhonesNew", "ShapeNetPhonesNew/shapenet_phones.json"),
}


def register_pix3d(dataset_name, json_file, image_root, root="datasets"):
    DatasetCatalog.register(
        dataset_name, lambda: load_pix3d_json(json_file, image_root, dataset_name)
    )
    things_ids = [k["id"] for k in get_pix3d_metadata()]
    if "bookcase" in dataset_name and "tool" in dataset_name:
        thing_classes = ["bookcase", "tool"]
        thing_colors = [[230, 25, 75], [70, 240, 240]]
        thing_dataset_id_to_contiguous_id = {1: 0, 2: 1}
    elif "bookcase" in dataset_name:
        thing_classes = ["bookcase"]
        thing_colors = [[230, 25, 75]]
        thing_dataset_id_to_contiguous_id = {1: 0}
    else:
        thing_classes = [k["name"] for k in get_pix3d_metadata()]
        thing_colors = [k["color"] for k in get_pix3d_metadata()]
        thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(things_ids)}
    json_file = os.path.join(root, json_file)
    image_root = os.path.join(root, image_root)
    metadata = {
        "thing_classes": thing_classes,
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_colors": thing_colors,
    }
    MetadataCatalog.get(dataset_name).set(
        json_file=json_file, image_root=image_root, evaluator_type="pix3d", **metadata
    )


def register_shapenet(dataset_name, json_file, image_root, root="datasets"):
    DatasetCatalog.register(
        dataset_name, lambda: load_shapenet_json(json_file, image_root, dataset_name)
    )
    things_ids = [k["id"] for k in get_shapenet_metadata()]
    thing_classes = [k["name"] for k in get_shapenet_metadata()]
    thing_colors = [k["color"] for k in get_shapenet_metadata()]

    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(things_ids)}
    json_file = os.path.join(root, json_file)
    image_root = os.path.join(root, image_root)
    metadata = {
        "thing_classes": thing_classes,
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_colors": thing_colors,
    }

    MetadataCatalog.get(dataset_name).set(
        json_file=json_file, image_root=image_root, evaluator_type="shapenet", **metadata
    )


for key, (data_root, anno_file) in PIX3D_SPLITS.items():
    register_pix3d(key, anno_file, data_root)

for key, (data_root, anno_file) in SHAPENET_SPLITS.items():
    register_shapenet(key, anno_file, data_root)
