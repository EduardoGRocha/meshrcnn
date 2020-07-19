import json
from pathlib import Path
import os

POINTS_FILE_NAME = 'points.npz'
POINTCLOUD_FILE_NAME = 'pointcloud.npz'

# FILE_PATH = 'pix3d_occupancies/pix3d_s1_train_bookcase_tool.json'
FILE_PATH = 'pix3d/pix3d_s1_test.json'
OUT_PATH = 'pix3d/pix3d_s1_occ_test.json'

with open(FILE_PATH, 'r') as in_file:
    data = json.load(in_file)
    
annotations = data['annotations']

annotations_new = []

for annotation in annotations:
    dir_path = Path(annotation['voxel']).parent
    points_path = str(dir_path / POINTS_FILE_NAME).replace('model', 'occupancies')
    pointcloud_path = str(dir_path / POINTCLOUD_FILE_NAME).replace('model', 'occupancies')
    annotation['points'] = points_path
    annotation['pointcloud'] = pointcloud_path
    
    if os.path.isfile('pix3d/' + annotation['points']):
        annotations_new.append(annotation)
    
data['annotations'] = annotations_new

with open(OUT_PATH, 'w') as out_file:
    json.dump(data, out_file)
    
