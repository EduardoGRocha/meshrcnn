from input_shapenet_dataset import InputShapenetDataset
from pathlib import Path
import json

SOURCE_FOLDER = 'ShapeNetPhones'
TARGET_FOLDER = 'ShapeNetPhonesNew'


if __name__ == "__main__":
    input_directory = Path(SOURCE_FOLDER)
    output_directory = Path(TARGET_FOLDER)

    input_dataset = InputShapenetDataset(
        input_directory, input_directory / 'metadata.yaml'
    )

    categories, annotations, images = input_dataset.get_data(output_directory)
    data = {
        "categories": categories,
        "annotations": annotations,
        "licenses": {},
        "images": images,
        "info": {}
    }

    with open(TARGET_FOLDER + '/shapenet_phones.json', 'w') as output_json:
        json.dump(data, output_json)
