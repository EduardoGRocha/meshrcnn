import json

INPUT_FILE_PATH = 'pix3d_s1_occ_test.json'
OUTPUT_FILE_PATH = 'pix3d_s1_occ_test_small.json'
NUM_MODELS = 12

# Very ugly code. Don't do this at home, kids
with open(INPUT_FILE_PATH, 'r') as in_file:
    in_data = json.load(in_file)
    
categories = [c['name'] for c in in_data['categories']]

annotations_per_categories = {}

for c in categories:
    annotations_per_categories[c] = [a for a in in_data['annotations'] if c in a['model']]


models_per_category = {i:[a['model'] for a in j] for i,j in annotations_per_categories.items()}

for category, models in models_per_category.items():
    unique_models = list(set(models))
    unique_models = unique_models[0:min(NUM_MODELS, len(unique_models))]
    models_per_category[category] = unique_models

new_annotations = []
for category in in_data['categories']:
    category = category['name']
    models = models_per_category[category]
    model_annotations = [a for a in in_data['annotations'] if a['model'] in models]
    new_annotations += model_annotations
    
image_ids = [a['image_id'] for a in new_annotations]
new_images = [img for img in in_data['images'] if img['id'] in image_ids]

in_data['images'] = new_images
in_data['annotations'] = new_annotations

with open(OUTPUT_FILE_PATH, 'w') as out_:
    json.dump(in_data, out_)
