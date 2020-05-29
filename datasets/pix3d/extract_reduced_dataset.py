import json


CATEGORIES = ['bookcase']
INPUT_FILE = 'pix3d_s1_train'
OUTPUT_FILE = INPUT_FILE + "_" + "_".join(CATEGORIES)

with open(INPUT_FILE + '.json', 'r') as input_file:
    data = json.load(input_file)
  
for category in data["categories"]:
    name = category["name"]
    id = category["id"]
    length = len([i for i in data["annotations"] if i["category_id"] == id])
    print("%s contains %i" % (name, length))
 
print("...")

reduced_data = {}
reduced_data["licenses"] = data["licenses"]
reduced_data["info"] = data["info"]
reduced_data["categories"] = \
    [i for i in data["categories"] if i["name"] in CATEGORIES]

category_ids = [i["id"] for i in reduced_data["categories"]]
reduced_data["annotations"] = \
    [i for i in data["annotations"] if i["category_id"] in category_ids]

image_ids = [i["image_id"] for i in reduced_data["annotations"]]
reduced_data["images"] = [i for i in data["images"] if i["id"] in image_ids]

print("Adding %i images..." % len(image_ids))

with open(OUTPUT_FILE + '.json', 'w') as output_file:
    json.dump(reduced_data, output_file)
