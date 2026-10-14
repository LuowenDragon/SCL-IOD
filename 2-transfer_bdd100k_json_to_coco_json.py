import json
from collections import defaultdict

bdd_json_path = "bdd100k_labels_images_daytime_clear_val.json"
coco_json_path = "bdd100k_labels_images_daytime_clear_val_coco_format.json"

keep_categories = ['bike', 'bus', 'car', 'motor', 'person', 'rider', 'truck']

with open(bdd_json_path, "r") as f:
    bdd_data = json.load(f)

images = []
annotations = []
categories = []
category2id = {}

for i, cat in enumerate(keep_categories):
    category2id[cat] = i + 1
    categories.append({
        "id": i + 1,
        "name": cat
    })

category_count = defaultdict(int)

image_id = 0
annotation_id = 0

for item in bdd_data:

    image_id += 1

    images.append({
        "id": image_id,
        "file_name": item["name"],
        "width": 1280,
        "height": 720
    })

    if "labels" not in item:
        continue

    for label in item["labels"]:

        if "box2d" not in label:
            continue

        cat = label["category"]

        if cat not in keep_categories:
            continue

        box = label["box2d"]

        x1 = box["x1"]
        y1 = box["y1"]
        x2 = box["x2"]
        y2 = box["y2"]

        w = x2 - x1
        h = y2 - y1

        annotation_id += 1

        annotations.append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category2id[cat],
            "bbox": [x1, y1, w, h],
            "area": w * h,
            "iscrowd": 0
        })

        category_count[cat] += 1

coco = {
    "images": images,
    "annotations": annotations,
    "categories": categories
}

with open(coco_json_path, "w") as f:
    json.dump(coco, f)

print("===== Category Statistics =====")
total = 0
for cat in keep_categories:
    num = category_count[cat]
    print(f"{cat}: {num}")
    total += num

print("-------------------------------")
print("Total instances:", total)
print("Saved to:", coco_json_path)