"""
coco_annotations.py
===================
COCO JSON export.
"""

import json
import os

CAR_PARTS_CLASSES = [
    "diggi_back_door", "diggi_back_door_glass", "front_bumper", "front_windshield_glass", "grill",
    "hood_bonnet", "left_fender", "left_front_door", "left_front_door_glass", "left_headlight",
    "left_quarter_panel", "left_rear_door", "left_rear_door_glass", "left_running_board",
    "left_side_mirror", "left_taillight", "rear_bumper", "right_fender", "right_front_door",
    "right_front_door_glass", "right_headlight", "right_quarter_panel", "right_rear_door",
    "right_rear_door_glass", "right_running_board", "right_side_mirror", "right_taillight", "roof", "tyre",
]
CAR_DAMAGES_CLASSES = ["crack", "dent", "glass_shatter", "lamp_broken", "scratch", "tire_flat"]

COCO_TEMPLATE = {
    "licenses": [{"name": "", "id": 0, "url": ""}],
    "info":     {"contributor": "", "date_created": "", "description": "", "url": "", "version": "", "year": ""},
    "categories": [], "images": [], "annotations": [],
}


def _polygon_area(polygon):
    if polygon and isinstance(polygon[0], (list, tuple)):
        if isinstance(polygon[0][0], (list, tuple)):
            polygon = polygon[0]
    area, n = 0, len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2


def store_annotations_to_json(image_path, width, height, annotations):
    import copy
    coco = copy.deepcopy(COCO_TEMPLATE)

    existing = {cat["name"] for cat in coco["categories"]}
    for name in CAR_PARTS_CLASSES:
        if name not in existing:
            coco["categories"].append({"id": len(coco["categories"]), "name": name, "supercategory": "car_parts"})
            existing.add(name)
    for name in CAR_DAMAGES_CLASSES:
        if name not in existing:
            coco["categories"].append({"id": len(coco["categories"]), "name": name, "supercategory": "car_damages"})
            existing.add(name)

    image_name = os.path.basename(image_path)
    image_id   = len(coco["images"])
    coco["images"].append({
        "id": image_id, "width": width, "height": height,
        "file_name": image_name, "license": 0,
        "flickr_url": "", "coco_url": "", "date_captured": 0,
    })

    for ann in annotations:
        cat_name = ann["class_name"]
        cat_id   = next((c["id"] for c in coco["categories"] if c["name"] == cat_name), None)
        if cat_id is None:
            continue
        seg  = ann.get("segmentation", [[]])
        area = _polygon_area(seg[0]) if seg and seg[0] else 0
        coco["annotations"].append({
            "id":           len(coco["annotations"]),
            "image_id":     image_id,
            "category_id":  cat_id,
            "segmentation": seg[0] if seg else [],
            "area":         area,
            "bbox":         ann["bounding_box"],
            "iscrowd":      0,
            "attributes":   {"occluded": False},
        })

    return coco

