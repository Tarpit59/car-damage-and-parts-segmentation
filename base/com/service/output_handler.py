"""
output_handler.py
=================
Full pipeline orchestrator. Returns raw detections + analytics for the
controller to use — rendering is done separately per the frontend request.
"""

import cv2
import numpy as np

from .image_analysis          import analyze_single_car_image
from .car_parts_segmentation  import (
    get_car_parts_detections, draw_parts_on_canvas,
    FINAL_CLASSES as PARTS_CLASSES,
    PART_COLORS_BGR, PART_COLORS_HEX,
)
from .car_damage_segmentation import (
    get_car_damage_detections, draw_damage_on_canvas,
    CLASS_NAMES as DAMAGE_CLASSES,
    DAMAGE_COLORS_BGR, DAMAGE_COLORS_HEX,
    draw_curved_mask, draw_label_on_image,
)
from .damage_validation       import filter_false_damage_predictions
from .coco_annotations        import store_annotations_to_json, save_coco_to_disk
from .damage_analytics        import generate_damage_report

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from config import PADDING_RATIO, SAVE_COCO_DEFAULT


# ──────────────────────────────────────────────────────────────────────────────
# Mask helpers
# ──────────────────────────────────────────────────────────────────────────────
def _mask_to_polygon(mask):
    mask_u8   = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons  = []
    for cnt in contours:
        if len(cnt) < 3:
            continue
        polygons.append(cnt.reshape(-1, 2).tolist())
    return polygons


def _box_from_mask(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()-xs.min()), int(ys.max()-ys.min())]


def _build_annotations(detections, class_names, masks):
    annotations = []
    for mask, cid, conf in zip(masks, detections.class_id, detections.confidence):
        polygons   = _mask_to_polygon(mask)
        if not polygons:
            continue
        class_name = class_names[int(cid)].lower().replace(" ", "_")
        annotations.append({
            "class_name":    class_name,
            "bounding_box":  _box_from_mask(mask),
            "segmentation":  polygons,
            "score":         float(round(float(conf), 4)),
        })
    return annotations


def _draw_damage_from_detections(canvas, detections, labels,
                                  damage_filled=True, damage_labels=True, mask_alpha=0.35):
    if detections is None or detections.mask is None or len(detections) == 0:
        return canvas
    for mask, cid in zip(detections.mask, detections.class_id):
        color  = DAMAGE_COLORS_BGR[int(cid) % len(DAMAGE_COLORS_BGR)]
        canvas = draw_curved_mask(canvas, mask, color, alpha=mask_alpha, filled=damage_filled)
    if damage_labels:
        for mask, label, cid in zip(detections.mask, labels, detections.class_id):
            color = DAMAGE_COLORS_BGR[int(cid) % len(DAMAGE_COLORS_BGR)]
            draw_label_on_image(canvas, label, mask, color)
    return canvas

