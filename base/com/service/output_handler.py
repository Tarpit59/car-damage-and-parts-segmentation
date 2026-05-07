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


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────
def process_image(
    image_path,
    padding_ratio=PADDING_RATIO,
    show_parts=True,
    parts_filled=True,
    parts_labels=True,
    show_damage=True,
    damage_filled=True,
    damage_labels=True,
    mask_alpha=0.25,
    save_coco=SAVE_COCO_DEFAULT,
    return_coco=False,
    override_side=None,
):
    empty = dict(
        success=False, coords=None, image_array=None,
        annotated_image=None, coco_data=None, coco_path=None,
        analytics=None, det_parts=None, det_damage=None,
        labels_parts=None, labels_damage=None,
        parts_warning="",
    )

    success, result = analyze_single_car_image(image_path, padding_ratio=padding_ratio)
    if not success:
        return empty

    coords   = result["coords"]
    original = cv2.imread(image_path)
    if original is None:
        return empty

    img_h, img_w = original.shape[:2]
    canvas       = original.copy()
    all_annotations = []
    det_parts = det_damage = None
    labels_parts = labels_damage = []
    parts_warning = ""

    # ── Parts  (ALWAYS run inference; show_parts only controls drawing)
    parts_result = get_car_parts_detections(image_path, coords, override_side=override_side)
    if parts_result is not None:
        det_parts, labels_parts, _, parts_warning = parts_result
        if det_parts.mask is not None:
            all_annotations += _build_annotations(det_parts, PARTS_CLASSES, det_parts.mask)
        if show_parts:
            canvas, _warn = draw_parts_on_canvas(
                canvas, image_path, coords,
                show_parts=True, parts_filled=parts_filled,
                parts_labels=parts_labels, mask_alpha=mask_alpha,
                override_side=override_side,
            )

    # ── Damage  (ALWAYS run inference; show_damage only controls drawing)
    damage_result = get_car_damage_detections(image_path, coords)
    if damage_result is not None:
        det_damage, labels_damage = damage_result
        det_damage, labels_damage = filter_false_damage_predictions(
            det_damage, labels_damage, det_parts,
        )
        if det_damage is not None and len(det_damage) > 0 and det_damage.mask is not None:
            all_annotations += _build_annotations(det_damage, DAMAGE_CLASSES, det_damage.mask)
        if show_damage:
            canvas = _draw_damage_from_detections(
                canvas, det_damage, labels_damage,
                damage_filled=damage_filled, damage_labels=damage_labels,
                mask_alpha=mask_alpha,
            )

    # ── Analytics
    analytics = generate_damage_report(
        det_damage, det_parts, DAMAGE_CLASSES, PARTS_CLASSES,
    )

    # ── COCO
    coco_data = coco_path = None
    if (save_coco or return_coco) and all_annotations:
        coco_data_result = store_annotations_to_json(
            image_path, img_w, img_h, all_annotations,
        )
        if save_coco:
            coco_path = save_coco_to_disk(image_path, coco_data_result)
        if return_coco:
            coco_data = coco_data_result

    return dict(
        success=True, coords=coords,
        image_array=original, annotated_image=canvas,
        coco_data=coco_data, coco_path=coco_path,
        analytics=analytics,
        det_parts=det_parts, det_damage=det_damage,
        labels_parts=labels_parts, labels_damage=labels_damage,
        parts_warning=parts_warning,
    )


def render_on_demand(
    image_path,
    coords,
    det_parts,
    labels_parts,
    det_damage,
    labels_damage,
    show_parts=True,
    parts_filled=True,
    parts_labels=True,
    show_damage=True,
    damage_filled=True,
    damage_labels=True,
    mask_alpha=0.35,
):

    original = cv2.imread(image_path)
    if original is None:
        return None
    canvas = original.copy()

    if show_parts and det_parts is not None and det_parts.mask is not None:
        for mask, cid in zip(det_parts.mask, det_parts.class_id):
            color  = PART_COLORS_BGR[int(cid) % len(PART_COLORS_BGR)]
            canvas = draw_curved_mask(canvas, mask, color, alpha=mask_alpha, filled=parts_filled)
        if parts_labels:
            for mask, label, cid in zip(det_parts.mask, labels_parts, det_parts.class_id):
                color = PART_COLORS_BGR[int(cid) % len(PART_COLORS_BGR)]
                draw_label_on_image(canvas, label, mask, color)

    if show_damage:
        canvas = _draw_damage_from_detections(
            canvas, det_damage, labels_damage,
            damage_filled=damage_filled, damage_labels=damage_labels,
            mask_alpha=mask_alpha,
        )

    return canvas
