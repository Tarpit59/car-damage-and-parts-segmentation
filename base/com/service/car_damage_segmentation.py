"""
car_damage_segmentation.py
==========================
RF-DETR-based car-damage segmentation with tile inference.
"""

import cv2
import math
import numpy as np
from PIL import Image
from rfdetr import RFDETRSegMedium

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from config import (
    DAMAGE_CHECKPOINT, DAMAGE_RESOLUTION, DAMAGE_THRESHOLD,
    TILE_THRESHOLD, TOUCH_GAP_PIXELS,
    USE_TILE_INFERENCE, GRID_ROWS, GRID_COLS, OVERLAP_RATIO, MIN_TILE_SIZE,
    TILE_DAMAGE_NAMES,
    BOX_IOU_THRESHOLD, MASK_IOU_THRESHOLD, MASK_COVERAGE_THRESHOLD,
)

# ──────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "crack", "dent", "glass shatter", "lamp broken", "scratch", "tire flat",
]

DAMAGE_COLOR_HEX = {
    "crack":         "#FF0000",
    "dent":          "#0000FF",
    "glass shatter": "#00FFFF",
    "lamp broken":   "#FFA500",
    "scratch":       "#00FF00",
    "tire flat":     "#800080",
}

def _hex_to_bgr(h):
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)

DAMAGE_COLORS_BGR = [_hex_to_bgr(DAMAGE_COLOR_HEX[c]) for c in CLASS_NAMES]
DAMAGE_COLORS_HEX = [DAMAGE_COLOR_HEX[c] for c in CLASS_NAMES]

TILE_ALLOWED_IDS = {
    i for i, name in enumerate(CLASS_NAMES)
    if name.lower() in TILE_DAMAGE_NAMES
}

# ──────────────────────────────────────────────────────────────────────────────
model = RFDETRSegMedium(pretrain_weights=DAMAGE_CHECKPOINT, resolution=DAMAGE_RESOLUTION)
try:
    import torch
    model.optimize_for_inference(compile=True, batch_size=1, dtype=torch.float32)
    print("[INFO] Damage model optimized for inference (torch.jit.trace, fp32).")
except Exception as e:
    print(f"[WARNING] Damage model optimization skipped: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# TILE HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def compute_tiles(width, height, rows=2, cols=2, overlap=0.2):
    tiles  = []
    tile_w = max(MIN_TILE_SIZE, math.ceil(width  / cols))
    tile_h = max(MIN_TILE_SIZE, math.ceil(height / rows))
    step_x = max(1, int(tile_w * (1 - overlap)))
    step_y = max(1, int(tile_h * (1 - overlap)))

    xs, ys = [], []
    x = 0
    while x < width:
        xs.append(x)
        if x + tile_w >= width:
            break
        x += step_x
    y = 0
    while y < height:
        ys.append(y)
        if y + tile_h >= height:
            break
        y += step_y

    for yy in ys:
        for xx in xs:
            x2 = min(xx + tile_w, width)
            y2 = min(yy + tile_h, height)
            x1 = max(0, x2 - tile_w)
            y1 = max(0, y2 - tile_h)
            tiles.append((x1, y1, x2, y2))
    return tiles


def box_iou(b1, b2):
    xa = max(b1[0], b2[0]); ya = max(b1[1], b2[1])
    xb = min(b1[2], b2[2]); yb = min(b1[3], b2[3])
    inter = max(0, xb - xa) * max(0, yb - ya)
    a1 = max(0, b1[2]-b1[0]) * max(0, b1[3]-b1[1])
    a2 = max(0, b2[2]-b2[0]) * max(0, b2[3]-b2[1])
    return inter / (a1 + a2 - inter + 1e-6)


def mask_iou(m1, m2):
    if m1 is None or m2 is None:
        return 0.0
    return np.logical_and(m1, m2).sum() / (np.logical_or(m1, m2).sum() + 1e-6)


def mask_coverage(m1, m2):
    if m1 is None or m2 is None:
        return 0.0
    inter   = np.logical_and(m1, m2).sum()
    smaller = min(m1.sum(), m2.sum()) + 1e-6
    return inter / smaller


def merge_masks(a, b):
    if a is None: return b
    if b is None: return a
    return np.logical_or(a, b)


def append_detection(det, offset_x, offset_y, full_h, full_w,
                     all_boxes, all_scores, all_class_ids, all_masks,
                     allowed_ids=None):
    if len(det) == 0:
        return
    for i in range(len(det)):
        cid = int(det.class_id[i])
        if allowed_ids is not None and cid not in allowed_ids:
            continue
        box = det.xyxy[i].copy()
        box[0] += offset_x; box[2] += offset_x
        box[1] += offset_y; box[3] += offset_y
        all_boxes.append(box)
        all_scores.append(float(det.confidence[i]))
        all_class_ids.append(cid)
        if det.mask is not None:
            m = det.mask[i].astype(bool)
            blank = np.zeros((full_h, full_w), dtype=bool)
            mh, mw = m.shape
            blank[offset_y:offset_y + mh, offset_x:offset_x + mw] = m
            all_masks.append(blank)
        else:
            all_masks.append(None)


def masks_touch_or_overlap(m1, m2, gap=3):
    if m1 is None or m2 is None:
        return False
    if np.logical_and(m1, m2).any():
        return True
    kernel = np.ones((gap * 2 + 1, gap * 2 + 1), np.uint8)
    dilated = cv2.dilate((m1.astype(np.uint8)) * 255, kernel, iterations=1).astype(bool)
    return np.logical_and(dilated, m2).any()


def smart_merge_nms(boxes, scores, class_ids, masks):
    idxs  = list(np.argsort(scores)[::-1])
    used  = set()
    fb, fs, fc, fm = [], [], [], []

    for i in idxs:
        if i in used:
            continue
        cb, cs, cc, cm = boxes[i].copy(), scores[i], class_ids[i], masks[i]
        used.add(i)
        changed = True
        while changed:
            changed = False
            for j in idxs:
                if j in used or class_ids[j] != cc:
                    continue
                should_merge = (
                    box_iou(cb, boxes[j]) > BOX_IOU_THRESHOLD or
                    mask_iou(cm, masks[j]) > MASK_IOU_THRESHOLD or
                    mask_coverage(cm, masks[j]) > MASK_COVERAGE_THRESHOLD or
                    masks_touch_or_overlap(cm, masks[j], TOUCH_GAP_PIXELS)
                )
                if should_merge:
                    cb[0] = min(cb[0], boxes[j][0]); cb[1] = min(cb[1], boxes[j][1])
                    cb[2] = max(cb[2], boxes[j][2]); cb[3] = max(cb[3], boxes[j][3])
                    cs = max(cs, scores[j])
                    cm = merge_masks(cm, masks[j])
                    used.add(j); changed = True
        fb.append(cb); fs.append(cs); fc.append(cc); fm.append(cm)

    return (
        np.array(fb, dtype=np.float32),
        np.array(fs, dtype=np.float32),
        np.array(fc, dtype=np.int32),
        fm,
    )


# ──────────────────────────────────────────────────────────────────────────────
# DRAW HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def smooth_mask_contour(mask, smoothing=0.003):
    mask_uint8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    smoothed = []
    for cnt in contours:
        if len(cnt) < 5:
            smoothed.append(cnt); continue
        epsilon = smoothing * cv2.arcLength(cnt, True)
        smoothed.append(cv2.approxPolyDP(cnt, epsilon, True))
    return smoothed


def draw_curved_mask(canvas, mask, color_bgr, alpha=0.35, filled=True):
    contours = smooth_mask_contour(mask)
    if not contours:
        return canvas
    if filled:
        overlay = canvas.copy()
        cv2.fillPoly(overlay, contours, color=color_bgr)
        canvas = cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0)
    H, W      = canvas.shape[:2]
    thickness = max(1, int(min(H, W) / 500))
    cv2.polylines(canvas, contours, isClosed=True,
                  color=(255, 255, 255), thickness=thickness)
    return canvas


def draw_label_on_image(canvas, label, mask, color_bgr):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return
    cx = int(np.mean(xs)); cy = int(np.mean(ys))
    H, W      = canvas.shape[:2]
    scale     = max(0.3, min(1.2, min(H, W) / 1500))
    thickness = max(1, int(scale * 2))
    padding   = max(2, int(min(H, W) / 300))
    font      = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
    tx = max(padding, min(cx - tw // 2, W - tw - padding))
    ty = max(th + padding, min(cy + th // 2, H - baseline - padding))
    b, g, r  = color_bgr
    bg_color = (int(b * 0.25), int(g * 0.25), int(r * 0.25))
    cv2.rectangle(canvas, (tx - padding, ty - th - padding),
                  (tx + tw + padding, ty + baseline + padding), bg_color, -1)
    cv2.putText(canvas, label, (tx, ty), font, scale, color_bgr, thickness, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN INFERENCE
# ──────────────────────────────────────────────────────────────────────────────
def get_car_damage_detections(image_path, coords):
    x1, y1, x2, y2 = coords
    full_bgr = cv2.imread(image_path)
    if full_bgr is None:
        return None

    H, W     = full_bgr.shape[:2]
    roi_bgr  = full_bgr[y1:y2, x1:x2]
    if roi_bgr.size == 0:
        return None

    roi_h, roi_w = roi_bgr.shape[:2]
    roi_pil  = Image.fromarray(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB))

    if not USE_TILE_INFERENCE:
        detections = model.predict(roi_pil, threshold=DAMAGE_THRESHOLD)
        if len(detections) == 0:
            return None
        detections.xyxy[:, [0, 2]] += x1
        detections.xyxy[:, [1, 3]] += y1
        if detections.mask is not None:
            full_masks = []
            for m in detections.mask:
                blank = np.zeros((H, W), dtype=bool)
                mh, mw = m.shape
                blank[y1:y1 + mh, x1:x1 + mw] = m
                full_masks.append(blank)
            detections.mask = np.array(full_masks)
        labels = [f"{CLASS_NAMES[cid]} {conf:.2f}"
                  for cid, conf in zip(detections.class_id, detections.confidence)]
        return detections, labels

    all_boxes, all_scores, all_class_ids, all_masks = [], [], [], []

    det_full = model.predict(roi_pil, threshold=DAMAGE_THRESHOLD)
    append_detection(det_full, x1, y1, H, W,
                     all_boxes, all_scores, all_class_ids, all_masks, allowed_ids=None)

    for tx1, ty1, tx2, ty2 in compute_tiles(roi_w, roi_h, GRID_ROWS, GRID_COLS, OVERLAP_RATIO):
        tile     = roi_pil.crop((tx1, ty1, tx2, ty2))
        det_tile = model.predict(tile, threshold=TILE_THRESHOLD)
        append_detection(det_tile, x1 + tx1, y1 + ty1, H, W,
                         all_boxes, all_scores, all_class_ids, all_masks,
                         allowed_ids=TILE_ALLOWED_IDS)

    if len(all_boxes) == 0:
        return None

    boxes, scores, class_ids, masks = smart_merge_nms(
        np.array(all_boxes, dtype=np.float32),
        np.array(all_scores, dtype=np.float32),
        np.array(all_class_ids, dtype=np.int32),
        all_masks,
    )

    detections = type(det_full)(
        xyxy=boxes, confidence=scores, class_id=class_ids,
        mask=np.array(masks) if masks[0] is not None else None,
    )
    labels = [f"{CLASS_NAMES[cid]} {conf:.2f}"
              for cid, conf in zip(class_ids, scores)]
    return detections, labels


def draw_damage_on_canvas(
    canvas, image_path, coords,
    show_damage=True, damage_filled=True,
    damage_labels=True, mask_alpha=0.35,
):
    if not show_damage:
        return canvas
    result = get_car_damage_detections(image_path, coords)
    if result is None:
        return canvas
    detections, labels = result
    if detections.mask is None:
        return canvas
    for mask, cid in zip(detections.mask, detections.class_id):
        color  = DAMAGE_COLORS_BGR[cid % len(DAMAGE_COLORS_BGR)]
        canvas = draw_curved_mask(canvas, mask, color, alpha=mask_alpha, filled=damage_filled)
    if damage_labels:
        for mask, label, cid in zip(detections.mask, labels, detections.class_id):
            color = DAMAGE_COLORS_BGR[cid % len(DAMAGE_COLORS_BGR)]
            draw_label_on_image(canvas, label, mask, color)
    return canvas
