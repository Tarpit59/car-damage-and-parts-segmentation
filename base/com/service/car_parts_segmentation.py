"""
car_parts_segmentation.py
=========================
RF-DETR-based car-parts segmentation.
"""

import os
import cv2
import numpy as np
from PIL import Image
from rfdetr import RFDETRSegNano

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from config import PARTS_CHECKPOINT, PARTS_RESOLUTION, PARTS_THRESHOLD

# ──────────────────────────────────────────────────────────────────────────────
# CLASS NAMES
# ──────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    'Diggi_Back_Door', 'Diggi_Back_Door_Glass', 'Fender', 'Front_Bumper',
    'Front_Door', 'Front_Door_Glass', 'Front_Windshield_Glass', 'Grill',
    'Headlight', 'Hood_Bonnet', 'Quarter_Panel', 'Rear_Bumper',
    'Rear_Door', 'Rear_Door_Glass', 'Roof', 'Running_Board',
    'Side_Mirror', 'Taillight', 'tyre',
]

FINAL_CLASSES = [
    'Diggi_Back_Door', 'Diggi_Back_Door_Glass', 'Front_Bumper',
    'Front_Windshield_Glass', 'Grill', 'Hood_Bonnet',
    'Left_Fender', 'Left_Front_Door', 'Left_Front_Door_Glass',
    'Left_Headlight', 'Left_Quarter_Panel', 'Left_Rear_Door',
    'Left_Rear_Door_Glass', 'Left_Running_Board', 'Left_Side_Mirror',
    'Left_Taillight', 'Rear_Bumper',
    'Right_Fender', 'Right_Front_Door', 'Right_Front_Door_Glass',
    'Right_Headlight', 'Right_Quarter_Panel', 'Right_Rear_Door',
    'Right_Rear_Door_Glass', 'Right_Running_Board',
    'Right_Side_Mirror', 'Right_Taillight',
    'Roof', 'tyre',
]

# Classes that can appear AT MOST ONCE per image — enforced after conversion.
# If the model produces multiple detections for the same singleton class,
# only the highest-confidence one is kept.
SINGLETON_FINAL_CLASSES = {
    'Diggi_Back_Door_Glass', 'Front_Windshield_Glass',
    'Left_Fender',       'Left_Front_Door',       'Left_Front_Door_Glass',
    'Left_Headlight',    'Left_Quarter_Panel',    'Left_Rear_Door',
    'Left_Rear_Door_Glass', 'Left_Running_Board', 'Left_Side_Mirror',
    'Left_Taillight',
    'Right_Fender',      'Right_Front_Door',      'Right_Front_Door_Glass',
    'Right_Headlight',   'Right_Quarter_Panel',   'Right_Rear_Door',
    'Right_Rear_Door_Glass', 'Right_Running_Board',
    'Right_Side_Mirror', 'Right_Taillight',
}

# ──────────────────────────────────────────────────────────────────────────────
# COLORS
# ──────────────────────────────────────────────────────────────────────────────
_HEX_COLORS = [
    "#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231",
    "#48F90A", "#92CC17", "#3DDB86", "#1A9334", "#00D4BB",
    "#2C99A8", "#00C2FF", "#344593", "#6473FF", "#0018EC",
    "#8438FF", "#520085", "#CB38FF", "#FF95C8", "#A52A2A",
    "#00FF7F", "#4682B4", "#DA70D6", "#FFD700", "#7FFF00",
    "#DC143C", "#00CED1", "#8A2BE2", "#FF1493",
]

def _hex_to_bgr(h):
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)

def _hex_to_rgb_tuple(h):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))

PART_COLORS_BGR = [_hex_to_bgr(c) for c in _HEX_COLORS[:len(FINAL_CLASSES)]]
PART_COLORS_HEX = _HEX_COLORS[:len(FINAL_CLASSES)]

# ──────────────────────────────────────────────────────────────────────────────
# MODEL (loaded once)
# ──────────────────────────────────────────────────────────────────────────────
model = RFDETRSegNano(pretrain_weights=PARTS_CHECKPOINT, resolution=PARTS_RESOLUTION)
try:
    import torch
    model.optimize_for_inference(compile=True, batch_size=1, dtype=torch.float32)
    print("[INFO] Parts model optimized for inference (torch.jit.trace, fp32).")
except Exception as e:
    print(f"[WARNING] Parts model optimization skipped: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def get_car_side(image_path, override_side=None):
    """Return the car side inferred from the image filename, or None if not determinable.

    Args:
        image_path:    Path to the image file.
        override_side: Optional side string provided externally (e.g. from a UI dropdown).
                       Used as a fallback ONLY when the filename does not encode the side.
                       Filename always takes priority.
    """
    name = os.path.basename(image_path).lower()
    valid_sides = [
        "front_left", "front_right", "front",
        "left", "rear_left", "rear_right", "rear", "right",
    ]
    for side in valid_sides:
        if name.startswith(side):
            return side          # Filename wins — ignore override
    # Filename has no side info; try the override (dropdown selection)
    if override_side and override_side.strip() in valid_sides:
        return override_side.strip()
    return None  # Side is not determinable


def get_lr_from_box(view_side, box, img_w):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2
    if view_side.startswith("front"):
        return "Right" if cx < img_w / 2 else "Left"
    return "Left" if cx < img_w / 2 else "Right"


def count_class_instances(detections, target_id):
    return int(np.sum(detections.class_id == target_id))


# Classes that require knowing the car's Left/Right side
_SIDE_DEPENDENT_CLASSES = {
    "Fender", "Front_Door", "Front_Door_Glass",
    "Headlight", "Quarter_Panel", "Rear_Door",
    "Rear_Door_Glass", "Running_Board", "Side_Mirror", "Taillight",
}


def convert_class(cls_name, side, box=None, img_w=None, class_count=1):
    """Convert a detected class name to its final Left/Right prefixed form.

    Returns None if the class requires a car-side context that is not available.
    """
    fixed_classes = {
        "Diggi_Back_Door", "Diggi_Back_Door_Glass", "Front_Bumper",
        "Front_Windshield_Glass", "Grill", "Hood_Bonnet",
        "Rear_Bumper", "Roof", "tyre",
    }
    if cls_name in fixed_classes:
        return cls_name

    # If side is unknown, skip side-dependent classes instead of guessing
    if side is None:
        if cls_name in _SIDE_DEPENDENT_CLASSES:
            return None  # Cannot determine Left/Right — skip this detection
        return cls_name  # Non-side-dependent, return as-is

    if box is None:
        return cls_name

    if side in ["front_left", "left", "rear_left"]:
        prefix = "Left"
    elif side in ["front_right", "right", "rear_right"]:
        prefix = "Right"
    else:
        prefix = get_lr_from_box(side, box, img_w)

    mapping = {
        "Fender":          f"{prefix}_Fender",
        "Front_Door":      f"{prefix}_Front_Door",
        "Front_Door_Glass":f"{prefix}_Front_Door_Glass",
        "Headlight":       f"{prefix}_Headlight",
        "Quarter_Panel":   f"{prefix}_Quarter_Panel",
        "Rear_Door":       f"{prefix}_Rear_Door",
        "Rear_Door_Glass": f"{prefix}_Rear_Door_Glass",
        "Running_Board":   f"{prefix}_Running_Board",
        "Side_Mirror":     f"{prefix}_Side_Mirror",
        "Taillight":       f"{prefix}_Taillight",
    }
    return mapping.get(cls_name, cls_name)


def smooth_mask_contour(mask, smoothing=0.003):
    mask_uint8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    smoothed = []
    for cnt in contours:
        if len(cnt) < 5:
            smoothed.append(cnt)
            continue
        perimeter = cv2.arcLength(cnt, True)
        epsilon   = smoothing * perimeter
        approx    = cv2.approxPolyDP(cnt, epsilon, True)
        smoothed.append(approx)
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
    cx = int(np.mean(xs))
    cy = int(np.mean(ys))
    H, W      = canvas.shape[:2]
    scale     = max(0.3, min(1.2, min(H, W) / 1500))
    thickness = max(1, int(scale * 2))
    padding   = max(2, int(min(H, W) / 300))
    font      = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
    tx = max(padding, min(cx - tw // 2, W - tw - padding))
    ty = max(th + padding, min(cy + th // 2, H - baseline - padding))
    b, g, r   = color_bgr
    bg_color  = (int(b * 0.25), int(g * 0.25), int(r * 0.25))
    cv2.rectangle(canvas, (tx - padding, ty - th - padding),
                  (tx + tw + padding, ty + baseline + padding), bg_color, -1)
    cv2.putText(canvas, label, (tx, ty), font, scale, color_bgr, thickness, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# DEDUPLICATION HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def _suppress_duplicate_raw_parts(class_ids, confidences, masks, iou_threshold=0.30):
    """
    Within each raw MODEL class (19 classes), greedily suppress overlapping
    detections whose mask IoU exceeds `iou_threshold`, keeping the
    highest-confidence one.  Runs BEFORE convert_class() so that the
    forced_names / left-right assignment logic sees clean inputs.

    Returns a sorted list of kept indices.
    """
    n = len(class_ids)
    if n == 0:
        return []

    order = list(np.argsort(confidences)[::-1])   # highest conf first
    suppressed = set()
    kept = []

    for i in order:
        if i in suppressed:
            continue
        kept.append(i)
        mi = masks[i] if (masks is not None and i < len(masks)) else None
        if mi is None:
            continue
        mi_sum = float(mi.sum())
        for j in order:
            if j in suppressed or j == i:
                continue
            if class_ids[j] != class_ids[i]:
                continue   # only suppress same raw class
            mj = masks[j] if (masks is not None and j < len(masks)) else None
            if mj is None:
                continue
            inter = float(np.logical_and(mi, mj).sum())
            union = float(np.logical_or(mi, mj).sum())
            if union > 0 and inter / union > iou_threshold:
                suppressed.add(j)

    return sorted(kept)


# ──────────────────────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ──────────────────────────────────────────────────────────────────────────────
def get_car_parts_detections(image_path, coords, override_side=None):
    """
    Detect car parts in the given image crop.

    Args:
        image_path:    Path to the image.
        coords:        (x1, y1, x2, y2) bounding box of the car ROI.
        override_side: Optional side string from an external source (e.g. UI dropdown).
                       Used as a fallback when the filename does not encode the side.
                       Filename always takes priority.

    Returns:
        (detections, labels, new_ids, warning_message)  on success
        None                                              if image unreadable or no detections

    warning_message is a non-empty string when the car side could not be determined
    from either the filename or override, and side-dependent classes were skipped.
    """
    x1, y1, x2, y2 = coords
    full_bgr = cv2.imread(image_path)
    if full_bgr is None:
        return None

    H, W     = full_bgr.shape[:2]
    roi_rgb  = cv2.cvtColor(full_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
    roi_pil  = Image.fromarray(roi_rgb)

    detections = model.predict(roi_pil, threshold=PARTS_THRESHOLD)
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

    # ── Step 1: Pre-conversion NMS — collapse duplicate raw-class masks ──────
    # Done BEFORE side assignment so that left/right forced-name logic works
    # on a clean, deduplicated set of detections.
    if detections.mask is not None:
        raw_masks = list(detections.mask)
        pre_kept  = _suppress_duplicate_raw_parts(
            detections.class_id, detections.confidence,
            raw_masks, iou_threshold=0.30,
        )
    else:
        pre_kept = list(range(len(detections.class_id)))

    if len(pre_kept) < len(detections.class_id):
        dropped = len(detections.class_id) - len(pre_kept)
        print(f"[INFO] Pre-conversion NMS: suppressed {dropped} duplicate raw-class mask(s).")
        detections.xyxy       = detections.xyxy[pre_kept]
        detections.confidence = detections.confidence[pre_kept]
        if detections.mask is not None:
            detections.mask   = detections.mask[pre_kept]
        detections.class_id   = detections.class_id[pre_kept]

    img_w = W
    side  = get_car_side(image_path, override_side=override_side)

    # ── Warn if car side is not determinable ────────────────────────────────
    warning_message = ""
    if side is None:
        warning_message = (
            f"Car side could not be determined from the image filename "
            f"'{os.path.basename(image_path)}'. "
            f"Side-dependent parts (e.g. Fender, Door, Headlight, Mirror, etc.) "
            f"will be skipped. Rename the image with a prefix such as "
            f"'front_', 'rear_', 'left_', 'right_', 'front_left_', etc. "
            f"to enable full part detection."
        )
        print(f"[WARNING] {warning_message}")

    TAILLIGHT_ID = CLASS_NAMES.index("Taillight")
    HEADLIGHT_ID = CLASS_NAMES.index("Headlight")
    MIRROR_ID    = CLASS_NAMES.index("Side_Mirror")

    forced_names = {}

    # only when side is known
    if side is not None:
        # Multiple taillights
        taillight_idx = [i for i, cid in enumerate(detections.class_id) if cid == TAILLIGHT_ID]
        if side in ["rear_left", "rear_right"] and len(taillight_idx) > 1:
            centers = sorted(
                [(i, (detections.xyxy[i][0] + detections.xyxy[i][2]) / 2) for i in taillight_idx],
                key=lambda x: x[1]
            )
            forced_names[centers[0][0]]  = "Left_Taillight"
            forced_names[centers[-1][0]] = "Right_Taillight"

        # Multiple headlights
        headlight_idx = [i for i, cid in enumerate(detections.class_id) if cid == HEADLIGHT_ID]
        if side in ["front_left", "front_right"] and len(headlight_idx) > 1:
            centers = sorted(
                [(i, (detections.xyxy[i][0] + detections.xyxy[i][2]) / 2) for i in headlight_idx],
                key=lambda x: x[1]
            )
            forced_names[centers[0][0]]  = "Right_Headlight"
            forced_names[centers[-1][0]] = "Left_Headlight"

        # Multiple mirrors
        mirror_idx = [i for i, cid in enumerate(detections.class_id) if cid == MIRROR_ID]
        if side in ["front_left", "front_right"] and len(mirror_idx) > 1:
            centers = sorted(
                [(i, (detections.xyxy[i][0] + detections.xyxy[i][2]) / 2) for i in mirror_idx],
                key=lambda x: x[1]
            )
            forced_names[centers[0][0]]  = "Right_Side_Mirror"
            forced_names[centers[-1][0]] = "Left_Side_Mirror"

    labels       = []
    new_ids      = []
    kept_indices = []   # track which original detections survived filtering

    for i, (box, cid, conf) in enumerate(zip(
        detections.xyxy, detections.class_id, detections.confidence
    )):
        old_name = CLASS_NAMES[cid]
        if i in forced_names:
            new_name = forced_names[i]
        else:
            count_same = count_class_instances(detections, cid)
            new_name   = convert_class(old_name, side, box, img_w, count_same)

        # Skip detections whose class couldn't be resolved (side unknown)
        if new_name is None:
            continue

        labels.append(f"{new_name} {conf:.2f}")
        new_ids.append(FINAL_CLASSES.index(new_name))
        kept_indices.append(i)

    # Nothing survived filtering
    if not kept_indices:
        return None

    # ── Step 2: Post-conversion singleton enforcement ─────────────────────────
    # For classes that can appear at most once per image, keep only the
    # highest-confidence detection.  Lower-confidence duplicates are dropped.
    singleton_best = {}   # final_class_name → result-list index of best survivor
    drop_result    = set()

    for res_idx, (lbl, nid, orig_idx) in enumerate(zip(labels, new_ids, kept_indices)):
        fname = FINAL_CLASSES[nid]
        if fname not in SINGLETON_FINAL_CLASSES:
            continue
        if fname not in singleton_best:
            singleton_best[fname] = res_idx
        else:
            prev = singleton_best[fname]
            prev_conf = float(detections.confidence[kept_indices[prev]])
            curr_conf = float(detections.confidence[orig_idx])
            if curr_conf > prev_conf:
                drop_result.add(prev)
                singleton_best[fname] = res_idx
            else:
                drop_result.add(res_idx)

    if drop_result:
        dropped = len(drop_result)
        print(f"[INFO] Singleton enforcement: removed {dropped} duplicate final-class mask(s).")
        labels       = [l for i, l in enumerate(labels)       if i not in drop_result]
        new_ids      = [n for i, n in enumerate(new_ids)      if i not in drop_result]
        kept_indices = [k for i, k in enumerate(kept_indices) if i not in drop_result]

    if not kept_indices:
        return None

    # Filter detections to only the kept indices
    detections.xyxy       = detections.xyxy[kept_indices]
    detections.confidence = detections.confidence[kept_indices]
    if detections.mask is not None:
        detections.mask = detections.mask[kept_indices]
    detections.class_id = np.array(new_ids)

    return detections, labels, new_ids, warning_message


def draw_parts_on_canvas(
    canvas, image_path, coords,
    show_parts=True, parts_filled=True,
    parts_labels=True, mask_alpha=0.35,
    override_side=None,
):
    """
    Draw detected car-part masks and labels onto *canvas*.

    Args:
        override_side: Optional side string from an external source (e.g. UI dropdown).
                       Forwarded to get_car_parts_detections as a fallback.

    Returns:
        (canvas, warning_message)  — warning_message is non-empty when the car
        side could not be determined and side-dependent parts were skipped.
    """
    if not show_parts:
        return canvas, ""
    result = get_car_parts_detections(image_path, coords, override_side=override_side)
    if result is None:
        return canvas, ""
    detections, labels, new_ids, warning_message = result
    for mask, cid in zip(detections.mask, detections.class_id):
        color  = PART_COLORS_BGR[cid % len(PART_COLORS_BGR)]
        canvas = draw_curved_mask(canvas, mask, color, alpha=mask_alpha, filled=parts_filled)
    if parts_labels:
        for mask, label, cid in zip(detections.mask, labels, detections.class_id):
            color = PART_COLORS_BGR[cid % len(PART_COLORS_BGR)]
            draw_label_on_image(canvas, label, mask, color)
    return canvas, warning_message
