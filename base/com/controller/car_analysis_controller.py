"""
car_analysis_controller.py
==========================
Flask blueprint for car damage analysis endpoints.
"""

import os
import base64
import json
import pickle
import tempfile

import cv2
import numpy as np
from flask import (
    Blueprint, request, jsonify, render_template,
    current_app, session,
)
from werkzeug.utils import secure_filename

from ..service.output_handler   import process_image, render_on_demand
from ..service.car_parts_segmentation import (
    PART_COLORS_HEX, FINAL_CLASSES as PARTS_CLASSES,
    get_car_side,
)
from ..service.car_damage_segmentation import DAMAGE_COLORS_HEX, CLASS_NAMES as DAMAGE_CLASSES

bp = Blueprint("car_analysis", __name__, url_prefix="/analysis")


# ──────────────────────────────────────────────────────────────────────────────
# numpy-safe JSON encoder  — fixes "int64 is not JSON serializable"
# ──────────────────────────────────────────────────────────────────────────────
def _sanitize(obj):
    """Recursively convert numpy scalars/arrays to plain Python types."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _allowed_file(filename):
    from config import ALLOWED_EXTENSIONS
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _auto_resize_image(image_path):
    """
    If the image's longest side exceeds MAX_INPUT_RESOLUTION, downscale it
    (preserving aspect ratio) and overwrite the file.  No-op if already small
    enough, or if MAX_INPUT_RESOLUTION is falsy (0 / None).
    Returns (actual_width, actual_height) after any resizing.
    """
    from config import MAX_INPUT_RESOLUTION
    if not MAX_INPUT_RESOLUTION:
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        return w, h

    img = cv2.imread(image_path)
    if img is None:
        return 0, 0
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= MAX_INPUT_RESOLUTION:
        return w, h   # already within limits

    scale  = MAX_INPUT_RESOLUTION / longest
    new_w  = max(1, int(round(w * scale)))
    new_h  = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(image_path, resized)
    print(f"[INFO] Image downscaled from {w}×{h} → {new_w}×{new_h} (MAX_INPUT_RESOLUTION={MAX_INPUT_RESOLUTION})")
    return new_w, new_h


def _bgr_to_b64(bgr_array):
    """Encode a BGR numpy array as base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", bgr_array, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _detections_to_json(detections, labels, class_names, colors_hex):
    """Serialize detections to a JSON-safe list (no numpy masks)."""
    if detections is None or len(detections) == 0:
        return []
    result = []
    for i, (cid, conf) in enumerate(zip(detections.class_id, detections.confidence)):
        name  = class_names[int(cid)]
        color = colors_hex[int(cid) % len(colors_hex)]
        label = labels[i] if i < len(labels) else name
        result.append({
            "id":         i,
            "class_id":   int(cid),
            "class_name": name,
            "confidence": round(float(conf), 3),
            "label":      label,
            "color":      color,
        })
    return result


def _store_detections(session_key, det_parts, det_damage,
                      labels_parts, labels_damage, image_path, coords):
    """Pickle detections into a temp file; store path in session."""
    payload = dict(
        image_path=image_path, coords=coords,
        det_parts=det_parts, det_damage=det_damage,
        labels_parts=labels_parts, labels_damage=labels_damage,
    )
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
    pickle.dump(payload, tmp)
    tmp.close()
    session[session_key] = tmp.name


def _load_detections(session_key):
    path = session.get(session_key)
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────
@bp.route("/", methods=["GET"])
def index():
    return render_template("car_analysis/index.html")


