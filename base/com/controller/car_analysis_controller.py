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


@bp.route("/upload", methods=["POST"])
def upload():
    """
    Receives image upload, runs full pipeline, returns:
      - original image (base64)
      - annotated image (base64)
      - detections metadata (for legend)
      - analytics JSON
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "" or not _allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    upload_dir = current_app.config["UPLOAD_FOLDER"]
    os.makedirs(upload_dir, exist_ok=True)
    filename   = secure_filename(file.filename)
    image_path = os.path.join(upload_dir, filename)
    file.save(image_path)

    # ── Auto-resize: prevent CUDA OOM for high-res images (e.g. 4608×3072) ──
    _auto_resize_image(image_path)

    # ── Resolve car side: filename takes priority, dropdown is fallback ───────
    override_side = (request.form.get("car_side") or "").strip() or None
    # Validate the override value against accepted sides
    _valid_sides = {
        "front", "rear", "left", "right",
        "front_left", "front_right", "rear_left", "rear_right",
    }
    if override_side and override_side not in _valid_sides:
        override_side = None

    # After resolving, hard-stop if no side is determinable at all
    resolved = get_car_side(image_path, override_side=override_side)
    if resolved is None:
        return jsonify({
            "error": (
                "Car side could not be determined. "
                "Please rename the image with a side prefix "
                "(e.g. front_, rear_, left_, right_, front_left_, front_right_, "
                "rear_left_, rear_right_) "
                "or select the car side from the dropdown before uploading."
            ),
            "side_required": True,
        }), 422

    # ── Read toggle states sent by the frontend ────────────────────────────
    def _bool(key, default=True):
        v = request.form.get(key)
        if v is None:
            return default
        return v.strip() not in ("0", "false", "no", "")

    show_parts    = _bool("show_parts",    True)
    parts_filled  = _bool("parts_filled",  True)
    parts_labels  = _bool("parts_labels",  True)
    show_damage   = _bool("show_damage",   True)
    damage_filled = _bool("damage_filled", True)
    damage_labels = _bool("damage_labels", True)
    try:
        mask_alpha = float(request.form.get("mask_alpha", 0.35))
    except (TypeError, ValueError):
        mask_alpha = 0.35

    result = process_image(
        image_path=image_path,
        show_parts=show_parts,
        parts_filled=parts_filled,
        parts_labels=parts_labels,
        show_damage=show_damage,
        damage_filled=damage_filled,
        damage_labels=damage_labels,
        mask_alpha=mask_alpha,
        save_coco=current_app.config.get("SAVE_COCO_DEFAULT", False),
        return_coco=False,
        override_side=override_side,
    )

    if not result["success"]:
        return jsonify({"error": "No car detected in the image"}), 422

    # Store detections in session for re-render
    _store_detections(
        "det_session",
        result["det_parts"], result["det_damage"],
        result["labels_parts"], result["labels_damage"],
        image_path, result["coords"],
    )

    return jsonify(_sanitize({
        "success":          True,
        "original_image":   _bgr_to_b64(result["image_array"]),
        "annotated_image":  _bgr_to_b64(result["annotated_image"]),
        "coords":           result["coords"],
        "parts_warning":    result.get("parts_warning", ""),
        "parts_detections": _detections_to_json(
            result["det_parts"], result["labels_parts"],
            PARTS_CLASSES, PART_COLORS_HEX,
        ),
        "damage_detections": _detections_to_json(
            result["det_damage"], result["labels_damage"],
            DAMAGE_CLASSES, DAMAGE_COLORS_HEX,
        ),
        "analytics":        result["analytics"],
    }))


@bp.route("/render", methods=["POST"])
def render():
    """
    Re-render the stored detections with new toggle/opacity settings.
    No model inference is run — pure drawing pass.
    """
    payload = _load_detections("det_session")
    if payload is None:
        return jsonify({"error": "No active session. Please upload an image first."}), 400

    data = request.get_json(force=True)

    canvas = render_on_demand(
        image_path    = payload["image_path"],
        coords        = payload["coords"],
        det_parts     = payload["det_parts"],
        labels_parts  = payload["labels_parts"],
        det_damage    = payload["det_damage"],
        labels_damage = payload["labels_damage"],
        show_parts    = data.get("show_parts",    True),
        parts_filled  = data.get("parts_filled",  True),
        parts_labels  = data.get("parts_labels",  True),
        show_damage   = data.get("show_damage",   True),
        damage_filled = data.get("damage_filled", True),
        damage_labels = data.get("damage_labels", True),
        mask_alpha    = float(data.get("mask_alpha", 0.35)),
    )

    if canvas is None:
        return jsonify({"error": "Render failed"}), 500

    return jsonify({
        "success":         True,
        "annotated_image": _bgr_to_b64(canvas),
    })
