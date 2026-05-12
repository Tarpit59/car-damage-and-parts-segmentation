"""
config.py
=========
Central configuration for the Car Damage AI project.
Edit paths and thresholds here — no need to touch model/logic files.
"""

import os

# ──────────────────────────────────────────────────────────────────────────────
# BASE PATHS
# ──────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

UPLOAD_FOLDER   = os.path.join(BASE_DIR, "uploads")
PROCESSED_FOLDER = os.path.join(UPLOAD_FOLDER, "processed")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_CONTENT_LENGTH = 20 * 1024 * 1024   # 20 MB

# Max longest-side (px) before the image is downscaled prior to inference.
# 4608×3072 (12 MP) → tiling creates ~14 MP float masks → CUDA OOM on 8 GB.
# 1920 px keeps peak VRAM well under 6 GB even with 2×2 tile inference.
# Increase carefully if you have more VRAM; set to 0 to disable.
MAX_INPUT_RESOLUTION = 1920

# ──────────────────────────────────────────────────────────────────────────────
# MODEL CHECKPOINTS  ← CHANGE THESE
# ──────────────────────────────────────────────────────────────────────────────
YOLO_CHECKPOINT = r"path\to\yolov8m.pt"

PARTS_CHECKPOINT = r"path\to\checkpoint_parts.pth"
PARTS_RESOLUTION = 960
PARTS_THRESHOLD  = 0.45

DAMAGE_CHECKPOINT  = r"path\to\checkpoint_damage.pth"
DAMAGE_RESOLUTION  = 960
DAMAGE_THRESHOLD   = 0.40
TILE_THRESHOLD     = 0.55

# ──────────────────────────────────────────────────────────────────────────────
# INFERENCE SETTINGS
# ──────────────────────────────────────────────────────────────────────────────
PADDING_RATIO = 0.02

USE_TILE_INFERENCE = True
GRID_ROWS   = 2
GRID_COLS   = 2
OVERLAP_RATIO   = 0.35
MIN_TILE_SIZE   = 300

TOUCH_GAP_PIXELS = 3

BOX_IOU_THRESHOLD      = 0.35
MASK_IOU_THRESHOLD     = 0.25
MASK_COVERAGE_THRESHOLD = 0.80

TILE_DAMAGE_NAMES = {"dent", "scratch", "crack"}

# ──────────────────────────────────────────────────────────────────────────────
# ANALYTICS
# ──────────────────────────────────────────────────────────────────────────────
MIN_EXTERNAL_PIXELS = 150

SEVERITY_RULES = {
    "scratch":       [(1.5, "Minor"), (4.0, "Moderate"), (8.0, "Major"),  (999, "Critical")],
    "dent":          [(2.0, "Minor"), (5.0, "Moderate"), (10.0, "Major"), (999, "Critical")],
    "crack":         [(1.0, "Minor"), (3.0, "Moderate"), (6.0, "Major"),  (999, "Critical")],
    "glass shatter": [(5.0, "Major"), (999, "Critical")],
    "lamp broken":   [(5.0, "Major"), (999, "Critical")],
    "tire flat":     [(999, "Critical")],
}

SEVERITY_PENALTY = {
    "Minor":    3,
    "Moderate": 6,
    "Major":    10,
    "Critical": 15,
}

# ──────────────────────────────────────────────────────────────────────────────
# FLASK
# ──────────────────────────────────────────────────────────────────────────────
SECRET_KEY  = "car-damage-ai-secret-2026"
DEBUG       = True
HOST        = "0.0.0.0"
PORT        = 5000

# ──────────────────────────────────────────────────────────────────────────────
# COCO EXPORT
# ──────────────────────────────────────────────────────────────────────────────
SAVE_COCO_DEFAULT = False
