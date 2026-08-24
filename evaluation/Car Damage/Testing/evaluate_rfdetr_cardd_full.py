"""
evaluate_rfdetr_cardd_full.py
==============================
Evaluates RF-DETR segmentation on the CarDD test set and compares against
DCN+ (ResNet-101) from the CarDD paper.

METRICS COMPUTED
────────────────
  Mask AP / AP50 / AP75 / per-category   ← instance segmentation (masks)
  Box  AP / AP50 / AP75 / per-category   ← object detection (bounding boxes)

GRAPHS GENERATED (saved to ./plots/)
──────────────────────────────────────
  1.  pr_curve_per_category_mask.png         Per-category PR curves (mask)
  2.  pr_curve_per_category_box.png          Per-category PR curves (box)
  3.  f1_vs_threshold_mask.png               F1 vs confidence threshold (mask)
  4.  f1_vs_threshold_box.png                F1 vs confidence threshold (box)
  5.  ap_comparison_bar_mask.png             Per-category bar: RF-DETR vs DCN+ (mask)
  6.  ap_comparison_bar_box.png              Per-category bar: RF-DETR vs DCN+ (box)
  7.  overall_radar.png                      Radar: AP / AP50 / AP75 both tasks
  8.  precision_recall_f1_summary.png        Summary bar: P / R / F1 at best threshold
  9.  iou_threshold_ap_curve.png             AP vs IoU threshold (mask + box)
  10. confusion_heatmap_mask/box.png         Predicted vs GT category heatmap
  11. rfdetr_mask_vs_box_overall.png         RF-DETR only: Mask vs Box AP/AP50/AP75 grouped bar
  12. rfdetr_mask_vs_box_per_category.png    RF-DETR only: Mask vs Box per-category grouped bar
  13. comparison_overall_mask_box.png        4-way bar: RF-DETR Mask/Box vs DCN+ Mask/Box (AP/AP50/AP75)
  14. comparison_per_category_mask_box.png   4-way per-category: RF-DETR Mask/Box vs DCN+ Mask/Box

USAGE
─────
  python evaluate_rfdetr_cardd_full.py \
      --images_dir  /path/to/test/images \
      --annotations /path/to/_annotations.coco.json \
      --checkpoint  /path/to/checkpoint_best_total.pth \
      --resolution  960 \
      --threshold   0.001 \
      --output_json results_full.json \
      --plots_dir   ./plots
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── DEFAULTS ─────────────────────────────────────────────────────────────────
IMAGES_DIR  = "/path/to/test/images"
ANNOTATIONS = "/path/to/test/_annotations.coco.json"
CHECKPOINT  = "/path/to/checkpoint_best_total.pth"
RESOLUTION  = 960
THRESHOLD   = 0.001
OUTPUT_JSON = "results_full.json"
PLOTS_DIR   = "./plots"
# ─────────────────────────────────────────────────────────────────────────────

CLASSES = ["crack", "dent", "glass shatter", "lamp broken", "scratch", "tire flat"]

# ── DCN+ (ResNet-101) reference values from CarDD paper ──────────────────────
# Table IV  → AP / AP50 / AP75  (format: mask / box)
# Table V   → per-category      (format: mask / box)
DCN_PLUS = {
    "mask": {
        "AP":   57.0,
        "AP50": 77.7,
        "AP75": 58.4,
        "per_category": {
            "crack":         16.6,
            "dent":          40.5,
            "glass shatter": 89.6,
            "lamp broken":   70.8,
            "scratch":       34.3,
            "tire flat":     90.0,
        },
    },
    "box": {
        "AP":   60.6,
        "AP50": 78.8,
        "AP75": 64.8,
        "per_category": {
            "crack":         29.6,
            "dent":          42.2,
            "glass shatter": 90.1,
            "lamp broken":   69.5,
            "scratch":       42.3,
            "tire flat":     90.2,
        },
    },
}

IOU_THRESHOLDS = np.arange(0.50, 1.00, 0.05)   # 10 thresholds

# ── COLOUR PALETTE ────────────────────────────────────────────────────────────
CAT_COLORS = {
    "crack":         "#E63946",
    "dent":          "#F4A261",
    "glass shatter": "#2A9D8F",
    "lamp broken":   "#457B9D",
    "scratch":       "#A8DADC",
    "tire flat":     "#6A0572",
}
RF_COLOR  = "#1A73E8"
DCN_COLOR = "#E8710A"

# ── MASK UTILITIES ────────────────────────────────────────────────────────────

def polygon_to_mask(segmentation, height, width):
    try:
        rles = maskUtils.frPyObjects(segmentation, height, width)
        rle  = maskUtils.merge(rles)
        return maskUtils.decode(rle).astype(bool)
    except Exception:
        from PIL import ImageDraw
        m = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(m)
        for poly in segmentation:
            pts = [(poly[i], poly[i+1]) for i in range(0, len(poly), 2)]
            if len(pts) >= 3:
                draw.polygon(pts, outline=1, fill=1)
        return np.array(m, dtype=bool)


def pred_mask_to_full(local_mask, full_h, full_w):
    blank = np.zeros((full_h, full_w), dtype=bool)
    mh, mw = local_mask.shape
    h, w = min(mh, full_h), min(mw, full_w)
    blank[:h, :w] = local_mask[:h, :w]
    return blank


def mask_to_bbox(mask_bool):
    """Return (x1,y1,x2,y2) tight bounding box from a boolean mask."""
    rows = np.any(mask_bool, axis=1)
    cols = np.any(mask_bool, axis=0)
    if not rows.any():
        return None
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return float(x1), float(y1), float(x2 + 1), float(y2 + 1)


def project_prediction_to_original(mask_model, orig_w, orig_h):
    """
    Resize a mask from the model's inference space to the original image
    coordinate space using bilinear interpolation + 0.5 threshold.

    Parameters
    ----------
    mask_model : np.ndarray   — boolean or uint8 mask at model output resolution
    orig_w, orig_h : int      — original image width and height

    Returns
    -------
    rle       : dict    — COCO RLE dict with size=[orig_h, orig_w]
    mask_orig : uint8   — binary mask at shape (orig_h, orig_w)
    """
    mask_float   = mask_model.astype(np.float32)
    mask_resized = cv2.resize(mask_float, (orig_w, orig_h),
                              interpolation=cv2.INTER_LINEAR)
    mask_orig    = (mask_resized >= 0.5).astype(np.uint8)

    rle = maskUtils.encode(np.asfortranarray(mask_orig))
    rle["counts"] = rle["counts"].decode("utf-8")
    rle["size"]   = [orig_h, orig_w]
    return rle, mask_orig


def decode_roboflow_filename(fname: str) -> str:
    """
    Recover the ORIGINAL filename from a Roboflow-exported filename.

    Roboflow renames files on export using the pattern:
        {original_stem}_{original_ext}.rf.{32_char_md5}.{export_ext}

    Examples
    --------
    '003283_jpg.rf.b013652adf93f662a5d15f8eb7bf3e8d.jpg'  →  '003283.jpg'
    'img_001_png.rf.abcdef1234567890abcdef1234567890.jpg'  →  'img_001.png'
    'normal_name.jpg'                                      →  'normal_name.jpg'  (unchanged)

    Returns the decoded filename, or fname unchanged if the pattern does not match.
    """
    import re as _re
    _RF = _re.compile(
        r'^(.+?)_(jpe?g|png|bmp|tiff?)\.rf\.[a-f0-9]{32}\.(jpe?g|png|bmp|tiff?)$',
        _re.IGNORECASE,
    )
    m = _RF.match(fname)
    if m:
        orig_stem = m.group(1)
        orig_ext  = m.group(2).lower().replace("jpeg", "jpg")
        return f"{orig_stem}.{orig_ext}"
    return fname


def box_iou(b1, b2):
    """Compute IoU between two (x1,y1,x2,y2) boxes."""
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

# ── COCOEVAL ────────────────────────────────────────────────────────

def run_cocoeval(coco_gt, coco_results_list, iou_type, label="RF-DETR"):
    """
    Run pycocotools COCOeval on projected predictions.

    Parameters
    ----------
    coco_gt           : pycocotools.coco.COCO  — ground-truth COCO object
    coco_results_list : list of dicts          — COCO-format detection results
    iou_type          : str                    — "segm" or "bbox"
    label             : str                    — display label for logging

    Returns
    -------
    dict with keys AP, AP50, AP75, per_category (name → AP×100);
    empty dict if no predictions were available.
    """
    if not coco_results_list:
        print(f"[WARN] No predictions for {iou_type} eval — skipping.")
        return {}

    coco_dt = coco_gt.loadRes(coco_results_list)
    ev = COCOeval(coco_gt, coco_dt, iou_type)
    # CarDD paper uses non-standard area ranges: small < 128², medium 128²-256², large > 256²
    ev.params.areaRng = [[0, 1e5 ** 2], [0, 128 ** 2], [128 ** 2, 256 ** 2], [256 ** 2, 1e5 ** 2]]
    ev.evaluate()
    ev.accumulate()
    print(f"\n── Pycocotools  [{label}  iouType={iou_type}] ──")
    ev.summarize()

    stats  = ev.stats   # [AP, AP50, AP75, APs, APm, APl, ...]
    result = {
        "AP":           round(float(stats[0]) * 100, 1),
        "AP50":         round(float(stats[1]) * 100, 1),
        "AP75":         round(float(stats[2]) * 100, 1),
        "APs":          round(float(stats[3]) * 100, 1) if stats[3] != -1 else 0.0,
        "APm":          round(float(stats[4]) * 100, 1) if stats[4] != -1 else 0.0,
        "APl":          round(float(stats[5]) * 100, 1) if stats[5] != -1 else 0.0,
        "per_category": {},
        "_per_cat_ap50": {},
        "_per_cat_ap75": {},
        "_curves": {},
    }

    # Per-category AP (mean over IoU 0.50:0.95)
    for cat_id, cat_info in coco_gt.cats.items():
        cname = cat_info["name"]
        ev2 = COCOeval(coco_gt, coco_dt, iou_type)
        ev2.params.catIds = [cat_id]
        # CarDD paper uses non-standard area ranges
        ev2.params.areaRng = [[0, 1e5 ** 2], [0, 128 ** 2], [128 ** 2, 256 ** 2], [256 ** 2, 1e5 ** 2]]
        ev2.evaluate()
        ev2.accumulate()
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            ev2.summarize()
            
        if len(ev2.stats) > 0 and ev2.stats[0] != -1:
            result["per_category"][cname]  = round(float(ev2.stats[0]) * 100, 1)
            result["_per_cat_ap50"][cname] = round(float(ev2.stats[1]) * 100, 1)
            result["_per_cat_ap75"][cname] = round(float(ev2.stats[2]) * 100, 1)
            
            # Extract PR curves directly from pycocotools evaluation
            recalls = ev2.params.recThrs
            cat_curves = {}
            # T (iou) x R (recall) x K (cat) x A (area) x M (maxDets)
            precisions = ev2.eval["precision"]
            for t_idx, iou_v in enumerate(np.arange(0.50, 1.00, 0.05)):
                key = round(float(iou_v), 2)
                # T=t_idx, R=all, K=0 (only 1 cat evaluated), A=0 (all area), M=2 (maxDets=100)
                prec = precisions[t_idx, :, 0, 0, 2]
                valid = prec > -1
                if valid.any():
                    cat_curves[key] = (recalls[valid], prec[valid])
                else:
                    cat_curves[key] = (np.array([0.0]), np.array([0.0]))
            result["_curves"][cname] = cat_curves
        else:
            result["per_category"][cname]  = 0.0
            result["_per_cat_ap50"][cname] = 0.0
            result["_per_cat_ap75"][cname] = 0.0
            result["_curves"][cname] = {round(v, 2): (np.array([0.0]), np.array([0.0])) for v in np.arange(0.50, 1.00, 0.05)}

    return result


def compute_f1_vs_threshold(preds_by_cat, gts_by_cat, use_box=False, n_steps=50):
    """Sweep confidence threshold; return (thresholds, precision, recall, f1)."""
    all_preds = []
    for cat in CLASSES:
        for p in preds_by_cat.get(cat, []):
            all_preds.append(p)

    thresholds = np.linspace(0.0, 1.0, n_steps)
    precs, recs, f1s = [], [], []

    total_gt = sum(len(v) for v in gts_by_cat.values())

    for thr in thresholds:
        tp_total = fp_total = fn_total = 0
        for cat in CLASSES:
            preds = [p for p in preds_by_cat.get(cat, []) if p["score"] >= thr]
            gts   = gts_by_cat.get(cat, [])
            if not gts:
                continue
            gt_by_img = defaultdict(list)
            for i, g in enumerate(gts):
                gt_by_img[g["image_id"]].append((i, g))

            preds_s = sorted(preds, key=lambda x: -x["score"])
            matched = set()
            tp = 0
            for pred in preds_s:
                best_iou = 0.50 - 1e-9
                best_gi  = -1
                for gi, g in gt_by_img.get(pred["image_id"], []):
                    if gi in matched:
                        continue
                    if use_box:
                        iou = box_iou(pred["bbox"], g["bbox"])
                    else:
                        iou = maskUtils.iou([pred["mask"]], [g["mask"]], [0])[0][0]
                    if iou > best_iou:
                        best_iou = iou
                        best_gi  = gi
                if best_gi >= 0:
                    tp += 1
                    matched.add(best_gi)

            fp = len(preds) - tp
            fn = len(gts) - tp
            tp_total += tp
            fp_total += fp
            fn_total += fn

        p = tp_total / (tp_total + fp_total + 1e-9)
        r = tp_total / (tp_total + fn_total + 1e-9)
        f = 2*p*r / (p + r + 1e-9)
        precs.append(p); recs.append(r); f1s.append(f)

    return thresholds, np.array(precs), np.array(recs), np.array(f1s)


def compute_confusion_data(preds_by_cat, gts_by_cat, iou_thresh=0.50, use_box=False):
    """Build confusion matrix: rows=GT cat, cols=pred cat."""
    n = len(CLASSES)
    cat2idx = {c: i for i, c in enumerate(CLASSES)}
    mat = np.zeros((n, n), dtype=int)
    fn_row = np.zeros(n, dtype=int)

    for cat in CLASSES:
        preds = preds_by_cat.get(cat, [])
        gts   = gts_by_cat.get(cat, [])
        if not gts:
            continue

        gt_by_img = defaultdict(list)
        for i, g in enumerate(gts):
            gt_by_img[g["image_id"]].append((i, g))

        preds_s = sorted(preds, key=lambda x: -x["score"])
        matched_gts = set()
        for pred in preds_s:
            best_iou = iou_thresh - 1e-9
            best_gi  = -1
            for gi, g in gt_by_img.get(pred["image_id"], []):
                if gi in matched_gts:
                    continue
                if use_box:
                    iou = box_iou(pred["bbox"], g["bbox"])
                else:
                    iou = maskUtils.iou([pred["mask"]], [g["mask"]], [0])[0][0]
                if iou > best_iou:
                    best_iou = iou
                    best_gi  = gi
            if best_gi >= 0:
                matched_gts.add(best_gi)
                mat[cat2idx[cat], cat2idx[cat]] += 1

        fn_row[cat2idx[cat]] = len(gts) - len(matched_gts)

    return mat, fn_row

# ── PLOTTING ──────────────────────────────────────────────────────────────────

def _style():
    plt.rcParams.update({
        "font.family":    "DejaVu Sans",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":     True,
        "grid.alpha":    0.3,
        "grid.linestyle": "--",
        "figure.facecolor": "#0D1117",
        "axes.facecolor":   "#161B22",
        "axes.labelcolor":  "#C9D1D9",
        "xtick.color":      "#C9D1D9",
        "ytick.color":      "#C9D1D9",
        "text.color":       "#C9D1D9",
        "axes.titlecolor":  "#E6EDF3",
        "legend.facecolor": "#21262D",
        "legend.edgecolor": "#30363D",
        "grid.color":       "#21262D",
    })

def _savefig(fig, path, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [PLOT] Saved → {path}")


# ── 1 & 2: PR Curves per category ────────────────────────────────────────────
def plot_pr_curves(results, plots_dir, mode="mask"):
    """
    Each subplot shows TWO things for one category:
      - Solid line  : PR curve at IoU=0.50  (the standard AP50 threshold)
      - Dashed lines: PR curves at IoU=0.55, 0.65, 0.75, 0.85, 0.95
                      (faint — shows how precision/recall degrades at stricter IoU)

    Corner annotations:
      AP50 in title  = area under the solid curve (IoU=0.50 only)
      AP   in corner = mean area under ALL 10 IoU curves (0.50→0.95), i.e. the primary metric
    """
    _style()
    mode_label = "Mask" if mode == "mask" else "Box"
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f"Precision–Recall Curves per Category  [{mode_label}]\n"
        f"Solid = IoU 0.50  |  Dashed = IoU 0.55 → 0.95  |  "
        f"AP50 = area under solid curve  |  AP = mean over all IoU thresholds",
        fontsize=11, color="#C9D1D9", y=1.02
    )

    curves = results["_curves"]
    for i, cat in enumerate(CLASSES):
        ax = axes[i//3][i%3]
        color = CAT_COLORS[cat]

        # Dashed lines first (background) — stricter IoU thresholds
        for iou_v in [0.55, 0.65, 0.75, 0.85, 0.95]:
            key = round(iou_v, 2)
            if key in curves[cat]:
                r2, p2 = curves[cat][key]
                ax.plot(r2, p2, color=color, lw=0.9, alpha=0.30, ls="--")

        # Solid line — IoU=0.50 (primary / most lenient)
        rec, prec = curves[cat][0.50]
        ax.fill_between(rec, prec, alpha=0.12, color=color)
        ax.plot(rec, prec, color=color, lw=2.5,
                label=f"IoU=0.50 (AP50={results['_per_cat_ap50'][cat]:.1f})")

        ax.set_title(f"{cat}", fontsize=11, fontweight="bold", color=color)
        ax.set_xlabel("Recall", fontsize=9)
        ax.set_ylabel("Precision", fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="upper right")

        # AP (mean over all IoU) in bottom-left corner
        ax.text(0.02, 0.04,
                f"AP (0.50:0.95) = {results['per_category'][cat]:.1f}",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=9, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#21262D",
                          edgecolor=color, alpha=0.7))

    _savefig(fig, plots_dir / f"pr_curve_per_category_{mode}.png")


# ── 3 & 4: F1 vs Confidence Threshold ────────────────────────────────────────
def plot_f1_vs_threshold(preds_by_cat, gts_by_cat, plots_dir, mode="mask"):
    _style()
    use_box = (mode == "box")
    thresholds, precs, recs, f1s = compute_f1_vs_threshold(
        preds_by_cat, gts_by_cat, use_box=use_box)

    best_idx = np.argmax(f1s)
    best_thr = thresholds[best_idx]
    best_f1  = f1s[best_idx]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(thresholds, precs, color="#58A6FF", lw=2.5, label="Precision")
    ax.plot(thresholds, recs,  color="#3FB950", lw=2.5, label="Recall")
    ax.plot(thresholds, f1s,   color="#F78166", lw=2.5, label="F1")
    ax.axvline(best_thr, color="#F78166", ls="--", alpha=0.6,
               label=f"Best F1={best_f1:.3f} @ thr={best_thr:.2f}")
    ax.scatter([best_thr], [best_f1], color="#F78166", s=80, zorder=5)

    ax.fill_between(thresholds, f1s, alpha=0.1, color="#F78166")
    ax.set_xlabel("Confidence Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Precision / Recall / F1 vs Confidence Threshold  [{mode.upper()}]",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    _savefig(fig, plots_dir / f"f1_vs_threshold_{mode}.png")
    return best_thr, float(precs[best_idx]), float(recs[best_idx]), float(best_f1)


# ── 5 & 6: Per-Category AP Comparison Bar Chart ───────────────────────────────
def plot_ap_comparison_bar(rf_results, mode="mask", plots_dir=None):
    _style()
    dcn_vals = [DCN_PLUS[mode]["per_category"][c] for c in CLASSES]
    rf_vals  = [rf_results["per_category"][c] for c in CLASSES]

    x = np.arange(len(CLASSES))
    w = 0.38

    fig, ax = plt.subplots(figsize=(13, 6))
    bars1 = ax.bar(x - w/2, dcn_vals, w, label="DCN+ (ResNet-101)",
                   color=DCN_COLOR, alpha=0.85, edgecolor="#30363D")
    bars2 = ax.bar(x + w/2, rf_vals,  w, label="RF-DETR-Seg",
                   color=RF_COLOR, alpha=0.85, edgecolor="#30363D")

    for bar in bars1:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9, color=DCN_COLOR)
    for bar in bars2:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.8,
                f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9, color=RF_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(" ", "\n") for c in CLASSES], fontsize=10)
    ax.set_ylabel(f"{'Mask' if mode=='mask' else 'Box'} AP (IoU 0.50:0.95)", fontsize=12)
    ax.set_title(f"Per-Category AP Comparison: RF-DETR vs DCN+  [{mode.upper()}]",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 105)

    # Δ annotations
    for i, (d, r) in enumerate(zip(dcn_vals, rf_vals)):
        delta = r - d
        col   = "#3FB950" if delta > 0 else "#F85149"
        sign  = "+" if delta >= 0 else ""
        ax.text(x[i], max(d, r)+4, f"{sign}{delta:.1f}",
                ha="center", fontsize=9, color=col, fontweight="bold")

    _savefig(fig, plots_dir / f"ap_comparison_bar_{mode}.png")


# ── 7: Overall Radar Chart ────────────────────────────────────────────────────
def plot_radar(rf_mask, rf_box, plots_dir):
    _style()
    labels    = ["Mask AP", "Mask AP50", "Mask AP75", "Box AP", "Box AP50", "Box AP75"]
    rf_vals   = [rf_mask["AP"], rf_mask["AP50"], rf_mask["AP75"],
                 rf_box["AP"],  rf_box["AP50"],  rf_box["AP75"]]
    dcn_vals  = [DCN_PLUS["mask"]["AP"], DCN_PLUS["mask"]["AP50"], DCN_PLUS["mask"]["AP75"],
                 DCN_PLUS["box"]["AP"],  DCN_PLUS["box"]["AP50"],  DCN_PLUS["box"]["AP75"]]

    n = len(labels)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    rf_vals_c  = rf_vals  + rf_vals[:1]
    dcn_vals_c = dcn_vals + dcn_vals[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_facecolor("#161B22")
    fig.patch.set_facecolor("#0D1117")

    ax.plot(angles, rf_vals_c,  color=RF_COLOR,  lw=2.5, label="RF-DETR-Seg")
    ax.fill(angles, rf_vals_c,  color=RF_COLOR,  alpha=0.15)
    ax.plot(angles, dcn_vals_c, color=DCN_COLOR, lw=2.5, label="DCN+ (R101)", ls="--")
    ax.fill(angles, dcn_vals_c, color=DCN_COLOR, alpha=0.10)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=11, color="#C9D1D9")
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], color="#8B949E", fontsize=8)
    ax.grid(color="#21262D", linestyle="--", alpha=0.5)
    ax.spines["polar"].set_color("#30363D")

    ax.set_title("Overall AP Radar: Mask & Box\nRF-DETR vs DCN+ (ResNet-101)",
                 fontsize=13, fontweight="bold", color="#E6EDF3", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=11)

    _savefig(fig, plots_dir / "overall_radar.png", tight=False)


# ── 8: Precision / Recall / F1 Summary Bar ───────────────────────────────────
def plot_prf_summary(mask_best, box_best, plots_dir):
    """
    mask_best / box_best: (best_thr, precision, recall, f1)
    """
    _style()
    metrics   = ["Precision", "Recall", "F1"]
    mask_vals = [mask_best[1], mask_best[2], mask_best[3]]
    box_vals  = [box_best[1],  box_best[2],  box_best[3]]

    x = np.arange(len(metrics))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, vals, mode, color, best in zip(
            axes, [mask_vals, box_vals], ["Mask", "Box"],
            [RF_COLOR, "#A371F7"], [mask_best, box_best]):
        bars = ax.bar(x, vals, w*2, color=color, alpha=0.8, edgecolor="#30363D")
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=11,
                    color=color, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=12)
        ax.set_ylim(0, 1.15)
        ax.set_title(f"{mode}  (@ best threshold={best[0]:.2f})",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("Score", fontsize=11)

    fig.suptitle("Precision / Recall / F1 at Best Confidence Threshold",
                 fontsize=14, fontweight="bold", color="#E6EDF3")
    _savefig(fig, plots_dir / "precision_recall_f1_summary.png")


# ── 9: AP vs IoU Threshold Curve ─────────────────────────────────────────────
def plot_ap_vs_iou(rf_mask_results, rf_box_results, preds_mask, gts_mask,
                   preds_box, gts_box, plots_dir):
    _style()
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for mode, preds_by_cat, gts_by_cat, color, label, use_box in [
        ("mask", preds_mask, gts_mask, RF_COLOR,    "RF-DETR Mask", False),
        ("box",  preds_box,  gts_box,  "#A371F7",   "RF-DETR Box",  True),
    ]:
        ap_per_iou = []
        for iou in IOU_THRESHOLDS:
            aps = []
            for cat in CLASSES:
                ap_v, _, _ = ap_and_curve(
                    preds_by_cat.get(cat, []), gts_by_cat.get(cat, []), iou, use_box)
                aps.append(ap_v)
            ap_per_iou.append(np.mean(aps))

        ax.plot(IOU_THRESHOLDS, ap_per_iou, color=color, lw=2.5,
                marker="o", markersize=5, label=label)
        ax.fill_between(IOU_THRESHOLDS, ap_per_iou, alpha=0.1, color=color)

    # DCN+ reference points (AP50 and AP75)
    for mode, color, label in [("mask", DCN_COLOR, "DCN+ Mask"), ("box", "#F4A261", "DCN+ Box")]:
        ax.scatter([0.50, 0.75],
                   [DCN_PLUS[mode]["AP50"], DCN_PLUS[mode]["AP75"]],
                   color=color, s=80, zorder=5, marker="D",
                   label=f"{label} (paper ref)")

    ax.set_xlabel("IoU Threshold", fontsize=12)
    ax.set_ylabel("Mean AP across categories", fontsize=12)
    ax.set_title("AP vs IoU Threshold  (RF-DETR vs DCN+ reference points)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(0.48, 0.97)
    _savefig(fig, plots_dir / "iou_threshold_ap_curve.png")


# ── 10: Confusion Heatmap ─────────────────────────────────────────────────────
def plot_confusion(preds_by_cat, gts_by_cat, plots_dir, mode="mask"):
    _style()
    use_box = (mode == "box")
    mat, fn_row = compute_confusion_data(preds_by_cat, gts_by_cat, use_box=use_box)

    # Normalize by GT count
    gt_counts = np.array([len(gts_by_cat.get(c, [])) for c in CLASSES], dtype=float)
    gt_counts[gt_counts == 0] = 1
    mat_norm = mat / gt_counts[:, None]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat_norm, cmap="YlOrRd", vmin=0, vmax=1,
                   aspect="auto")
    plt.colorbar(im, ax=ax, label="Fraction of GT matched")

    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(CLASSES, fontsize=9)
    ax.set_xlabel("Predicted Category", fontsize=11)
    ax.set_ylabel("Ground-Truth Category", fontsize=11)
    ax.set_title(f"Detection Heatmap (TP rate per GT class)  [{mode.upper()} @IoU=0.50]",
                 fontsize=12, fontweight="bold")

    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{mat_norm[i,j]:.2f}",
                    ha="center", va="center", fontsize=9,
                    color="black" if mat_norm[i,j] > 0.5 else "white")

    _savefig(fig, plots_dir / f"confusion_heatmap_{mode}.png")


# ── 11: RF-DETR only — Mask vs Box overall AP/AP50/AP75 ──────────────────────
def plot_rfdetr_mask_vs_box_overall(rf_mask, rf_box, plots_dir):
    """
    Single-model chart: grouped bar showing RF-DETR Mask AP vs Box AP
    for each of AP, AP50, AP75.
    """
    _style()
    metrics   = ["AP", "AP50", "AP75"]
    mask_vals = [rf_mask["AP"], rf_mask["AP50"], rf_mask["AP75"]]
    box_vals  = [rf_box["AP"],  rf_box["AP50"],  rf_box["AP75"]]

    x = np.arange(len(metrics))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars_m = ax.bar(x - w/2, mask_vals, w, label="Mask AP",
                    color=RF_COLOR, alpha=0.88, edgecolor="#30363D")
    bars_b = ax.bar(x + w/2, box_vals,  w, label="Box AP (APbb)",
                    color="#A371F7", alpha=0.88, edgecolor="#30363D")

    for bar in bars_m:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.6,
                f"{bar.get_height():.1f}", ha="center", va="bottom",
                fontsize=12, color=RF_COLOR, fontweight="bold")
    for bar in bars_b:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.6,
                f"{bar.get_height():.1f}", ha="center", va="bottom",
                fontsize=12, color="#A371F7", fontweight="bold")

    # Δ between mask and box
    for i, (m, b) in enumerate(zip(mask_vals, box_vals)):
        delta = b - m
        sign  = "+" if delta >= 0 else ""
        col   = "#3FB950" if delta >= 0 else "#F85149"
        ax.annotate(f"Δ={sign}{delta:.1f}",
                    xy=(x[i], max(m, b) + 2.5),
                    ha="center", fontsize=10, color=col, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=13)
    ax.set_ylabel("AP Score", fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_title("RF-DETR  ·  Mask AP vs Box AP  (Overall)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)

    _savefig(fig, plots_dir / "rfdetr_mask_vs_box_overall.png")


# ── 12: RF-DETR only — Mask vs Box per-category ───────────────────────────────
def plot_rfdetr_mask_vs_box_per_category(rf_mask, rf_box, plots_dir):
    """
    Single-model chart: for each damage category, show RF-DETR Mask AP
    vs Box AP side by side.
    """
    _style()
    mask_vals = [rf_mask["per_category"][c] for c in CLASSES]
    box_vals  = [rf_box["per_category"][c]  for c in CLASSES]

    x = np.arange(len(CLASSES))
    w = 0.38

    fig, ax = plt.subplots(figsize=(14, 6))
    bars_m = ax.bar(x - w/2, mask_vals, w, label="Mask AP",
                    color=RF_COLOR, alpha=0.88, edgecolor="#30363D")
    bars_b = ax.bar(x + w/2, box_vals,  w, label="Box AP (APbb)",
                    color="#A371F7", alpha=0.88, edgecolor="#30363D")

    for bar in bars_m:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.7,
                f"{bar.get_height():.1f}", ha="center", va="bottom",
                fontsize=9, color=RF_COLOR, fontweight="bold")
    for bar in bars_b:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.7,
                f"{bar.get_height():.1f}", ha="center", va="bottom",
                fontsize=9, color="#A371F7", fontweight="bold")

    # Δ annotations (box − mask)
    for i, (m, b) in enumerate(zip(mask_vals, box_vals)):
        delta = b - m
        sign  = "+" if delta >= 0 else ""
        col   = "#3FB950" if delta >= 0 else "#F85149"
        ax.text(x[i], max(m, b) + 3.5, f"{sign}{delta:.1f}",
                ha="center", fontsize=9, color=col, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(" ", "\n") for c in CLASSES], fontsize=10)
    ax.set_ylabel("AP (IoU 0.50:0.95)", fontsize=12)
    ax.set_ylim(0, 108)
    ax.set_title("RF-DETR  ·  Mask AP vs Box AP  per Category",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.text(0.99, 0.98, "Δ = Box − Mask",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="#8B949E")

    _savefig(fig, plots_dir / "rfdetr_mask_vs_box_per_category.png")


# ── 13: 4-way comparison — overall AP/AP50/AP75 ───────────────────────────────
def plot_comparison_overall_mask_box(rf_mask, rf_box, plots_dir):
    """
    4-bar grouped chart per metric (AP, AP50, AP75):
      RF-DETR Mask | RF-DETR Box | DCN+ Mask | DCN+ Box
    """
    _style()
    metrics = ["AP", "AP50", "AP75"]

    rf_mask_vals  = [rf_mask["AP"],            rf_mask["AP50"],            rf_mask["AP75"]]
    rf_box_vals   = [rf_box["AP"],             rf_box["AP50"],             rf_box["AP75"]]
    dcn_mask_vals = [DCN_PLUS["mask"]["AP"],   DCN_PLUS["mask"]["AP50"],   DCN_PLUS["mask"]["AP75"]]
    dcn_box_vals  = [DCN_PLUS["box"]["AP"],    DCN_PLUS["box"]["AP50"],    DCN_PLUS["box"]["AP75"]]

    x = np.arange(len(metrics))
    w = 0.20

    fig, ax = plt.subplots(figsize=(11, 6))

    colors = [RF_COLOR, "#A371F7", DCN_COLOR, "#F4A261"]
    labels = ["RF-DETR Mask", "RF-DETR Box", "DCN+ Mask", "DCN+ Box"]
    all_vals = [rf_mask_vals, rf_box_vals, dcn_mask_vals, dcn_box_vals]
    offsets  = [-1.5*w, -0.5*w, 0.5*w, 1.5*w]

    bar_groups = []
    for vals, color, label, offset in zip(all_vals, colors, labels, offsets):
        bars = ax.bar(x + offset, vals, w, label=label,
                      color=color, alpha=0.88, edgecolor="#30363D")
        bar_groups.append(bars)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}", ha="center", va="bottom",
                    fontsize=8.5, color=color, fontweight="bold", rotation=0)

    # Separator lines between metric groups
    for xi in [0.5, 1.5]:
        ax.axvline(xi, color="#30363D", lw=1, ls="--", alpha=0.5)

    # Per-metric IoU footnotes below each group
    iou_notes = ["IoU 0.50:0.95", "IoU = 0.50", "IoU = 0.75"]
    for xi, note in zip(x, iou_notes):
        ax.text(xi, -7, note, ha="center", va="top",
                fontsize=8, color="#8B949E", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=13)
    ax.set_ylabel("AP Score", fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_title("AP / AP50 / AP75 Comparison: RF-DETR Mask & Box  vs  DCN+ Mask & Box",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, ncol=2, loc="upper left")

    _savefig(fig, plots_dir / "comparison_overall_mask_box.png")


# ── 14: 4-way comparison — per-category ──────────────────────────────────────
def plot_comparison_per_category_mask_box(rf_mask, rf_box, plots_dir):
    """
    For each damage category show 4 bars:
      RF-DETR Mask | RF-DETR Box | DCN+ Mask | DCN+ Box
    """
    _style()

    rf_mask_vals  = [rf_mask["per_category"][c]          for c in CLASSES]
    rf_box_vals   = [rf_box["per_category"][c]           for c in CLASSES]
    dcn_mask_vals = [DCN_PLUS["mask"]["per_category"][c] for c in CLASSES]
    dcn_box_vals  = [DCN_PLUS["box"]["per_category"][c]  for c in CLASSES]

    x = np.arange(len(CLASSES))
    w = 0.20

    colors = [RF_COLOR, "#A371F7", DCN_COLOR, "#F4A261"]
    labels = ["RF-DETR Mask", "RF-DETR Box", "DCN+ Mask", "DCN+ Box"]
    all_vals = [rf_mask_vals, rf_box_vals, dcn_mask_vals, dcn_box_vals]
    offsets  = [-1.5*w, -0.5*w, 0.5*w, 1.5*w]

    fig, ax = plt.subplots(figsize=(16, 6.5))

    for vals, color, label, offset in zip(all_vals, colors, labels, offsets):
        bars = ax.bar(x + offset, vals, w, label=label,
                      color=color, alpha=0.88, edgecolor="#30363D")
        for bar in bars:
            h = bar.get_height()
            if h > 2:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.6,
                        f"{h:.1f}", ha="center", va="bottom",
                        fontsize=7.5, color=color, fontweight="bold", rotation=90)

    # Category separator lines
    for xi in np.arange(0.5, len(CLASSES)-0.5, 1):
        ax.axvline(xi, color="#30363D", lw=0.8, ls="--", alpha=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace(" ", "\n") for c in CLASSES], fontsize=10)
    ax.set_ylabel("AP (IoU 0.50:0.95)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_title("Per-Category AP: RF-DETR Mask & Box  vs  DCN+ Mask & Box",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, ncol=2, loc="upper right")

    _savefig(fig, plots_dir / "comparison_per_category_mask_box.png")



# ── PRINT TABLE ───────────────────────────────────────────────────────────────

def print_table(mask_results, box_results):
    SEP = "=" * 92

    for mode, results, label in [
        ("mask", mask_results, "Mask AP"),
        ("box",  box_results,  "Box AP (APbb)"),
    ]:
        dcn = DCN_PLUS[mode]
        print(f"\n{SEP}")
        print(f"  CarDD TEST SET — {label} Comparison")
        print(f"  Coordinate space : ORIGINAL image coordinates (predictions projected)")
        print(f"  Headline evaluator: pycocotools COCOeval")
        print(f"  Paper: Table IV & V  (DCN+ ResNet-101, test split)")
        print(SEP)

        if results:
            print(f"\n  {'Metric':<10} {'DCN+(R101)':>12} "
                  f"{'RF-DETR(pycoco)':>17} "
                  f"{'Δ(pycoco-DCN+)':>16}")
            print(f"  {'─'*74}")
            for m in ["AP", "AP50", "AP75"]:
                d    = dcn[m]
                r_off = results.get(m, 0.0)
                delta = r_off - d
                sign  = "+" if delta >= 0 else ""
                win   = "←RF-DETR" if delta > 0.5 else ("←DCN+" if delta < -0.5 else "≈tie")
                print(f"  {m:<10} {d:>12.1f} {r_off:>17.1f} "
                      f"{sign+str(round(delta,1)):>16}  {win}")

            print(f"\n  {'Category':<18} {'DCN+':>8} "
                  f"{'RF-DETR(pycoco)':>17} {'Δ(pycoco)':>12}")
            print(f"  {'─'*74}")
            for cat in CLASSES:
                d     = dcn["per_category"][cat]
                r_off = results.get("per_category", {}).get(cat, 0.0)
                delta = r_off - d
                sign  = "+" if delta >= 0 else ""
                bar   = "▲" if delta > 1 else ("▼" if delta < -1 else "–")
                print(f"  {cat:<18} {d:>8.1f} {r_off:>17.1f} "
                      f"{sign+str(round(delta,1)):>12}  {bar}")
        else:
            print("[WARN] No metrics to display.")

        print(f"  {'─'*50}")
        mean_d = np.mean(list(dcn["per_category"].values()))
        if results:
            mean_r = np.mean(list(results.get("per_category", {}).values()))
            delta  = mean_r - mean_d
            sign   = "+" if delta >= 0 else ""
            print(f"  {'Mean':<18} {mean_d:>10.1f} {mean_r:>10.1f} {sign+str(round(delta,1)):>8}")
        print(SEP)


# ── INFERENCE ─────────────────────────────────────────────────────────────────

def run_inference(model, img_path, threshold, idx_to_name, orig_W=None, orig_H=None):
    """
    Run RF-DETR inference and project all outputs to the ORIGINAL image
    coordinate space before returning.

    - Masks are resized from the model's output resolution to (orig_H, orig_W)
      using bilinear interpolation + 0.5 threshold (see project_prediction_to_original).
    - Bounding boxes are derived from the projected mask (tight bbox in original coords).

    Parameters
    ----------
    idx_to_name : dict  — maps model class-index (0-based int) → class name str.
                  Built from the RESIZED dataset's category list sorted by id,
                  so it exactly matches the model's training label order.
    orig_W, orig_H : int, optional — True original dimensions. If None, falls
                     back to reading the physical image file dimensions.

    Returns
    -------
    results : list of dicts — each with keys class_name, score, mask (RLE in
              original coords), bbox (x1,y1,x2,y2 in original coords), cat
    orig_W  : int — original image width
    orig_H  : int — original image height
    """
    img_pil = Image.open(img_path).convert("RGB")
    if orig_W is None or orig_H is None:
        orig_W, orig_H = img_pil.size
    det = model.predict(img_pil, threshold=threshold)
    if det is None or len(det) == 0:
        return [], orig_W, orig_H

    indices  = np.argsort(-det.confidence)
    MAX_DETS = 100
    results  = []

    for idx in indices[:MAX_DETS]:
        cid   = int(det.class_id[idx])
        # Use idx_to_name (derived from resized dataset) — NOT the hardcoded
        # CLASSES list — so the mapping survives any category-order difference.
        name  = idx_to_name.get(cid, f"cls_{cid}")
        score = float(det.confidence[idx])

        # ── MASK: project from model output space → original image coords ──
        if det.mask is not None:
            raw_mask = det.mask[idx]   # shape varies by model impl (e.g. 1152×1152)
            projected_rle, mask_orig = project_prediction_to_original(
                raw_mask, orig_W, orig_H)
        else:
            # No mask output: synthesise a rectangular mask from model bbox.
            # Scale bbox from img_pil dimensions to orig_W x orig_H
            scale_x = orig_W / img_pil.width
            scale_y = orig_H / img_pil.height
            x1, y1, x2, y2 = (float(v) for v in det.xyxy[idx])
            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
            mask_orig = np.zeros((orig_H, orig_W), dtype=np.uint8)
            mask_orig[max(0, y1):min(orig_H, y2),
                      max(0, x1):min(orig_W, x2)] = 1
            projected_rle = maskUtils.encode(np.asfortranarray(mask_orig))
            projected_rle["counts"] = projected_rle["counts"].decode("utf-8")
            projected_rle["size"]   = [orig_H, orig_W]

        # ── BOUNDING BOX: tight bbox derived from projected mask ───────────
        bbox_from_mask = mask_to_bbox(mask_orig)
        if bbox_from_mask is not None:
            bbox = bbox_from_mask
        else:
            # Fallback if mask is empty (very rare)
            scale_x = orig_W / img_pil.width
            scale_y = orig_H / img_pil.height
            x1, y1, x2, y2 = (float(v) for v in det.xyxy[idx])
            bbox = (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)

        results.append({
            "class_name": name,
            "score":      score,
            "mask":       projected_rle,   # RLE in ORIGINAL image coords
            "bbox":       bbox,            # (x1,y1,x2,y2) in ORIGINAL image coords
            "cat":        name,
        })

    del det
    return results, orig_W, orig_H


# ── MAIN ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate RF-DETR segmentation on CarDD test set using "
                    "pycocotools COCOeval in original image coordinates.")
    p.add_argument("--images_dir",  default=IMAGES_DIR)
    p.add_argument("--annotations", default=ANNOTATIONS,
                   help="COCO annotation JSON for the inference image list "
                        "(may be the resized dataset).")
    p.add_argument("--orig_annotations", default=None,
                   help="Path to the ORIGINAL (un-resized) CarDD test annotation "
                        "JSON used for pycocotools evaluation and "
                        "image_id / category_id mapping. "
                        "If omitted, falls back to --annotations "
                        "(coordinate reprojection still uses true image H×W).")
    p.add_argument("--checkpoint",  default=CHECKPOINT)
    p.add_argument("--resolution",  type=int,   default=RESOLUTION)
    p.add_argument("--threshold",   type=float, default=THRESHOLD)
    p.add_argument("--output_json", default=OUTPUT_JSON)
    p.add_argument("--plots_dir",   default=PLOTS_DIR)
    return p.parse_args()


def main():
    args = parse_args()
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # ── Load resized annotations (provides the image file-name list) ──────
    print(f"\n[INFO] Loading resized annotations: {args.annotations}")
    with open(args.annotations) as f:
        coco_resized = json.load(f)

    images_resized = {img["id"]: img for img in coco_resized["images"]}
    anns_resized   = coco_resized["annotations"]

    if len(images_resized) != 374:
        print(f"[WARN] Expected 374 images in resized annotations, "
              f"got {len(images_resized)}")
    if len(anns_resized) != 785:
        print(f"[WARN] Expected 785 annotations in resized set, "
              f"got {len(anns_resized)}")
    print(f"[INFO] Resized set: {len(images_resized)} images, "
          f"{len(anns_resized)} annotations")

    # ── Build model class-index → name from RESIZED dataset category order ─
    # The model's class_id output is 0-based and corresponds to the sorted
    # category-id order of the dataset it was trained on (the resized file).
    # This may DIFFER from the hardcoded CLASSES list if categories were
    # assigned IDs in a different order during dataset preparation.
    resized_cats_sorted = sorted(coco_resized["categories"], key=lambda c: c["id"])
    model_idx_to_name: dict = {i: c["name"] for i, c in enumerate(resized_cats_sorted)}
    print(f"\n[INFO] Model class-index → name (from resized dataset): {model_idx_to_name}")

    # Warn if the resized category order differs from the hardcoded CLASSES list
    hardcoded_order = {i: c for i, c in enumerate(CLASSES)}
    if model_idx_to_name != hardcoded_order:
        print(f"[WARN] Resized dataset category order DIFFERS from hardcoded CLASSES list!")
        print(f"       Hardcoded CLASSES : {hardcoded_order}")
        print(f"       Resized file order: {model_idx_to_name}")
        print(f"       Using resized file order for class-name mapping (correct).")
    else:
        print(f"[INFO] ✓ Resized category order matches hardcoded CLASSES list.")

    # ── Load ORIGINAL annotations (for pycocotools + ID/category mapping) ─
    orig_ann_path = args.orig_annotations if args.orig_annotations else args.annotations
    print(f"\n[INFO] Loading original annotations: {orig_ann_path}")
    coco_gt_orig = COCO(orig_ann_path)

    # filename (basename) → original image_id
    # (image order / IDs may differ between resized and original datasets)
    fname_to_orig_id: dict = {}
    for img_id, img_info in coco_gt_orig.imgs.items():
        fname_to_orig_id[Path(img_info["file_name"]).name] = img_id

    # class_name → original category_id (name-based lookup, not positional)
    # Original dataset may have different numeric IDs for the same class names.
    name_to_orig_cat_id: dict = {
        cat["name"]: cat["id"]
        for cat in coco_gt_orig.dataset["categories"]
    }
    print(f"[INFO] Original annotations: {len(coco_gt_orig.imgs)} images, "
          f"{len(coco_gt_orig.anns)} annotations")
    print(f"[INFO] Original category → ID: {name_to_orig_cat_id}")

    # Cross-check: every class the model can predict must exist in original annotations
    model_class_names = set(model_idx_to_name.values())
    missing_in_orig   = model_class_names - set(name_to_orig_cat_id.keys())
    if missing_in_orig:
        print(f"[WARN] Classes in model NOT found in original annotations: {missing_in_orig}")
        print(f"       Predictions for these classes will be EXCLUDED from COCOeval!")
    else:
        print(f"[INFO] ✓ All model class names found in original annotations.")

    # Also warn about any mismatch in the original category order vs resized
    orig_cats_sorted   = sorted(coco_gt_orig.dataset["categories"], key=lambda c: c["id"])
    orig_name_order    = [c["name"] for c in orig_cats_sorted]
    resized_name_order = [c["name"] for c in resized_cats_sorted]
    if orig_name_order != resized_name_order:
        print(f"[WARN] Category NAME ORDER differs between original and resized datasets!")
        print(f"       Original order : {orig_name_order}")
        print(f"       Resized  order : {resized_name_order}")
        print(f"       Name-based lookup (name_to_orig_cat_id) handles this correctly.")
    else:
        print(f"[INFO] ✓ Category name order matches between original and resized datasets.")

    # ── Decode GT masks + boxes from ORIGINAL annotations ────────────────
    # GT must be in original image coordinates to match projected predictions.
    gts_mask_by_cat = defaultdict(list)
    gts_box_by_cat  = defaultdict(list)
    print("[INFO] Decoding GT masks from original annotations …")
    decode_errors = 0

    for ann_id, ann in coco_gt_orig.anns.items():
        img_info = coco_gt_orig.imgs[ann["image_id"]]
        H, W     = img_info["height"], img_info["width"]
        cat_name = coco_gt_orig.cats[ann["category_id"]]["name"]
        if cat_name not in CLASSES:
            continue
        cname = cat_name

        seg = ann["segmentation"]
        try:
            if isinstance(seg, dict):
                gt_mask = maskUtils.decode(seg).astype(bool)
            elif isinstance(seg, list) and len(seg) > 0:
                gt_mask = polygon_to_mask(seg, H, W)
            else:
                decode_errors += 1
                continue
        except Exception:
            decode_errors += 1
            continue

        rle = maskUtils.encode(np.asfortranarray(gt_mask.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("utf-8")

        entry_mask = {"image_id": ann["image_id"], "mask": rle, "cat": cname}
        gts_mask_by_cat[cname].append(entry_mask)

        # GT box from annotation (COCO format: x,y,w,h → convert to x1,y1,x2,y2)
        x, y, w, h = ann["bbox"]
        entry_box  = {"image_id": ann["image_id"],
                      "bbox": (x, y, x + w, y + h), "cat": cname}
        gts_box_by_cat[cname].append(entry_box)

    if decode_errors:
        print(f"[WARN] {decode_errors} GT annotations could not be decoded")
    print(f"[INFO] GT per category: { {c: len(v) for c,v in gts_mask_by_cat.items()} }")

    # ── Load model ────────────────────────────────────────────────────────
    if not os.path.isfile(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}"); sys.exit(1)

    print(f"\n[INFO] Loading model …")
    from rfdetr import RFDETRSegMedium
    model = RFDETRSegMedium(pretrain_weights=args.checkpoint, resolution=args.resolution)
    try:
        import torch
        model.optimize_for_inference(compile=True, batch_size=1, dtype=torch.float32)
        print("[INFO] Model optimized (fp32).")
    except Exception as e:
        print(f"[WARN] Optimization skipped: {e}")

    # ── Inference ─────────────────────────────────────────────────────────
    images_dir = Path(args.images_dir)
    preds_mask_by_cat = defaultdict(list)
    preds_box_by_cat  = defaultdict(list)
    coco_results_segm: list = []   # For COCOeval (segmentation)
    coco_results_bbox: list = []   # For COCOeval (bounding box)
    failed = 0

    print(f"\n[INFO] Running inference on {len(images_resized)} images …")
    print(f"[INFO] Confidence threshold  : {args.threshold}")
    print(f"[INFO] Coordinate space      : predictions projected to original image space")
    t0 = time.time()

    for img_id, img_info in tqdm(images_resized.items(), desc="Inference", ncols=70):
        img_path = images_dir / img_info["file_name"]
        if not img_path.exists():
            hits = list(images_dir.rglob(img_info["file_name"]))
            img_path = hits[0] if hits else None
        if img_path is None or not img_path.exists():
            failed += 1; continue

        # Resolve the ORIGINAL image_id by filename matching.
        # Three-step fallback handles Roboflow-renamed files:
        #   Step 1: exact basename match  (both datasets use the same filename)
        #   Step 2: Roboflow-decoded match (strip '.rf.<hash>' suffix → original name)
        #   Step 3: stem-only match        (last resort, e.g. extension changed)
        fname        = Path(img_info["file_name"]).name
        decoded_fname = decode_roboflow_filename(fname)

        orig_img_id = fname_to_orig_id.get(fname)                     # Step 1
        if orig_img_id is None and decoded_fname != fname:
            orig_img_id = fname_to_orig_id.get(decoded_fname)         # Step 2
            if orig_img_id is not None:
                pass  # matched via Roboflow decode
        if orig_img_id is None:
            # Step 3: stem-only scan (slow, only for edge cases)
            fname_stem   = Path(fname).stem
            decoded_stem = Path(decoded_fname).stem
            for k, v in fname_to_orig_id.items():
                if Path(k).stem in (fname_stem, decoded_stem):
                    orig_img_id = v
                    break
        if orig_img_id is None:
            print(f"\n[WARN] '{fname}' (decoded: '{decoded_fname}') not found in "
                  f"original annotations — predictions omitted from COCOeval")

        # Get true original dimensions for projection
        if orig_img_id is not None:
            orig_info = coco_gt_orig.imgs[orig_img_id]
            true_W, true_H = orig_info["width"], orig_info["height"]
        else:
            true_W, true_H = img_info["width"], img_info["height"]

        try:
            preds, orig_W, orig_H = run_inference(
                model, img_path, args.threshold, model_idx_to_name, true_W, true_H)
        except Exception as e:
            print(f"\n[WARN] {img_info['file_name']}: {e}")
            failed += 1; continue

        # Effective image_id (use orig when available)
        eff_img_id = orig_img_id if orig_img_id is not None else img_id

        for p in preds:
            cat = p["class_name"]
            if cat not in CLASSES:
                continue

            x1, y1, x2, y2 = p["bbox"]   # already in original image coords
            bbox_xywh = [float(x1), float(y1),
                         float(x2 - x1), float(y2 - y1)]

            # ── Extract results ──────────────
            preds_mask_by_cat[cat].append({
                "image_id": eff_img_id,
                "score":    p["score"],
                "mask":     p["mask"],          # projected RLE in orig coords
                "cat":      cat,
            })
            preds_box_by_cat[cat].append({
                "image_id": eff_img_id,
                "score":    p["score"],
                "bbox":     (x1, y1, x2, y2),  # (x1,y1,x2,y2) in orig coords
                "cat":      cat,
            })

            # ── COCO results (only when orig ID is resolved) ──
            if orig_img_id is not None:
                orig_cat_id = name_to_orig_cat_id.get(cat)
                if orig_cat_id is not None:
                    coco_results_segm.append({
                        "image_id":     orig_img_id,
                        "category_id":  orig_cat_id,
                        "segmentation": p["mask"],    # projected RLE
                        "score":        float(p["score"]),
                        "bbox":         bbox_xywh,    # COCO format: [x,y,w,h]
                    })
                    coco_results_bbox.append({
                        "image_id":    orig_img_id,
                        "category_id": orig_cat_id,
                        "bbox":        bbox_xywh,     # COCO format: [x,y,w,h]
                        "score":       float(p["score"]),
                    })

    elapsed = time.time() - t0
    total_p = sum(len(v) for v in preds_mask_by_cat.values())
    print(f"[INFO] Done in {elapsed:.1f}s  |  Total preds: {total_p}  |  Failed: {failed}")
    print(f"[INFO] COCO results entries : {len(coco_results_segm)} segm / "
          f"{len(coco_results_bbox)} bbox")

    # ── Pycocotools evaluation (headline numbers) ───────────────
    print("\n[INFO] Running pycocotools evaluation …")
    mask_results = run_cocoeval(
        coco_gt_orig, coco_results_segm, "segm", label="RF-DETR")
    box_results  = run_cocoeval(
        coco_gt_orig, coco_results_bbox, "bbox", label="RF-DETR")

    print_table(mask_results, box_results)

    # ── Save COCO results JSONs ───────────────────────────────────────────
    out_base  = str(args.output_json).replace(".json", "")
    segm_json = out_base + "_coco_segm.json"
    bbox_json = out_base + "_coco_bbox.json"
    with open(segm_json, "w") as f:
        json.dump(coco_results_segm, f, indent=2)
    with open(bbox_json, "w") as f:
        json.dump(coco_results_bbox, f, indent=2)
    print(f"\n[INFO] COCO segm results  → {segm_json}")
    print(f"[INFO] COCO bbox results  → {bbox_json}")

    # ── Generate all plots ────────────────────────────────────────────────
    print("\n[INFO] Generating plots …")

    plot_pr_curves(mask_results, plots_dir, mode="mask")
    plot_pr_curves(box_results,  plots_dir, mode="box")

    mask_best = plot_f1_vs_threshold(dict(preds_mask_by_cat), dict(gts_mask_by_cat),
                                     plots_dir, mode="mask")
    box_best  = plot_f1_vs_threshold(dict(preds_box_by_cat),  dict(gts_box_by_cat),
                                     plots_dir, mode="box")

    plot_ap_comparison_bar(mask_results, mode="mask", plots_dir=plots_dir)
    plot_ap_comparison_bar(box_results,  mode="box",  plots_dir=plots_dir)

    plot_radar(mask_results, box_results, plots_dir)

    plot_prf_summary(mask_best, box_best, plots_dir)



    plot_confusion(dict(preds_mask_by_cat), dict(gts_mask_by_cat),
                   plots_dir, mode="mask")
    plot_confusion(dict(preds_box_by_cat),  dict(gts_box_by_cat),
                   plots_dir, mode="box")

    # ── Mask vs Box charts ─────────────────────
    plot_rfdetr_mask_vs_box_overall(mask_results, box_results, plots_dir)
    plot_rfdetr_mask_vs_box_per_category(mask_results, box_results, plots_dir)
    plot_comparison_overall_mask_box(mask_results, box_results, plots_dir)
    plot_comparison_per_category_mask_box(mask_results, box_results, plots_dir)

    # ── Save full JSON (all chart values) ────────────────────────────────
    # PR curve data (IoU=0.50, sampled at 20 points for readability)
    def sample_curve(rec, prec, n=20):
        idx = np.linspace(0, len(rec)-1, min(n, len(rec)), dtype=int)
        return {"recall":    [round(float(rec[i]),  3) for i in idx],
                "precision": [round(float(prec[i]), 3) for i in idx]}

    pr_curves_mask, pr_curves_box = {}, {}
    for cat in CLASSES:
        r_m, p_m = mask_results["_curves"][cat][0.50]
        r_b, p_b = box_results["_curves"][cat][0.50]
        pr_curves_mask[cat] = sample_curve(r_m, p_m)
        pr_curves_box[cat]  = sample_curve(r_b, p_b)

    off_m_per = (mask_results or {}).get("per_category", {})
    off_b_per = (box_results  or {}).get("per_category", {})

    output = {
        "model":      "RF-DETR-Seg-Medium",
        "checkpoint": str(args.checkpoint),
        "resolution": args.resolution,
        "threshold":  args.threshold,
        "eval_methodology": {
            "coordinate_space":   "original image coordinates (projected from model output)",
            "headline_evaluator": "pycocotools.cocoeval.COCOeval",
            "orig_annotations":   orig_ann_path,
        },
        "dataset": {
            "images":      len(images_resized),
            "annotations": len(anns_resized),
            "note":        "CarDD test split (image list from resized annotation file)",
        },

        # ── HEADLINE: pycocotools results ────────────────────────
        "mask_ap": {
            "AP":           mask_results.get("AP",   None),
            "AP50":         mask_results.get("AP50", None),
            "AP75":         mask_results.get("AP75", None),
            "APs":          mask_results.get("APs",  None),
            "APm":          mask_results.get("APm",  None),
            "APl":          mask_results.get("APl",  None),
            "per_category": mask_results.get("per_category", {}),
        } if mask_results else None,
        "box_ap": {
            "AP":           box_results.get("AP",   None),
            "AP50":         box_results.get("AP50", None),
            "AP75":         box_results.get("AP75", None),
            "APs":          box_results.get("APs",  None),
            "APm":          box_results.get("APm",  None),
            "APl":          box_results.get("APl",  None),
            "per_category": box_results.get("per_category", {}),
        } if box_results else None,

        # ── DCN+ reference ────────────────────────────────────────────────
        "dcnplus_reference": DCN_PLUS,

        # ── Delta vs DCN+ ───────
        "delta_vs_dcnplus": {
            "mask": {
                "AP":   round(mask_results["AP"]   - DCN_PLUS["mask"]["AP"],   1),
                "AP50": round(mask_results["AP50"] - DCN_PLUS["mask"]["AP50"], 1),
                "AP75": round(mask_results["AP75"] - DCN_PLUS["mask"]["AP75"], 1),
                "per_category": {
                    cat: round(off_m_per.get(cat, 0) - DCN_PLUS["mask"]["per_category"][cat], 1)
                    for cat in CLASSES
                },
            } if mask_results else {},
            "box": {
                "AP":   round(box_results["AP"]   - DCN_PLUS["box"]["AP"],   1),
                "AP50": round(box_results["AP50"] - DCN_PLUS["box"]["AP50"], 1),
                "AP75": round(box_results["AP75"] - DCN_PLUS["box"]["AP75"], 1),
                "per_category": {
                    cat: round(off_b_per.get(cat, 0) - DCN_PLUS["box"]["per_category"][cat], 1)
                    for cat in CLASSES
                },
            } if box_results else {},
        },

        # ── Mask vs Box delta ──
        "mask_vs_box_delta": {
            "AP":   round(box_results.get("AP", 0)   - mask_results.get("AP", 0),   1),
            "AP50": round(box_results.get("AP50", 0) - mask_results.get("AP50", 0), 1),
            "AP75": round(box_results.get("AP75", 0) - mask_results.get("AP75", 0), 1),
            "per_category": {
                cat: round(off_b_per.get(cat, 0) - off_m_per.get(cat, 0), 1)
                for cat in CLASSES
            },
        } if box_results and mask_results else {},

        # ── F1 / Precision / Recall at best threshold ─────────────────────
        "best_threshold_metrics": {
            "mask": {
                "threshold":  round(mask_best[0], 3),
                "precision":  round(mask_best[1], 4),
                "recall":     round(mask_best[2], 4),
                "f1":         round(mask_best[3], 4),
            },
            "box": {
                "threshold":  round(box_best[0], 3),
                "precision":  round(box_best[1], 4),
                "recall":     round(box_best[2], 4),
                "f1":         round(box_best[3], 4),
            },
        },

        # ── PR curve data (IoU=0.50, charts 1 & 2) ───────────────────────
        "pr_curves_at_iou50": {
            "mask": pr_curves_mask,
            "box":  pr_curves_box,
        },

        # ── Inference stats ───────────────────────────────────────────────
        "inference_stats": {
            "time_seconds":      round(elapsed, 1),
            "failed_images":     failed,
            "total_predictions": total_p,
            "predictions_per_category": {
                c: len(preds_mask_by_cat.get(c, [])) for c in CLASSES},
        },
    }

    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[INFO] Results saved → {args.output_json}")

    print(f"\n[INFO] ✓ All plots saved to: {plots_dir}/")
    print("[INFO] Files generated:")
    for f in sorted(plots_dir.glob("*.png")):
        print(f"        {f.name}")


if __name__ == "__main__":
    main()