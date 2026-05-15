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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors

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
    return float(x1), float(y1), float(x2), float(y2)


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

# ── AP ENGINE ─────────────────────────────────────────────────────────────────

def ap_101point(recalls, precisions):
    r = np.concatenate(([0.0], recalls,    [1.0]))
    p = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(p)-2, -1, -1):
        p[i] = max(p[i], p[i+1])
    idx = np.where(r[1:] != r[:-1])[0]
    return float(np.sum((r[idx+1]-r[idx]) * p[idx+1]))


def _match_preds_gts(preds_sorted, gt_by_img, iou_thresh, use_box=False):
    """Return (tp, fp, n_gt, recalls, precisions, matched_cat_ids)."""
    tp = np.zeros(len(preds_sorted))
    fp = np.zeros(len(preds_sorted))
    matched_gts = set()
    matched_cats = []   # (pred_cat, gt_cat) for confusion matrix

    for pi, pred in enumerate(preds_sorted):
        best_iou = iou_thresh - 1e-9
        best_gi  = -1
        best_gt_cat = None

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
                best_gt_cat = g.get("cat", pred["cat"])

        if best_gi >= 0:
            tp[pi] = 1
            matched_gts.add(best_gi)
            matched_cats.append((pred["cat"], best_gt_cat))
        else:
            fp[pi] = 1
            matched_cats.append((pred["cat"], None))

    return tp, fp, matched_cats


def ap_and_curve(preds, gts, iou_thresh, use_box=False):
    """
    Returns (AP, recalls_array, precisions_array).
    """
    if not preds or not gts:
        return 0.0, np.array([0.0]), np.array([0.0])

    gt_by_img = defaultdict(list)
    for i, g in enumerate(gts):
        gt_by_img[g["image_id"]].append((i, g))

    preds_sorted = sorted(preds, key=lambda x: -x["score"])
    tp, fp, _ = _match_preds_gts(preds_sorted, gt_by_img, iou_thresh, use_box)

    n_gt = len(gts)
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls    = tp_cum / (n_gt + 1e-6)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-6)
    ap = ap_101point(recalls, precisions) * 100.0
    return ap, recalls, precisions


def compute_metrics(preds_by_cat, gts_by_cat, use_box=False):
    """Compute AP, AP50, AP75, per-category AP and raw PR curves."""
    per_cat_ap   = {}
    per_cat_ap50 = {}
    per_cat_ap75 = {}
    curves = {}   # cat → {iou → (recalls, precisions)}

    for cat in CLASSES:
        preds = preds_by_cat.get(cat, [])
        gts   = gts_by_cat.get(cat, [])

        aps_all = []
        cat_curves = {}
        for iou in IOU_THRESHOLDS:
            ap_val, rec, prec = ap_and_curve(preds, gts, iou, use_box)
            aps_all.append(ap_val)
            cat_curves[round(float(iou), 2)] = (rec, prec)

        per_cat_ap[cat]   = float(np.mean(aps_all))
        per_cat_ap50[cat] = cat_curves[0.50][0], cat_curves[0.50][1]  # will be overwritten below
        per_cat_ap75[cat] = cat_curves[0.75]
        curves[cat] = cat_curves

        # Scalar AP50 / AP75
        per_cat_ap50[cat] = ap_and_curve(preds, gts, 0.50, use_box)[0]
        per_cat_ap75[cat] = ap_and_curve(preds, gts, 0.75, use_box)[0]

    AP   = float(np.mean(list(per_cat_ap.values())))
    AP50 = float(np.mean(list(per_cat_ap50.values())))
    AP75 = float(np.mean(list(per_cat_ap75.values())))

    return {
        "AP":           round(AP,   1),
        "AP50":         round(AP50, 1),
        "AP75":         round(AP75, 1),
        "per_category": {c: round(v, 1) for c, v in per_cat_ap.items()},
        "_per_cat_ap50": per_cat_ap50,
        "_per_cat_ap75": per_cat_ap75,
        "_curves":       curves,
    }


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

    # Subtle annotation strip
    ax.axhspan(0, 5, color="#161B22", zorder=0)
    ax.text(0.99, 0.02, "CarDD Test Set · 374 images",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="#8B949E")

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

def print_table(rf_mask, rf_box):
    SEP = "=" * 72

    for mode, rf, label in [("mask", rf_mask, "Mask AP"), ("box", rf_box, "Box AP (APbb)")]:
        dcn = DCN_PLUS[mode]
        print(f"\n{SEP}")
        print(f"  CarDD TEST SET — {label} Comparison")
        print(f"  374 images · 785 instances · IoU 0.50:0.95")
        print(f"  Paper: Table IV & V  (DCN+ ResNet-101, test split)")
        print(SEP)
        print(f"\n  {'Metric':<10} {'DCN+ (R101)':>14} {'RF-DETR-Seg':>14} {'Δ':>8}")
        print(f"  {'─'*50}")
        for m in ["AP", "AP50", "AP75"]:
            d = dcn[m]; r = rf[m]; delta = r - d
            sign = "+" if delta >= 0 else ""
            win  = "←RF-DETR" if delta>0.5 else ("←DCN+" if delta<-0.5 else "≈tie")
            print(f"  {m:<10} {d:>14.1f} {r:>14.1f} {sign+str(round(delta,1)):>8}  {win}")
        print(f"\n  {'Category':<18} {'DCN+':>10} {'RF-DETR':>10} {'Δ':>8}")
        print(f"  {'─'*50}")
        for cat in CLASSES:
            d = dcn["per_category"][cat]; r = rf["per_category"][cat]; delta = r-d
            sign = "+" if delta >= 0 else ""
            bar  = "▲" if delta>1 else ("▼" if delta<-1 else "–")
            print(f"  {cat:<18} {d:>10.1f} {r:>10.1f} {sign+str(round(delta,1)):>8}  {bar}")
        print(f"  {'─'*50}")
        mean_d = np.mean(list(dcn["per_category"].values()))
        mean_r = np.mean(list(rf["per_category"].values()))
        delta  = mean_r - mean_d
        sign   = "+" if delta >= 0 else ""
        print(f"  {'Mean':<18} {mean_d:>10.1f} {mean_r:>10.1f} {sign+str(round(delta,1)):>8}")
        print(SEP)


# ── INFERENCE ─────────────────────────────────────────────────────────────────

def run_inference(model, img_path, threshold):
    img_pil = Image.open(img_path).convert("RGB")
    W, H = img_pil.size
    det = model.predict(img_pil, threshold=threshold)
    if det is None or len(det) == 0:
        return []

    indices = np.argsort(-det.confidence)
    MAX_DETS = 100
    results = []

    for idx in indices[:MAX_DETS]:
        cid   = int(det.class_id[idx])
        name  = CLASSES[cid] if 0 <= cid < len(CLASSES) else f"cls_{cid}"
        score = float(det.confidence[idx])

        # ── MASK ──
        if det.mask is not None:
            full_mask = pred_mask_to_full(det.mask[idx].astype(bool), H, W)
        else:
            x1, y1, x2, y2 = (int(v) for v in det.xyxy[idx])
            full_mask = np.zeros((H, W), dtype=np.uint8)
            full_mask[max(0,y1):min(H,y2), max(0,x1):min(W,x2)] = 1

        rle = maskUtils.encode(np.asfortranarray(full_mask.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("utf-8")

        # ── BOUNDING BOX ──
        # Derive from mask if available (tighter); fall back to model bbox
        bbox_from_mask = mask_to_bbox(full_mask)
        if bbox_from_mask is not None:
            bbox = bbox_from_mask
        else:
            x1, y1, x2, y2 = (float(v) for v in det.xyxy[idx])
            bbox = (x1, y1, x2, y2)

        results.append({
            "class_name": name,
            "score":      score,
            "mask":       rle,
            "bbox":       bbox,   # (x1,y1,x2,y2)
            "cat":        name,
        })

    del det
    return results


# ── MAIN ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir",  default=IMAGES_DIR)
    p.add_argument("--annotations", default=ANNOTATIONS)
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

    # ── Load annotations ──────────────────────────────────────────────────
    print(f"\n[INFO] Loading annotations: {args.annotations}")
    with open(args.annotations) as f:
        coco = json.load(f)

    images    = {img["id"]: img for img in coco["images"]}
    anns      = coco["annotations"]
    coco_cats = {c["id"]: c["name"] for c in coco["categories"]}

    assert len(images) == 374, f"Expected 374 images, got {len(images)}"
    assert len(anns)   == 785, f"Expected 785 annotations, got {len(anns)}"
    print(f"[INFO] ✓ 374 images, 785 annotations")

    # ── Decode GT masks + GT boxes ────────────────────────────────────────
    gts_mask_by_cat = defaultdict(list)
    gts_box_by_cat  = defaultdict(list)
    print("[INFO] Decoding GT masks …")
    decode_errors = 0

    for ann in tqdm(anns, desc="GT masks", ncols=70):
        img   = images[ann["image_id"]]
        H, W  = img["height"], img["width"]
        cname = coco_cats.get(ann["category_id"])
        if cname not in CLASSES:
            continue

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

        # GT box from annotation (COCO format: x,y,w,h)
        x, y, w, h = ann["bbox"]
        entry_box  = {"image_id": ann["image_id"],
                      "bbox": (x, y, x+w, y+h), "cat": cname}
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
    failed = 0

    print(f"\n[INFO] Running inference on {len(images)} images …")
    print(f"[INFO] Confidence threshold: {args.threshold}")
    t0 = time.time()

    for img_id, img_info in tqdm(images.items(), desc="Inference", ncols=70):
        img_path = images_dir / img_info["file_name"]
        if not img_path.exists():
            hits = list(images_dir.rglob(img_info["file_name"]))
            img_path = hits[0] if hits else None
        if img_path is None or not img_path.exists():
            failed += 1; continue

        try:
            preds = run_inference(model, img_path, args.threshold)
        except Exception as e:
            print(f"\n[WARN] {img_info['file_name']}: {e}")
            failed += 1; continue

        for p in preds:
            cat = p["class_name"]
            if cat not in CLASSES:
                continue
            preds_mask_by_cat[cat].append({
                "image_id": img_id, "score": p["score"],
                "mask": p["mask"], "cat": cat})
            preds_box_by_cat[cat].append({
                "image_id": img_id, "score": p["score"],
                "bbox": p["bbox"], "cat": cat})

    elapsed = time.time() - t0
    total_p = sum(len(v) for v in preds_mask_by_cat.values())
    print(f"[INFO] Done in {elapsed:.1f}s  |  Total preds: {total_p}  |  Failed: {failed}")

    # ── Compute metrics ───────────────────────────────────────────────────
    print("\n[INFO] Computing MASK metrics …")
    rf_mask = compute_metrics(dict(preds_mask_by_cat), dict(gts_mask_by_cat), use_box=False)

    print("[INFO] Computing BOX metrics …")
    rf_box  = compute_metrics(dict(preds_box_by_cat),  dict(gts_box_by_cat),  use_box=True)

    print_table(rf_mask, rf_box)

    # ── Save JSON ─────────────────────────────────────────────────────────
    output = {
        "model": "RF-DETR-Seg-Medium", "checkpoint": str(args.checkpoint),
        "resolution": args.resolution, "threshold": args.threshold,
        "dataset": {"images": 374, "annotations": 785,
                    "note": "CarDD test split"},
        "mask_ap": {
            "AP": rf_mask["AP"], "AP50": rf_mask["AP50"], "AP75": rf_mask["AP75"],
            "per_category": rf_mask["per_category"],
        },
        "box_ap": {
            "AP": rf_box["AP"], "AP50": rf_box["AP50"], "AP75": rf_box["AP75"],
            "per_category": rf_box["per_category"],
        },
        "dcnplus_reference": DCN_PLUS,
        "inference_stats": {
            "time_seconds": round(elapsed,1),
            "failed_images": failed, "total_predictions": total_p,
        },
    }
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[INFO] Results saved → {args.output_json}")

    # ── Generate all plots ────────────────────────────────────────────────
    print("\n[INFO] Generating plots …")

    plot_pr_curves(rf_mask, plots_dir, mode="mask")
    plot_pr_curves(rf_box,  plots_dir, mode="box")

    mask_best = plot_f1_vs_threshold(dict(preds_mask_by_cat), dict(gts_mask_by_cat),
                                     plots_dir, mode="mask")
    box_best  = plot_f1_vs_threshold(dict(preds_box_by_cat),  dict(gts_box_by_cat),
                                     plots_dir, mode="box")

    plot_ap_comparison_bar(rf_mask, mode="mask", plots_dir=plots_dir)
    plot_ap_comparison_bar(rf_box,  mode="box",  plots_dir=plots_dir)

    plot_radar(rf_mask, rf_box, plots_dir)

    plot_prf_summary(mask_best, box_best, plots_dir)

    plot_ap_vs_iou(rf_mask, rf_box,
                   dict(preds_mask_by_cat), dict(gts_mask_by_cat),
                   dict(preds_box_by_cat),  dict(gts_box_by_cat),
                   plots_dir)

    plot_confusion(dict(preds_mask_by_cat), dict(gts_mask_by_cat),
                   plots_dir, mode="mask")
    plot_confusion(dict(preds_box_by_cat),  dict(gts_box_by_cat),
                   plots_dir, mode="box")

    # ── NEW: Mask vs Box charts ───────────────────────────────────────────
    plot_rfdetr_mask_vs_box_overall(rf_mask, rf_box, plots_dir)
    plot_rfdetr_mask_vs_box_per_category(rf_mask, rf_box, plots_dir)
    plot_comparison_overall_mask_box(rf_mask, rf_box, plots_dir)
    plot_comparison_per_category_mask_box(rf_mask, rf_box, plots_dir)

    # ── Save full JSON (all chart values) ────────────────────────────────
    # AP vs IoU curve data
    ap_vs_iou_mask, ap_vs_iou_box = [], []
    for iou in IOU_THRESHOLDS:
        aps_m, aps_b = [], []
        for cat in CLASSES:
            av_m, _, _ = ap_and_curve(dict(preds_mask_by_cat).get(cat, []),
                                      dict(gts_mask_by_cat).get(cat, []), iou, False)
            av_b, _, _ = ap_and_curve(dict(preds_box_by_cat).get(cat, []),
                                      dict(gts_box_by_cat).get(cat, []), iou, True)
            aps_m.append(av_m); aps_b.append(av_b)
        ap_vs_iou_mask.append(round(float(np.mean(aps_m)), 2))
        ap_vs_iou_box.append(round(float(np.mean(aps_b)), 2))

    # PR curve data (IoU=0.50, sampled at 20 points for readability)
    def sample_curve(rec, prec, n=20):
        idx = np.linspace(0, len(rec)-1, min(n, len(rec)), dtype=int)
        return {"recall": [round(float(rec[i]),3) for i in idx],
                "precision": [round(float(prec[i]),3) for i in idx]}

    pr_curves_mask, pr_curves_box = {}, {}
    for cat in CLASSES:
        r_m, p_m = rf_mask["_curves"][cat][0.50]
        r_b, p_b = rf_box["_curves"][cat][0.50]
        pr_curves_mask[cat] = sample_curve(r_m, p_m)
        pr_curves_box[cat]  = sample_curve(r_b, p_b)

    # Delta tables
    def delta_table(rf_res, mode):
        dcn = DCN_PLUS[mode]["per_category"]
        return {cat: round(rf_res["per_category"][cat] - dcn[cat], 1) for cat in CLASSES}

    output = {
        "model":      "RF-DETR-Seg-Medium",
        "checkpoint": str(args.checkpoint),
        "resolution": args.resolution,
        "threshold":  args.threshold,
        "dataset":    {"images": 374, "annotations": 785, "note": "CarDD test split"},

        # ── Core AP metrics ──────────────────────────────────────────────
        "mask_ap": {
            "AP":   rf_mask["AP"],
            "AP50": rf_mask["AP50"],
            "AP75": rf_mask["AP75"],
            "per_category": rf_mask["per_category"],
            "per_category_AP50": {c: round(v,1) for c,v in rf_mask["_per_cat_ap50"].items()},
            "per_category_AP75": {c: round(v,1) for c,v in rf_mask["_per_cat_ap75"].items()},
        },
        "box_ap": {
            "AP":   rf_box["AP"],
            "AP50": rf_box["AP50"],
            "AP75": rf_box["AP75"],
            "per_category": rf_box["per_category"],
            "per_category_AP50": {c: round(v,1) for c,v in rf_box["_per_cat_ap50"].items()},
            "per_category_AP75": {c: round(v,1) for c,v in rf_box["_per_cat_ap75"].items()},
        },

        # ── DCN+ reference ───────────────────────────────────────────────
        "dcnplus_reference": DCN_PLUS,

        # ── Delta vs DCN+ ────────────────────────────────────────────────
        "delta_vs_dcnplus": {
            "mask": {
                "AP":   round(rf_mask["AP"]   - DCN_PLUS["mask"]["AP"],   1),
                "AP50": round(rf_mask["AP50"] - DCN_PLUS["mask"]["AP50"], 1),
                "AP75": round(rf_mask["AP75"] - DCN_PLUS["mask"]["AP75"], 1),
                "per_category": delta_table(rf_mask, "mask"),
            },
            "box": {
                "AP":   round(rf_box["AP"]   - DCN_PLUS["box"]["AP"],   1),
                "AP50": round(rf_box["AP50"] - DCN_PLUS["box"]["AP50"], 1),
                "AP75": round(rf_box["AP75"] - DCN_PLUS["box"]["AP75"], 1),
                "per_category": delta_table(rf_box, "box"),
            },
        },

        # ── Mask vs Box delta (RF-DETR internal) ─────────────────────────
        "mask_vs_box_delta": {
            "AP":   round(rf_box["AP"]   - rf_mask["AP"],   1),
            "AP50": round(rf_box["AP50"] - rf_mask["AP50"], 1),
            "AP75": round(rf_box["AP75"] - rf_mask["AP75"], 1),
            "per_category": {
                cat: round(rf_box["per_category"][cat] - rf_mask["per_category"][cat], 1)
                for cat in CLASSES
            },
        },

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

        # ── AP vs IoU threshold curve (chart 9) ───────────────────────────
        "ap_vs_iou_curve": {
            "iou_thresholds": [round(float(v), 2) for v in IOU_THRESHOLDS],
            "rf_detr_mask":   ap_vs_iou_mask,
            "rf_detr_box":    ap_vs_iou_box,
            "dcnplus_mask_ref_points": {"0.50": DCN_PLUS["mask"]["AP50"],
                                        "0.75": DCN_PLUS["mask"]["AP75"]},
            "dcnplus_box_ref_points":  {"0.50": DCN_PLUS["box"]["AP50"],
                                        "0.75": DCN_PLUS["box"]["AP75"]},
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