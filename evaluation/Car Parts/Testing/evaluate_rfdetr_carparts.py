"""
evaluate_rfdetr_carparts.py
============================
Evaluates RF-DETR segmentation on the Car Parts test set.
No external baseline comparison — all charts are self-contained RF-DETR metrics.

19 CAR PART CLASSES
────────────────────
  Diggi_Back_Door, Diggi_Back_Door_Glass, Fender,
  Front_Bumper, Front_Door, Front_Door_Glass,
  Front_Windshield_Glass, Grill, Headlight,
  Hood_Bonnet, Quarter_Panel, Rear_Bumper,
  Rear_Door, Rear_Door_Glass, Roof,
  Running_Board, Side_Mirror, Taillight, tyre

METRICS COMPUTED
────────────────
  Mask AP / AP50 / AP75 / per-category   (instance segmentation)
  Box  AP / AP50 / AP75 / per-category   (bounding box)

CHARTS GENERATED (saved to --plots_dir, default ./plots_parts/)
────────────────────────────────────────────────────────────────
  1.  pr_curve_mask_page1/2.png          PR curves per category (mask) — 10 per page
  2.  pr_curve_box_page1/2.png           PR curves per category (box)
  3.  f1_vs_threshold_mask.png           F1 / P / R vs confidence threshold (mask)
  4.  f1_vs_threshold_box.png            F1 / P / R vs confidence threshold (box)
  5.  ap_per_category_mask.png           Horizontal bar — per-category mask AP (sorted)
  6.  ap_per_category_box.png            Horizontal bar — per-category box AP (sorted)
  7.  mask_vs_box_overall.png            Grouped bar — Mask vs Box for AP/AP50/AP75
  8.  mask_vs_box_per_category.png       Grouped bar — Mask vs Box per category
  9.  iou_ap_curve.png                   AP vs IoU threshold (mask + box)
  10. category_group_ap.png              AP by region group (Front/Rear/Side/Other)
  11. ap50_ap75_scatter.png              Scatter: AP50 vs AP75 per category (mask)
  12. confusion_heatmap_mask.png         TP-rate heatmap (mask @IoU=0.50)
  13. confidence_distribution.png        Histogram of predicted confidence scores

JSON OUTPUT
───────────
  All chart data, AP metrics, PR curves, F1 sweep, AP-vs-IoU
  saved to --output_json (default: rfdetr_carparts_results.json)

USAGE
─────
  python evaluate_rfdetr_carparts.py \\
      --images_dir  /path/to/test/images \\
      --annotations /path/to/_annotations.coco.json \\
      --checkpoint  /path/to/checkpoint_best_total.pth \\
      --resolution  960 \\
      --threshold   0.001 \\
      --output_json rfdetr_carparts_results.json \\
      --plots_dir   ./plots_parts
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

# ── DEFAULTS ──────────────────────────────────────────────────────────────────
IMAGES_DIR  = "/path/to/test/images"
ANNOTATIONS = "/path/to/test/annotations.coco.json"
CHECKPOINT  = "/path/to/checkpoint_best_total.pth"
RESOLUTION  = 960
THRESHOLD   = 0.001
OUTPUT_JSON = "rfdetr_carparts_results.json"
PLOTS_DIR   = "./plots_parts"
# ─────────────────────────────────────────────────────────────────────────────

# All 19 car part classes (order matches annotation file)
CLASSES = [
    "Diggi_Back_Door", "Diggi_Back_Door_Glass", "Fender",
    "Front_Bumper", "Front_Door", "Front_Door_Glass",
    "Front_Windshield_Glass", "Grill", "Headlight",
    "Hood_Bonnet", "Quarter_Panel", "Rear_Bumper",
    "Rear_Door", "Rear_Door_Glass", "Roof",
    "Running_Board", "Side_Mirror", "Taillight",
    "tyre",
]

# Per-category colours (cycle through a palette)
_PALETTE = [
    "#E63946","#F4A261","#2A9D8F","#457B9D","#A8DADC","#6A0572",
    "#1A73E8","#E8710A","#3FB950","#F78166","#58A6FF","#FF7B72",
    "#D29922","#79C0FF","#56D364","#FFA657","#FF6E96","#A5D6FF",
    "#7EE787","#FFD700","#C9D1D9","#8B949E","#E6EDF3","#B1BAC4",
    "#30363D","#21262D","#161B22","#0D1117","#0969DA",
]
CAT_COLORS = {c: _PALETTE[i % len(_PALETTE)] for i, c in enumerate(CLASSES)}

IOU_THRESHOLDS = np.arange(0.50, 1.00, 0.05)   # 10 thresholds


# ── STYLE ─────────────────────────────────────────────────────────────────────
def _style():
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linestyle":    "--",
        "figure.facecolor":  "#0D1117",
        "axes.facecolor":    "#161B22",
        "axes.labelcolor":   "#C9D1D9",
        "xtick.color":       "#C9D1D9",
        "ytick.color":       "#C9D1D9",
        "text.color":        "#C9D1D9",
        "axes.titlecolor":   "#E6EDF3",
        "legend.facecolor":  "#21262D",
        "legend.edgecolor":  "#30363D",
        "grid.color":        "#21262D",
    })

def _savefig(fig, path, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [PLOT] → {path.name}")


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



def mask_to_bbox(mask_bool):
    rows = np.any(mask_bool, axis=1)
    cols = np.any(mask_bool, axis=0)
    if not rows.any():
        return None
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return float(x1), float(y1), float(x2), float(y2)

def box_iou(b1, b2):
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    iw = max(0, ix2-ix1);    ih = max(0, iy2-iy1)
    inter = iw * ih
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def project_prediction_to_original(mask_model, orig_w, orig_h):
    """
    Resize a mask from the model's inference space to the original image
    coordinate space using bilinear interpolation + 0.5 threshold.
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
    """Recover the ORIGINAL filename from a Roboflow-exported filename."""
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


def run_cocoeval(coco_gt, coco_results_list, iou_type, label="RF-DETR"):
    if not coco_results_list:
        print(f"[WARN] No predictions for {iou_type} eval — skipping.")
        return {}

    coco_dt = coco_gt.loadRes(coco_results_list)
    ev = COCOeval(coco_gt, coco_dt, iou_type)
    ev.evaluate()
    ev.accumulate()
    print(f"\n── Pycocotools  [{label}  iouType={iou_type}] ──")
    ev.summarize()

    stats  = ev.stats
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

    for cat_id, cat_info in coco_gt.cats.items():
        cname = cat_info["name"]
        if cname not in CLASSES: continue
        ev2 = COCOeval(coco_gt, coco_dt, iou_type)
        ev2.params.catIds = [cat_id]
        ev2.evaluate()
        ev2.accumulate()
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            ev2.summarize()
            
        if len(ev2.stats) > 0 and ev2.stats[0] != -1:
            result["per_category"][cname]  = round(float(ev2.stats[0]) * 100, 1)
            result["_per_cat_ap50"][cname] = round(float(ev2.stats[1]) * 100, 1)
            result["_per_cat_ap75"][cname] = round(float(ev2.stats[2]) * 100, 1)
            
            recalls = ev2.params.recThrs
            cat_curves = {}
            precisions = ev2.eval["precision"]
            for t_idx, iou_v in enumerate(np.arange(0.50, 1.00, 0.05)):
                key = round(float(iou_v), 2)
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

def compute_f1_sweep(preds_by_cat, gts_by_cat, use_box=False, n_steps=50):
    thresholds = np.linspace(0.0, 1.0, n_steps)
    precs, recs, f1s = [], [], []
    for thr in thresholds:
        tp_t = fp_t = fn_t = 0
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
                best = 0.50 - 1e-9; best_gi = -1
                for gi, g in gt_by_img.get(pred["image_id"], []):
                    if gi in matched: continue
                    iou = (box_iou(pred["bbox"], g["bbox"]) if use_box
                           else maskUtils.iou([pred["mask"]], [g["mask"]], [0])[0][0])
                    if iou > best: best = iou; best_gi = gi
                if best_gi >= 0: tp += 1; matched.add(best_gi)
            tp_t += tp; fp_t += len(preds) - tp; fn_t += len(gts) - tp
        p = tp_t / (tp_t + fp_t + 1e-9)
        r = tp_t / (tp_t + fn_t + 1e-9)
        f = 2*p*r / (p + r + 1e-9)
        precs.append(p); recs.append(r); f1s.append(f)
    return thresholds, np.array(precs), np.array(recs), np.array(f1s)


# ── CHART 1 & 2 : PR curves (2 pages of 10/9) ────────────────────────────────
def plot_pr_curves_paged(results, plots_dir, mode="mask"):
    _style()
    curves   = results["_curves"]
    mode_lbl = "Mask" if mode == "mask" else "Box"
    pages    = [CLASSES[:10], CLASSES[10:]]   # 10 + 9

    for pg, cats in enumerate(pages, 1):
        rows = 2; cols = 5
        fig, axes = plt.subplots(rows, cols, figsize=(20, 9))
        fig.suptitle(
            f"Precision–Recall Curves — Car Parts [{mode_lbl}]  (Page {pg}/2)\n"
            f"Solid = IoU 0.50  |  Dashed = IoU 0.55→0.95  |  "
            f"AP50 = area under solid  |  AP = mean over all IoU thresholds",
            fontsize=10, color="#C9D1D9", y=1.02,
        )
        for i, cat in enumerate(cats):
            ax = axes[i//cols][i%cols]
            color = CAT_COLORS[cat]

            # Faint dashed curves for stricter IoU
            for iou_v in [0.55, 0.65, 0.75, 0.85, 0.95]:
                key = round(iou_v, 2)
                if key in curves[cat]:
                    r2, p2 = curves[cat][key]
                    ax.plot(r2, p2, color=color, lw=0.8, alpha=0.28, ls="--")

            # Solid at IoU=0.50
            rec, prec = curves[cat][0.50]
            ax.fill_between(rec, prec, alpha=0.12, color=color)
            ax.plot(rec, prec, color=color, lw=2.2,
                    label=f"AP50={results['_per_cat_ap50'][cat]:.1f}")

            short = cat.replace("_", "\n")
            ax.set_title(short, fontsize=8, fontweight="bold", color=color)
            ax.set_xlabel("Recall", fontsize=7)
            ax.set_ylabel("Precision", fontsize=7)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7, loc="upper right")
            ax.text(0.03, 0.05,
                    f"AP={results['per_category'][cat]:.1f}",
                    transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=8, color=color, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="#21262D",
                              edgecolor=color, alpha=0.7))

        # Hide unused axes on page 2 (last page may have 9 cats, 1 empty slot)
        for j in range(len(cats), rows*cols):
            axes[j//cols][j%cols].set_visible(False)

        _savefig(fig, plots_dir / f"pr_curve_{mode}_page{pg}.png")


# ── CHART 3 & 4 : F1 vs threshold ────────────────────────────────────────────
def plot_f1_threshold(preds_by_cat, gts_by_cat, plots_dir, mode="mask"):
    _style()
    use_box  = (mode == "box")
    mode_lbl = "Mask" if mode == "mask" else "Box"
    thresholds, precs, recs, f1s = compute_f1_sweep(
        preds_by_cat, gts_by_cat, use_box=use_box)

    best_idx = int(np.argmax(f1s))
    best_thr = float(thresholds[best_idx])
    best_f1  = float(f1s[best_idx])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(thresholds, precs, color="#58A6FF", lw=2.5, label="Precision")
    ax.plot(thresholds, recs,  color="#3FB950", lw=2.5, label="Recall")
    ax.plot(thresholds, f1s,   color="#F78166", lw=2.5, label="F1")
    ax.fill_between(thresholds, f1s, alpha=0.10, color="#F78166")
    ax.axvline(best_thr, color="#F78166", ls="--", alpha=0.6)
    ax.scatter([best_thr], [best_f1], color="#F78166", s=90, zorder=5,
               label=f"Best F1={best_f1:.3f} @ thr={best_thr:.2f}")
    ax.set_xlabel("Confidence Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Precision / Recall / F1 vs Confidence Threshold  [{mode_lbl}]",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    _savefig(fig, plots_dir / f"f1_vs_threshold_{mode}.png")
    return best_thr, float(precs[best_idx]), float(recs[best_idx]), best_f1


# ── CHART 5 & 6 : Per-category horizontal bar (sorted) ───────────────────────
def plot_ap_per_category(results, plots_dir, mode="mask"):
    _style()
    mode_lbl = "Mask" if mode == "mask" else "Box"
    per_cat  = results["per_category"]

    # Sort descending by AP
    cats_sorted = sorted(per_cat.keys(), key=lambda c: per_cat[c], reverse=True)
    vals   = [per_cat[c] for c in cats_sorted]
    colors = [CAT_COLORS[c] for c in cats_sorted]

    fig, ax = plt.subplots(figsize=(10, 13))
    bars = ax.barh(range(len(cats_sorted)), vals, color=colors,
                   alpha=0.85, edgecolor="#30363D", height=0.72)
    ax.set_yticks(range(len(cats_sorted)))
    ax.set_yticklabels([c.replace("_", " ") for c in cats_sorted], fontsize=9)
    ax.set_xlabel(f"{mode_lbl} AP (IoU 0.50:0.95)", fontsize=11)
    ax.set_xlim(0, 108)
    ax.set_title(f"Per-Category {mode_lbl} AP  (sorted)  —  RF-DETR Car Parts",
                 fontsize=12, fontweight="bold")

    # Value labels
    for bar, val in zip(bars, vals):
        ax.text(val + 0.8, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}", va="center", fontsize=8.5,
                color=colors[cats_sorted.index(
                    cats_sorted[list(vals).index(val)])])

    # Mean line
    mean_val = float(np.mean(vals))
    ax.axvline(mean_val, color="#F78166", ls="--", lw=1.5,
               label=f"Mean AP = {mean_val:.1f}")
    ax.legend(fontsize=10)
    ax.invert_yaxis()
    _savefig(fig, plots_dir / f"ap_per_category_{mode}.png")


# ── CHART 7 : Mask vs Box overall ─────────────────────────────────────────────
def plot_mask_vs_box_overall(rf_mask, rf_box, plots_dir):
    _style()
    metrics   = ["AP", "AP50", "AP75"]
    mask_vals = [rf_mask["AP"], rf_mask["AP50"], rf_mask["AP75"]]
    box_vals  = [rf_box["AP"],  rf_box["AP50"],  rf_box["AP75"]]

    x = np.arange(len(metrics)); w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars_m = ax.bar(x - w/2, mask_vals, w, label="Mask AP",
                    color="#1A73E8", alpha=0.88, edgecolor="#30363D")
    bars_b = ax.bar(x + w/2, box_vals,  w, label="Box AP",
                    color="#A371F7", alpha=0.88, edgecolor="#30363D")

    for bar, v in zip(bars_m, mask_vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.6,
                f"{v:.1f}", ha="center", va="bottom",
                fontsize=12, color="#1A73E8", fontweight="bold")
    for bar, v in zip(bars_b, box_vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.6,
                f"{v:.1f}", ha="center", va="bottom",
                fontsize=12, color="#A371F7", fontweight="bold")

    # Δ between mask and box
    for i, (m, b) in enumerate(zip(mask_vals, box_vals)):
        delta = b - m
        sign  = "+" if delta >= 0 else ""
        col   = "#3FB950" if delta >= 0 else "#F85149"
        ax.annotate(f"Δ={sign}{delta:.1f}",
                    xy=(x[i], max(m, b)+2.5),
                    ha="center", fontsize=10, color=col, fontweight="bold")

    # IoU note per metric
    iou_notes = ["IoU 0.50:0.95", "IoU = 0.50", "IoU = 0.75"]
    for xi, note in zip(x, iou_notes):
        ax.text(xi, -7, note, ha="center", va="top",
                fontsize=8, color="#8B949E", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=13)
    ax.set_ylabel("AP Score", fontsize=12)
    ax.set_ylim(0, 108)
    ax.set_title("RF-DETR Car Parts  ·  Mask AP vs Box AP  (Overall)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    _savefig(fig, plots_dir / "mask_vs_box_overall.png")


# ── CHART 8 : Mask vs Box per category (single page — 19 classes) ─────────────
def plot_mask_vs_box_per_category(rf_mask, rf_box, plots_dir):
    _style()
    mask_vals = [rf_mask["per_category"][c] for c in CLASSES]
    box_vals  = [rf_box["per_category"][c]  for c in CLASSES]

    x = np.arange(len(CLASSES)); w = 0.38
    fig, ax = plt.subplots(figsize=(18, 6))
    bars_m = ax.bar(x - w/2, mask_vals, w, label="Mask AP",
                    color="#1A73E8", alpha=0.88, edgecolor="#30363D")
    bars_b = ax.bar(x + w/2, box_vals,  w, label="Box AP",
                    color="#A371F7", alpha=0.88, edgecolor="#30363D")

    for bar, v in zip(bars_m, mask_vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.7,
                f"{v:.1f}", ha="center", va="bottom", fontsize=7.5,
                color="#1A73E8", fontweight="bold")
    for bar, v in zip(bars_b, box_vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.7,
                f"{v:.1f}", ha="center", va="bottom", fontsize=7.5,
                color="#A371F7", fontweight="bold")

    # Δ annotations
    for i, (m, b) in enumerate(zip(mask_vals, box_vals)):
        delta = b - m
        sign  = "+" if delta >= 0 else ""
        col   = "#3FB950" if delta >= 0 else "#F85149"
        ax.text(x[i], max(m, b)+3.2, f"{sign}{delta:.1f}",
                ha="center", fontsize=7, color=col, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", "\n") for c in CLASSES], fontsize=8)
    ax.set_ylabel("AP (IoU 0.50:0.95)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_title("RF-DETR Car Parts  ·  Mask AP vs Box AP  per Category",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.text(0.99, 0.98, "Δ = Box − Mask",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="#8B949E")
    _savefig(fig, plots_dir / "mask_vs_box_per_category.png")


# ── CHART 11 : AP50 vs AP75 grouped bar per category ─────────────────────────
def plot_ap50_ap75_bar(rf_mask, plots_dir):
    """
    For each category show two bars side by side:
      Blue  = AP50  (how well model detects parts at relaxed IoU=0.50)
      Green = AP75  (how well model detects parts at strict  IoU=0.75)
    A large gap between the two bars means the model finds the part
    but the mask shape is imprecise. A small gap means tight masks.
    """
    _style()
    ap50_vals = [rf_mask["_per_cat_ap50"][c] for c in CLASSES]
    ap75_vals = [rf_mask["_per_cat_ap75"][c] for c in CLASSES]

    x = np.arange(len(CLASSES)); w = 0.38
    fig, ax = plt.subplots(figsize=(18, 6))

    bars50 = ax.bar(x - w/2, ap50_vals, w, label="AP50  (IoU ≥ 0.50)",
                    color="#1A73E8", alpha=0.88, edgecolor="#30363D")
    bars75 = ax.bar(x + w/2, ap75_vals, w, label="AP75  (IoU ≥ 0.75)",
                    color="#3FB950", alpha=0.88, edgecolor="#30363D")

    for bar, v in zip(bars50, ap50_vals):
        if v > 1:
            ax.text(bar.get_x()+bar.get_width()/2, v+0.8,
                    f"{v:.1f}", ha="center", va="bottom",
                    fontsize=8, color="#1A73E8", fontweight="bold")
    for bar, v in zip(bars75, ap75_vals):
        if v > 1:
            ax.text(bar.get_x()+bar.get_width()/2, v+0.8,
                    f"{v:.1f}", ha="center", va="bottom",
                    fontsize=8, color="#3FB950", fontweight="bold")

    # Gap annotation: AP50 − AP75 (mask quality drop)
    for i, (v50, v75) in enumerate(zip(ap50_vals, ap75_vals)):
        gap = v50 - v75
        col = "#F85149" if gap > 20 else ("#F78166" if gap > 10 else "#8B949E")
        ax.text(x[i], max(v50, v75) + 3.5, f"↓{gap:.0f}",
                ha="center", fontsize=7.5, color=col, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ") for c in CLASSES],
                       rotation=35, ha="right", fontsize=8.5)
    ax.set_ylabel("AP Score", fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_title(
        "AP50 vs AP75 per Category  —  RF-DETR Car Parts  [Mask]\n"
        "↓ number = drop from AP50→AP75  "
        "(red = large drop → imprecise mask shape,  grey = small drop → tight masks)",
        fontsize=11, fontweight="bold")
    ax.legend(fontsize=11)
    _savefig(fig, plots_dir / "ap50_ap75_bar.png")


# ── CHART 12 : Confusion heatmap ──────────────────────────────────────────────
def plot_confusion_heatmap(preds_by_cat, gts_by_cat, plots_dir):
    _style()
    n = len(CLASSES)
    cat2idx = {c: i for i, c in enumerate(CLASSES)}
    mat = np.zeros((n, n), dtype=int)

    for cat in CLASSES:
        preds = preds_by_cat.get(cat, [])
        gts   = gts_by_cat.get(cat, [])
        if not gts:
            continue
        gt_by_img = defaultdict(list)
        for i, g in enumerate(gts):
            gt_by_img[g["image_id"]].append((i, g))
        preds_s = sorted(preds, key=lambda x: -x["score"])
        matched = set()
        for pred in preds_s:
            best = 0.50 - 1e-9; best_gi = -1
            for gi, g in gt_by_img.get(pred["image_id"], []):
                if gi in matched: continue
                iou = maskUtils.iou([pred["mask"]], [g["mask"]], [0])[0][0]
                if iou > best: best = iou; best_gi = gi
            if best_gi >= 0:
                mat[cat2idx[cat], cat2idx[cat]] += 1
                matched.add(best_gi)

    gt_counts = np.array([len(gts_by_cat.get(c, [])) for c in CLASSES], dtype=float)
    gt_counts[gt_counts == 0] = 1
    mat_norm = mat / gt_counts[:, None]

    short = [c.replace("_", " ").replace("Left ", "L.").replace("Right ", "R.")
             for c in CLASSES]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(mat_norm, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="TP rate (fraction of GT matched)")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(short, fontsize=7)
    ax.set_xlabel("Predicted Category", fontsize=10)
    ax.set_ylabel("Ground-Truth Category", fontsize=10)
    ax.set_title("Detection Heatmap (TP rate per GT class)  [Mask @IoU=0.50]\nRF-DETR Car Parts",
                 fontsize=12, fontweight="bold")
    for i in range(n):
        for j in range(n):
            v = mat_norm[i, j]
            if v > 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="black" if v > 0.5 else "white")
    _savefig(fig, plots_dir / "confusion_heatmap_mask.png", tight=False)


# ── INFERENCE ─────────────────────────────────────────────────────────────────
def run_inference(model, img_path, threshold, classes):
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
        name  = classes[cid] if 0 <= cid < len(classes) else f"cls_{cid}"
        score = float(det.confidence[idx])

        # ── MASK: project from model output space → original image coords ──
        if det.mask is not None:
            raw_mask = det.mask[idx]
            rle, mask_orig = project_prediction_to_original(raw_mask, W, H)
        else:
            scale_x = W / img_pil.width
            scale_y = H / img_pil.height
            x1, y1, x2, y2 = (float(v) for v in det.xyxy[idx])
            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
            mask_orig = np.zeros((H, W), dtype=np.uint8)
            mask_orig[max(0, y1):min(H, y2),
                      max(0, x1):min(W, x2)] = 1
            rle = maskUtils.encode(np.asfortranarray(mask_orig))
            rle["counts"] = rle["counts"].decode("utf-8")
            rle["size"]   = [H, W]

        # ── BOUNDING BOX: tight bbox derived from projected mask ───────────
        bbox_from_mask = mask_to_bbox(mask_orig)
        if bbox_from_mask is not None:
            bbox = bbox_from_mask
        else:
            scale_x = W / img_pil.width
            scale_y = H / img_pil.height
            x1, y1, x2, y2 = (float(v) for v in det.xyxy[idx])
            bbox = (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
            
        # COCO requires [x, y, width, height]
        coco_bbox = [bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]]

        results.append({"class_name": name, "score": score,
                        "mask": rle, "bbox": bbox, "coco_bbox": coco_bbox,
                        "category_id": cid}) # Need category_id for coco
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
    
    # We need category_id mapping from string to id to build coco results
    name_to_id = {c["name"]: c["id"] for c in coco["categories"]}
    coco_cats = {c["id"]: c["name"] for c in coco["categories"]}
    
    n_images  = len(images)
    n_anns    = len(anns)
    print(f"[INFO] {n_images} images, {n_anns} annotations, "
          f"{len(coco_cats)} categories")

    per_cat_gt = defaultdict(int)
    for a in anns:
        per_cat_gt[coco_cats[a["category_id"]]] += 1
    print(f"[INFO] GT per category: {dict(per_cat_gt)}")

    # ── Decode GT masks + boxes ───────────────────────────────────────────
    gts_mask_by_cat = defaultdict(list)
    gts_box_by_cat  = defaultdict(list)
    decode_errors   = 0
    print("[INFO] Decoding GT masks …")
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
                decode_errors += 1; continue
        except Exception:
            decode_errors += 1; continue

        rle = maskUtils.encode(np.asfortranarray(gt_mask.astype(np.uint8)))
        rle["counts"] = rle["counts"].decode("utf-8")
        gts_mask_by_cat[cname].append({"image_id": ann["image_id"], "mask": rle})

        x, y, bw, bh = ann["bbox"]
        gts_box_by_cat[cname].append(
            {"image_id": ann["image_id"], "bbox": (x, y, x+bw, y+bh)})

    if decode_errors:
        print(f"[WARN] {decode_errors} GT annotations could not be decoded")

    # ── Load model ────────────────────────────────────────────────────────
    if not os.path.isfile(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}"); sys.exit(1)

    print(f"\n[INFO] Loading model …")
    
    from rfdetr import RFDETRSegNano
    model = RFDETRSegNano(pretrain_weights=args.checkpoint,
                           resolution=args.resolution)
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
    
    coco_results_segm = []
    coco_results_bbox = []
    
    failed = 0

    print(f"\n[INFO] Running inference on {n_images} images …")
    t0 = time.time()
    for img_id, img_info in tqdm(images.items(), desc="Inference", ncols=70):
        # Handle original filenames like CarDD
        orig_fname = decode_roboflow_filename(img_info["file_name"])
        img_path = images_dir / img_info["file_name"]
        
        if not img_path.exists():
            img_path = images_dir / orig_fname
        if not img_path.exists():
            orig = img_info.get("extra", {}).get("name", "")
            if orig: img_path = images_dir / orig
        if not img_path.exists():
            hits = list(images_dir.rglob(img_info["file_name"]))
            if not hits: hits = list(images_dir.rglob(orig_fname))
            img_path = hits[0] if hits else None
            
        if img_path is None or not img_path.exists():
            failed += 1; continue

        try:
            preds = run_inference(model, img_path, args.threshold, CLASSES)
        except Exception as e:
            print(f"\n[WARN] {img_info['file_name']}: {e}")
            failed += 1; continue

        for p in preds:
            cat = p["class_name"]
            if cat not in CLASSES:
                continue
            preds_mask_by_cat[cat].append(
                {"image_id": img_id, "score": p["score"], "mask": p["mask"]})
            preds_box_by_cat[cat].append(
                {"image_id": img_id, "score": p["score"], "bbox": p["bbox"]})
                
            cat_id = name_to_id.get(cat)
            if cat_id is not None:
                coco_results_segm.append({
                    "image_id": img_id,
                    "category_id": cat_id,
                    "segmentation": p["mask"],
                    "score": p["score"],
                })
                coco_results_bbox.append({
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": p["coco_bbox"],
                    "score": p["score"],
                })

    elapsed = time.time() - t0
    total_p = sum(len(v) for v in preds_mask_by_cat.values())
    print(f"[INFO] Done in {elapsed:.1f}s  |  Preds: {total_p}  |  Failed: {failed}")

    # ── Pycocotools evaluation ───────────────────────────────────
    print("\n[INFO] Running pycocotools evaluation …")
    coco_gt_orig = COCO(args.annotations)
    
    rf_mask = run_cocoeval(coco_gt_orig, coco_results_segm, "segm", label="RF-DETR")
    rf_box  = run_cocoeval(coco_gt_orig, coco_results_bbox, "bbox", label="RF-DETR")

    print(f"\n  Mask AP={rf_mask.get('AP')}  AP50={rf_mask.get('AP50')}  AP75={rf_mask.get('AP75')}")
    print(f"  Box  AP={rf_box.get('AP')}   AP50={rf_box.get('AP50')}   AP75={rf_box.get('AP75')}")

    # ── Generate charts ───────────────────────────────────────────────────
    print("\n[INFO] Generating charts …")

    plot_pr_curves_paged(rf_mask, plots_dir, mode="mask")
    plot_pr_curves_paged(rf_box,  plots_dir, mode="box")

    mask_best = plot_f1_threshold(dict(preds_mask_by_cat), dict(gts_mask_by_cat),
                                  plots_dir, mode="mask")
    box_best  = plot_f1_threshold(dict(preds_box_by_cat),  dict(gts_box_by_cat),
                                  plots_dir, mode="box")

    plot_ap_per_category(rf_mask, plots_dir, mode="mask")
    plot_ap_per_category(rf_box,  plots_dir, mode="box")

    plot_mask_vs_box_overall(rf_mask, rf_box, plots_dir)
    plot_mask_vs_box_per_category(rf_mask, rf_box, plots_dir)
    plot_ap50_ap75_bar(rf_mask, plots_dir)
    plot_confusion_heatmap(dict(preds_mask_by_cat), dict(gts_mask_by_cat), plots_dir)

    # ── Build full JSON ───────────────────────────────────────────────────
    print("\n[INFO] Building JSON …")

    # PR curve samples (IoU=0.50)
    def sample_curve(rec, prec, n=20):
        idx = np.linspace(0, len(rec)-1, min(n, len(rec)), dtype=int)
        return {"recall":    [round(float(rec[i]),  3) for i in idx],
                "precision": [round(float(prec[i]), 3) for i in idx]}

    pr_mask, pr_box = {}, {}
    for cat in CLASSES:
        r_m, p_m = rf_mask["_curves"][cat][0.50]
        r_b, p_b = rf_box["_curves"][cat][0.50]
        pr_mask[cat] = sample_curve(r_m, p_m)
        pr_box[cat]  = sample_curve(r_b, p_b)

    output = {
        "model":      "RF-DETR-Seg-Nano",
        "checkpoint": str(args.checkpoint),
        "resolution": args.resolution,
        "threshold":  args.threshold,
        "dataset": {
            "images":      n_images,
            "annotations": n_anns,
            "categories":  len(CLASSES),
            "class_list":  CLASSES,
        },

        # ── Core AP ──────────────────────────────────────────────────────
        "mask_ap": {
            "AP":   rf_mask.get("AP"),
            "AP50": rf_mask.get("AP50"),
            "AP75": rf_mask.get("AP75"),
            "APs":  rf_mask.get("APs"),
            "APm":  rf_mask.get("APm"),
            "APl":  rf_mask.get("APl"),
            "per_category":      rf_mask.get("per_category"),
            "per_category_AP50": {c: round(v,1) for c,v in
                                  rf_mask["_per_cat_ap50"].items()},
            "per_category_AP75": {c: round(v,1) for c,v in
                                  rf_mask["_per_cat_ap75"].items()},
        },
        "box_ap": {
            "AP":   rf_box.get("AP"),
            "AP50": rf_box.get("AP50"),
            "AP75": rf_box.get("AP75"),
            "APs":  rf_box.get("APs"),
            "APm":  rf_box.get("APm"),
            "APl":  rf_box.get("APl"),
            "per_category":      rf_box["per_category"],
            "per_category_AP50": {c: round(v,1) for c,v in
                                  rf_box["_per_cat_ap50"].items()},
            "per_category_AP75": {c: round(v,1) for c,v in
                                  rf_box["_per_cat_ap75"].items()},
        },

        # ── Mask vs Box delta ─────────────────────────────────────────────
        "mask_vs_box_delta": {
            "AP":   round(rf_box["AP"]   - rf_mask["AP"],   1),
            "AP50": round(rf_box["AP50"] - rf_mask["AP50"], 1),
            "AP75": round(rf_box["AP75"] - rf_mask["AP75"], 1),
            "per_category": {
                c: round(rf_box["per_category"][c] - rf_mask["per_category"][c], 1)
                for c in CLASSES
            },
        },

        # ── Best-threshold F1 ─────────────────────────────────────────────
        "best_threshold_metrics": {
            "mask": {"threshold": round(mask_best[0],3), "precision": round(mask_best[1],4),
                     "recall": round(mask_best[2],4), "f1": round(mask_best[3],4)},
            "box":  {"threshold": round(box_best[0],3),  "precision": round(box_best[1],4),
                     "recall": round(box_best[2],4),  "f1": round(box_best[3],4)},
        },

        # ── PR curves @IoU=0.50 ───────────────────────────────────────────
        "pr_curves_at_iou50": {
            "mask": pr_mask,
            "box":  pr_box,
        },

        # ── AP50 vs AP75 per category ─────────────────────────────────────
        "ap50_vs_ap75_mask": {
            c: {"AP50": round(rf_mask["_per_cat_ap50"][c], 1),
                "AP75": round(rf_mask["_per_cat_ap75"][c], 1),
                "drop": round(rf_mask["_per_cat_ap50"][c] - rf_mask["_per_cat_ap75"][c], 1)}
            for c in CLASSES
        },

        # ── GT distribution ───────────────────────────────────────────────
        "gt_per_category": dict(per_cat_gt),

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
    print(f"[INFO] Results saved → {args.output_json}")

    print(f"\n[INFO] ✓ All charts saved to: {plots_dir}/")
    print("[INFO] Files generated:")
    for fp in sorted(plots_dir.glob("*.png")):
        print(f"        {fp.name}")


if __name__ == "__main__":
    main()