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
import copy
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# A Windows console defaults to cp1252, which cannot encode the arrows, box
# characters and check marks used throughout the messages below: the very first
# print raises UnicodeEncodeError and the script dies before doing any work.
# Force UTF-8 on the streams that support it.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ── DEFAULTS ──────────────────────────────────────────────────────────────────
IMAGES_DIR  = "/path/to/test/images"
ANNOTATIONS = "/path/to/test/annotations.coco.json"
CHECKPOINT  = "/path/to/checkpoint_best_total.pth"
RESOLUTION  = 960
# COLLECTION threshold: what is handed to COCOeval. Must stay very low.
# COCOeval sweeps the score axis itself when it builds the precision-recall
# curve, so every detection has to reach it; raising this truncates the recall
# axis and UNDERSTATES AP. This is standard COCO practice, not a bug.
THRESHOLD   = 0.001

# REPORTING threshold: where single-operating-point precision / recall / F1 and
# the confusion matrix are evaluated. These are supplementary to AP and need a
# threshold a person would actually deploy at. Quoting them at THRESHOLD would
# report precision over ~47 detections per ground-truth instance -- on this
# project's own predictions that is precision 0.0201 and F1 0.0394, versus
# 0.7835 and 0.6909 at 0.50. 0.50 is a conventional value fixed in advance, so
# it is not tuned on the test set.
REPORT_THRESHOLD = 0.50
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

# This dataset is not CarDD, so COCO's DEFAULT area ranges (small < 32^2,
# medium 32^2-96^2, large > 96^2) are the right ones and are left untouched.
# Stated explicitly because the sibling damage script deliberately overrides
# them with CarDD's, and a reader moving between the two files will want to
# know the difference is intentional.

# Below this many ground-truth instances a per-class AP describes a handful of
# annotations rather than the model. Such classes are still reported, always
# with their support attached, and are excluded from any mean.
MIN_SUPPORT_FOR_AP = 10

# COCO's default size bins, spelled out so the support table and COCOeval
# cannot drift apart.
COCO_AREA_RNG  = [[0, 1e5 ** 2], [0, 32 ** 2], [32 ** 2, 96 ** 2],
                  [96 ** 2, 1e5 ** 2]]
AREA_BIN_NAMES = ["small", "medium", "large"]


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
    # +1 because np.where returns the LAST occupied index and a COCO box is
    # half-open: a mask spanning columns 10..20 is 11 px wide, not 10. Without
    # this every predicted box is 1 px short in width and height, which biases
    # box IoU downward for small parts. Matches evaluate_rfdetr_cardd_full.py.
    return float(x1), float(y1), float(x2 + 1), float(y2 + 1)

def box_iou(b1, b2):
    ix1 = max(b1[0], b2[0]); iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]); iy2 = min(b1[3], b2[3])
    iw = max(0, ix2-ix1);    ih = max(0, iy2-iy1)
    inter = iw * ih
    a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
    a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


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


def gt_support_table(coco_gt, classes=None, area_rng=None):
    """
    Ground-truth instance counts per category and per COCO area bin, measured
    from the DECODED MASK area -- the quantity COCOeval bins by.

    With 19 part classes on a modest test split, several classes have only a
    handful of instances. Their per-class AP is then a property of those few
    annotations rather than of the model, and a reader cannot tell the two
    apart from the AP alone. Publishing the table without the supports is how a
    per-class result becomes unreproducible.
    """
    classes  = list(CLASSES) if classes is None else classes
    area_rng = COCO_AREA_RNG if area_rng is None else area_rng
    counts = {c: {"all": 0, "small": 0, "medium": 0, "large": 0} for c in classes}
    undecodable = 0

    for ann in coco_gt.anns.values():
        cname = coco_gt.cats[ann["category_id"]]["name"]
        if cname not in counts:
            continue
        info = coco_gt.imgs[ann["image_id"]]
        seg  = ann.get("segmentation")
        try:
            if isinstance(seg, list):
                rle = maskUtils.merge(
                    maskUtils.frPyObjects(seg, info["height"], info["width"]))
            elif isinstance(seg, dict) and isinstance(seg.get("counts"), list):
                rle = maskUtils.frPyObjects(seg, info["height"], info["width"])
            else:
                rle = seg
            area = float(maskUtils.area(rle))
        except Exception:
            undecodable += 1
            continue
        counts[cname]["all"] += 1
        # First matching bin. COCOeval treats both ends as inclusive, so an
        # instance of area exactly 32^2 is scored in BOTH small and medium
        # there; here it is counted once, in the smaller bin.
        for name, (lo, hi) in zip(AREA_BIN_NAMES, area_rng[1:]):
            if lo <= area <= hi:
                counts[cname][name] += 1
                break

    if undecodable:
        print(f"[WARN] gt_support_table: {undecodable} annotations could not be "
              f"decoded and are absent from the support counts")
    return counts


def print_gt_support(counts):
    print("\n" + "=" * 74)
    print("  GROUND-TRUTH SUPPORT  (instances per category x COCO area bin)")
    print("  small < 32^2   medium 32^2-96^2   large > 96^2   (mask area)")
    print("=" * 74)
    print(f"  {'Category':<26} {'all':>6} {'small':>7} {'medium':>7} {'large':>7}")
    print("  " + "-" * 58)
    thin = []
    for cat in CLASSES:
        c = counts.get(cat, {})
        n_all = c.get("all", 0)
        mark = "  <-- thin" if 0 <= n_all < MIN_SUPPORT_FOR_AP else ""
        print(f"  {cat:<26} {n_all:>6} {c.get('small', 0):>7} "
              f"{c.get('medium', 0):>7} {c.get('large', 0):>7}{mark}")
        if n_all < MIN_SUPPORT_FOR_AP:
            thin.append((cat, n_all))
    if thin:
        print(f"\n  [CAUTION] {len(thin)} class(es) below {MIN_SUPPORT_FOR_AP} "
              f"instances. Their AP describes a few annotations, not the model;")
        print("            quote them with the count attached and keep them out "
              "of any mean.")
    print("=" * 74)


def run_cocoeval(coco_gt, coco_results_list, iou_type, label="RF-DETR"):
    if not coco_results_list:
        print(f"[WARN] No predictions for {iou_type} eval — skipping.")
        return {}

    # loadRes MUTATES the caller's list, writing "area", "bbox", "id" and
    # "iscrowd" into every dict. Copy first so the results this script holds
    # (and anything later dumped from them) stay exactly what was predicted.
    # A "bbox" key written back into a segm result is not cosmetic: on a
    # re-evaluation it makes loadRes bin segm detections by BOX area, because
    # it tests `if 'bbox' in anns[0]` before `elif 'segmentation' in anns[0]`.
    coco_dt = coco_gt.loadRes(copy.deepcopy(coco_results_list))
    ev = COCOeval(coco_gt, coco_dt, iou_type)
    ev.evaluate()
    ev.accumulate()
    print(f"\n── Pycocotools  [{label}  iouType={iou_type}] ──")
    ev.summarize()

    stats  = ev.stats

    # pycocotools returns -1 where a cell has no ground truth. That is
    # "undefined", not "the model scored zero"; coercing it to 0.0 publishes a
    # zero the model never earned and drags every mean over it downwards.
    def _stat(i):
        v = float(stats[i])
        return None if v == -1 else round(v * 100, 1)

    result = {
        "AP":           _stat(0),
        "AP50":         _stat(1),
        "AP75":         _stat(2),
        "APs":          _stat(3),
        "APm":          _stat(4),
        "APl":          _stat(5),
        "per_category": {},
        "per_category_valid": {},
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
            result["per_category_valid"][cname] = True
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
            # No ground truth for this class in this split. The 0.0 exists only
            # so the bar charts have something to draw; per_category_valid is
            # what every aggregate and the output JSON must consult.
            result["per_category"][cname]  = 0.0
            result["per_category_valid"][cname] = False
            result["_per_cat_ap50"][cname] = 0.0
            result["_per_cat_ap75"][cname] = 0.0
            result["_curves"][cname] = {round(v, 2): (np.array([0.0]), np.array([0.0])) for v in np.arange(0.50, 1.00, 0.05)}

    return result

def prf_at_threshold(preds_by_cat, gts_by_cat, thr, use_box=False,
                     iou_thresh=0.50):
    """
    Micro-averaged precision / recall / F1 across all classes at one confidence
    threshold and one IoU. Matching is within class, greedy by descending
    score, each ground-truth instance claimable once.

    Returns dict with precision, recall, f1, tp, fp, fn.
    """
    tp_t = fp_t = fn_t = 0
    for cat in CLASSES:
        preds = [p for p in preds_by_cat.get(cat, []) if p["score"] >= thr]
        gts   = gts_by_cat.get(cat, [])

        if not gts:
            # Predictions of a class with no ground truth are false positives.
            # Skipping the class outright (as the previous version did) removed
            # them from the denominator and inflated precision -- and with 19
            # part classes, several are absent from any given test split.
            fp_t += len(preds)
            continue

        gt_by_img = defaultdict(list)
        for i, g in enumerate(gts):
            gt_by_img[g["image_id"]].append((i, g))

        matched = set()
        tp = 0
        for pred in sorted(preds, key=lambda x: -x["score"]):
            best = iou_thresh - 1e-9
            best_gi = -1
            for gi, g in gt_by_img.get(pred["image_id"], []):
                if gi in matched:
                    continue
                iou = (box_iou(pred["bbox"], g["bbox"]) if use_box
                       else maskUtils.iou([pred["mask"]], [g["mask"]], [0])[0][0])
                if iou > best:
                    best = iou
                    best_gi = gi
            if best_gi >= 0:
                tp += 1
                matched.add(best_gi)

        tp_t += tp
        fp_t += len(preds) - tp
        fn_t += len(gts) - tp

    p = tp_t / (tp_t + fp_t) if (tp_t + fp_t) else 0.0
    r = tp_t / (tp_t + fn_t) if (tp_t + fn_t) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"threshold": float(thr), "precision": p, "recall": r, "f1": f,
            "tp": int(tp_t), "fp": int(fp_t), "fn": int(fn_t)}


def compute_f1_sweep(preds_by_cat, gts_by_cat, use_box=False, n_steps=50):
    thresholds = np.linspace(0.0, 1.0, n_steps)
    precs, recs, f1s = [], [], []
    for thr in thresholds:
        m = prf_at_threshold(preds_by_cat, gts_by_cat, thr, use_box=use_box)
        precs.append(m["precision"]); recs.append(m["recall"]); f1s.append(m["f1"])
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
def plot_f1_threshold(preds_by_cat, gts_by_cat, plots_dir, mode="mask",
                      operating_thr=None):
    """
    Returns (oracle, operating), each a dict from prf_at_threshold().

    ORACLE is the threshold that maximises F1 ON THE TEST SET ITSELF -- an
    upper bound on what a perfectly tuned threshold could reach, not a score
    the model achieves. Selecting it on the same data it is scored against is
    tuning on the evaluation set, so it is never the number to quote. The
    honest figure is OPERATING, measured at REPORT_THRESHOLD: a conventional
    value fixed before the test set was seen. A validation split would give a
    legitimately tuned threshold; this script has none, which is exactly why
    the two are kept apart rather than merged into one "best" number.

    Note REPORT_THRESHOLD, NOT --threshold. The latter is the COLLECTION
    threshold handed to COCOeval and is deliberately ~0.001, because COCOeval
    sweeps the score axis itself and needs every detection. Evaluating a single
    operating point there measures precision over tens of detections per
    ground-truth instance and reports a near-zero F1 for a healthy model.
    """
    _style()
    use_box  = (mode == "box")
    mode_lbl = "Mask" if mode == "mask" else "Box"
    thresholds, precs, recs, f1s = compute_f1_sweep(
        preds_by_cat, gts_by_cat, use_box=use_box)

    best_idx = int(np.argmax(f1s))
    best_thr = float(thresholds[best_idx])
    best_f1  = float(f1s[best_idx])

    oracle    = prf_at_threshold(preds_by_cat, gts_by_cat, best_thr, use_box=use_box)
    operating = (prf_at_threshold(preds_by_cat, gts_by_cat, operating_thr,
                                  use_box=use_box)
                 if operating_thr is not None else None)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(thresholds, precs, color="#58A6FF", lw=2.5, label="Precision")
    ax.plot(thresholds, recs,  color="#3FB950", lw=2.5, label="Recall")
    ax.plot(thresholds, f1s,   color="#F78166", lw=2.5, label="F1")
    ax.fill_between(thresholds, f1s, alpha=0.10, color="#F78166")
    ax.axvline(best_thr, color="#F78166", ls="--", alpha=0.6)
    ax.scatter([best_thr], [best_f1], color="#F78166", s=90, zorder=5,
               label=f"Oracle (selected ON TEST): F1={best_f1:.3f} @ thr={best_thr:.2f}")
    if operating is not None:
        ax.axvline(operating_thr, color="#C9D1D9", ls=":", alpha=0.9,
                   label=f"Operating thr={operating_thr:g}: F1={operating['f1']:.3f}")
    ax.set_xlabel("Confidence Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Precision / Recall / F1 vs Confidence Threshold  [{mode_lbl}]\n"
                 f"the oracle marker is an upper bound, not an achieved score",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    _savefig(fig, plots_dir / f"f1_vs_threshold_{mode}.png")
    return oracle, operating



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
def compute_confusion_data(preds_by_cat, gts_by_cat, iou_thresh=0.50,
                           use_box=False, score_thresh=0.0):
    """
    Cross-class confusion at a fixed IoU.

    Returns (mat, fn_col, bg_fp):
        mat[i, j]  ground truth of class i claimed by a prediction of class j
        fn_col[i]  ground truth of class i claimed by nothing
        bg_fp[j]   predictions of class j that claimed no ground truth

    Matching is greedy over ALL predictions of ALL classes together, sorted by
    descending score, each ground-truth instance claimable once -- so a
    prediction can claim ground truth of a DIFFERENT class, which is the only
    way an off-diagonal cell becomes non-zero.

    The previous implementation looped over one class at a time and matched
    that class's predictions against that class's ground truth only, writing
    every count to mat[cat, cat]. Every off-diagonal cell was structurally
    zero: it was a per-class recall column drawn as a 19x19 square and titled
    "confusion", and it could not have revealed a single confusion whatever the
    model predicted. That matters most here -- Front_Door vs Rear_Door,
    Front_Bumper vs Rear_Bumper, Quarter_Panel vs Fender are exactly the
    confusions a car-parts model makes, and this figure was structurally
    incapable of showing any of them. Do not reintroduce a per-class loop.
    """
    n = len(CLASSES)
    cat2idx = {c: i for i, c in enumerate(CLASSES)}
    mat      = np.zeros((n, n), dtype=int)
    bg_fp    = np.zeros(n, dtype=int)
    gt_count = np.zeros(n, dtype=int)

    gt_by_img = defaultdict(list)
    running = 0
    for cat in CLASSES:
        gt_count[cat2idx[cat]] = len(gts_by_cat.get(cat, []))
        for g in gts_by_cat.get(cat, []):
            gt_by_img[g["image_id"]].append((running, cat2idx[cat], g))
            running += 1

    all_preds = []
    for cat in CLASSES:
        for p in preds_by_cat.get(cat, []):
            if p["score"] >= score_thresh:
                all_preds.append((cat2idx[cat], p))
    all_preds.sort(key=lambda t: -t[1]["score"])

    matched = set()
    for pred_idx, pred in all_preds:
        best = iou_thresh - 1e-9
        best_gi = best_gt_idx = -1
        for gi, gt_idx, g in gt_by_img.get(pred["image_id"], []):
            if gi in matched:
                continue
            iou = (box_iou(pred["bbox"], g["bbox"]) if use_box
                   else maskUtils.iou([pred["mask"]], [g["mask"]], [0])[0][0])
            if iou > best:
                best = iou
                best_gi = gi
                best_gt_idx = gt_idx
        if best_gi >= 0:
            matched.add(best_gi)
            mat[best_gt_idx, pred_idx] += 1
        else:
            bg_fp[pred_idx] += 1

    fn_col = gt_count - mat.sum(axis=1)
    assert (fn_col >= 0).all(), "confusion matrix over-counted ground truth"
    return mat, fn_col, bg_fp


def plot_confusion_heatmap(preds_by_cat, gts_by_cat, plots_dir, score_thresh=0.0):
    _style()
    n = len(CLASSES)
    mat, fn_col, bg_fp = compute_confusion_data(
        preds_by_cat, gts_by_cat, score_thresh=score_thresh)

    gt_counts = np.array([len(gts_by_cat.get(c, [])) for c in CLASSES], dtype=float)
    denom = gt_counts.copy()
    denom[denom == 0] = 1.0

    # A "(missed)" column makes every row sum to exactly 1, so the diagonal is
    # per-class recall and a light row is immediately readable as either
    # mislabelled (mass off-diagonal) or undetected (mass in the last column).
    full      = np.concatenate([mat, fn_col[:, None]], axis=1).astype(float)
    full_norm = full / denom[:, None]

    short = [c.replace("_", " ") for c in CLASSES]

    fig, ax = plt.subplots(figsize=(15, 12))
    im = ax.imshow(full_norm, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Fraction of that class's ground truth")
    ax.set_xticks(range(n + 1)); ax.set_yticks(range(n))
    ax.set_xticklabels(short + ["(missed)"], rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels([f"{c}  (n={int(g)})" for c, g in zip(short, gt_counts)],
                       fontsize=7)
    ax.axvline(n - 0.5, color="#30363D", lw=2)
    ax.set_xlabel("Predicted Category   ·   final column = claimed by no prediction",
                  fontsize=10)
    ax.set_ylabel("Ground-Truth Category", fontsize=10)
    ax.set_title(f"Confusion Matrix  [Mask @ IoU=0.50, score ≥ {score_thresh:g}]\n"
                 f"RF-DETR Car Parts — class-agnostic greedy matching, rows sum to 1",
                 fontsize=12, fontweight="bold")
    for i in range(n):
        for j in range(n + 1):
            v = full_norm[i, j]
            if v > 0.005:
                # YlOrRd is LIGHT at low values and dark red at high ones, so
                # dark text belongs on the pale cells. With 19 classes the
                # faint off-diagonal cells are the entire point of the figure,
                # so they have to stay legible.
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6, color="white" if v > 0.6 else "black")

    # Placed in FIGURE coordinates, below the axes, so it cannot collide with
    # the x-axis label the way an axes-relative offset did.
    top_bg = sorted(zip(CLASSES, bg_fp), key=lambda t: -t[1])[:6]
    fig.text(0.5, 0.005,
             "Background FP (prediction claimed no GT), top 6:   "
             + "   ".join(f"{c}={int(v)}" for c, v in top_bg if v > 0),
             ha="center", va="bottom", fontsize=8, color="#8B949E")

    _savefig(fig, plots_dir / "confusion_heatmap_mask.png", tight=False)

    return {
        "classes":         CLASSES,
        "gt_per_class":    {c: int(g) for c, g in zip(CLASSES, gt_counts)},
        "matrix":          mat.tolist(),
        "missed":          {c: int(v) for c, v in zip(CLASSES, fn_col)},
        "background_fp":   {c: int(v) for c, v in zip(CLASSES, bg_fp)},
        "iou_threshold":   0.50,
        "score_threshold": float(score_thresh),
        "note": "rows = ground-truth class, columns = predicted class; "
                "class-agnostic greedy matching by descending score",
    }


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

        # MASK: already in the source image's coordinate space. RF-DETR's
        # postprocess interpolates the mask LOGITS to target_sizes -- the size
        # of the image handed to predict() -- and thresholds them there, so no
        # projection is needed. Assert the shape rather than rescaling it: a
        # mismatch means predictions and ground truth are in different
        # coordinate spaces, which has to fail loudly.
        if det.mask is not None:
            mask_orig = det.mask[idx].astype(np.uint8)
            if mask_orig.shape != (H, W):
                raise RuntimeError(
                    f"{Path(img_path).name}: model returned a {mask_orig.shape} "
                    f"mask for a ({H}, {W}) image. Predictions and ground truth "
                    f"would be in different coordinate spaces.")
            rle = maskUtils.encode(np.asfortranarray(mask_orig))
            rle["counts"] = rle["counts"].decode("utf-8")
            rle["size"]   = [H, W]
        else:
            # No mask head output: fall back to the detection box, filled.
            # det.xyxy is already in source-image coordinates, so there is
            # nothing to rescale: W and H came from img_pil.size.
            x1, y1, x2, y2 = (int(round(float(v))) for v in det.xyxy[idx])
            mask_orig = np.zeros((H, W), dtype=np.uint8)
            mask_orig[max(0, y1):min(H, y2),
                      max(0, x1):min(W, x2)] = 1
            rle = maskUtils.encode(np.asfortranarray(mask_orig))
            rle["counts"] = rle["counts"].decode("utf-8")
            rle["size"]   = [H, W]

        # BOUNDING BOX: tight box derived from the mask.
        bbox_from_mask = mask_to_bbox(mask_orig)
        if bbox_from_mask is not None:
            bbox = bbox_from_mask
        else:
            bbox = tuple(float(v) for v in det.xyxy[idx])
            
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
    p.add_argument("--threshold",   type=float, default=THRESHOLD,
                   help="Confidence threshold for collecting detections. Keep "
                        "it very low (default %(default)s): COCOeval sweeps the "
                        "score axis itself, so raising this truncates the recall "
                        "axis and understates AP.")
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

    # The model emits a 0-based class index that run_inference resolves
    # POSITIONALLY against the hardcoded CLASSES list. That list therefore has
    # to be the exact category order the model was trained on, and no
    # annotation file can prove it -- but a disagreement with the file being
    # scored is certain evidence that one of the two is wrong, and continuing
    # would relabel every prediction while still producing a plausible-looking
    # table. Roboflow exports also carry a dummy supercategory at id 0, which
    # shifts every index by one if it is enumerated.
    file_cats = [c["name"] for c in sorted(coco["categories"], key=lambda c: c["id"])
                 if c["name"] in CLASSES]
    missing = sorted(set(CLASSES) - set(file_cats))
    extra   = sorted({c["name"] for c in coco["categories"]} - set(CLASSES))
    if missing:
        sys.exit(f"[FATAL] {args.annotations} is missing {len(missing)} of the "
                 f"{len(CLASSES)} part classes: {missing}. The class-index -> "
                 f"name map cannot be built from it.")
    if file_cats != CLASSES:
        sys.exit("[FATAL] Category order in the annotation file disagrees with "
                 "the hardcoded CLASSES list.\n"
                 f"        CLASSES        : {CLASSES}\n"
                 f"        Annotation file: {file_cats}\n"
                 "        Point --annotations at the file the model was TRAINED "
                 "on, or correct CLASSES. Continuing would relabel every "
                 "prediction.")
    if extra:
        print(f"[INFO] Annotation file carries {len(extra)} non-part "
              f"categor{'y' if len(extra) == 1 else 'ies'} that are not scored: "
              f"{extra}")
    print(f"[INFO] Category order matches CLASSES ({len(CLASSES)} classes).")
    
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
    # Every annotation this loop discards is counted. An uncounted `continue`
    # silently shrinks the ground truth, which raises recall without leaving a
    # trace anywhere in the output.
    gt_dropped_unknown_class = 0
    print("[INFO] Decoding GT masks …")
    for ann in tqdm(anns, desc="GT masks", ncols=70):
        img   = images[ann["image_id"]]
        H, W  = img["height"], img["width"]
        cname = coco_cats.get(ann["category_id"])
        if cname not in CLASSES:
            gt_dropped_unknown_class += 1
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
    if gt_dropped_unknown_class:
        print(f"[WARN] {gt_dropped_unknown_class} GT annotations belong to a "
              f"category outside CLASSES and are not scored")
    n_gt_scored = sum(len(v) for v in gts_mask_by_cat.values())
    print(f"[INFO] Ground truth scored: {n_gt_scored} of {n_anns} annotations "
          f"({decode_errors} undecodable, {gt_dropped_unknown_class} out-of-class)")

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
    pred_dropped_unknown_class = 0   # model emitted a class not in CLASSES
    pred_dropped_no_cat_id     = 0   # class has no id in the annotation file

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
                pred_dropped_unknown_class += 1
                continue
            preds_mask_by_cat[cat].append(
                {"image_id": img_id, "score": p["score"], "mask": p["mask"]})
            preds_box_by_cat[cat].append(
                {"image_id": img_id, "score": p["score"], "bbox": p["bbox"]})
                
            cat_id = name_to_id.get(cat)
            if cat_id is None:
                pred_dropped_no_cat_id += 1
            else:
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
    if pred_dropped_unknown_class:
        print(f"[WARN] {pred_dropped_unknown_class} predictions carried a class "
              f"name outside CLASSES and were discarded. The model's class index "
              f"may not match CLASSES.")
    if pred_dropped_no_cat_id:
        print(f"[WARN] {pred_dropped_no_cat_id} predictions had no category id "
              f"in the annotation file and are absent from COCOeval")
    print(f"[INFO] COCO results entries: {len(coco_results_segm)} segm / "
          f"{len(coco_results_bbox)} bbox")

    # ── Pycocotools evaluation ───────────────────────────────────
    print("\n[INFO] Running pycocotools evaluation …")
    coco_gt_orig = COCO(args.annotations)

    # Roboflow COCO exports write the BOUNDING-BOX area into ann["area"], not
    # the polygon area. COCOeval bins ground truth by that field, so APs/APm/APl
    # would be computed over the wrong instances: on this dataset 565 of 4591
    # annotations (12.3%) land in the wrong size bin, and APs ends up averaging
    # over 9 instances instead of 44. Recompute from the segmentation.
    _n_fixed, _max_ratio = 0, 0.0
    for _a in coco_gt_orig.dataset["annotations"]:
        _seg = _a.get("segmentation")
        if not _seg:
            continue
        _info = coco_gt_orig.imgs[_a["image_id"]]
        _h, _w = _info["height"], _info["width"]
        try:
            if isinstance(_seg, list):
                _rle = maskUtils.merge(maskUtils.frPyObjects(_seg, _h, _w))
            elif isinstance(_seg.get("counts"), list):
                _rle = maskUtils.frPyObjects(_seg, _h, _w)
            else:
                _rle = _seg
            _true = float(maskUtils.area(_rle))
        except Exception:
            continue
        if _true <= 0:
            continue
        _old = float(_a.get("area") or 0.0)
        if _old > 0:
            _max_ratio = max(_max_ratio, _old / _true)
        if abs(_old - _true) > 1e-6:
            _a["area"] = _true
            _n_fixed += 1
    if _n_fixed:
        coco_gt_orig.createIndex()
        print(f"[INFO] Recomputed GT 'area' from segmentation for {_n_fixed} "
              f"annotations (max stored/true ratio was {_max_ratio:.2f}). "
              f"Size-stratified AP is now binned by MASK area, matching the "
              f"detections.")
    else:
        print("[INFO] GT 'area' fields already match mask area — no change.")

    # Support counts behind every per-class and size-stratified AP below.
    gt_support = gt_support_table(coco_gt_orig)
    print_gt_support(gt_support)

    rf_mask = run_cocoeval(coco_gt_orig, coco_results_segm, "segm", label="RF-DETR")
    rf_box  = run_cocoeval(coco_gt_orig, coco_results_bbox, "bbox", label="RF-DETR")

    print(f"\n  Mask AP={rf_mask.get('AP')}  AP50={rf_mask.get('AP50')}  AP75={rf_mask.get('AP75')}")
    print(f"  Box  AP={rf_box.get('AP')}   AP50={rf_box.get('AP50')}   AP75={rf_box.get('AP75')}")

    # ── Generate charts ───────────────────────────────────────────────────
    print("\n[INFO] Generating charts …")

    plot_pr_curves_paged(rf_mask, plots_dir, mode="mask")
    plot_pr_curves_paged(rf_box,  plots_dir, mode="box")

    mask_best = plot_f1_threshold(dict(preds_mask_by_cat), dict(gts_mask_by_cat),
                                  plots_dir, mode="mask",
                                  operating_thr=REPORT_THRESHOLD)
    box_best  = plot_f1_threshold(dict(preds_box_by_cat),  dict(gts_box_by_cat),
                                  plots_dir, mode="box",
                                  operating_thr=REPORT_THRESHOLD)

    plot_ap_per_category(rf_mask, plots_dir, mode="mask")
    plot_ap_per_category(rf_box,  plots_dir, mode="box")

    plot_mask_vs_box_overall(rf_mask, rf_box, plots_dir)
    plot_mask_vs_box_per_category(rf_mask, rf_box, plots_dir)
    plot_ap50_ap75_bar(rf_mask, plots_dir)
    confusion_mask = plot_confusion_heatmap(
        dict(preds_mask_by_cat), dict(gts_mask_by_cat), plots_dir,
        score_thresh=REPORT_THRESHOLD)

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

    def _sub(a, b):
        return None if (a is None or b is None) else round(a - b, 1)

    def _round_prf(m):
        if m is None:
            return None
        return {"threshold": round(float(m["threshold"]), 4),
                "precision": round(m["precision"], 4),
                "recall":    round(m["recall"], 4),
                "f1":        round(m["f1"], 4),
                "tp": m["tp"], "fp": m["fp"], "fn": m["fn"]}

    def _per_cat(res):
        """per_category with None where the class has no ground truth."""
        valid = res.get("per_category_valid", {})
        return {c: (res["per_category"].get(c) if valid.get(c, True) else None)
                for c in CLASSES}

    def _thin():
        """Classes whose AP rests on fewer than MIN_SUPPORT_FOR_AP instances."""
        return sorted(c for c in CLASSES
                      if 0 < gt_support[c]["all"] < MIN_SUPPORT_FOR_AP)

    output = {
        "model":      "RF-DETR-Seg-Nano",
        "checkpoint": str(args.checkpoint),
        "resolution": args.resolution,
        # See the constants at the top: these are two different things.
        "threshold":         args.threshold,          # COCOeval collection
        "report_threshold":  REPORT_THRESHOLD,   # P/R/F1 + confusion

        # Exactly what produced these numbers. Without it the JSON cannot be
        # tied back to a run: a rerun against a different annotation file or
        # image directory yields a file that looks identical.
        "inputs": {
            "images_dir":  str(Path(args.images_dir).resolve()),
            "annotations": str(Path(args.annotations).resolve()),
            "checkpoint":  str(Path(args.checkpoint).resolve()),
            "resolution":  args.resolution,
            "threshold":   args.threshold,
            "report_threshold": REPORT_THRESHOLD,
            "command":     " ".join(sys.argv),
        },
        "eval_methodology": {
            "coordinate_space":   "native source image coordinates; images are "
                                  "fed to the model unresized and masks are used "
                                  "as returned",
            "headline_evaluator": "pycocotools.cocoeval.COCOeval",
            "area_ranges":        "COCO defaults: small<32^2, medium 32^2-96^2, "
                                  "large>96^2 (this is not CarDD)",
            "gt_area":            "recomputed from segmentation; the Roboflow "
                                  "export stores bounding-box area, which bins "
                                  "instances into the wrong size ranges",
            "segm_result_keys":   "image_id, category_id, segmentation, score "
                                  "(no bbox key: it would make loadRes bin segm "
                                  "detections by box area)",
        },
        "dataset": {
            "images":      n_images,
            "annotations": n_anns,
            "categories":  len(CLASSES),
            "class_list":  CLASSES,
            "annotations_scored":       n_gt_scored,
            "annotations_undecodable":  decode_errors,
            "annotations_out_of_class": gt_dropped_unknown_class,
        },

        # Instance counts behind every per-class and size-stratified AP.
        "gt_support": gt_support,
        "min_support_for_ap": MIN_SUPPORT_FOR_AP,
        "low_support_classes": _thin(),

        # ── Core AP ──────────────────────────────────────────────────────
        # null anywhere below means "no ground truth in that cell", never
        # "the model scored zero".
        "mask_ap": {
            "AP":   rf_mask.get("AP"),
            "AP50": rf_mask.get("AP50"),
            "AP75": rf_mask.get("AP75"),
            "APs":  rf_mask.get("APs"),
            "APm":  rf_mask.get("APm"),
            "APl":  rf_mask.get("APl"),
            "per_category":      _per_cat(rf_mask),
            "per_category_AP50": {c: round(v, 1) for c, v in
                                  rf_mask["_per_cat_ap50"].items()},
            "per_category_AP75": {c: round(v, 1) for c, v in
                                  rf_mask["_per_cat_ap75"].items()},
            "per_category_gt":   {c: gt_support[c]["all"] for c in CLASSES},
        },
        "box_ap": {
            "AP":   rf_box.get("AP"),
            "AP50": rf_box.get("AP50"),
            "AP75": rf_box.get("AP75"),
            "APs":  rf_box.get("APs"),
            "APm":  rf_box.get("APm"),
            "APl":  rf_box.get("APl"),
            "per_category":      _per_cat(rf_box),
            "per_category_AP50": {c: round(v, 1) for c, v in
                                  rf_box["_per_cat_ap50"].items()},
            "per_category_AP75": {c: round(v, 1) for c, v in
                                  rf_box["_per_cat_ap75"].items()},
            "per_category_gt":   {c: gt_support[c]["all"] for c in CLASSES},
        },

        # ── Mask vs Box delta ─────────────────────────────────────────────
        "mask_vs_box_delta": {
            "AP":   _sub(rf_box.get("AP"),   rf_mask.get("AP")),
            "AP50": _sub(rf_box.get("AP50"), rf_mask.get("AP50")),
            "AP75": _sub(rf_box.get("AP75"), rf_mask.get("AP75")),
            "per_category": {
                c: _sub(_per_cat(rf_box)[c], _per_cat(rf_mask)[c])
                for c in CLASSES
            },
        },

        # operating_point : the fixed --threshold, chosen before the test set
        #                   was seen. THIS is the model's precision/recall/F1
        #                   and the only one of the two that may be quoted.
        # oracle_upper_bound : the threshold maximising F1 ON THE TEST SET.
        #                   Selected on the data it is scored against, so it
        #                   bounds the achievable operating point rather than
        #                   reporting one. Quoting it as a headline is tuning on
        #                   the evaluation set. Repairing it properly needs a
        #                   validation split, which this script does not have;
        #                   the key is named so no reader can mistake it for one
        #                   that does.
        "threshold_metrics": {
            "operating_point": {
                "mask": _round_prf(mask_best[1]),
                "box":  _round_prf(box_best[1]),
            },
            "oracle_upper_bound": {
                "selection": "argmax F1 over a 50-point sweep, evaluated on the "
                             "test set itself",
                "mask": _round_prf(mask_best[0]),
                "box":  _round_prf(box_best[0]),
            },
        },

        # Class-agnostic greedy matching @ IoU 0.50: rows are ground-truth
        # classes, columns predicted classes, plus missed GT and background FP.
        "confusion_mask": confusion_mask,

        # ── PR curves @IoU=0.50 ───────────────────────────────────────────
        "pr_curves_at_iou50": {
            "mask": pr_mask,
            "box":  pr_box,
        },

        # ── AP50 vs AP75 per category ─────────────────────────────────────
        "ap50_vs_ap75_mask": {
            c: {"AP50": round(rf_mask["_per_cat_ap50"][c], 1),
                "AP75": round(rf_mask["_per_cat_ap75"][c], 1),
                "drop": round(rf_mask["_per_cat_ap50"][c]
                              - rf_mask["_per_cat_ap75"][c], 1),
                "gt":   gt_support[c]["all"],
                "low_support": gt_support[c]["all"] < MIN_SUPPORT_FOR_AP}
            for c in CLASSES
        },

        # ── GT distribution ───────────────────────────────────────────────
        "gt_per_category": dict(per_cat_gt),

        # ── Inference stats ───────────────────────────────────────────────
        "inference_stats": {
            "time_seconds":      round(elapsed, 1),
            "failed_images":     failed,
            "total_predictions": total_p,
            "predictions_dropped_unknown_class": pred_dropped_unknown_class,
            "predictions_dropped_no_category_id": pred_dropped_no_cat_id,
            "coco_results_segm": len(coco_results_segm),
            "coco_results_bbox": len(coco_results_bbox),
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