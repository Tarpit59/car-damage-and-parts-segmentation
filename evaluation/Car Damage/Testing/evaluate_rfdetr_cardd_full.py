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
  8.  precision_recall_f1_summary.png        Summary bar: P / R / F1 at the fixed
                                             operating threshold (oracle drawn behind)
  9.  iou_threshold_ap_curve.png             AP vs IoU threshold (mask + box)
  10. confusion_heatmap_mask/box.png         Confusion matrix, class-agnostic
                                             matching, with a (missed) column
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


# ── DEFAULTS ─────────────────────────────────────────────────────────────────
IMAGES_DIR  = "/path/to/test/images"
# Only used as a hint in --help; --annotations now defaults to None so
# that omitting it means "skip the check", not "open a dummy path".
ANNOTATIONS = "/path/to/training_export/_annotations.coco.json"
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
OUTPUT_JSON = "results_full.json"
PLOTS_DIR   = "./plots"
# ─────────────────────────────────────────────────────────────────────────────

CLASSES = ["crack", "dent", "glass shatter", "lamp broken", "scratch", "tire flat"]

# ── DCN+ (ResNet-101) reference values from CarDD paper ──────────────────────
# Table IV  → AP / AP50 / AP75 / APS / APM / APL   (format: mask / box)
# Table V   → per-category                          (format: mask / box)
#
# CarDD defines size bins differently from COCO: small < 128^2,
# medium 128^2-256^2, large > 256^2 (see CARDD_AREA_RNG). The size-stratified
# numbers below are only comparable under those ranges.
DCN_PLUS = {
    "mask": {
        "AP":   57.0,
        "AP50": 77.7,
        "AP75": 58.4,
        "APs":  34.6,
        "APm":  44.0,
        "APl":  71.6,
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
        "APs":  37.1,
        "APm":  48.0,
        "APl":  66.0,
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

# CarDD uses NON-STANDARD area ranges: small < 128^2, medium 128^2-256^2,
# large > 256^2. COCO's defaults are 32^2 / 96^2, and using them would produce
# APs/APm/APl that cannot be compared against any published CarDD number.
# Defined once so that every call site shares one definition.
CARDD_AREA_RNG = [[0, 1e5 ** 2], [0, 128 ** 2], [128 ** 2, 256 ** 2],
                  [256 ** 2, 1e5 ** 2]]
AREA_BIN_NAMES = ["small", "medium", "large"]

# Below this many ground-truth instances a per-cell AP is a property of a
# handful of annotations rather than of the model, and is reported with an
# explicit support count rather than as a comparable score.
MIN_SUPPORT_FOR_AP = 10

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


def mask_to_bbox(mask_bool):
    """Return (x1,y1,x2,y2) tight bounding box from a boolean mask."""
    rows = np.any(mask_bool, axis=1)
    cols = np.any(mask_bool, axis=0)
    if not rows.any():
        return None
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return float(x1), float(y1), float(x2 + 1), float(y2 + 1)


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

# ── GROUND-TRUTH SUPPORT ─────────────────────────────────────────────────────

def gt_support_table(coco_gt, classes=None, area_rng=None):
    """
    Ground-truth instance counts per category and per CarDD area bin, measured
    from the DECODED MASK area -- the same quantity COCOeval bins by.

    Every size-stratified AP in this evaluation is an average over the cells of
    this table. Without it a reader cannot tell an APl of 11.8 computed over a
    single annotation from one computed over two hundred, and the two are not
    the same claim. Publishing size-stratified AP without the supports is the
    most common way a per-class table becomes unreproducible.
    """
    classes  = list(CLASSES) if classes is None else classes
    area_rng = CARDD_AREA_RNG if area_rng is None else area_rng
    counts = {c: {"all": 0, "small": 0, "medium": 0, "large": 0} for c in classes}
    undecodable = 0

    for ann in coco_gt.anns.values():
        cname = coco_gt.cats[ann["category_id"]]["name"]
        if cname not in counts:
            continue
        info = coco_gt.imgs[ann["image_id"]]
        seg  = ann.get("segmentation")
        try:
            rle = (maskUtils.merge(
                       maskUtils.frPyObjects(seg, info["height"], info["width"]))
                   if isinstance(seg, list) else seg)
            area = float(maskUtils.area(rle))
        except Exception:
            undecodable += 1
            continue
        counts[cname]["all"] += 1
        # First matching bin. COCOeval treats both ends as inclusive, so an
        # instance of area exactly 128^2 is scored in BOTH small and medium
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
    print("  GROUND-TRUTH SUPPORT  (instances per category x CarDD area bin)")
    print("  small < 128^2   medium 128^2-256^2   large > 256^2   (mask area)")
    print("=" * 74)
    print(f"  {'Category':<18} {'all':>7} {'small':>8} {'medium':>8} {'large':>8}")
    print("  " + "-" * 54)
    thin = []
    for cat in CLASSES:
        c = counts.get(cat, {})
        print(f"  {cat:<18} {c.get('all', 0):>7} {c.get('small', 0):>8} "
              f"{c.get('medium', 0):>8} {c.get('large', 0):>8}")
        for b in AREA_BIN_NAMES:
            n = c.get(b, 0)
            if 0 < n < MIN_SUPPORT_FOR_AP:
                thin.append((cat, b, n))
    if thin:
        print("\n  [CAUTION] cells with fewer than "
              f"{MIN_SUPPORT_FOR_AP} instances -- the AP for these is a "
              "property of a\n            few annotations, not of the model. "
              "Quote them with the count attached:")
        for cat, b, n in thin:
            print(f"              {cat:<16} {b:<7} n={n}")
    print("=" * 74)


# ── COCOEVAL ────────────────────────────────────────────────────────

def run_cocoeval(coco_gt, coco_results_list, iou_type, label="RF-DETR"):
    """
    Run pycocotools COCOeval on predictions in original image coordinates.

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

    # loadRes MUTATES the list it is given: it writes back "area", "bbox",
    # "id" and "iscrowd" into every caller-owned dict. The same list is later
    # dumped to results_coco_segm.json, so without this copy the shipped
    # detections carry a "bbox" key -- which is exactly the key that makes a
    # re-evaluation of that file bin segm detections by box area. Copy first.
    coco_dt = coco_gt.loadRes(copy.deepcopy(coco_results_list))

    # GUARD: for a segm evaluation the detections must be binned by MASK area,
    # the same criterion COCOeval uses for the ground truth. loadRes silently
    # switches to BOX area if any result dict carries a "bbox" key, because it
    # tests `if 'bbox' in anns[0]` before `elif 'segmentation' in anns[0]`.
    # That mismatch understated this project's mask APl by 10.5 points before
    # it was found, while leaving AP/AP50/AP75 untouched — i.e. it is invisible
    # in the headline number. Fail loudly rather than let it recur.
    if iou_type == "segm" and coco_dt.anns:
        _a = next(iter(coco_dt.anns.values()))
        _mask_area = float(maskUtils.area(_a["segmentation"]))
        if abs(float(_a["area"]) - _mask_area) > 1e-6:
            raise RuntimeError(
                f"segm detections are binned by area={_a['area']:.1f} but their "
                f"mask area is {_mask_area:.1f}. A \"bbox\" key has been added "
                f"back to coco_results_segm; remove it (see the note at the "
                f"coco_results_segm.append call). Size-stratified mask AP would "
                f"otherwise be computed over mismatched partitions.")

    ev = COCOeval(coco_gt, coco_dt, iou_type)
    ev.params.areaRng = CARDD_AREA_RNG
    ev.evaluate()
    ev.accumulate()
    print(f"\n── Pycocotools  [{label}  iouType={iou_type}] ──")
    ev.summarize()

    stats  = ev.stats   # [AP, AP50, AP75, APs, APm, APl, ...]

    # pycocotools returns -1 for a size bin containing no ground truth. That is
    # "undefined", NOT "the model scored zero": coercing it to 0.0 publishes
    # a zero the model never earned and drags any mean computed over it down.
    # Carry it as None.
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

    # Per-category AP (mean over IoU 0.50:0.95). Restricted to CLASSES: a
    # Roboflow dummy category in the GT file would otherwise appear as a
    # seventh "class" with AP 0 and be swept into the mean.
    for cat_id, cat_info in coco_gt.cats.items():
        cname = cat_info["name"]
        if cname not in CLASSES:
            continue
        ev2 = COCOeval(coco_gt, coco_dt, iou_type)
        ev2.params.catIds = [cat_id]
        ev2.params.areaRng = CARDD_AREA_RNG
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
            # No ground truth for this class in this split. 0.0 is written only
            # so the bar charts have a number to draw; per_category_valid is
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
    Micro-averaged precision / recall / F1 over all classes at one confidence
    threshold and one IoU.

    Matching is within class (a prediction can only satisfy ground truth of its
    own class), greedy by descending score, each ground-truth instance
    claimable once.

    Returns dict with precision, recall, f1, tp, fp, fn.
    """
    tp_total = fp_total = fn_total = 0

    for cat in CLASSES:
        preds = [p for p in preds_by_cat.get(cat, []) if p["score"] >= thr]
        gts   = gts_by_cat.get(cat, [])

        if not gts:
            # Predictions of a class with no ground truth are false positives.
            # Skipping the class entirely would drop them from the
            # denominator and inflate precision.
            fp_total += len(preds)
            continue

        gt_by_img = defaultdict(list)
        for i, g in enumerate(gts):
            gt_by_img[g["image_id"]].append((i, g))

        matched = set()
        tp = 0
        for pred in sorted(preds, key=lambda x: -x["score"]):
            best_iou = iou_thresh - 1e-9
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

        tp_total += tp
        fp_total += len(preds) - tp
        fn_total += len(gts) - tp

    p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
    r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"threshold": float(thr), "precision": p, "recall": r, "f1": f,
            "tp": int(tp_total), "fp": int(fp_total), "fn": int(fn_total)}


def compute_f1_vs_threshold(preds_by_cat, gts_by_cat, use_box=False, n_steps=50):
    """Sweep confidence threshold; return (thresholds, precision, recall, f1)."""
    thresholds = np.linspace(0.0, 1.0, n_steps)
    precs, recs, f1s = [], [], []
    for thr in thresholds:
        m = prf_at_threshold(preds_by_cat, gts_by_cat, thr, use_box=use_box)
        precs.append(m["precision"]); recs.append(m["recall"]); f1s.append(m["f1"])
    return thresholds, np.array(precs), np.array(recs), np.array(f1s)


def compute_confusion_data(preds_by_cat, gts_by_cat, iou_thresh=0.50,
                           use_box=False, score_thresh=0.0):
    """
    Cross-class confusion at a fixed IoU.

    Returns
    -------
    mat    : (n, n) int   mat[i, j] = ground truth of class i claimed by a
                          prediction of class j
    fn_col : (n,)   int   ground truth of class i claimed by nothing
    bg_fp  : (n,)   int   predictions of class j that claimed no ground truth
                          (background false positives)

    Matching is greedy over ALL predictions of ALL classes together, sorted by
    descending score, each ground-truth instance claimable once. A prediction
    may therefore claim a ground-truth instance of a DIFFERENT class — which is
    the only mechanism by which an off-diagonal cell can ever be non-zero.

    An earlier version looped `for cat in CLASSES` and matched that class's
    predictions against that class's ground truth only, writing the count to
    mat[cat, cat]. Every off-diagonal cell was therefore structurally zero: the
    figure was a per-class recall column drawn as a square and labelled
    "confusion matrix", and could not have revealed a single confusion no
    matter what the model predicted. Do not reintroduce a per-class outer loop.
    """
    n = len(CLASSES)
    cat2idx = {c: i for i, c in enumerate(CLASSES)}
    mat   = np.zeros((n, n), dtype=int)
    bg_fp = np.zeros(n, dtype=int)
    gt_count = np.zeros(n, dtype=int)

    # Ground truth pooled per image and class-agnostic; the running index makes
    # each instance claimable exactly once across every class.
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
        best_iou    = iou_thresh - 1e-9
        best_gi     = -1
        best_gt_idx = -1
        for gi, gt_idx, g in gt_by_img.get(pred["image_id"], []):
            if gi in matched:
                continue
            if use_box:
                iou = box_iou(pred["bbox"], g["bbox"])
            else:
                iou = maskUtils.iou([pred["mask"]], [g["mask"]], [0])[0][0]
            if iou > best_iou:
                best_iou    = iou
                best_gi     = gi
                best_gt_idx = gt_idx
        if best_gi >= 0:
            matched.add(best_gi)
            mat[best_gt_idx, pred_idx] += 1
        else:
            bg_fp[pred_idx] += 1

    # Each ground-truth instance is either claimed (counted somewhere in its
    # row) or missed, so this identity is exact.
    fn_col = gt_count - mat.sum(axis=1)
    assert (fn_col >= 0).all(), "confusion matrix over-counted ground truth"

    return mat, fn_col, bg_fp

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
def plot_f1_vs_threshold(preds_by_cat, gts_by_cat, plots_dir, mode="mask",
                         operating_thr=None):
    """
    Sweep the confidence threshold and plot P/R/F1.

    Returns (oracle, operating), each a dict from prf_at_threshold().

    ORACLE is the threshold that maximises F1 ON THE TEST SET ITSELF. It is a
    selected-on-test quantity: an UPPER BOUND on what this model could reach
    had the threshold been tuned perfectly, not a score the model achieves in
    deployment. It is reported because it bounds the achievable operating
    point, and it is labelled as such everywhere it appears. It must never be
    quoted as the model's precision/recall/F1 -- that is tuning on the
    evaluation set. The honest number is OPERATING, measured at
    REPORT_THRESHOLD, a conventional value fixed before the test set was
    seen.

    Note REPORT_THRESHOLD, NOT --threshold. The latter is the collection
    threshold handed to COCOeval and is deliberately ~0.001; evaluating a
    single operating point there measures precision over roughly fifty
    detections per ground-truth instance and reports a near-zero F1 for a model
    that is performing normally.

    A validation-split sweep would yield a legitimately tuned threshold; this
    script has no such split, which is precisely why the distinction is drawn
    here instead of being papered over.
    """
    _style()
    use_box = (mode == "box")
    thresholds, precs, recs, f1s = compute_f1_vs_threshold(
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
    ax.axvline(best_thr, color="#F78166", ls="--", alpha=0.6,
               label=f"Oracle (selected ON TEST): F1={best_f1:.3f} @ thr={best_thr:.2f}")
    ax.scatter([best_thr], [best_f1], color="#F78166", s=80, zorder=5)
    if operating is not None:
        ax.axvline(operating_thr, color="#C9D1D9", ls=":", alpha=0.9,
                   label=f"Operating thr={operating_thr:g}: F1={operating['f1']:.3f}")

    ax.fill_between(thresholds, f1s, alpha=0.1, color="#F78166")
    ax.set_xlabel("Confidence Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Precision / Recall / F1 vs Confidence Threshold  [{mode.upper()}]\n"
                 f"the oracle line is an upper bound, not an achieved score",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    _savefig(fig, plots_dir / f"f1_vs_threshold_{mode}.png")
    return oracle, operating


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
def plot_prf_summary(mask_pair, box_pair, plots_dir):
    """
    mask_pair / box_pair: (oracle, operating) dicts from plot_f1_vs_threshold.

    The bars are the OPERATING point -- the fixed threshold chosen before the
    test set was seen, and the only one of the two that may be quoted as this
    model's precision / recall / F1. The oracle point is drawn as a dashed
    outline behind it so the headroom stays visible without the two being
    confused. If no operating threshold was supplied the oracle is shown
    instead and the title says so.
    """
    _style()
    metrics = ["Precision", "Recall", "F1"]
    x = np.arange(len(metrics))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
    for ax, pair, mode, color in zip(
            axes, [mask_pair, box_pair], ["Mask", "Box"],
            [RF_COLOR, "#A371F7"]):
        oracle, operating = pair
        shown = operating if operating is not None else oracle
        oracle_only = operating is None
        vals  = [shown["precision"],  shown["recall"],  shown["f1"]]
        ovals = [oracle["precision"], oracle["recall"], oracle["f1"]]

        if not oracle_only:
            ax.bar(x, ovals, 0.70, facecolor="none", edgecolor="#F78166",
                   lw=1.5, ls="--", label="oracle (selected on test)")
        bars = ax.bar(x, vals, 0.70, color=color, alpha=0.85,
                      edgecolor="#30363D",
                      label=f"operating (thr={shown['threshold']:g})")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom",
                    fontsize=11, color=color, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=12)
        ax.set_ylim(0, 1.15)
        ax.set_title(
            (f"{mode}  (ORACLE thr={shown['threshold']:g} -- UPPER BOUND)"
             if oracle_only else
             f"{mode}  (operating thr={shown['threshold']:g};  "
             f"TP={shown['tp']}  FP={shown['fp']}  FN={shown['fn']})"),
            fontsize=11, fontweight="bold")
        ax.set_ylabel("Score", fontsize=11)
        ax.legend(fontsize=9, loc="upper right")

    fig.suptitle("Precision / Recall / F1 at the Fixed Operating Threshold",
                 fontsize=14, fontweight="bold", color="#E6EDF3")
    _savefig(fig, plots_dir / "precision_recall_f1_summary.png")


# ── 9: AP vs IoU Threshold Curve ──────────────────────────────
def plot_ap_vs_iou(rf_mask_results, rf_box_results, plots_dir):
    """
    Mean AP across categories as a function of the IoU threshold.

    Derived from the interpolated precision curves COCOeval already produced,
    so this chart and the headline table are the same numbers. The previous
    version called a second, independent AP implementation (ap_and_curve) that
    no longer exists in this file, so the chart advertised as "9." in the
    header was never actually generated -- the call was simply absent from
    main() and nothing failed. A figure list that promises a plot the script
    cannot produce is worse than no list.

    COCO's AP at one IoU is the mean of the 101-point interpolated precision
    vector, which is exactly what _curves stores per category.
    """
    _style()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    plotted = False

    for results, color, label in [
        (rf_mask_results, RF_COLOR,  "RF-DETR Mask"),
        (rf_box_results,  "#A371F7", "RF-DETR Box"),
    ]:
        if not results or not results.get("_curves"):
            continue
        valid = results.get("per_category_valid", {})
        cats  = [c for c in CLASSES if valid.get(c, True) and c in results["_curves"]]
        if not cats:
            continue

        ap_per_iou = []
        for iou in IOU_THRESHOLDS:
            key = round(float(iou), 2)
            aps = []
            for cat in cats:
                prec = results["_curves"][cat].get(key)
                aps.append(float(np.mean(prec[1])) * 100 if prec is not None else 0.0)
            ap_per_iou.append(float(np.mean(aps)))

        ax.plot(IOU_THRESHOLDS, ap_per_iou, color=color, lw=2.5,
                marker="o", markersize=5, label=f"{label}  (n={len(cats)} classes)")
        ax.fill_between(IOU_THRESHOLDS, ap_per_iou, alpha=0.1, color=color)
        plotted = True

    if not plotted:
        plt.close(fig)
        print("  [PLOT] skipped iou_threshold_ap_curve.png (no per-category curves)")
        return

    # DCN+ published only AP50 and AP75, so only those two points exist to
    # compare against -- they are drawn as markers, not joined into a curve
    # the paper never reported.
    for mode, color, label in [("mask", DCN_COLOR, "DCN+ Mask"),
                               ("box", "#F4A261", "DCN+ Box")]:
        ax.scatter([0.50, 0.75],
                   [DCN_PLUS[mode]["AP50"], DCN_PLUS[mode]["AP75"]],
                   color=color, s=80, zorder=5, marker="D",
                   label=f"{label} (paper, AP50/AP75 only)")

    ax.set_xlabel("IoU Threshold", fontsize=12)
    ax.set_ylabel("Mean AP across categories  (%)", fontsize=12)
    ax.set_title("AP vs IoU Threshold  (RF-DETR vs DCN+ reference points)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9.5)
    ax.set_xlim(0.48, 0.97)
    ax.set_ylim(0, 100)
    _savefig(fig, plots_dir / "iou_threshold_ap_curve.png")


# ── 10: Confusion Heatmap ─────────────────────────────────────────────────────
def plot_confusion(preds_by_cat, gts_by_cat, plots_dir, mode="mask",
                   score_thresh=0.0):
    _style()
    use_box = (mode == "box")
    mat, fn_col, bg_fp = compute_confusion_data(
        preds_by_cat, gts_by_cat, use_box=use_box, score_thresh=score_thresh)

    gt_counts = np.array([len(gts_by_cat.get(c, [])) for c in CLASSES], dtype=float)
    denom = gt_counts.copy()
    denom[denom == 0] = 1.0

    # A "(missed)" column is appended so every row sums to exactly 1 and the
    # diagonal reads as per-class recall. Without it the rows sum to the recall
    # and the reader cannot tell a miss from a mislabel.
    full      = np.concatenate([mat, fn_col[:, None]], axis=1).astype(float)
    full_norm = full / denom[:, None]
    col_labels = CLASSES + ["(missed)"]

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(full_norm, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Fraction of that class's ground truth")

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(CLASSES)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels([f"{c}  (n={int(g)})" for c, g in zip(CLASSES, gt_counts)],
                       fontsize=9)
    ax.axvline(len(CLASSES) - 0.5, color="#30363D", lw=2)
    ax.set_xlabel("Predicted Category   ·   final column = claimed by no prediction",
                  fontsize=11)
    ax.set_ylabel("Ground-Truth Category", fontsize=11)
    ax.set_title(f"Confusion Matrix  [{mode.upper()} @ IoU=0.50, score ≥ {score_thresh:g}]\n"
                 f"class-agnostic greedy matching — rows sum to 1",
                 fontsize=12, fontweight="bold")

    for i in range(len(CLASSES)):
        for j in range(len(col_labels)):
            v = full_norm[i, j]
            # YlOrRd is LIGHT at low values and dark red at high ones, so
            # dark text belongs on the pale cells. The small-but-non-zero
            # cells are exactly the mislabels worth reading, so they have to
            # stay legible.
            ax.text(j, i, f"{v:.2f}\n({int(full[i, j])})",
                    ha="center", va="center", fontsize=7.5,
                    color="white" if v > 0.6 else "black")

    # Background false positives have no ground-truth row to be normalised
    # against, so they are reported as raw counts instead of being folded into
    # the matrix.
    # Placed in FIGURE coordinates, below the axes, so it cannot collide with
    # the x-axis label the way an axes-relative offset did.
    fig.text(0.5, 0.005,
             "Background FP (prediction claimed no GT):   "
             + "   ".join(f"{c}={int(v)}" for c, v in zip(CLASSES, bg_fp)),
             ha="center", va="bottom", fontsize=8.5, color="#8B949E")

    _savefig(fig, plots_dir / f"confusion_heatmap_{mode}.png")

    return {
        "classes":        CLASSES,
        "gt_per_class":   {c: int(g) for c, g in zip(CLASSES, gt_counts)},
        "matrix":         mat.tolist(),
        "missed":         {c: int(v) for c, v in zip(CLASSES, fn_col)},
        "background_fp":  {c: int(v) for c, v in zip(CLASSES, bg_fp)},
        "iou_threshold":  0.50,
        "score_threshold": float(score_thresh),
        "note": "rows = ground-truth class, columns = predicted class; "
                "class-agnostic greedy matching by descending score",
    }


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

def print_table(mask_results, box_results, gt_support=None):
    SEP = "=" * 92

    for mode, results, label in [
        ("mask", mask_results, "Mask AP"),
        ("box",  box_results,  "Box AP (APbb)"),
    ]:
        dcn = DCN_PLUS[mode]
        print(f"\n{SEP}")
        print(f"  CarDD TEST SET — {label} Comparison")
        print(f"  Coordinate space : native ORIGINAL image coordinates "
              f"(no resize, no re-projection)")
        print(f"  Area ranges      : CarDD (small<128², medium 128²-256², large>256²)")
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
                r_off = results.get(m)
                if r_off is None:
                    print(f"  {m:<10} {d:>12.1f} {'n/a':>17} "
                          f"{'—':>16}  (no ground truth)")
                    continue
                delta = r_off - d
                sign  = "+" if delta >= 0 else ""
                win   = "←RF-DETR" if delta > 0.5 else ("←DCN+" if delta < -0.5 else "≈tie")
                print(f"  {m:<10} {d:>12.1f} {r_off:>17.1f} "
                      f"{sign+str(round(delta,1)):>16}  {win}")

            # Size-stratified AP against CarDD Table IV, with the
            # ground-truth count each cell was averaged over. The count is not
            # decoration: an AP over a couple of instances is a coin flip and
            # must not be read as a difference between the two models.
            print(f"\n  {'Size bin':<18} {'DCN+':>8} {'RF-DETR':>10} "
                  f"{'Δ':>8} {'GT':>7}")
            print(f"  {'─'*56}")
            for key, bin_name in [("APs", "small"), ("APm", "medium"), ("APl", "large")]:
                v = results.get(key)
                d = dcn.get(key)
                n = (sum(gt_support[c][bin_name] for c in CLASSES)
                     if gt_support else None)
                v_s = "n/a" if v is None else f"{v:.1f}"
                d_s = "n/a" if d is None else f"{d:.1f}"
                if v is None or d is None:
                    delta_s, bar = "—", " "
                else:
                    delta = v - d
                    delta_s = ("+" if delta >= 0 else "") + f"{delta:.1f}"
                    bar = "▲" if delta > 1 else ("▼" if delta < -1 else "–")
                n_s = "?" if n is None else str(n)
                flag = "   <-- thin support" if (n is not None
                                                 and 0 < n < MIN_SUPPORT_FOR_AP) else ""
                print(f"  {key + '  (' + bin_name + ')':<18} {d_s:>8} {v_s:>10} "
                      f"{delta_s:>8} {bar} {n_s:>5}{flag}")
            print(f"  CarDD ranges: small<128², medium 128²-256², large>256² "
                  f"(paper Table IV)")

            print(f"\n  {'Category':<18} {'DCN+':>8} "
                  f"{'RF-DETR(pycoco)':>17} {'Δ(pycoco)':>12} {'GT':>6}")
            print(f"  {'─'*74}")
            valid = results.get("per_category_valid", {})
            for cat in CLASSES:
                d     = dcn["per_category"][cat]
                n_gt  = gt_support[cat]["all"] if gt_support else None
                if not valid.get(cat, True):
                    print(f"  {cat:<18} {d:>8.1f} {'n/a':>17} {'—':>12}     "
                          f"{'0':>6}   (absent from this split)")
                    continue
                r_off = results.get("per_category", {}).get(cat, 0.0)
                delta = r_off - d
                sign  = "+" if delta >= 0 else ""
                bar   = "▲" if delta > 1 else ("▼" if delta < -1 else "–")
                print(f"  {cat:<18} {d:>8.1f} {r_off:>17.1f} "
                      f"{sign+str(round(delta,1)):>12}  {bar} "
                      f"{('?' if n_gt is None else str(n_gt)):>5}")
        else:
            print("[WARN] No metrics to display.")

        print(f"  {'─'*50}")
        if results:
            # The mean must run over the SAME classes on both sides. Averaging
            # RF-DETR over six entries -- one of which is a 0.0 placeholder for
            # a class with no ground truth -- against DCN+'s six real numbers
            # compares two different quantities and understates RF-DETR.
            valid = results.get("per_category_valid", {})
            shared = [c for c in CLASSES
                      if valid.get(c, True) and c in dcn["per_category"]]
            mean_d = float(np.mean([dcn["per_category"][c] for c in shared])) if shared else 0.0
            mean_r = float(np.mean([results["per_category"][c] for c in shared])) if shared else 0.0
            delta  = mean_r - mean_d
            sign   = "+" if delta >= 0 else ""
            print(f"  {'Mean (n=' + str(len(shared)) + ')':<18} "
                  f"{mean_d:>10.1f} {mean_r:>10.1f} {sign+str(round(delta,1)):>8}")
            if len(shared) != len(CLASSES):
                print(f"  averaged over {len(shared)} of {len(CLASSES)} classes; "
                      f"the rest have no ground truth in this split")
        print(SEP)


# ── INFERENCE ─────────────────────────────────────────────────────────────────

def run_inference(model, img_path, threshold, idx_to_name, orig_W=None, orig_H=None):
    """
    Run RF-DETR inference and return all outputs in ORIGINAL image coordinates.

    img_path MUST be an ORIGINAL, un-resized image. RF-DETR squashes whatever it
    is given to a square `resolution` internally (rfdetr/detr.py:685) and returns
    masks at the INPUT image size, so feeding the original yields
    original-coordinate masks with no projection step. A mismatched shape raises.

    - Masks are used as returned; no resampling.
    - Bounding boxes are the tight bbox of that mask (original coords).

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
    elif img_pil.size != (orig_W, orig_H):
        # orig_W/orig_H come from the annotation file; img_pil is the file on
        # disk. Every coordinate below assumes they agree -- the model is given
        # img_pil and returns masks at ITS size, which are then scored against
        # ground truth expressed in the annotation file's frame. If the two
        # disagree the run is invalid, so say so instead of quietly rescaling.
        raise RuntimeError(
            f"{Path(img_path).name}: annotations say {orig_W}x{orig_H} but the "
            f"image on disk is {img_pil.size[0]}x{img_pil.size[1]}. "
            f"--images_dir and --orig_annotations describe different images.")
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

        # ── MASK: already in ORIGINAL image coordinates ────────────────────
        # RF-DETR's postprocess (rfdetr/models/postprocess.py:66-72) bilinearly
        # interpolates the mask LOGITS to the size of the image it was handed
        # and only then thresholds, so masks come back at the input image size.
        # Feeding the ORIGINAL image therefore yields original-coordinate masks
        # with no projection step at all.
        #
        # Do NOT reintroduce a resize-and-project step here. Resampling an
        # already-binarised mask and re-thresholding it is strictly lossier
        # than interpolating the logits once, and the loss lands on the thin
        # structures -- crack, scratch -- that are hardest to segment anyway.
        if det.mask is not None:
            mask_orig = det.mask[idx].astype(np.uint8)
            if mask_orig.shape != (orig_H, orig_W):
                raise RuntimeError(
                    f"Mask is {mask_orig.shape} but the original image is "
                    f"({orig_H}, {orig_W}). --images_dir must point at the "
                    f"ORIGINAL, un-resized images so that RF-DETR returns "
                    f"masks in original coordinates. Pointing it at a resized "
                    f"copy would require re-projecting a binarised mask, which "
                    f"loses boundary accuracy on thin damage.")
            mask_rle = maskUtils.encode(np.asfortranarray(mask_orig))
            mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
            mask_rle["size"]   = [orig_H, orig_W]
        else:
            # No mask head output: synthesise a rectangular mask from the box.
            # det.xyxy is already in img_pil coordinates, and the check at the
            # top of this function has established img_pil.size == (orig_W,
            # orig_H), so there is nothing to rescale.
            x1, y1, x2, y2 = (int(round(float(v))) for v in det.xyxy[idx])
            mask_orig = np.zeros((orig_H, orig_W), dtype=np.uint8)
            mask_orig[max(0, y1):min(orig_H, y2),
                      max(0, x1):min(orig_W, x2)] = 1
            mask_rle = maskUtils.encode(np.asfortranarray(mask_orig))
            mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
            mask_rle["size"]   = [orig_H, orig_W]

        # ── BOUNDING BOX: tight bbox derived from the mask ────────────────
        bbox_from_mask = mask_to_bbox(mask_orig)
        if bbox_from_mask is not None:
            bbox = bbox_from_mask
        else:
            # Fallback if the mask is empty (very rare). Already in original
            # coordinates -- see the note in the branch above.
            bbox = tuple(float(v) for v in det.xyxy[idx])

        results.append({
            "class_name": name,
            "score":      score,
            "mask":       mask_rle,        # RLE in ORIGINAL image coords
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
    p.add_argument("--images_dir",  default=IMAGES_DIR,
                   help="ORIGINAL, un-resized test images. RF-DETR returns "
                        "masks at the input image size, so this must be the "
                        "original directory for predictions to land in original "
                        "coordinates. Pointing it at a resized copy now raises "
                        "instead of silently re-projecting a binarised mask.")
    p.add_argument("--annotations", default=None,
                   help="OPTIONAL. The annotation JSON of the dataset the model "
                        "was TRAINED on (the resized export). It is not scored "
                        "and contributes no images: it is used only to verify "
                        "that the hardcoded CLASSES list really is the model's "
                        "label order, which no other file can establish -- the "
                        "original CarDD categories are in a different order "
                        "(dent, scratch, crack, ...). Omit it and CLASSES is "
                        "taken on trust.")
    p.add_argument("--orig_annotations", default=None,
                   help="REQUIRED. Path to the ORIGINAL (un-resized) CarDD test "
                        "annotation JSON. It supplies the scored ground truth, "
                        "the image_id / category_id mapping and the true image "
                        "dimensions. There is no fallback: scoring against the "
                        "resized annotations would use 1152x1152 coordinates "
                        "and bounding-box 'area' fields, silently invalidating "
                        "every size-stratified metric.")
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

    # ── Load the ORIGINAL annotations: image list AND scored ground truth ─
    # One file is the sole source of both, so the images evaluated and the
    # ground truth scored cannot drift apart. Predictions are made on these
    # exact images at their native size, and RF-DETR returns masks at the
    # input size, so everything below is in one coordinate space throughout.
    if not args.orig_annotations:
        sys.exit(
            "[FATAL] --orig_annotations is required. It must point at the "
            "ORIGINAL (un-resized) CarDD test annotations, which supply the "
            "image list, the scored ground truth and the true image sizes.")
    orig_ann_path = args.orig_annotations
    print(f"\n[INFO] Loading original annotations: {orig_ann_path}")
    coco_gt_orig = COCO(orig_ann_path)

    images_to_score = dict(coco_gt_orig.imgs)
    print(f"[INFO] Test set: {len(images_to_score)} images, "
          f"{len(coco_gt_orig.anns)} annotations")
    if len(images_to_score) != 374 or len(coco_gt_orig.anns) != 785:
        print(f"[WARN] Expected the CarDD test split to have 374 images and "
              f"785 annotations; got {len(images_to_score)} and "
              f"{len(coco_gt_orig.anns)}.")

    # GUARD: the scored GT must be binned by true mask area. A Roboflow export
    # stores bounding-box area instead, which silently moves instances between
    # the small/medium/large bins.
    _bad, _checked = 0, 0
    for _a in list(coco_gt_orig.anns.values())[:200]:
        _seg, _st = _a.get("segmentation"), float(_a.get("area") or 0.0)
        if not _seg or _st <= 0:
            continue
        _i = coco_gt_orig.imgs[_a["image_id"]]
        try:
            _r = (maskUtils.merge(maskUtils.frPyObjects(_seg, _i["height"], _i["width"]))
                  if isinstance(_seg, list) else _seg)
            _true = float(maskUtils.area(_r))
        except Exception:
            continue
        _checked += 1
        if _true > 0 and abs(_st - _true) / _true > 0.05:
            _bad += 1
    if _checked and _bad / _checked > 0.10:
        sys.exit(
            f"[FATAL] {_bad}/{_checked} sampled ground-truth 'area' fields "
            f"disagree with the decoded mask area by more than 5%. This file "
            f"stores bounding-box area, so APs/APm/APl would be computed over "
            f"the wrong instances. Use the original CarDD annotations.")

    # ── Model class-index → name ──────────────────────────────────────────
    # The model emits a 0-based class index in the order of the dataset it was
    # TRAINED on. That order is NOT the original CarDD category order, which
    # runs dent, scratch, crack, glass shatter, lamp broken, tire flat -- so it
    # cannot be recovered from the file above. It is the hardcoded CLASSES
    # list, and no annotation file can prove it; only the training export can
    # corroborate it, which is what --annotations is for.
    model_idx_to_name: dict = {i: c for i, c in enumerate(CLASSES)}
    print(f"\n[INFO] Model class-index → name: {model_idx_to_name}")

    if args.annotations:
        print(f"[INFO] Verifying that order against: {args.annotations}")
        with open(args.annotations) as f:
            coco_train = json.load(f)
        # Replicate RF-DETR's label mapping EXACTLY. From
        # rfdetr/datasets/coco.py, build_roboflow_from_coco passes
        # remap_category_ids=True, which does:
        #
        #     cat2label = {cat_id: i
        #                  for i, cat_id in enumerate(sorted(coco.cats.keys()))}
        #
        # Note what it does NOT do: it does not skip a dummy supercategory.
        # A Roboflow export that carries {"id": 0, "name": "damage"} therefore
        # trains a model whose label 0 IS "damage", label 1 is "crack", and so
        # on -- every real class shifted by one.
        #
        # Do NOT filter the dummy out before comparing -- that inverts the
        # check's purpose. It would compare ['crack', 'dent', ...] against
        # CLASSES, report a match, and wave through the one export that
        # actually breaks the mapping. The enumeration below is over ALL
        # categories, unfiltered.
        _id_to_name = {c["id"]: c["name"] for c in coco_train["categories"]}
        label_to_name = {i: _id_to_name[cid]
                         for i, cid in enumerate(sorted(_id_to_name))}
        train_order = [label_to_name[i] for i in range(len(label_to_name))]

        if train_order != CLASSES:
            extra = [n for n in train_order if n not in CLASSES]
            hint = ""
            if extra:
                hint = (f"\n        The export carries {len(extra)} category "
                        f"that is not a damage class ({extra}). RF-DETR "
                        f"enumerates it too, so every real class is shifted. "
                        f"Either point at the export without it, or set "
                        f"CLASSES to the full {len(train_order)}-entry order "
                        f"above if the model really was trained that way.")
            sys.exit(
                f"[FATAL] The label order RF-DETR would derive from "
                f"{args.annotations} disagrees with the hardcoded CLASSES "
                f"list.\n"
                f"        CLASSES        : {CLASSES}\n"
                f"        Training export: {train_order}\n"
                f"        Continuing would relabel every prediction while "
                f"still producing a plausible-looking table.{hint}")
        print(f"[INFO] ✓ CLASSES matches the label order RF-DETR derives from "
              f"the training export.")
        _n_train_imgs = len(coco_train.get("images", []))
        if _n_train_imgs != len(images_to_score):
            print(f"[WARN] The training export lists {_n_train_imgs} images for "
                  f"this split but the original annotations list "
                  f"{len(images_to_score)}. Only the latter are evaluated.")
    else:
        print("[INFO] --annotations not given; the label order above is taken "
              "on trust from the hardcoded CLASSES list. Pass the training "
              "export to have it verified.")

    # class_name → original category_id (name-based, never positional, because
    # the two orders genuinely differ)
    name_to_orig_cat_id: dict = {
        cat["name"]: cat["id"]
        for cat in coco_gt_orig.dataset["categories"]
    }
    print(f"[INFO] Original category → ID: {name_to_orig_cat_id}")

    missing_in_orig = set(model_idx_to_name.values()) - set(name_to_orig_cat_id)
    if missing_in_orig:
        sys.exit(
            f"[FATAL] Classes the model can predict are absent from the scored "
            f"annotations: {sorted(missing_in_orig)}. Their predictions could "
            f"not be evaluated, so the reported AP would silently exclude them.")
    print(f"[INFO] ✓ All model class names found in the scored annotations.")

    # ── Decode GT masks + boxes from ORIGINAL annotations ────────────────
    # GT must be in original image coordinates, which is where the model's
    # masks already are -- the images are fed to it unresized.
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

    # Support counts behind every size-stratified AP reported below.
    gt_support = gt_support_table(coco_gt_orig)
    print_gt_support(gt_support)

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
    pred_dropped_unknown_class = 0

    print(f"\n[INFO] Running inference on {len(images_to_score)} images …")
    print(f"[INFO] Collection threshold  : {args.threshold}  "
          f"(fed to COCOeval; it sweeps the score axis itself)")
    print(f"[INFO] Reporting threshold   : {REPORT_THRESHOLD}  "
          f"(P/R/F1 summary and confusion matrix only)")
    print(f"[INFO] Coordinate space      : native original image coordinates "
          f"(images fed unresized; masks used as returned)")
    t0 = time.time()

    # The image list comes from the RESIZED annotation file, whose file names
    # are Roboflow-mangled ("000625_jpg.rf.<32 hex>.jpg"), while --images_dir
    # now points at the ORIGINAL images, named "000625.jpg". Resolving only the
    # verbatim name would fail on every single image and report 374 failures.
    # The same decode is already applied when resolving the original image_id;
    # it has to be applied to the path too.
    for orig_img_id, img_info in tqdm(images_to_score.items(),
                                      desc="Inference", ncols=70):
        # img_info comes from the annotation file that is also the ground
        # truth, so orig_img_id is already correct -- no filename decoding
        # and no id lookup, so there is no way to lose an image silently.
        listed = img_info["file_name"]
        img_path = images_dir / listed
        if not img_path.exists():
            # Tolerate a nested layout or a Roboflow-renamed copy.
            cand = decode_roboflow_filename(listed)
            for probe in (images_dir / cand, ):
                if probe.exists():
                    img_path = probe
                    break
            else:
                hits = (list(images_dir.rglob(listed))
                        or list(images_dir.rglob(cand)))
                img_path = hits[0] if hits else None
        if img_path is None or not img_path.exists():
            if failed < 5:
                print(f"\n[WARN] not found in {images_dir}: '{listed}'")
            failed += 1
            continue

        true_W, true_H = img_info["width"], img_info["height"]

        try:
            preds, orig_W, orig_H = run_inference(
                model, img_path, args.threshold, model_idx_to_name, true_W, true_H)
        except Exception as e:
            print(f"\n[WARN] {img_info['file_name']}: {e}")
            failed += 1; continue


        for p in preds:
            cat = p["class_name"]
            if cat not in CLASSES:
                # run_inference falls back to "cls_<n>" for a class index
                # outside idx_to_name. That means the model and CLASSES
                # disagree on the number of classes, which is worth reporting
                # rather than dropping in silence.
                pred_dropped_unknown_class += 1
                continue

            x1, y1, x2, y2 = p["bbox"]   # already in original image coords
            bbox_xywh = [float(x1), float(y1),
                         float(x2 - x1), float(y2 - y1)]

            # ── Extract results ──────────────
            preds_mask_by_cat[cat].append({
                "image_id": orig_img_id,
                "score":    p["score"],
                "mask":     p["mask"],          # RLE in original image coords
                "cat":      cat,
            })
            preds_box_by_cat[cat].append({
                "image_id": orig_img_id,
                "score":    p["score"],
                "bbox":     (x1, y1, x2, y2),  # (x1,y1,x2,y2) in orig coords
                "cat":      cat,
            })

            # ── COCO results ─────────────────────────────────────────
            # orig_img_id is a key of the scored annotations by construction,
            # and every class in CLASSES was verified against
            # name_to_orig_cat_id at startup. The lookup below is therefore
            # a plain index rather than a .get(): if either invariant is ever
            # violated it must raise, not silently skip the detection.
            orig_cat_id = name_to_orig_cat_id[cat]
            # NOTE: no "bbox" key here, deliberately.
            # pycocotools COCO.loadRes tests `if 'bbox' in anns[0]`
            # BEFORE `elif 'segmentation' in anns[0]`, so a bbox key on
            # a SEGM result makes it take the bbox branch and set
            # ann['area'] = w*h (box area) instead of
            # maskUtils.area(RLE) (mask area). COCOeval then bins
            # detections by box area while ground truth is binned by
            # mask area, and the size-stratified mask metrics are
            # computed over mismatched partitions.
            # Measured on this project's own shipped predictions:
            #   with the key    APs 41.16  APm 54.41  APl 62.67
            #   without the key APs 39.22  APm 55.06  APl 73.19
            # (AP/AP50/AP75 bit-identical at 59.93/79.84/60.14).
            # The bbox RESULTS list below must keep its bbox key —
            # for box AP that branch is the correct COCO convention.
            coco_results_segm.append({
                "image_id":     orig_img_id,
                "category_id":  orig_cat_id,
                "segmentation": p["mask"],    # RLE, original coords
                "score":        float(p["score"]),
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

    # A handful of unreadable images is a data problem; most of them missing
    # means --images_dir is wrong, and scoring the remainder would produce a
    # complete-looking results.json computed over a fraction of the test set.
    if failed:
        frac = failed / max(1, len(images_to_score))
        msg = (f"{failed} of {len(images_to_score)} images could not be read "
               f"({frac:.0%}).")
        if frac > 0.10:
            sys.exit(f"[FATAL] {msg} Check that --images_dir contains the "
                     f"ORIGINAL CarDD test images.")
        print(f"[WARN] {msg} The metrics below cover the remaining "
              f"{len(images_to_score) - failed}.")
    print(f"[INFO] COCO results entries : {len(coco_results_segm)} segm / "
          f"{len(coco_results_bbox)} bbox")
    if pred_dropped_unknown_class:
        print(f"[WARN] {pred_dropped_unknown_class} predictions carried a class "
              f"index outside CLASSES and were discarded. The model and CLASSES "
              f"disagree on the class count.")

    # ── Pycocotools evaluation (headline numbers) ───────────────
    print("\n[INFO] Running pycocotools evaluation …")
    mask_results = run_cocoeval(
        coco_gt_orig, coco_results_segm, "segm", label="RF-DETR")
    box_results  = run_cocoeval(
        coco_gt_orig, coco_results_bbox, "bbox", label="RF-DETR")

    for _name, _res in (("mask", mask_results), ("box", box_results)):
        if _res and _res.get("AP") is None:
            sys.exit(f"[FATAL] {_name} AP over the full area range is undefined "
                     f"(-1), which means COCOeval saw no ground truth at all. "
                     f"Check that --orig_annotations and the prediction "
                     f"category IDs refer to the same categories.")

    print_table(mask_results, box_results, gt_support=gt_support)

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
                                     plots_dir, mode="mask",
                                     operating_thr=REPORT_THRESHOLD)
    box_best  = plot_f1_vs_threshold(dict(preds_box_by_cat),  dict(gts_box_by_cat),
                                     plots_dir, mode="box",
                                     operating_thr=REPORT_THRESHOLD)

    plot_ap_comparison_bar(mask_results, mode="mask", plots_dir=plots_dir)
    plot_ap_comparison_bar(box_results,  mode="box",  plots_dir=plots_dir)

    plot_radar(mask_results, box_results, plots_dir)

    plot_prf_summary(mask_best, box_best, plots_dir)

    plot_ap_vs_iou(mask_results, box_results, plots_dir)



    confusion_mask = plot_confusion(dict(preds_mask_by_cat), dict(gts_mask_by_cat),
                                    plots_dir, mode="mask",
                                    score_thresh=REPORT_THRESHOLD)
    confusion_box  = plot_confusion(dict(preds_box_by_cat),  dict(gts_box_by_cat),
                                    plots_dir, mode="box",
                                    score_thresh=REPORT_THRESHOLD)

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

    output = {
        "model":      "RF-DETR-Seg-Medium",
        "checkpoint": str(args.checkpoint),
        "resolution": args.resolution,
        # See the constants at the top: these are two different things.
        "threshold":         args.threshold,          # COCOeval collection
        "report_threshold":  REPORT_THRESHOLD,   # P/R/F1 + confusion

        # Exactly what produced these numbers. Without this block the JSON
        # cannot be tied back to a run: a rerun against a different annotation
        # file or image directory yields a file that looks identical.
        "inputs": {
            "images_dir":       str(Path(args.images_dir).resolve()),
            "training_export":  (str(Path(args.annotations).resolve())
                                 if args.annotations else None),
            "orig_annotations": str(Path(orig_ann_path).resolve()),
            "checkpoint":       str(Path(args.checkpoint).resolve()),
            "resolution":       args.resolution,
            "threshold":        args.threshold,
            "report_threshold": REPORT_THRESHOLD,
            "command":          " ".join(sys.argv),
        },
        "eval_methodology": {
            "coordinate_space":   "native original image coordinates; images are "
                                  "fed to the model unresized and masks are used "
                                  "as returned, at the source resolution",
            "headline_evaluator": "pycocotools.cocoeval.COCOeval",
            "area_ranges":        "CarDD: small<128^2, medium 128^2-256^2, large>256^2",
            "segm_result_keys":   "image_id, category_id, segmentation, score "
                                  "(no bbox key: it would make loadRes bin segm "
                                  "detections by box area)",
            "orig_annotations":   orig_ann_path,
        },
        "dataset": {
            "images":      len(images_to_score),
            "annotations": len(coco_gt_orig.anns),
            "note":        "CarDD test split; image list and scored ground truth both come from --orig_annotations",
        },

        # Instance counts behind every per-class and size-stratified AP above.
        "gt_support": gt_support,
        "min_support_for_ap": MIN_SUPPORT_FOR_AP,

        # ── HEADLINE: pycocotools results ────────────────────────
        # null in per_category / APs / APm / APl means "no ground truth in
        # that cell", never "the model scored zero".
        "mask_ap": {
            "AP":           mask_results.get("AP"),
            "AP50":         mask_results.get("AP50"),
            "AP75":         mask_results.get("AP75"),
            "APs":          mask_results.get("APs"),
            "APm":          mask_results.get("APm"),
            "APl":          mask_results.get("APl"),
            "per_category": _per_cat(mask_results),
        } if mask_results else None,
        "box_ap": {
            "AP":           box_results.get("AP"),
            "AP50":         box_results.get("AP50"),
            "AP75":         box_results.get("AP75"),
            "APs":          box_results.get("APs"),
            "APm":          box_results.get("APm"),
            "APl":          box_results.get("APl"),
            "per_category": _per_cat(box_results),
        } if box_results else None,

        # ── DCN+ reference ────────────────────────────────────────────────
        "dcnplus_reference": DCN_PLUS,

        # ── Delta vs DCN+ ───────
        # A delta against an undefined AP is undefined, not zero, so every
        # subtraction here goes through _sub().
        "delta_vs_dcnplus": {
            "mask": {
                "AP":   _sub(mask_results.get("AP"),   DCN_PLUS["mask"]["AP"]),
                "AP50": _sub(mask_results.get("AP50"), DCN_PLUS["mask"]["AP50"]),
                "AP75": _sub(mask_results.get("AP75"), DCN_PLUS["mask"]["AP75"]),
                "APs":  _sub(mask_results.get("APs"),  DCN_PLUS["mask"]["APs"]),
                "APm":  _sub(mask_results.get("APm"),  DCN_PLUS["mask"]["APm"]),
                "APl":  _sub(mask_results.get("APl"),  DCN_PLUS["mask"]["APl"]),
                "per_category": {
                    cat: _sub(_per_cat(mask_results)[cat],
                              DCN_PLUS["mask"]["per_category"][cat])
                    for cat in CLASSES
                },
            } if mask_results else {},
            "box": {
                "AP":   _sub(box_results.get("AP"),   DCN_PLUS["box"]["AP"]),
                "AP50": _sub(box_results.get("AP50"), DCN_PLUS["box"]["AP50"]),
                "AP75": _sub(box_results.get("AP75"), DCN_PLUS["box"]["AP75"]),
                "APs":  _sub(box_results.get("APs"),  DCN_PLUS["box"]["APs"]),
                "APm":  _sub(box_results.get("APm"),  DCN_PLUS["box"]["APm"]),
                "APl":  _sub(box_results.get("APl"),  DCN_PLUS["box"]["APl"]),
                "per_category": {
                    cat: _sub(_per_cat(box_results)[cat],
                              DCN_PLUS["box"]["per_category"][cat])
                    for cat in CLASSES
                },
            } if box_results else {},
        },

        # ── Mask vs Box delta ──
        "mask_vs_box_delta": {
            "AP":   _sub(box_results.get("AP"),   mask_results.get("AP")),
            "AP50": _sub(box_results.get("AP50"), mask_results.get("AP50")),
            "AP75": _sub(box_results.get("AP75"), mask_results.get("AP75")),
            "per_category": {
                cat: _sub(_per_cat(box_results)[cat], _per_cat(mask_results)[cat])
                for cat in CLASSES
            },
        } if box_results and mask_results else {},

        # ── Precision / Recall / F1 ──────────────────────────────
        # operating_point : the fixed --threshold, chosen before the test set
        #                   was seen. THIS is the model's precision/recall/F1
        #                   and the only one of the two that may be quoted.
        # oracle_upper_bound : the threshold that maximises F1 ON THE TEST SET.
        #                   Selected on the data it is scored against, so it is
        #                   an upper bound on the achievable operating point,
        #                   not an achieved result. Quoting it as a headline is
        #                   tuning on the evaluation set. Repairing it properly
        #                   needs a validation split, which this script does
        #                   not have; the key is named so no reader can mistake
        #                   it for one that does.
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

        # ── Confusion (class-agnostic greedy matching @ IoU 0.50) ─────────
        "confusion": {
            "mask": confusion_mask,
            "box":  confusion_box,
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
            "predictions_dropped_unknown_class": pred_dropped_unknown_class,
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