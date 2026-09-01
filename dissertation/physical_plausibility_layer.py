"""
physical_plausibility_layer.py
==============================
Does part-context validation reduce false positives in car damage instance
segmentation, and by how much?

Two independent RF-DETR segmentation models are run over the same photograph -
one predicts damage, one predicts car parts - and the second is used to audit
the first: a damage prediction is rejected when the part it lies on cannot
physically sustain that kind of damage.

SELF-CONTAINED BY DESIGN
------------------------
This file imports nothing from research/ and reads no cache. It runs both
models itself, applies the rules itself, scores itself, and computes its own
confidence intervals. Every number it prints comes from this one file and one
inference pass, so any table in the dissertation can be reproduced with a
single command and audited by reading a single script.

WHAT IT MEASURES
----------------
  1. baseline precision / recall / F1 of the damage model
  2. the same after the validation layer removes anatomically impossible
     predictions
  3. false positives removed, true positives destroyed, rejection precision
  4. the per-damage-class impossible rate
  5. 95% bootstrap confidence intervals on all of the above
  6. per-class AP (CarDD only, COCOeval with CarDD's own area ranges)
  7. the rank correlation between rule restrictiveness and per-class AP
  8. every TRUE positive the layer destroyed, named individually, with the
     parts detected beneath it - so the recall cost is evidence rather than a
     number (TABLE 4)

All of 1-5 are reported at EVERY threshold in --sweep and for ALL FOUR
policies, not at one tuned operating point for one chosen policy.

WHY A SWEEP AND NOT A SINGLE OPERATING POINT
--------------------------------------------
A confidence threshold picked by maximising F1 on the same split you then
report is a hyperparameter selected on test, and no result resting on it is
defensible. This script therefore takes no single threshold as authoritative:
--operating_threshold only chooses which row TABLE 2 and TABLE 3 expand, and
every headline number is reported across the whole sweep.

There is a second reason, and it is the substantive one. Anatomically
impossible predictions live in the LOW-CONFIDENCE TAIL - a confident detector
rarely places a dent on a windscreen. Measured only at a high threshold the
layer has almost nothing to act on, and the honest reading of a null result
there is "not tested", not "no effect". The sweep shows where the layer acts.

WHY ALL FOUR POLICIES
---------------------
set / dominant / majority / analytics are four readings of the same evidence.
Reporting only the one that happened to look best is a garden-of-forking-paths
problem, so every run prints all four. --policy names the one declared in
advance as primary; it changes no number, only which row is expanded.

CLASS-INDEX SPACES  (verify these before trusting any output)
-------------------------------------------------------------
DAMAGE model, 6 classes, 0-based, alphabetical:
    0 crack | 1 dent | 2 glass shatter | 3 lamp broken | 4 scratch | 5 tire flat

PARTS model, 19 UNSIDED classes, 0-based - the RAW model space, before the
left/right assignment step used in the deployed app. Side information is not
needed here: the rules only ask whether a part is glass / lamp / tyre / other,
and Left_Headlight versus Right_Headlight makes no difference to that. Working
in raw space also avoids the 19->29 conversion, which drops every side-dependent
class when the car's orientation is unknown - and it is unknown for CarDD,
whose filenames are '000012.jpg'.

    0  Diggi_Back_Door        7  Grill              14 Roof
    1  Diggi_Back_Door_Glass  8  Headlight          15 Running_Board
    2  Fender                 9  Hood_Bonnet        16 Side_Mirror
    3  Front_Bumper           10 Quarter_Panel      17 Taillight
    4  Front_Door             11 Rear_Bumper        18 tyre
    5  Front_Door_Glass       12 Rear_Door
    6  Front_Windshield_Glass 13 Rear_Door_Glass

THE THREE-VALUED RULE  (this is the part that matters)
------------------------------------------------------
Each rule asks "is this damage on a part that permits it?". Written naively as
a yes/no test it answers NO when the parts model detected nothing at all - so
"I cannot see" is scored identically to "wrong part", and the prediction is
discarded. On close-up imagery that happens to one prediction in five and
destroys mostly correct detections.

Every rule here therefore returns one of THREE values:

    True     plausible   sits on a part the rules permit
    False    impossible  sits on a NAMED part the rules forbid   <- the signal
    None     abstain     no part evidence; no anatomical statement is possible

Abstain is NOT a rejection. --no_evidence controls it (default keep) and
--unknown controls the separate "mostly on unsegmented area" test (default
keep). Both are exposed so the alternative can be measured rather than assumed.

PARTS PRE-PROCESSING - ORDER MATTERS
------------------------------------
    step 1  spatial dedup  same class, mask IoU > 0.30, keep highest confidence
    step 2  count caps     max 1 windshield, 2 headlights, 4 tyres, ...

If capping ran first, two detections on the SAME headlight (0.90, 0.85) would
fill the "max 2" quota and a genuine second headlight at 0.60 would be dropped.
Deduplicating first makes survivors spatially distinct, so capping by
confidence cannot delete a duplicate-free part OF ONE CAR. Note the limit of
that claim: CAPS describes a single vehicle while apply_caps runs per IMAGE,
so a photograph containing two cars can still lose genuine parts. Neither
CarDD nor HIL is a multi-vehicle set, but the caps would need to be per-car
before this ran on street scenes.

BOOTSTRAP
---------
Images, not predictions, are the resampling unit: predictions on one photograph
share a parts segmentation and a viewpoint, so they are not independent, and
resampling predictions would produce intervals that are too narrow. B=5000
resamples of precomputed per-image contributions. Intervals are 2.5/97.5
percentiles.

DATASETS
--------
    cardd   CarDD test split, COCO format. All 6 damage classes are present
            and every ground-truth instance carries exactly one label.

    hil     Humans-in-the-Loop. NOTE the published folders are SWAPPED: the
            directory named "Car parts dataset" is the one holding the 814
            DAMAGE-annotated images.
            Only 5 of the 6 classes are testable - no HIL label corresponds
            to tire flat - and "Broken part" is accepted as glass shatter OR
            lamp broken, a loose criterion that cannot separate the two, so
            that pair must be read as one merged class.
            Paint chip / Missing part / Flaking / Corrosion carry no meaning
            for this study and become IGNORE regions: a prediction lying
            mostly inside one is counted as neither a hit nor an error.

Both runs print, and both result JSONs record, exactly which classes were
evaluated, which were not and why, how many ground-truth instances each class
contributed, and how source labels map onto damage classes. No number in
either file has to be interpreted without that context.

USAGE
-----
    python dissertation/physical_plausibility_layer.py --dataset cardd
    python dissertation/physical_plausibility_layer.py --dataset hil

    # the self-check suite (no GPU needed; add --with_models for group H)
    python dissertation/physical_plausibility_layer.py --self_check

    # a different sweep
    python dissertation/physical_plausibility_layer.py --dataset cardd \
        --sweep 0.05,0.10,0.15,0.20,0.30,0.40,0.50,0.60
"""

import argparse
import contextlib
import io
import json
import math
import sys
from collections import Counter, defaultdict
from itertools import permutations
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ── local paths ──────────────────────────────────────────────────────────────
# Where the checkpoints and datasets live differs per machine and is no part of
# any result, so it is read from local_paths.json - which is NOT tracked by git -
# rather than written into this file. Copy local_paths.example.json, edit it, and
# every command below runs with no arguments. Any value can still be overridden
# on the command line.
_CFG = Path(__file__).resolve().parent / "local_paths.json"
_LOCAL = json.load(open(_CFG, encoding="utf-8")) if _CFG.exists() else {}


def _p(key):
    return _LOCAL.get(key, "")


DAMAGE_CLASSES = ["crack", "dent", "glass shatter", "lamp broken",
                  "scratch", "tire flat"]
CRACK, DENT, GLASS_SHATTER, LAMP_BROKEN, SCRATCH, TIRE_FLAT = range(6)

PART_CLASSES = [
    "Diggi_Back_Door", "Diggi_Back_Door_Glass", "Fender", "Front_Bumper",
    "Front_Door", "Front_Door_Glass", "Front_Windshield_Glass", "Grill",
    "Headlight", "Hood_Bonnet", "Quarter_Panel", "Rear_Bumper",
    "Rear_Door", "Rear_Door_Glass", "Roof", "Running_Board",
    "Side_Mirror", "Taillight", "tyre",
]
GLASS_PARTS = {1, 5, 6, 13}
LAMP_PARTS = {8, 17}
TYRE_PARTS = {18}

# Which parts each damage class may sit on. allowed=None means "any named part".
ALLOWED = {CRACK: None, DENT: None, GLASS_SHATTER: GLASS_PARTS,
           LAMP_BROKEN: LAMP_PARTS, SCRATCH: None, TIRE_FLAT: TYRE_PARTS}
EXCLUDED = {CRACK: set(), DENT: GLASS_PARTS | LAMP_PARTS | TYRE_PARTS,
            GLASS_SHATTER: set(), LAMP_BROKEN: set(),
            SCRATCH: GLASS_PARTS, TIRE_FLAT: set()}

# How many of each part a car can have.
CAPS = {"Front_Windshield_Glass": 1, "Diggi_Back_Door_Glass": 1, "Grill": 1,
        "Hood_Bonnet": 1, "Roof": 1, "Front_Bumper": 1, "Rear_Bumper": 1,
        "Diggi_Back_Door": 1, "Headlight": 2, "Taillight": 2, "Side_Mirror": 2,
        "Fender": 2, "Front_Door": 2, "Rear_Door": 2, "Quarter_Panel": 2,
        "Front_Door_Glass": 2, "Rear_Door_Glass": 2, "Running_Board": 2,
        "tyre": 4}
CAP_BY_ID = {i: CAPS.get(n, 99) for i, n in enumerate(PART_CLASSES)}

POLICIES = ["set", "dominant", "majority", "analytics"]

# CarDD uses NON-STANDARD COCO area ranges. Using the COCO defaults gives
# APs/APm/APl that are not comparable with any published CarDD number.
CARDD_AREA_RNG = [[0, 1e5 ** 2], [0, 128 ** 2], [128 ** 2, 256 ** 2],
                  [256 ** 2, 1e5 ** 2]]

HIL_DIRECT = {"Scratch": {SCRATCH}, "Dent": {DENT}, "Cracked": {CRACK}}
HIL_IGNORE = {"Paint chip", "Missing part", "Flaking", "Corrosion"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["cardd", "hil"], default="cardd")
    p.add_argument("--damage_ckpt",
                   default=_p("damage_ckpt"))
    p.add_argument("--parts_ckpt",
                   default=_p("parts_ckpt"))
    p.add_argument("--images_dir",
                   default=_p("images_dir"))
    p.add_argument("--annotations",
                   default=_p("annotations"))
    p.add_argument("--data_root", default=_p("data_root"))
    p.add_argument("--broken_part", choices=["ignore", "superclass"],
                   default="superclass")
    p.add_argument("--ignore_overlap", type=float, default=0.50)

    p.add_argument("--damage_resolution", type=int, default=960)
    p.add_argument("--parts_resolution", type=int, default=960)
    p.add_argument("--infer_threshold", type=float, default=0.05)
    p.add_argument("--parts_threshold", type=float, default=0.45)
    p.add_argument("--operating_threshold", type=float, default=0.50,
                   help="the row TABLE 2 and TABLE 3 expand. Declared in "
                        "advance, NOT tuned: 0.50 is the same point the "
                        "evaluation chapter reports at, so the two are "
                        "directly comparable. No claim rests on it - every "
                        "statistic is reported across --sweep.")
    p.add_argument("--iou_threshold", type=float, default=0.50)

    p.add_argument("--policy", default="majority", choices=POLICIES)
    p.add_argument("--part_nms_iou", type=float, default=0.30)
    p.add_argument("--overlap_threshold", type=float, default=0.20)
    p.add_argument("--attribution_overlap", type=float, default=0.15)
    p.add_argument("--unknown_dominance", type=float, default=0.60)
    p.add_argument("--unknown", choices=["reject", "keep"], default="keep")
    p.add_argument("--no_evidence", choices=["reject", "keep"], default="keep")

    p.add_argument("--sweep", default="0.05,0.10,0.20,0.30,0.40,0.50",
                   help="confidence thresholds to report the layer's effect "
                        "at. A single tuned threshold cannot show where the "
                        "layer acts, because anatomically impossible "
                        "predictions live in the low-confidence tail.")
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no_fp16", dest="fp16", action="store_false")
    p.add_argument("--n_images", type=int, default=0, help="0 = all")
    p.add_argument("--out_dir", default=str(Path(__file__).parent / "results"))

    # self-check mode: verify the script instead of running the study
    p.add_argument("--self_check", action="store_true",
                   help="run the built-in check suite (groups A-G) and exit. "
                        "No GPU and no model weights are needed unless "
                        "--with_models is also given.")
    p.add_argument("--with_models", action="store_true",
                   help="add group H to --self_check: load both checkpoints "
                        "and verify class-id ranges and mask geometry.")
    p.add_argument("--n_sample", type=int, default=40,
                   help="how many annotation files the self-check samples.")
    p.add_argument("--hil_parts_root",
                   default=_p("hil_parts_root"),
                   help="self-check only: the HIL folder that holds PART "
                        "labels. The published folders are swapped, and group "
                        "A proves it rather than assuming it.")
    return p.parse_args()


# ── rules ────────────────────────────────────────────────────────────────────

def part_ok(pid, dcid):
    if pid in EXCLUDED.get(dcid, set()):
        return False
    a = ALLOWED.get(dcid)
    return a is None or pid in a


def judge(dcid, fracs, overlap_thr, attrib_thr):
    """
    fracs : [(part_id, fraction_of_the_damage_mask_lying_on_that_part), ...]
    returns {policy: True plausible | False impossible | None abstain}
    """
    if dcid == CRACK:                       # no anatomical rule exists
        return {p: True for p in POLICIES}

    op = {p for p, f in fracs if f >= overlap_thr}
    res = {}

    # set - the strict reading: which parts does the damage touch at all?
    if not op:
        res["set"] = None
    elif dcid == DENT:
        res["set"] = bool(op - EXCLUDED[DENT])
    elif dcid == GLASS_SHATTER:
        res["set"] = bool(op & GLASS_PARTS)
    elif dcid == LAMP_BROKEN:
        res["set"] = bool(op & LAMP_PARTS) and not bool(op - LAMP_PARTS)
    elif dcid == SCRATCH:
        res["set"] = (not (op & GLASS_PARTS)) and bool(op - GLASS_PARTS)
    else:
        res["set"] = bool(op & TYRE_PARTS)

    # dominant - judge only the single largest part among those clearing
    # overlap_thr. Taking the max over ALL fracs would let a part covering
    # 0.1% of the mask decide the verdict while every other policy ignores it.
    dom = [(p, f) for p, f in fracs if f >= overlap_thr]
    if not dom:
        res["dominant"] = None
    else:
        top = max(f for _, f in dom)
        verdicts = {part_ok(p, dcid) for p, f in dom if f == top}
        # An exact tie between a permitted and a forbidden part is genuinely
        # ambiguous. max() would resolve it by position in `fracs`, i.e. by
        # the parts model's detection order, which is not evidence about
        # anatomy - so abstain instead.
        res["dominant"] = verdicts.pop() if len(verdicts) == 1 else None

    # majority - do the permitted parts carry more of the mask than the rest?
    if not op:
        res["majority"] = None
    else:
        v = sum(f for p, f in fracs if f >= overlap_thr and part_ok(p, dcid))
        iv = sum(f for p, f in fracs if f >= overlap_thr and not part_ok(p, dcid))
        res["majority"] = v > iv

    # analytics - multi-part credit at a lower bar
    credited = [(p, f) for p, f in fracs if f >= attrib_thr]
    res["analytics"] = (None if not credited
                        else any(part_ok(p, dcid) for p, _ in credited))
    return res


# ── parts pre-processing ─────────────────────────────────────────────────────

def suppress_duplicate_parts(masks, cids, confs, iou_thr):
    """Same class, mask IoU > iou_thr, keep the higher confidence."""
    order = list(np.argsort(confs)[::-1])
    suppressed, kept = set(), []
    for i in order:
        if i in suppressed:
            continue
        kept.append(i)
        mi = masks[i]
        for j in order:
            if j in suppressed or j == i or cids[j] != cids[i]:
                continue
            inter = int(np.logical_and(mi, masks[j]).sum())
            if not inter:
                continue
            union = int(np.logical_or(mi, masks[j]).sum())
            if union and inter / union > iou_thr:
                suppressed.add(j)
    return sorted(kept)


def apply_caps(idxs, cids, confs):
    """At most CAP_BY_ID[c] per class. Safe only AFTER dedup."""
    by_cls = defaultdict(list)
    for i in idxs:
        by_cls[cids[i]].append(i)
    out = []
    for c, lst in by_cls.items():
        lst.sort(key=lambda i: -confs[i])
        out += lst[:CAP_BY_ID.get(c, 99)]
    return sorted(out)


# ── datasets ─────────────────────────────────────────────────────────────────

def poly_to_mask(pts, H, W):
    ext = pts.get("exterior") or []
    if len(ext) < 3:
        return None
    im = Image.new("1", (W, H), 0)
    dr = ImageDraw.Draw(im)
    dr.polygon([(float(x), float(y)) for x, y in ext], fill=1)
    for hole in pts.get("interior") or []:
        if len(hole) >= 3:
            dr.polygon([(float(x), float(y)) for x, y in hole], fill=0)
    return np.array(im, dtype=bool)


def decode_coco(ann, H, W):
    seg = ann["segmentation"]
    if isinstance(seg, list):
        rle = maskUtils.merge(maskUtils.frPyObjects(seg, H, W))
    elif isinstance(seg.get("counts"), list):
        rle = maskUtils.frPyObjects(seg, H, W)
    else:
        rle = seg
    return maskUtils.decode(rle).astype(bool)


def load_cardd(a):
    """-> items, evaluable, coco  (coco kept so AP can be computed later)"""
    coco = COCO(a.annotations)
    id2n = {c["id"]: c["name"] for c in coco.dataset["categories"]}
    missing = set(DAMAGE_CLASSES) - set(id2n.values())
    if missing:
        print("[WARN] classes absent from annotations: %s" % missing)
    anns = defaultdict(list)
    for an in coco.dataset["annotations"]:
        anns[an["image_id"]].append(an)
    ids = list(coco.imgs.keys())
    if a.n_images:
        ids = ids[:a.n_images]
    img_dir = Path(a.images_dir)
    items, n_missing, n_decode_fail = [], 0, 0
    for iid in ids:
        info = coco.imgs[iid]
        fp = img_dir / Path(info["file_name"]).name
        if not fp.exists():
            n_missing += 1
            continue
        H, W = info["height"], info["width"]
        masks, accept = [], []
        for an in anns.get(iid, []):
            nm = id2n[an["category_id"]]
            if nm not in DAMAGE_CLASSES:
                continue
            # decode first, append second: appending the mask and then
            # failing would desynchronise masks from accept
            try:
                dm = decode_coco(an, H, W)
            except Exception:
                n_decode_fail += 1
                continue
            masks.append(dm)
            accept.append({DAMAGE_CLASSES.index(nm)})
        items.append({"path": fp, "image_id": iid, "gt_masks": masks,
                      "gt_accept": accept, "ignore": None})
    sup = cardd_support(coco, [it["image_id"] for it in items])
    desc = {
        "name": "CarDD",
        "split": "test",
        "annotations_file": str(a.annotations),
        "images_dir": str(a.images_dir),
        "images_listed": len(ids),
        "images_missing_on_disk": n_missing,
        "images_scored": len(items),
        "gt_instances_scored": sum(len(it["gt_accept"]) for it in items),
        "gt_annotations_that_failed_to_decode": n_decode_fail,
        "classes_evaluated": list(DAMAGE_CLASSES),
        "classes_not_evaluated": [],
        "gt_instances_per_class": sup,
        "label_map": {c: [c] for c in DAMAGE_CLASSES},
        "ignored_labels": [],
        "notes": ["Every ground-truth instance carries exactly one damage "
                  "label, so a prediction matches only its own class."],
    }
    return items, set(range(len(DAMAGE_CLASSES))), coco, desc


def load_hil(a):
    root = Path(a.data_root)
    gt_map = dict(HIL_DIRECT)
    if a.broken_part == "superclass":
        gt_map["Broken part"] = {GLASS_SHATTER, LAMP_BROKEN}
    evaluable = set().union(*gt_map.values())
    ann_files = sorted((root / "ann").glob("*.json"))
    if a.n_images:
        ann_files = ann_files[:a.n_images]
    items, hist, n_missing = [], Counter(), 0
    for af in ann_files:
        fp = root / "img" / af.name[:-5]
        if not fp.exists():
            n_missing += 1
            continue
        d = json.load(open(af, encoding="utf-8"))
        H, W = d["size"]["height"], d["size"]["width"]
        masks, accept, ign = [], [], []
        for o in d.get("objects", []):
            t = o.get("classTitle")
            drop_broken = (t == "Broken part" and a.broken_part == "ignore")
            if t not in gt_map and t not in HIL_IGNORE and not drop_broken:
                continue
            m = poly_to_mask(o.get("points", {}), H, W)
            if m is None:
                continue
            if t in gt_map:
                hist[t] += 1
                masks.append(m)
                accept.append(gt_map[t])
            else:
                ign.append(m)
        items.append({"path": fp, "image_id": None, "gt_masks": masks,
                      "gt_accept": accept,
                      "ignore": np.any(ign, axis=0) if ign else None})
    print("[INFO] HIL ground truth: %s" % dict(hist))
    # An instance labelled with a superclass ('Broken part') is ONE instance
    # that either of two classes may match. Adding it to both counters would
    # make this column sum to more than the ground-truth total and overstate
    # the support behind each class, so ambiguous labels get their own
    # combined row and every instance is counted exactly once.
    per_class = Counter()
    for t, n in hist.items():
        names = sorted(DAMAGE_CLASSES[c] for c in gt_map[t])
        per_class[" | ".join(names)] += n
    absent = [DAMAGE_CLASSES[c] for c in range(len(DAMAGE_CLASSES))
              if c not in evaluable]
    notes = ["The published HIL folders are SWAPPED: the directory named "
             "'Car parts dataset' is the one holding damage annotations.",
             "No HIL label corresponds to tire flat, so that class is not "
             "evaluated here and its predictions are discarded before "
             "scoring."]
    if a.broken_part == "superclass":
        notes.append(
            "The label 'Broken part' is accepted as EITHER glass shatter OR "
            "lamp broken. HIL does not distinguish the two, so neither "
            "class's precision on this dataset is a clean measurement, and "
            "the pair should be read as one merged class.")
    else:
        notes.append("'Broken part' instances were dropped (--broken_part "
                     "ignore), so glass shatter and lamp broken have no "
                     "ground truth here.")
    notes.append("Labels %s carry no damage-class meaning for this study and "
                 "become IGNORE regions: a prediction overlapping one by >= "
                 "%.2f of its own area is counted as neither a hit nor an "
                 "error." % (sorted(HIL_IGNORE), a.ignore_overlap))
    desc = {
        "name": "Humans-in-the-Loop (HIL)",
        "split": "all annotated images",
        "data_root": str(a.data_root),
        "images_listed": len(ann_files),
        "images_missing_on_disk": n_missing,
        "images_scored": len(items),
        "gt_instances_scored": sum(len(it["gt_accept"]) for it in items),
        "classes_evaluated": sorted(DAMAGE_CLASSES[c] for c in evaluable),
        "classes_not_evaluated": absent,
        "gt_instances_per_source_label": dict(hist),
        "gt_instances_per_class": {k: int(v) for k, v in per_class.items()},
        "gt_instances_per_class_note":
            "Each ground-truth instance is counted once. A key joined by '|' "
            "is a source label that either of those classes may match, so the "
            "two cannot be told apart and share that support.",
        "label_map": {t: sorted(DAMAGE_CLASSES[c] for c in cs)
                      for t, cs in gt_map.items()},
        "ignored_labels": sorted(HIL_IGNORE),
        "notes": notes,
    }
    return items, evaluable, None, desc


# ── matching ─────────────────────────────────────────────────────────────────

def greedy_match(items, accept, iou, thr):
    """items: [(index, score, class)]. accept[j]: set of classes matching gt j."""
    taken, tp = set(), set()
    for i, _, c in sorted(items, key=lambda t: -t[1]):
        bj, best = -1, thr
        for j in range(len(accept)):
            if j in taken or c not in accept[j]:
                continue
            v = iou[i][j] if len(iou) else 0.0
            if v >= best:
                best, bj = v, j
        if bj >= 0:
            taken.add(bj)
            tp.add(i)
    return tp


# ── statistics ───────────────────────────────────────────────────────────────

def ci_excludes_zero(lo, hi):
    """True when a 95% CI excludes zero.

    An absent bootstrap yields (nan, nan). NaN comparisons are all False, so
    the negated form `not (lo <= 0 <= hi)` would silently mark every row
    significant - exactly backwards. Guard it here, once.
    """
    if lo is None or hi is None or math.isnan(lo) or math.isnan(hi):
        return False
    return not (lo <= 0 <= hi)


def jnum(v, nd):
    """Round for JSON, but write null rather than a bare NaN literal.

    json.dump emits NaN unquoted, which is not valid JSON and is rejected by
    strict parsers - including the ones a reviewer is likely to use.
    """
    if v is None:
        return None
    v = float(v)
    return None if math.isnan(v) or math.isinf(v) else round(v, nd)


def thr_keys(values):
    """Distinct string keys for the sweep, at the coarsest precision that
    keeps them unique. Three decimals reads well; more is used only if two
    thresholds would otherwise collide and silently overwrite each other."""
    for nd in (3, 4, 6, 10):
        keys = ["%.*f" % (nd, v) for v in values]
        if len(set(keys)) == len(keys):
            return keys
    return [repr(v) for v in values]


def pct_ci(samples, lo=2.5, hi=97.5):
    s = np.asarray([x for x in samples if x is not None and np.isfinite(x)])
    if not len(s):
        return (float("nan"), float("nan"))
    return float(np.percentile(s, lo)), float(np.percentile(s, hi))


def spearman(x, y):
    def rank(v):
        v = np.asarray(v, float)
        order = np.argsort(v)
        r = np.empty(len(v), float)
        r[order] = np.arange(len(v), dtype=float)
        for u in np.unique(v):
            m = v == u
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    rx, ry = rank(x), rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    d = math.sqrt(float((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / d) if d else 0.0


def exact_perm_p(x, y, observed):
    """Exact two-sided p by enumerating all n! orderings. n<=8."""
    n = len(x)
    if n > 8:
        return None
    hit = tot = 0
    for perm in permutations(range(n)):
        if abs(spearman(x, [y[i] for i in perm])) >= abs(observed) - 1e-12:
            hit += 1
        tot += 1
    return hit / tot


# ── inference ────────────────────────────────────────────────────────────────

def run(a, items, evaluable):
    import torch
    from rfdetr import RFDETRSegMedium, RFDETRSegNano
    dt = torch.float16 if a.fp16 else torch.float32
    print("[INFO] loading models ...")
    dmg = RFDETRSegMedium(pretrain_weights=a.damage_ckpt,
                          resolution=a.damage_resolution)
    prt = RFDETRSegNano(pretrain_weights=a.parts_ckpt,
                        resolution=a.parts_resolution)
    for nm, m in (("damage", dmg), ("parts", prt)):
        try:
            m.optimize_for_inference(compile=False, batch_size=1, dtype=dt)
        except Exception as e:
            print("[WARN] %s optimize_for_inference failed: %s" % (nm, e))

    rows, coco_dets = [], []
    removed = Counter()
    n_zero_parts = 0
    part_counts, part_class_counts = [], []

    for n, it in enumerate(items, 1):
        img = Image.open(it["path"]).convert("RGB")
        W, H = img.size

        # parts: predict -> dedup -> cap  (this order, see module docstring)
        pdet = prt.predict(img, threshold=a.parts_threshold)
        if pdet is None or len(pdet) == 0 or pdet.mask is None:
            pmasks, pcids = [], []
            n_zero_parts += 1
        else:
            pm = pdet.mask.astype(bool)
            pc = [int(c) for c in pdet.class_id]
            pf = [float(c) for c in pdet.confidence]
            k1 = suppress_duplicate_parts(pm, pc, pf, a.part_nms_iou)
            removed["dedup"] += len(pc) - len(k1)
            keep = apply_caps(k1, pc, pf)
            removed["caps"] += len(k1) - len(keep)
            pmasks = [pm[i] for i in keep]
            pcids = [pc[i] for i in keep]
        # the number of PARTS kept, not the number of distinct part
        # classes: caps allow 2 headlights and 4 tyres, and each of them is
        # separate evidence for the rules
        part_counts.append(len(pcids))
        part_class_counts.append(len(set(pcids)))
        union = np.any(pmasks, axis=0) if pmasks else None

        ddet = dmg.predict(img, threshold=a.infer_threshold)
        preds, dmasks = [], []
        if ddet is not None and len(ddet) and ddet.mask is not None:
            for m, cid, conf in zip(ddet.mask.astype(bool), ddet.class_id,
                                    ddet.confidence):
                cid = int(cid)
                if cid not in evaluable:
                    continue
                area = int(m.sum())
                if area == 0:
                    continue
                # Ignore regions (Paint chip / Missing part / Flaking /
                # Corrosion) mark damage this study cannot name. A prediction
                # lying mostly inside one must count as NEITHER a hit nor an
                # error - but that verdict may only be reached AFTER matching.
                # Deleting it here instead would also delete predictions that
                # would have matched a real, scored instance sitting inside an
                # ignore polygon, leaving that instance in the recall
                # denominator with nothing able to match it. On the real HIL
                # set 51 of 6482 scored instances are fully inside the ignore
                # union, so that is not hypothetical. COCO resolves this by
                # matching first and excusing only unmatched detections; so
                # does image_stats() now. Record the fraction, keep the
                # prediction.
                ig = 0.0
                if it["ignore"] is not None:
                    ig = float(np.logical_and(m, it["ignore"]).sum()) / area
                kf = (float(np.logical_and(m, union).sum()) / area
                      if union is not None else 0.0)
                fr = []
                for pmk, pc_ in zip(pmasks, pcids):
                    inter = int(np.logical_and(m, pmk).sum())
                    if inter:
                        fr.append((pc_, inter / area))
                fr.sort(key=lambda t: -t[1])
                preds.append({"cls": cid, "conf": float(conf), "kf": kf,
                              "fracs": fr, "ig": ig})
                dmasks.append(m)
                if it["image_id"] is not None:
                    rle = maskUtils.encode(np.asfortranarray(m.astype(np.uint8)))
                    rle["counts"] = rle["counts"].decode("utf-8")
                    coco_dets.append({"image_id": it["image_id"],
                                      "cls_name": DAMAGE_CLASSES[cid],
                                      "segmentation": rle,
                                      "score": float(conf)})

        iou = np.zeros((len(preds), len(it["gt_accept"])), np.float32)
        for i, m in enumerate(dmasks):
            pa = int(m.sum())
            for j, g in enumerate(it["gt_masks"]):
                inter = int(np.logical_and(m, g).sum())
                if inter:
                    u = pa + int(g.sum()) - inter
                    iou[i][j] = inter / u if u else 0.0

        rows.append({"preds": preds, "accept": it["gt_accept"], "iou": iou,
                     "n_parts": len(pcids),
                     "n_part_classes": len(set(pcids))})
        if n % 25 == 0:
            print("  ... %d/%d" % (n, len(items)))

    return (rows, coco_dets, dict(removed), n_zero_parts, part_counts,
            part_class_counts)


# ── scoring ──────────────────────────────────────────────────────────────────

def image_stats(rows, a, policy, thr):
    """One record per image at confidence threshold `thr`.

    Bootstrapping resamples these records, so every statistic below is a sum
    of per-image contributions and nothing crosses an image boundary.
    """
    cut = 1.0 - a.unknown_dominance
    ignore_thr = a.ignore_overlap
    out = []
    for r in rows:
        above, keep = [], []
        ignorable = set()
        imp = np.zeros(len(DAMAGE_CLASSES))
        jud = np.zeros(len(DAMAGE_CLASSES))
        abst = np.zeros(len(DAMAGE_CLASSES))
        for i, p in enumerate(r["preds"]):
            if p["conf"] < thr:
                continue
            c = p["cls"]
            t = (i, p["conf"], c)
            above.append(t)
            if p.get("ig", 0.0) >= ignore_thr:
                ignorable.add(i)
            if a.unknown == "reject" and p["kf"] < cut:
                continue
            v = judge(c, p["fracs"], a.overlap_threshold,
                      a.attribution_overlap)[policy]
            if v is None:
                abst[c] += 1
                if a.no_evidence == "keep":
                    keep.append(t)
            elif v is False:
                imp[c] += 1
                jud[c] += 1
            else:
                jud[c] += 1
                keep.append(t)
        tb = greedy_match(above, r["accept"], r["iou"], a.iou_threshold)
        tk = greedy_match(keep, r["accept"], r["iou"], a.iou_threshold)
        dropped = [t for t in above if t not in keep]
        # COCO ignore semantics: a prediction that matched nothing AND lies
        # mostly inside an ignore region is excused - it is neither a hit nor
        # an error, so it leaves the precision denominator. One that DID match
        # a scored instance keeps its true positive; the ignore region does
        # not take it away.
        ig_b = sum(1 for t in above if t[0] in ignorable and t[0] not in tb)
        ig_l = sum(1 for t in keep if t[0] in ignorable and t[0] not in tk)
        out.append({
            "gt": len(r["accept"]),
            "TP_base": len(tb), "FP_base": len(above) - len(tb) - ig_b,
            "TP_layer": len(tk), "FP_layer": len(keep) - len(tk) - ig_l,
            "excused_base": ig_b, "excused_layer": ig_l,
            "impossible": imp, "judged": jud, "abstain": abst,
            # a dropped prediction that was excused is neither a rejected
            # true positive nor a rejected false positive
            "rej_tp": sum(1 for t in dropped if t[0] in tb),
            "rej_fp": sum(1 for t in dropped
                          if t[0] not in tb and t[0] not in ignorable)})
    return out


def pack(st):
    """Per-image records -> arrays, so the bootstrap can be vectorised."""
    A = {k: np.array([r[k] for r in st], float)
         for k in ("gt", "TP_base", "FP_base", "TP_layer", "FP_layer",
                   "rej_tp", "rej_fp", "excused_base", "excused_layer")}
    for k in ("impossible", "judged", "abstain"):
        A[k] = np.array([r[k] for r in st], float)      # (n_images, n_classes)
    A["n"] = len(st)
    return A


def _derive(g, tb, fb, tl, fl):
    """Precision / recall / F1 for both arms from summed counts."""
    with np.errstate(divide="ignore", invalid="ignore"):
        pb = np.where(tb + fb > 0, tb / np.where(tb + fb > 0, tb + fb, 1), 0.0)
        pl = np.where(tl + fl > 0, tl / np.where(tl + fl > 0, tl + fl, 1), 0.0)
        rb = np.where(g > 0, tb / np.where(g > 0, g, 1), 0.0)
        rl = np.where(g > 0, tl / np.where(g > 0, g, 1), 0.0)
        fb1 = np.where(pb + rb > 0, 2 * pb * rb / np.where(pb + rb > 0, pb + rb, 1), 0.0)
        fl1 = np.where(pl + rl > 0, 2 * pl * rl / np.where(pl + rl > 0, pl + rl, 1), 0.0)
        cut = np.where(fb > 0, 100.0 * (fb - fl) / np.where(fb > 0, fb, 1), 0.0)
    return pb, pl, rb, rl, fb1, fl1, cut


def point(A):
    """Point estimate over every image."""
    tot = {k: A[k].sum(0) for k in
           ("gt", "TP_base", "FP_base", "TP_layer", "FP_layer",
            "rej_tp", "rej_fp", "excused_base", "excused_layer",
            "impossible", "judged", "abstain")}
    pb, pl, rb, rl, f1b, f1l, cut = _derive(
        *(np.array([tot[k]]) for k in
          ("gt", "TP_base", "FP_base", "TP_layer", "FP_layer")))
    out = {k: tot[k] for k in tot}
    out.update({"P_base": pb[0], "P_layer": pl[0],
                "R_base": rb[0], "R_layer": rl[0],
                "F1_base": f1b[0], "F1_layer": f1l[0],
                "dP": pl[0] - pb[0], "dR": rl[0] - rb[0],
                "dF1": f1l[0] - f1b[0], "FP_cut_pct": cut[0]})
    return out


def boot(A, idx):
    """Paired bootstrap over IMAGES, on a caller-supplied resample matrix.

    Both arms are recomputed on the same resample, so every delta below is
    paired. Predictions inside one photograph share a parts segmentation and a
    viewpoint and are not independent, which is why the image is the unit.

    idx is built ONCE and reused for every policy and every threshold. Drawing
    a fresh set per row would leave the four policies bootstrapped on
    different resamples, so a difference between two policies could not be
    read as a difference in the policies.
    """
    g = A["gt"][idx].sum(1)
    tb, fb = A["TP_base"][idx].sum(1), A["FP_base"][idx].sum(1)
    tl, fl = A["TP_layer"][idx].sum(1), A["FP_layer"][idx].sum(1)
    pb, pl, rb, rl, f1b, f1l, cut = _derive(g, tb, fb, tl, fl)
    rtp, rfp = A["rej_tp"][idx].sum(1), A["rej_fp"][idx].sum(1)
    tt = rtp + rfp
    out = {"dP": pl - pb, "dR": rl - rb, "dF1": f1l - f1b, "FP_cut_pct": cut,
           "rej_prec": np.where(tt > 0, rfp / np.where(tt > 0, tt, 1), np.nan),
           "rate": []}
    for c in range(A["impossible"].shape[1]):
        im = A["impossible"][:, c][idx].sum(1)
        ju = A["judged"][:, c][idx].sum(1)
        out["rate"].append(
            np.where(ju > 0, 100.0 * im / np.where(ju > 0, ju, 1), np.nan))
    return out


def per_class_ap(coco, dets, a, scored_ids):
    """COCOeval per category, with CarDD's own area ranges."""
    if coco is None or not dets:
        return {}
    name_to_cat = {c["name"]: c["id"] for c in coco.dataset["categories"]}
    res = [{"image_id": d["image_id"], "category_id": name_to_cat[d["cls_name"]],
            "segmentation": d["segmentation"], "score": d["score"]}
           for d in dets if d["cls_name"] in name_to_cat]
    if not res:
        return {}
    out = {}
    # Every image that was SCORED, not only those that produced a detection.
    # COCOeval filters ground truth by params.imgIds, so restricting this to
    # images with detections would delete the ground truth of any image the
    # model was silent on - turning a total miss into a free pass and
    # inflating AP.
    img_ids = sorted(scored_ids)
    # loadRes/evaluate/accumulate/summarize between them print ~20 lines per
    # category, which would bury the table this function feeds. Keep the
    # numbers, drop the noise.
    quiet = lambda: contextlib.redirect_stdout(io.StringIO())  # noqa: E731
    with quiet():
        dt = coco.loadRes(res)
    for nm, cid in name_to_cat.items():
        if nm not in DAMAGE_CLASSES:
            continue
        ev = COCOeval(coco, dt, "segm")
        ev.params.areaRng = CARDD_AREA_RNG
        ev.params.imgIds = img_ids
        ev.params.catIds = [cid]
        with quiet():
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        v = float(ev.stats[0])
        # pycocotools returns -1 for a category with no ground truth among the
        # evaluated images. That means "undefined", NOT "the model scored
        # zero" - rounding it would publish -100.0 and poison any mean taken
        # over these values. Carry it as None instead.
        out[nm] = None if v == -1 else round(v * 100.0, 2)
    return out


def cardd_support(coco, scored_ids):
    """Ground-truth instances per damage class over the images actually
    scored - not over the whole annotation file. --n_images, a missing image
    file or a decode failure all shrink the scored set, and a support column
    that ignored them would describe a different dataset from every other
    number printed beside it."""
    keep = set(scored_ids)
    id2n = {c["id"]: c["name"] for c in coco.dataset["categories"]}
    sup = Counter()
    for an in coco.dataset["annotations"]:
        if an["image_id"] not in keep:
            continue
        nm = id2n.get(an["category_id"])
        if nm in DAMAGE_CLASSES:
            sup[nm] += 1
    return {k: int(sup.get(k, 0)) for k in DAMAGE_CLASSES}


def restrictiveness(dcid):
    """Fraction of the 19 part classes this rule forbids."""
    allowed, excluded = ALLOWED.get(dcid), EXCLUDED.get(dcid, set())
    ok = sum(1 for pid in range(len(PART_CLASSES))
             if pid not in excluded and (allowed is None or pid in allowed))
    return 1.0 - ok / len(PART_CLASSES)


# ── the recall cost, instance by instance ────────────────────────────────────
#
# Across the sweep the layer's rejected FALSE positives collapse while its
# rejected TRUE positives barely move, so a small, stable core of correct
# detections is ruled impossible at every threshold. A rejected true positive
# means the DAMAGE model was right - the prediction matched real ground truth
# at IoU >= 0.50 with the correct class - and the PARTS model disagreed. Two
# models, and only one of them has to be wrong. These functions name which, per
# instance, so the recall cost in TABLE 1 is evidence rather than inference.
#
# This runs on the predictions already in memory, so it costs no extra
# inference pass.

SELF = sys.modules[__name__]


def dominant_part(fracs, overlap_thr):
    """The part carrying the largest overlap above the bar, or None."""
    above = [(p, f) for p, f in fracs if f >= overlap_thr]
    if not above:
        return None
    return max(above, key=lambda t: t[1])


def audit_rejected_tp(a, rows, items, thr, policy):
    """Return one record per rejected TRUE positive."""
    out = []
    for it, r in zip(items, rows):
        above = [(i, p["conf"], p["cls"]) for i, p in enumerate(r["preds"])
                 if p["conf"] >= thr]
        if not above:
            continue
        tp_idx = greedy_match(above, r["accept"], r["iou"], a.iou_threshold)
        for i, conf, c in above:
            verdict = judge(c, r["preds"][i]["fracs"], a.overlap_threshold,
                            a.attribution_overlap)[policy]
            if verdict is not False or i not in tp_idx:
                continue                      # only rejected TRUE positives
            # best IoU against a ground truth this class may match
            best = 0.0
            for j, acc in enumerate(r["accept"]):
                if c in acc and len(r["iou"]):
                    best = max(best, float(r["iou"][i][j]))
            fr = sorted(r["preds"][i]["fracs"], key=lambda t: -t[1])
            dom = dominant_part(fr, a.overlap_threshold)
            out.append({
                "image": Path(it["path"]).name,
                "damage": DAMAGE_CLASSES[c],
                "damage_id": c,
                "confidence": round(float(conf), 4),
                "iou_with_gt": round(best, 4),
                "known_fraction": round(float(r["preds"][i].get("kf", 0.0)), 4),
                "parts_beneath": [(PART_CLASSES[p], round(f, 4)) for p, f in fr],
                "dominant_part": (PART_CLASSES[dom[0]] if dom else None),
                "dominant_fraction": (round(dom[1], 4) if dom else None),
                "dominant_is_forbidden": ((not part_ok(dom[0], c))
                                          if dom else None)})
    return out


def audit_without_scratch_on_glass(a, rows, items, thr, policy):
    """How many rejected true positives survive if scratch-on-glass is allowed?

    A windscreen scratched by a wiper blade is an ordinary event, so
    EXCLUDED[SCRATCH] = GLASS_PARTS is the one entry in the rule table that is
    anatomically arguable. Temporarily relax it and re-audit, so the cost of
    keeping it is measured rather than argued.
    """
    original = EXCLUDED[SCRATCH]
    try:
        EXCLUDED[SCRATCH] = set()
        return audit_rejected_tp(a, rows, items, thr, policy)
    finally:
        EXCLUDED[SCRATCH] = original


def print_rejected_tp(rec, cf, a):
    """TABLE 4 - the three views of the layer's recall cost."""
    print("\n" + "-" * 100)
    print("  TABLE 4  the recall cost: every CORRECT detection the layer "
          "rejected")
    print("           policy '%s', confidence %.3f   (%d rows)"
          % (a.policy, a.operating_threshold, len(rec)))
    print("-" * 100)
    if not rec:
        print("    none - the layer destroyed no true positives at this "
              "operating point.")
    else:
        print("    %-30s %-14s %5s %6s  %s"
              % ("image", "damage", "conf", "IoU", "parts beneath (fraction)"))
        for d in sorted(rec, key=lambda x: (x["damage"], -x["confidence"])):
            parts = ", ".join("%s %.2f" % (n, f)
                              for n, f in d["parts_beneath"][:4]) or "(none)"
            print("    %-30s %-14s %5.3f %6.3f  %s"
                  % (d["image"][:30], d["damage"], d["confidence"],
                     d["iou_with_gt"], parts))

    by_part = Counter((d["dominant_part"] or "(no part above the bar)",
                       d["damage"]) for d in rec)
    if by_part:
        print("\n    grouped by the part that carried the verdict:")
        print("    %-24s %-14s %6s" % ("dominant part", "damage", "count"))
        for (p, dmg), n in by_part.most_common():
            print("    %-24s %-14s %6d" % (p, dmg, n))
        print("\n    Reading it: a row whose dominant part is NOT the part a")
        print("    human would name is a PARTS-MODEL error - the damage model")
        print("    was right and the rules were misled. A row whose dominant")
        print("    part is correct and still forbidden is a RULE that does not")
        print("    hold in reality.")

    print("\n    counterfactual - allow scratches on glass:")
    print("      rejected true positives, rules as written : %d" % len(rec))
    print("      ... with EXCLUDED[scratch] relaxed        : %d" % len(cf))
    print("      recovered by relaxing that one rule       : %d"
          % (len(rec) - len(cf)))
    return by_part



# ═══════════════════════════════════════════════════════════════════════════
# SELF-CHECK SUITE  (--self_check)
# ═══════════════════════════════════════════════════════════════════════════
# Every number this script prints was produced by code that had been checked
# against nothing except its own output. A silent error in a class index, a
# mask decode or the matching loop would produce results that look entirely
# reasonable and are entirely wrong. These groups check the parts that could
# fail quietly, and print PASS or FAIL for each.
#
#   A  class index spaces      does the annotation file's category order really
#                              match the constants the rules assume?
#   B  ground truth decoding   do decoded masks agree with the stored areas and
#                              the actual image dimensions?
#   C  rule logic              exhaustive over synthetic part contexts
#   D  matching                hand-built cases with known answers
#   E  parts pre-processing    dedup-then-cap, including the case the ordering
#                              exists to protect
#   F  statistics              bootstrap, Spearman, exact permutation p
#   G  result consistency      arithmetic invariants inside the JSONs already
#                              produced - catches bookkeeping errors end to end
#   H  models (optional)       class-id ranges and mask geometry; needs the GPU

_HERE = Path(__file__).resolve().parent
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print("  [%s] %-52s %s" % ("PASS" if ok else "FAIL", name, detail))
    return ok


# ── A. class index spaces ────────────────────────────────────────────────────

def group_a(a, SELF):
    print("\nA. CLASS INDEX SPACES")

    try:
        coco = COCO(a.annotations)
    except Exception as e:
        check("A1 CarDD annotations load", False, str(e))
        return
    names = sorted(c["name"] for c in coco.dataset["categories"])
    check("A1 CarDD categories sorted == DAMAGE_CLASSES",
          names == SELF.DAMAGE_CLASSES,
          "file=%s" % names)

    # the constants must equal the position in the list, or every rule shifts
    consts = [("crack", SELF.CRACK), ("dent", SELF.DENT),
              ("glass shatter", SELF.GLASS_SHATTER),
              ("lamp broken", SELF.LAMP_BROKEN), ("scratch", SELF.SCRATCH),
              ("tire flat", SELF.TIRE_FLAT)]
    ok = all(SELF.DAMAGE_CLASSES[i] == n for n, i in consts)
    check("A2 damage class constants match list order", ok,
          "" if ok else str(consts))

    check("A3 PART_CLASSES has 19 entries", len(SELF.PART_CLASSES) == 19,
          "n=%d" % len(SELF.PART_CLASSES))

    glass = {SELF.PART_CLASSES[i] for i in SELF.GLASS_PARTS}
    lamp = {SELF.PART_CLASSES[i] for i in SELF.LAMP_PARTS}
    tyre = {SELF.PART_CLASSES[i] for i in SELF.TYRE_PARTS}
    check("A4 GLASS_PARTS index glass names",
          all("Glass" in n for n in glass), str(sorted(glass)))
    check("A5 LAMP_PARTS index lamp names",
          lamp == {"Headlight", "Taillight"}, str(sorted(lamp)))
    check("A6 TYRE_PARTS index tyre", tyre == {"tyre"}, str(sorted(tyre)))
    check("A7 PART_CLASSES is sorted (0-based enumeration assumption)",
          SELF.PART_CLASSES == sorted(SELF.PART_CLASSES),
          "" if SELF.PART_CLASSES == sorted(SELF.PART_CLASSES) else "NOT SORTED")

    # HIL: are the published folders really swapped?
    dmg_root, prt_root = Path(a.hil_damage_root), Path(a.hil_parts_root)
    for label, root, expect in [("A8 'Car parts dataset' holds DAMAGE labels",
                                 dmg_root, {"Scratch", "Dent"}),
                                ("A9 'Car damages dataset' holds PART labels",
                                 prt_root, None)]:
        files = sorted((root / "ann").glob("*.json"))[:20] if (root / "ann").exists() else []
        if not files:
            check(label, False, "no ann/*.json under %s" % root)
            continue
        titles = Counter()
        for f in files:
            for o in json.load(open(f, encoding="utf-8")).get("objects", []):
                titles[o.get("classTitle")] += 1
        got = set(titles)
        if expect is None:
            ok = not (got & {"Scratch", "Dent", "Cracked", "Broken part"})
        else:
            ok = bool(expect & got)
        check(label, ok, "titles=%s" % sorted(got)[:6])

    # every HIL title must be handled: mapped, ignored, or "Broken part"
    files = sorted((dmg_root / "ann").glob("*.json"))
    if files:
        titles = Counter()
        for f in files[:200]:
            for o in json.load(open(f, encoding="utf-8")).get("objects", []):
                titles[o.get("classTitle")] += 1
        handled = set(SELF.HIL_DIRECT) | SELF.HIL_IGNORE | {"Broken part"}
        unhandled = set(titles) - handled
        check("A10 every HIL class title is mapped or ignored",
              not unhandled, "unhandled=%s" % sorted(unhandled))


# ── B. ground truth decoding ─────────────────────────────────────────────────

def group_b(a, SELF):
    print("\nB. GROUND-TRUTH DECODING")
    try:
        coco = COCO(a.annotations)
    except Exception as e:
        check("B0 annotations load", False, str(e))
        return

    ratios, dim_ok, n = [], True, 0
    img_dir = Path(a.images_dir)
    for iid in list(coco.imgs.keys())[:a.n_sample]:
        info = coco.imgs[iid]
        H, W = info["height"], info["width"]
        fp = img_dir / Path(info["file_name"]).name
        if fp.exists():
            with Image.open(fp) as im:
                if im.size != (W, H):
                    dim_ok = False
        for an in coco.imgToAnns.get(iid, []):
            try:
                m = SELF.decode_coco(an, H, W)
            except Exception:
                continue
            stored = float(an.get("area", 0))
            if stored > 0:
                ratios.append(int(m.sum()) / stored)
                n += 1
    if ratios:
        med = float(np.median(ratios))
        # A ratio far from 1 means the annotations were resized without their
        # area field being recomputed - which silently breaks every size bin.
        check("B1 decoded mask area == stored COCO area", 0.9 <= med <= 1.1,
              "median ratio %.4f over %d instances" % (med, n))
    else:
        check("B1 decoded mask area == stored COCO area", False, "no instances")
    check("B2 annotation image dims == actual image files", dim_ok)

    # polygon rasteriser: 11x11 square (PIL polygon fill is inclusive) minus a
    # 4x4 hole = 121 - 16 = 105
    m = SELF.poly_to_mask({"exterior": [[0, 0], [10, 0], [10, 10], [0, 10]],
                          "interior": [[[2, 2], [5, 2], [5, 5], [2, 5]]]}, 20, 20)
    check("B3 poly_to_mask square-with-hole area", int(m.sum()) == 105,
          "area=%d expected 105" % int(m.sum()))
    check("B4 poly_to_mask rejects degenerate polygons",
          SELF.poly_to_mask({"exterior": [[0, 0], [1, 1]]}, 10, 10) is None)

    # HIL size field vs the real image
    root = Path(a.hil_damage_root)
    files = sorted((root / "ann").glob("*.json"))[:a.n_sample]
    bad = 0
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        fp = root / "img" / f.name[:-5]
        if not fp.exists():
            continue
        with Image.open(fp) as im:
            if im.size != (d["size"]["width"], d["size"]["height"]):
                bad += 1
    check("B5 HIL size field == actual image dims", bad == 0,
          "%d/%d mismatched" % (bad, len(files)))


# ── C. rule logic ────────────────────────────────────────────────────────────

def group_c(SELF):
    print("\nC. RULE LOGIC")
    GL, LA, TY, BUMP, DOOR = 6, 8, 18, 3, 4

    # no part evidence must ABSTAIN, never reject - the bug this design exists
    # to prevent
    ok = True
    for c in range(6):
        for pol, v in SELF.judge(c, [], 0.20, 0.15).items():
            if c == SELF.CRACK:
                ok &= (v is True)
            else:
                ok &= (v is None)
    check("C1 empty part context abstains for every class/policy", ok)

    # overlap below threshold is no evidence for ANY policy. dominant used to
    # be the exception: it took the max over all fractions, so a sliver on a
    # forbidden part could reject a prediction that every other policy
    # abstained on. C2b pins that down.
    v = SELF.judge(SELF.GLASS_SHATTER, [(GL, 0.05)], 0.20, 0.15)
    check("C2 sub-threshold overlap abstains for set/majority",
          v["set"] is None and v["majority"] is None, str(v))

    v = SELF.judge(SELF.DENT, [(GL, 0.10), (BUMP, 0.05)], 0.20, 0.15)
    check("C2b no policy rejects on sub-threshold evidence alone",
          all(x is None for x in v.values()), str(v))

    v = SELF.judge(SELF.DENT, [(GL, 0.30), (BUMP, 0.25)], 0.20, 0.15)
    check("C2c dominant still rejects when the forbidden part clears the bar",
          v["dominant"] is False, str(v))

    check("C3 crack is always plausible",
          all(all(x is True for x in SELF.judge(SELF.CRACK, f, 0.2, 0.15).values())
              for f in ([], [(GL, 1.0)], [(TY, 0.9)], [(BUMP, 0.5)])))

    v = SELF.judge(SELF.GLASS_SHATTER, [(BUMP, 0.9)], 0.20, 0.15)
    check("C4 glass shatter on a bumper is impossible (all policies)",
          all(x is False for x in v.values()), str(v))

    v = SELF.judge(SELF.LAMP_BROKEN, [(LA, 0.75), (BUMP, 0.25)], 0.20, 0.15)
    check("C5 lamp 75/25 lamp+bumper: set rejects, majority accepts",
          v["set"] is False and v["majority"] is True, str(v))

    v = SELF.judge(SELF.LAMP_BROKEN, [(LA, 0.25), (BUMP, 0.75)], 0.20, 0.15)
    check("C6 lamp 25/75 lamp+bumper: majority rejects",
          v["majority"] is False, str(v))

    v = SELF.judge(SELF.SCRATCH, [(DOOR, 0.85), (GL, 0.15)], 0.20, 0.15)
    check("C7 scratch on door grazing glass stays plausible",
          v["set"] is True and v["majority"] is True, str(v))

    check("C8 part_ok agrees with ALLOWED/EXCLUDED", all(
        SELF.part_ok(p, c) == (p not in SELF.EXCLUDED.get(c, set())
                              and (SELF.ALLOWED.get(c) is None
                                   or p in SELF.ALLOWED[c]))
        for c in range(6) for p in range(19)))

    r = {SELF.DAMAGE_CLASSES[i]: round(SELF.restrictiveness(i), 3)
         for i in range(6)}
    check("C9 restrictiveness: crack 0, tire flat highest",
          r["crack"] == 0.0 and r["tire flat"] == max(r.values()), str(r))

    # a policy must only ever return one of exactly three values
    vals = set()
    rng = np.random.default_rng(0)
    for _ in range(400):
        c = int(rng.integers(0, 6))
        k = int(rng.integers(0, 4))
        fr = [(int(rng.integers(0, 19)), float(rng.random())) for _ in range(k)]
        vals |= set(SELF.judge(c, fr, 0.20, 0.15).values())
    check("C10 judge only ever returns True/False/None",
          vals <= {True, False, None}, str(vals))


# ── D. matching ──────────────────────────────────────────────────────────────

def group_d(SELF):
    print("\nD. MATCHING")
    iou = np.array([[0.90, 0.10], [0.20, 0.80]])

    check("D1 both detections match their own gt",
          SELF.greedy_match([(0, 0.9, 4), (1, 0.8, 1)], [{4}, {1}], iou, 0.5)
          == {0, 1})
    check("D2 wrong class never matches",
          SELF.greedy_match([(0, 0.9, 1)], [{4}], iou, 0.5) == set())
    check("D3 IoU below threshold never matches",
          SELF.greedy_match([(0, 0.9, 4)], [{4}], np.array([[0.30]]), 0.5) == set())

    # one ground truth can be claimed only once, by the higher-scoring mask
    i2 = np.array([[0.80], [0.90]])
    tp = SELF.greedy_match([(0, 0.9, 2), (1, 0.5, 2)], [{2}], i2, 0.5)
    check("D4 a gt is claimed once, by the higher score", tp == {0}, str(tp))

    # and if that blocker is removed, the remaining mask may claim it - this is
    # how the layer can RAISE recall, so it has to work
    tp2 = SELF.greedy_match([(1, 0.5, 2)], [{2}], i2, 0.5)
    check("D5 removing the blocker frees the gt for a lower-scoring mask",
          tp2 == {1}, str(tp2))

    check("D6 superclass accept-set matches either class",
          SELF.greedy_match([(0, 0.9, 2)], [{2, 3}], np.array([[0.9]]), 0.5) == {0}
          and SELF.greedy_match([(0, 0.9, 3)], [{2, 3}], np.array([[0.9]]), 0.5) == {0})

    check("D7 empty inputs are safe",
          SELF.greedy_match([], [{0}], np.zeros((0, 1)), 0.5) == set()
          and SELF.greedy_match([(0, 0.9, 0)], [], np.zeros((1, 0)), 0.5) == set())


# ── E. parts pre-processing ──────────────────────────────────────────────────

def group_e(SELF):
    print("\nE. PARTS PRE-PROCESSING")

    def box(x0, y0, x1, y1, H=40, W=40):
        m = np.zeros((H, W), bool)
        m[y0:y1, x0:x1] = True
        return m

    # two heavily overlapping masks of the SAME class -> one survives
    masks = [box(0, 0, 20, 20), box(1, 1, 21, 21)]
    keep = SELF.suppress_duplicate_parts(masks, [8, 8], [0.9, 0.8], 0.30)
    check("E1 dedup removes a same-class duplicate", len(keep) == 1 and keep == [0],
          str(keep))

    # same geometry, DIFFERENT classes -> both survive (nested parts are normal)
    keep = SELF.suppress_duplicate_parts(masks, [8, 17], [0.9, 0.8], 0.30)
    check("E2 dedup never merges different classes", len(keep) == 2, str(keep))

    # disjoint masks of the same class -> both survive
    keep = SELF.suppress_duplicate_parts([box(0, 0, 10, 10), box(25, 25, 35, 35)],
                                        [8, 8], [0.9, 0.6], 0.30)
    check("E3 dedup keeps spatially distinct same-class parts",
          len(keep) == 2, str(keep))

    # THE CASE THE ORDERING EXISTS FOR: two boxes on headlight #1 (0.90, 0.85)
    # and a genuine headlight #2 (0.60). Cap is 2.
    masks = [box(0, 0, 10, 10), box(1, 1, 11, 11), box(25, 25, 35, 35)]
    cids, confs = [8, 8, 8], [0.90, 0.85, 0.60]
    capped_first = SELF.apply_caps(list(range(3)), cids, confs)
    k1 = SELF.suppress_duplicate_parts(masks, cids, confs, 0.30)
    correct = SELF.apply_caps(k1, cids, confs)
    check("E4 cap-then-dedup WOULD drop the real second headlight",
          2 not in capped_first, "caps-first kept %s" % capped_first)
    check("E5 dedup-then-cap keeps the real second headlight",
          2 in correct and len(correct) == 2, "kept %s" % correct)

    idx = list(range(6))
    check("E6 caps respect the per-class limit",
          len(SELF.apply_caps(idx, [6] * 6, [0.9] * 6)) == 1        # windshield
          and len(SELF.apply_caps(idx, [8] * 6, [0.9] * 6)) == 2    # headlight
          and len(SELF.apply_caps(idx, [18] * 6, [0.9] * 6)) == 4)  # tyre
    check("E7 CAP_BY_ID covers every part class",
          all(i in SELF.CAP_BY_ID for i in range(19)))


# ── F. statistics ────────────────────────────────────────────────────────────

def group_f(SELF):
    print("\nF. STATISTICS")
    rng = np.random.default_rng(0)
    x = rng.normal(5.0, 1.0, 400)
    boots = [float(np.mean(rng.choice(x, len(x)))) for _ in range(2000)]
    lo, hi = SELF.pct_ci(boots)
    check("F1 bootstrap CI brackets the point estimate",
          lo <= float(np.mean(x)) <= hi,
          "mean %.4f in [%.4f, %.4f]" % (float(np.mean(x)), lo, hi))

    r1 = np.random.default_rng(7).integers(0, 100, 50)
    r2 = np.random.default_rng(7).integers(0, 100, 50)
    check("F2 seeded resampling is deterministic", np.array_equal(r1, r2))

    check("F3 Spearman rho is +1 / -1 on monotone data",
          abs(SELF.spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1.0) < 1e-9
          and abs(SELF.spearman([1, 2, 3, 4], [40, 30, 20, 10]) + 1.0) < 1e-9)

    p = SELF.exact_perm_p([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], 1.0)
    check("F4 exact permutation p at rho=1, n=6 equals 2/720",
          abs(p - 2.0 / 720.0) < 1e-12, "p=%.6f" % p)

    check("F5 permutation p is never below 2/n!",
          SELF.exact_perm_p([1, 2, 3], [1, 2, 3], 1.0) >= 2.0 / 6.0 - 1e-12)

    check("F6 pct_ci survives empty / NaN input",
          all(math.isnan(v) for v in SELF.pct_ci([]))
          and all(math.isnan(v) for v in SELF.pct_ci([float("nan")])))


# ── G. result consistency ────────────────────────────────────────────────────

def group_g():
    print("\nG. RESULT CONSISTENCY (checks the JSONs already produced)")
    res = _HERE / "results"

    for ds in ("cardd", "hil"):
        fn = res / ("plausibility_%s.json" % ds)
        if not fn.exists():
            check("G0 plausibility_%s.json present" % ds, False, "not found")
            continue
        d = json.load(open(fn))
        for pol, v in d["policies"].items():
            tag = "%s/%s" % (ds, pol)
            if "TP_base" not in v:
                check("G1 %s has raw counts" % tag, False,
                      "re-run the study - counts were added after this file")
                continue
            tb, fb = v["TP_base"], v["FP_base"]
            tl, fl = v["TP_layer"], v["FP_layer"]
            rtp, rfp = v["rejected_tp"], v["rejected_fp"]
            # everything kept = everything above threshold minus what was cut
            check("G1 %s kept == above - rejected" % tag,
                  (tl + fl) == (tb + fb) - rtp - rfp,
                  "%d vs %d-%d" % (tl + fl, tb + fb, rtp + rfp))
            # rejecting can only ever free a gt, never lose one it did not touch
            check("G2 %s TP change is bounded by rejections" % tag,
                  -rtp <= (tl - tb) <= rfp,
                  "dTP=%d rejTP=%d rejFP=%d" % (tl - tb, rtp, rfp))
            for lab, tp, fp, rep in [("base", tb, fb, v["precision_base"]),
                                     ("layer", tl, fl, v["precision_layer"])]:
                p = tp / (tp + fp) if (tp + fp) else 0.0
                check("G3 %s %s precision recomputes" % (tag, lab),
                      abs(p - rep) < 5e-4, "%.5f vs %.5f" % (p, rep))
            check("G4 %s recall recomputes" % tag,
                  abs(tb / v["gt"] - v["recall_base"]) < 5e-4
                  and abs(tl / v["gt"] - v["recall_layer"]) < 5e-4)
            check("G5 %s FP cut %% recomputes" % tag,
                  abs(100.0 * (fb - fl) / fb - v["FP_cut_pct"]) < 0.01)
            for k in ("d_precision", "d_recall", "d_f1", "FP_cut_pct"):
                lo, hi = v[k + "_ci"] if k + "_ci" in v else v["FP_cut_ci"]
                check("G6 %s CI brackets %s" % (tag, k),
                      lo - 1e-6 <= v[k] <= hi + 1e-6,
                      "%.5f not in [%.5f, %.5f]" % (v[k], lo, hi))
            pc = v["per_class"]
            check("G7 %s per-class events sum is consistent" % tag,
                  all(r["events"] <= r["judged"] for r in pc.values()))
            check("G8 %s per-class rate recomputes" % tag,
                  all(abs(100.0 * r["events"] / r["judged"] - r["rate"]) < 0.01
                      for r in pc.values() if r["judged"]))
            check("G9 %s rejection precision recomputes" % tag,
                  v["rejection_precision"] is None
                  or abs(rfp / (rfp + rtp) - v["rejection_precision"]) < 5e-4)



# ── H. models (optional) ─────────────────────────────────────────────────────

def group_h(a, SELF):
    print("\nH. MODELS (requires GPU)")
    try:
        import torch  # noqa: F401
        from rfdetr import RFDETRSegMedium, RFDETRSegNano
    except Exception as e:
        check("H0 rfdetr/torch import", False, str(e))
        return
    img_dir = Path(a.images_dir)
    files = sorted(p for p in img_dir.glob("*.jpg"))[:12]
    if not files:
        check("H0 sample images found", False, str(img_dir))
        return

    prt = RFDETRSegNano(pretrain_weights=a.parts_ckpt, resolution=960)
    dmg = RFDETRSegMedium(pretrain_weights=a.damage_ckpt, resolution=960)

    bad_p = bad_d = bad_shape = 0
    portrait_seen = False
    for fp in files:
        img = Image.open(fp).convert("RGB")
        W, H = img.size
        if H > W:
            portrait_seen = True
        pd = prt.predict(img, threshold=0.30)
        if pd is not None and len(pd) and pd.mask is not None:
            if any(not (0 <= int(c) < 19) for c in pd.class_id):
                bad_p += 1
            if pd.mask.shape[1:] != (H, W):
                bad_shape += 1
        dd = dmg.predict(img, threshold=0.30)
        if dd is not None and len(dd) and dd.mask is not None:
            if any(not (0 <= int(c) < 6) for c in dd.class_id):
                bad_d += 1
            if dd.mask.shape[1:] != (H, W):
                bad_shape += 1

    check("H1 parts class ids all within [0,19)", bad_p == 0,
          "%d images out of range" % bad_p)
    check("H2 damage class ids all within [0,6)", bad_d == 0,
          "%d images out of range" % bad_d)
    check("H3 masks returned at the input image size", bad_shape == 0,
          "%d shape mismatches%s" % (bad_shape,
                                     "" if portrait_seen
                                     else "  (NOTE: no portrait image in sample)"))



def run_self_check(a):
    """Groups A-G (plus H with --with_models), then write self_check.json."""
    # the study calls the HIL damage root --data_root; group A needs both roots
    # so it can prove the published folders really are swapped
    a.hil_damage_root = a.data_root
    print("=" * 78)
    print("SELF-CHECK  -  physical_plausibility_layer.py")
    print("=" * 78)

    group_a(a, SELF)
    group_b(a, SELF)
    group_c(SELF)
    group_d(SELF)
    group_e(SELF)
    group_f(SELF)
    group_g()
    if a.with_models:
        group_h(a, SELF)
    else:
        print("\nH. MODELS  -  skipped (pass --with_models to run)")

    n = len(RESULTS)
    bad = [r for r in RESULTS if not r[1]]
    print("\n" + "=" * 78)
    print("%d checks, %d passed, %d FAILED" % (n, n - len(bad), len(bad)))
    if bad:
        print("\nFAILURES - do not report results until these are understood:")
        for name, _, detail in bad:
            print("  - %-50s %s" % (name, detail))
    print("=" * 78 + "\n")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json.dump([{"check": n2, "pass": ok, "detail": d}
               for n2, ok, d in RESULTS],
              open(out / "self_check.json", "w"), indent=2)
    return len(bad)



def shorten_path(v):
    """An absolute path reduced to its last two components.

    The result JSONs are published, and a machine-specific absolute path adds
    nothing a reader can use while exposing the author's directory layout. Two
    trailing components keep everything that identifies the artefact - the
    checkpoint filename, the split directory, the HIL folder whose name is the
    evidence for the swap - and drop the rest.
    """
    if not isinstance(v, str) or len(v) < 3:
        return v
    drive = v[1] == ":" and (v[2] == "/" or v[2] == chr(92))
    if not (drive or v[0] == "/"):
        return v                                   # already relative: leave it
    parts = [q for q in v.replace(chr(92), "/").split("/") if q]
    if "." in parts[-1][1:]:
        return parts[-1]        # a file: its name is what identifies it
    return ".../" + "/".join(parts[-2:]) if len(parts) > 2 else v


def shorten_paths(obj):
    """shorten_path applied to every string in a nested structure."""
    if isinstance(obj, dict):
        return {k: shorten_paths(x) for k, x in obj.items()}
    if isinstance(obj, list):
        return [shorten_paths(x) for x in obj]
    return shorten_path(obj)



def main():
    a = parse_args()
    if a.self_check:
        sys.exit(1 if run_self_check(a) else 0)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    items, evaluable, coco, desc = (load_cardd(a) if a.dataset == "cardd"
                                    else load_hil(a))

    sweep = sorted({float(x) for x in a.sweep.split(",") if x.strip()}
                   | {float(a.operating_threshold)})
    sweep_keys = thr_keys(sweep)

    print("\n" + "=" * 136)
    print("PHYSICAL-PLAUSIBILITY VALIDATION  |  dataset: %s (%s)"
          % (desc["name"], desc["split"]))
    print("=" * 136)
    print("  images scored          : %d   (listed %d, %d not found on disk)"
          % (desc["images_scored"], desc["images_listed"],
             desc["images_missing_on_disk"]))
    print("  ground-truth instances : %d" % desc["gt_instances_scored"])
    if desc.get("gt_annotations_that_failed_to_decode"):
        print("  WARNING: %d ground-truth annotations failed to decode and are"
              % desc["gt_annotations_that_failed_to_decode"])
        print("           excluded from every number below.")
    print("  classes evaluated      : %s (%d of %d)"
          % (", ".join(desc["classes_evaluated"]),
             len(desc["classes_evaluated"]), len(DAMAGE_CLASSES)))
    if desc["classes_not_evaluated"]:
        print("  classes NOT evaluated  : %s"
              % ", ".join(desc["classes_not_evaluated"]))
    print("  ground truth per class :")
    for k, v in desc["gt_instances_per_class"].items():
        print("        %-30s %6d" % (k, v))
    if a.dataset != "cardd":
        print("  source label -> class  :")
        for k, v in desc["label_map"].items():
            print("        %-16s -> %s" % (k, ", ".join(v)))
    if desc["ignored_labels"]:
        print("  labels held as IGNORE  : %s" % ", ".join(desc["ignored_labels"]))
    for note in desc["notes"]:
        print("  note: %s" % note)

    print("\n  policy (primary)       : %s   [all %d policies reported below]"
          % (a.policy, len(POLICIES)))
    print("  confidence sweep       : %s"
          % ", ".join("%.3f" % t for t in sweep))
    print("  abstain handling       : unknown=%s  no_evidence=%s"
          % (a.unknown, a.no_evidence))

    rows, dets, removed, n_zero, part_counts, part_cls_counts = run(
        a, items, evaluable)

    # ONE resample matrix, reused by every policy and every threshold, so all
    # rows in every table are bootstrapped on identical image draws.
    boot_idx = (rng.integers(0, len(rows), size=(a.bootstrap, len(rows)))
                if a.bootstrap > 0 and rows else np.zeros((0, 0), int))
    n_pred = sum(1 for r in rows for p in r["preds"]
                 if p["conf"] >= a.operating_threshold)

    summary = {"script": "physical_plausibility_layer",
               "dataset": desc,
               "parameters": vars(a),
               "primary": {
                   "policy": a.policy,
                   "threshold": a.operating_threshold,
                   "threshold_provenance":
                       "Reported as the primary point only. It was not "
                       "derived on a held-out split, so no claim rests on it "
                       "alone: every statistic is reported across the full "
                       "confidence sweep below."},
               "images": len(rows),
               "predictions_at_operating_threshold": n_pred,
               "parts_preproc_removed": removed,
               "images_with_zero_parts": n_zero,
               "median_parts_per_image": int(np.median(part_counts)),
               "median_part_classes_per_image": int(np.median(part_cls_counts)),
               "sweep": {}, "policies": {}}

    print("\n  parts pre-processing removed : %s" % removed)
    print("  images with zero parts       : %d / %d" % (n_zero, len(rows)))
    print("  median parts per image       : %d   (distinct classes: %d)"
          % (np.median(part_counts), np.median(part_cls_counts)))

    def ci(v):
        lo, hi = pct_ci(v)
        return lo, hi

    for policy in POLICIES:
        mark = "  <- primary" if policy == a.policy else ""
        print("\n" + "-" * 136)
        print("  TABLE 1  effect of the validation layer, policy '%s'%s"
              % (policy, mark))
        print("-" * 136)
        print("    %6s %7s %10s %20s %28s %28s %17s %9s"
              % ("thr", "preds", "FP b->l", "FP cut % [95% CI]",
                 "d precision [95% CI]", "d recall [95% CI]",
                 "rej prec [95% CI]", "rejFP/TP"))
        summary["sweep"][policy] = {}
        f1_rows = []
        for thr, tkey in zip(sweep, sweep_keys):
            st = image_stats(rows, a, policy, thr)
            A = pack(st)
            pt = point(A)
            bs = boot(A, boot_idx)
            dp_lo, dp_hi = ci(bs["dP"])
            dr_lo, dr_hi = ci(bs["dR"])
            df_lo, df_hi = ci(bs["dF1"])
            fc_lo, fc_hi = ci(bs["FP_cut_pct"])
            judged = float(pt["judged"].sum())
            imp = 100.0 * float(pt["impossible"].sum()) / judged if judged else 0.0
            npred = int(pt["TP_base"] + pt["FP_base"])
            # a trailing * marks a 95% CI that excludes zero
            sp = "*" if ci_excludes_zero(dp_lo, dp_hi) else " "
            sr = "*" if ci_excludes_zero(dr_lo, dr_hi) else " "
            rej_tot0 = pt["rej_fp"] + pt["rej_tp"]
            rp0 = float(pt["rej_fp"]) / rej_tot0 if rej_tot0 else float("nan")
            rp_l, rp_h = pct_ci(bs["rej_prec"])
            rp_txt = ("     n/a          " if math.isnan(rp0)
                      else "%5.3f [%.2f,%.2f]" % (rp0, rp_l, rp_h))
            print("    %6.3f %7d %4d->%-4d  %5.2f%% [%5.2f,%5.2f]"
                  "  %+.5f [%+.4f,%+.4f]%s  %+.5f [%+.4f,%+.4f]%s %s %4d/%-4d"
                  % (thr, npred, int(pt["FP_base"]), int(pt["FP_layer"]),
                     float(pt["FP_cut_pct"]), fc_lo, fc_hi,
                     pt["dP"], dp_lo, dp_hi, sp,
                     pt["dR"], dr_lo, dr_hi, sr,
                     rp_txt, int(pt["rej_fp"]), int(pt["rej_tp"])))
            f1_rows.append((thr, pt["F1_base"], pt["F1_layer"], pt["dF1"],
                            df_lo, df_hi, imp))

            rej_tot = pt["rej_fp"] + pt["rej_tp"]
            rp = float(pt["rej_fp"]) / rej_tot if rej_tot else float("nan")
            rp_lo, rp_hi = pct_ci(bs["rej_prec"])
            per_class = {}
            for c in range(len(DAMAGE_CLASSES)):
                j = float(pt["judged"][c])
                if j == 0 and pt["abstain"][c] == 0:
                    continue
                rate = 100.0 * float(pt["impossible"][c]) / j if j else float("nan")
                lo, hi = pct_ci(bs["rate"][c])
                per_class[DAMAGE_CLASSES[c]] = {
                    "events": int(pt["impossible"][c]), "judged": int(j),
                    "abstain": int(pt["abstain"][c]),
                    "rate": None if math.isnan(rate) else round(rate, 2),
                    "ci": [None if math.isnan(lo) else round(lo, 2),
                           None if math.isnan(hi) else round(hi, 2)]}
            block = {
                "threshold": round(thr, 4),
                "predictions_scored": npred,
                "precision_base": round(pt["P_base"], 4),
                "precision_layer": round(pt["P_layer"], 4),
                "d_precision": round(pt["dP"], 5),
                "d_precision_ci": [jnum(dp_lo, 5), jnum(dp_hi, 5)],
                "recall_base": round(pt["R_base"], 4),
                "recall_layer": round(pt["R_layer"], 4),
                "d_recall": round(pt["dR"], 5),
                "d_recall_ci": [jnum(dr_lo, 5), jnum(dr_hi, 5)],
                "f1_base": round(pt["F1_base"], 4),
                "f1_layer": round(pt["F1_layer"], 4),
                "d_f1": round(pt["dF1"], 5),
                "d_f1_ci": [jnum(df_lo, 5), jnum(df_hi, 5)],
                "gt": int(pt["gt"]),
                "TP_base": int(pt["TP_base"]), "TP_layer": int(pt["TP_layer"]),
                "FP_base": int(pt["FP_base"]), "FP_layer": int(pt["FP_layer"]),
                # predictions that matched nothing AND lay mostly inside an
                # ignore region: counted as neither hit nor error, so
                # TP + FP + excused == predictions scored at this threshold
                "excused_base": int(pt["excused_base"]),
                "excused_layer": int(pt["excused_layer"]),
                "FP_cut_pct": round(float(pt["FP_cut_pct"]), 3),
                "FP_cut_ci": [jnum(fc_lo, 3), jnum(fc_hi, 3)],
                "rejected_fp": int(pt["rej_fp"]),
                "rejected_tp": int(pt["rej_tp"]),
                "rejection_precision": None if math.isnan(rp) else round(rp, 4),
                "rejection_precision_ci": [jnum(rp_lo, 4), jnum(rp_hi, 4)],
                "impossible_pct_overall": round(imp, 3),
                "per_class": per_class}
            summary["sweep"][policy][tkey] = block
            if abs(thr - a.operating_threshold) < 1e-9:
                summary["policies"][policy] = block
        print("    (* = that 95% CI excludes zero;  FP b->l = false "
              "positives before -> after the layer)")
        # F1 is a secondary metric for this study: the layer is a precision
        # instrument, and a summary that averages a precision gain against a
        # recall loss hides the trade-off the result is about.
        print("\n    secondary - F1, and the share of judged predictions "
              "ruled impossible")
        print("    %6s %9s %9s %28s %8s"
              % ("thr", "F1_base", "F1_layer", "d F1 [95% CI]", "imp%"))
        for (t_, fb_, fl_, df_, dlo_, dhi_, imp_) in f1_rows:
            st_ = "*" if ci_excludes_zero(dlo_, dhi_) else " "
            print("    %6.3f %9.4f %9.4f  %+.5f [%+.4f,%+.4f]%s %7.2f%%"
                  % (t_, fb_, fl_, df_, dlo_, dhi_, st_, imp_))

    prim = summary["policies"].get(a.policy, {})
    print("\n" + "-" * 136)
    print("  TABLE 2  impossible rate by damage class  |  policy '%s', "
          "threshold %.3f" % (a.policy, a.operating_threshold))
    print("-" * 136)
    print("    %-15s %11s %9s %22s %9s"
          % ("class", "events", "rate", "95% CI", "abstain"))
    for nm, d_ in prim.get("per_class", {}).items():
        lo, hi = d_["ci"]
        star = "  *" if (lo is not None and lo > 0) else ""
        rate = "n/a" if d_["rate"] is None else "%.2f%%" % d_["rate"]
        span = ("n/a" if lo is None or hi is None
                else "[%.2f, %.2f]" % (lo, hi))
        print("    %-15s %5d/%-5d %9s %21s %9d%s"
              % (nm, d_["events"], d_["judged"], rate, span,
                 d_["abstain"], star))
    print("    (* = rate significantly above zero)")

    if a.dataset == "cardd":
        print("\n" + "-" * 136)
        print("  TABLE 3  mechanism: rule restrictiveness vs detector accuracy")
        print("-" * 136)
        scored_ids = [it["image_id"] for it in items
                      if it["image_id"] is not None]
        ap = per_class_ap(coco, dets, a, scored_ids)
        sup = cardd_support(coco, scored_ids)
        names, xs, aps, obs = [], [], [], []
        for c in range(len(DAMAGE_CLASSES)):
            nm = DAMAGE_CLASSES[c]
            pc = prim.get("per_class", {}).get(nm)
            if ap.get(nm) is None or not pc or pc["judged"] == 0:
                continue
            names.append(nm)
            xs.append(restrictiveness(c))
            aps.append(ap[nm])
            obs.append(100.0 * pc["events"] / pc["judged"])
        print("\n    %-15s %8s %8s %13s %13s"
              % ("class", "AP", "GT n", "forbidden%", "impossible%"))
        for k in range(len(names)):
            print("    %-15s %8.2f %8d %12.1f%% %12.2f%%"
                  % (names[k], aps[k], sup.get(names[k], 0),
                     100 * xs[k], obs[k]))
        if len(names) >= 3:
            rho = spearman(xs, aps)
            p = exact_perm_p(xs, aps, rho)
            print("\n    Spearman rho(restrictiveness, AP) = %+.4f" % rho)
            print("    exact permutation p = %s   (n=%d, %d orderings)"
                  % (("%.4f" % p) if p is not None else "n/a", len(names),
                     math.factorial(len(names))))
            floor_p = 2.0 / math.factorial(len(names))
            print("\n    A POSITIVE rho means the rules are most restrictive on")
            print("    the classes the detector already handles best - which is")
            print("    what bounds any gain this layer can deliver.")
            print("\n    POWER: with n=%d classes the smallest two-sided p this"
                  % len(names))
            print("    exact test can return is 2/%d! = %.4f. A p above 0.05"
                  % (len(names), floor_p))
            print("    therefore means these %d points cannot CERTIFY the"
                  % len(names))
            print("    relationship - not that there is none. Restrictiveness")
            print("    is also confounded with how visually distinctive a class")
            print("    is, so read this as a mechanism sketch and let the")
            print("    per-class impossible rates carry the empirical weight.")
            summary["mechanism"] = {
                "classes": names, "ap": aps,
                "gt_instances": [sup.get(n_, 0) for n_ in names],
                "restrictiveness": [round(x, 4) for x in xs],
                "impossible_pct": [round(x, 3) for x in obs],
                "spearman_rho": round(rho, 4),
                "p_exact": round(p, 4) if p is not None else None,
                "p_exact_floor": round(floor_p, 5),
                "caveat": "n=%d classes. The smallest two-sided p this exact "
                          "test can return is %.4f, so a p above 0.05 means "
                          "these points cannot certify the relationship, not "
                          "that there is none. Restrictiveness is also "
                          "confounded with class distinctiveness, so this "
                          "does not establish causation."
                          % (len(names), floor_p)}
        summary["per_class_ap"] = ap
        summary["per_class_ap_support"] = sup
        summary["per_class_ap_note"] = (
            "COCOeval segm AP at IoU 0.50:0.95 with CarDD area ranges, over "
            "detections collected at --infer_threshold. None means the class "
            "had no ground truth among the evaluated images.")

    rej = audit_rejected_tp(a, rows, items, a.operating_threshold, a.policy)
    cfa = audit_without_scratch_on_glass(a, rows, items, a.operating_threshold,
                                         a.policy)
    by_part = print_rejected_tp(rej, cfa, a)
    summary["rejected_true_positives"] = {
        "policy": a.policy,
        "threshold": a.operating_threshold,
        "n": len(rej),
        "by_damage_class": dict(Counter(d["damage"] for d in rej)),
        "by_dominant_part": {("%s|%s" % (p, d)): n
                             for (p, d), n in by_part.items()},
        "counterfactual_allow_scratch_on_glass": {
            "rejected_true_positives_after": len(cfa),
            "recovered": len(rej) - len(cfa)},
        "rows": rej}

    fn = out / ("plausibility_%s.json" % a.dataset)
    # published file: absolute paths carry no reproducible meaning, so record
    # only the trailing components that identify each artefact
    json.dump(shorten_paths(summary), open(fn, "w"), indent=2, default=float)
    print("\n  written: %s\n" % fn)


if __name__ == "__main__":
    main()
