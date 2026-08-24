"""
verify_class_maps.py
====================
PREREQUISITE CHECK - run this before any experiment.

Answers three questions empirically:

  Q1. What class_id values does each model actually output?
      (category-id space, or 0-based remapped space?)
  Q2. Do the hardcoded CLASS_NAMES lists decode those ids correctly?
  Q3. Do predicted masks come back at the INPUT image size?
      (settles whether variable-size input is safe)

Nothing here depends on config/config.py - all paths are argparse
defaults, so local config edits cannot change the result.
"""

import argparse
import json
from pathlib import Path

from PIL import Image

# -- Hardcoded lists copied from the app, for comparison only ----------------
DAMAGE_CLASS_NAMES_APP = [
    "crack", "dent", "glass shatter", "lamp broken", "scratch", "tire flat",
]

PARTS_CLASS_NAMES_APP = [
    "Diggi_Back_Door", "Diggi_Back_Door_Glass", "Fender", "Front_Bumper",
    "Front_Door", "Front_Door_Glass", "Front_Windshield_Glass", "Grill",
    "Headlight", "Hood_Bonnet", "Quarter_Panel", "Rear_Bumper",
    "Rear_Door", "Rear_Door_Glass", "Roof", "Running_Board",
    "Side_Mirror", "Taillight", "tyre",
]

D = r"D:\GTU\Mini Project"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--damage_ckpt",
                   default=D + r"\run_detr\T03_car_damage_training_models\checkpoint_best_total.pth")
    p.add_argument("--parts_ckpt",
                   default=D + r"\Car Parts\just for testing\checkpoint_best_total_02.1.pth")
    p.add_argument("--damage_train_ann",
                   default=D + r"\carDD_DETR_dataset\CarDD_1152_Coco_multiscale\train\_annotations.coco.json")
    p.add_argument("--parts_train_ann",
                   default=D + r"\Car Parts\Dataset\Car_parts_in 2.v1i.coco_1152_without_sides\train\_annotations.coco.json")
    p.add_argument("--test_images_dir",
                   default=D + r"\carDD_DETR_dataset\CarDD_COCO_original\test")
    p.add_argument("--test_ann",
                   default=D + r"\carDD_DETR_dataset\CarDD_COCO_original\test\_annotations.coco.json")
    p.add_argument("--damage_resolution", type=int, default=960)
    p.add_argument("--parts_resolution", type=int, default=960)
    p.add_argument("--damage_threshold", type=float, default=0.408)
    p.add_argument("--parts_threshold", type=float, default=0.45)
    p.add_argument("--n_images", type=int, default=5)
    return p.parse_args()


def cats_sorted(ann_path):
    with open(ann_path) as f:
        d = json.load(f)
    return sorted(d["categories"], key=lambda c: c["id"])


def show_dataset_order(label, ann_path, app_list):
    cats = cats_sorted(ann_path)
    print("\n" + "=" * 72)
    print(label)
    print("=" * 72)
    print("  file: " + str(ann_path))
    print("  %d categories, ids = %s" % (len(cats), [c["id"] for c in cats]))

    # Two competing hypotheses for how RF-DETR indexes classes
    by_catid = {c["id"]: c["name"] for c in cats}
    by_zero = {i: c["name"] for i, c in enumerate(cats)}

    for title, mapping in (
        ("Hypothesis A - model emits CATEGORY IDs as-is:", by_catid),
        ("Hypothesis B - model emits 0-BASED index into sorted categories:", by_zero),
    ):
        print("\n  " + title)
        for k in sorted(mapping):
            app = app_list[k] if 0 <= k < len(app_list) else "<INDEX ERROR>"
            flag = "OK " if app == mapping[k] else "BAD"
            print("    [%s] class_id=%-3d true='%s'  app says '%s'"
                  % (flag, k, mapping[k], app))

    return by_catid, by_zero


def main():
    a = parse_args()

    print("\n" + "#" * 72)
    print("# PART 1 - what the ANNOTATION FILES say")
    print("#" * 72)
    dmg_catid, dmg_zero = show_dataset_order(
        "DAMAGE training set", a.damage_train_ann, DAMAGE_CLASS_NAMES_APP)
    prt_catid, prt_zero = show_dataset_order(
        "PARTS training set", a.parts_train_ann, PARTS_CLASS_NAMES_APP)

    print("\n" + "#" * 72)
    print("# PART 2 - what the MODELS actually emit")
    print("#" * 72)

    from rfdetr import RFDETRSegMedium, RFDETRSegNano

    print("\n[INFO] Loading damage model ...")
    dmg_model = RFDETRSegMedium(pretrain_weights=a.damage_ckpt,
                                resolution=a.damage_resolution)
    print("[INFO] Loading parts model ...")
    prt_model = RFDETRSegNano(pretrain_weights=a.parts_ckpt,
                              resolution=a.parts_resolution)

    with open(a.test_ann) as f:
        test = json.load(f)
    img_dir = Path(a.test_images_dir)
    files = [img_dir / im["file_name"] for im in test["images"][:a.n_images]]

    seen_dmg, seen_prt = set(), set()
    shape_ok = True

    for fp in files:
        if not fp.exists():
            print("[WARN] missing image: " + str(fp))
            continue
        img = Image.open(fp).convert("RGB")
        W, H = img.size
        print("\n--- %s   input size = %dx%d (WxH) ---" % (fp.name, W, H))

        for tag, model, thr, catid_map, zero_map, app_list in (
            ("DAMAGE", dmg_model, a.damage_threshold, dmg_catid, dmg_zero,
             DAMAGE_CLASS_NAMES_APP),
            ("PARTS", prt_model, a.parts_threshold, prt_catid, prt_zero,
             PARTS_CLASS_NAMES_APP),
        ):
            det = model.predict(img, threshold=thr)
            if det is None or len(det) == 0:
                print("  %s: no detections" % tag)
                continue

            # Q3 - mask shape vs input size
            if det.mask is not None:
                mh, mw = det.mask[0].shape
                ok = (mh == H and mw == W)
                shape_ok = shape_ok and ok
                print("  %s: %d dets | mask shape = %dx%d (HxW) -> %s"
                      % (tag, len(det), mh, mw,
                         "MATCHES input" if ok else "*** DOES NOT MATCH INPUT ***"))
            else:
                print("  %s: %d dets | no masks returned" % (tag, len(det)))

            ids = [int(c) for c in det.class_id]
            (seen_dmg if tag == "DAMAGE" else seen_prt).update(ids)

            for cid, conf in sorted(zip(ids, det.confidence), key=lambda t: -t[1])[:6]:
                app = app_list[cid] if 0 <= cid < len(app_list) else "<INDEX ERROR>"
                print("      class_id=%-3d conf=%.3f | A(cat-id)='%s'  B(0-based)='%s'  APP='%s'"
                      % (cid, conf, catid_map.get(cid, "?"), zero_map.get(cid, "?"), app))

    print("\n" + "#" * 72)
    print("# PART 3 - VERDICT")
    print("#" * 72)
    print("\n  DAMAGE class_ids observed: %s  (app list valid indices 0..%d)"
          % (sorted(seen_dmg), len(DAMAGE_CLASS_NAMES_APP) - 1))
    print("  PARTS  class_ids observed: %s  (app list valid indices 0..%d)"
          % (sorted(seen_prt), len(PARTS_CLASS_NAMES_APP) - 1))
    if seen_dmg and max(seen_dmg) >= len(DAMAGE_CLASS_NAMES_APP):
        print("\n  >>> DAMAGE ids exceed the app list length.")
        print("      The app's CLASS_NAMES lookup is OFF BY ONE (dummy 'damage' category).")
    print("\n  Mask shapes match input size for every check: %s" % shape_ok)
    print("\n  Read the A / B / APP columns above: whichever column gives sensible")
    print("  names for the highest-confidence detections is the true mapping.\n")


if __name__ == "__main__":
    main()
