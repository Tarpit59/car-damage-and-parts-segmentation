# Car Parts — Model Evaluation

Evaluation of the **RF-DETR-Seg-Nano** model trained for **19-class car part instance segmentation**. Unlike the damage model, there is no published external baseline — all results are self-contained RF-DETR metrics.

---

## 🤖 Model & Dataset Download

> **🔗 [Google Drive — Models & Datasets](https://drive.google.com/drive/folders/1KxSQLXaXxa2vOlMYNeq6UL5Ua1hEDirn?usp=drive_link)**

| Item | Description |
|------|-------------|
| `checkpoint_best_total.pth` | Trained RF-DETR-Seg-Nano car parts model |
| Car Parts test set | 540 images · 4,591 annotations · 19 part classes |
| `metrics.csv` | Raw training log |

---

## 📦 Dataset Details

- **Source:** Multiple publicly available car part datasets were combined and unified into a single dataset covering 19 part classes.
- **Resolution:** all images are **1152 × 1152**. They are fed to the model at that native size — no further resizing takes place at evaluation time.
- **Format:** COCO JSON with polygon segmentation masks.

### Class Distribution

| Class | Train Images | Train Instances | Valid Images | Valid Instances | Test Images | Test Instances |
|------------------------|-------------|----------------|-------------|----------------|------------|----------------|
| Diggi_Back_Door | 1,228 | 1,240 | 172 | 172 | 97 | 97 |
| Diggi_Back_Door_Glass | 1,889 | 1,903 | 330 | 330 | 139 | 141 |
| Fender | 2,026 | 2,064 | 297 | 302 | 149 | 152 |
| Front_Bumper | 3,133 | 3,164 | 570 | 573 | 315 | 316 |
| Front_Door | 3,786 | 3,850 | 663 | 671 | 370 | 372 |
| Front_Door_Glass | 1,979 | 2,002 | 301 | 308 | 142 | 146 |
| Front_Windshield_Glass | 3,125 | 3,134 | 568 | 574 | 329 | 329 |
| Grill | 1,309 | 1,833 | 191 | 265 | 109 | 152 |
| Headlight | 3,696 | 5,275 | 697 | 1,007 | 383 | 558 |
| Hood_Bonnet | 3,249 | 3,270 | 592 | 596 | 327 | 329 |
| Quarter_Panel | 1,813 | 1,857 | 274 | 284 | 132 | 136 |
| Rear_Bumper | 1,946 | 1,963 | 333 | 335 | 137 | 138 |
| Rear_Door | 3,370 | 3,412 | 611 | 619 | 325 | 329 |
| Rear_Door_Glass | 1,920 | 1,972 | 290 | 299 | 141 | 146 |
| Roof | 1,813 | 1,842 | 251 | 253 | 128 | 134 |
| Running_Board | 1,587 | 1,606 | 245 | 251 | 104 | 105 |
| Side_Mirror | 2,448 | 2,969 | 336 | 395 | 259 | 315 |
| Taillight | 2,558 | 3,618 | 475 | 658 | 198 | 275 |
| tyre | 2,730 | 4,562 | 397 | 699 | 241 | 421 |
| **TOTAL** | **6,089** | **51,536** | **1,079** | **8,591** | **540** | **4,591** |

---

## 🏷️ Car Part Classes (19)

| # | Class | Human Label |
|---|-------|------------|
| 1 | `Diggi_Back_Door` | Diggi Back Door |
| 2 | `Diggi_Back_Door_Glass` | Diggi Door Glass |
| 3 | `Fender` | Fender |
| 4 | `Front_Bumper` | Front Bumper |
| 5 | `Front_Door` | Front Door |
| 6 | `Front_Door_Glass` | Front Door Glass |
| 7 | `Front_Windshield_Glass` | Front Windshield |
| 8 | `Grill` | Grill |
| 9 | `Headlight` | Headlight |
| 10 | `Hood_Bonnet` | Hood/Bonnet |
| 11 | `Quarter_Panel` | Quarter Panel |
| 12 | `Rear_Bumper` | Rear Bumper |
| 13 | `Rear_Door` | Rear Door |
| 14 | `Rear_Door_Glass` | Rear Door Glass |
| 15 | `Roof` | Roof |
| 16 | `Running_Board` | Running Board |
| 17 | `Side_Mirror` | Side Mirror |
| 18 | `Taillight` | Taillight |
| 19 | `tyre` | Tyre |

> In the AutoScan AI pipeline, the model's 19 raw classes are further expanded to **29 final classes** using car-side detection (Left / Right prefixes for side-dependent parts).

---

## 📂 Folder Structure

```
Car Parts/
├── Testing/
│   ├── evaluate_rfdetr_carparts.py     ← Full test-set evaluation script
│   └── plots_parts/                    ← Main evaluation results (12 charts + JSON)
│       ├── rfdetr_carparts_results.json
│       ├── ap_per_category_mask.png
│       ├── ap_per_category_box.png
│       ├── ap50_ap75_bar.png
│       ├── mask_vs_box_overall.png
│       ├── mask_vs_box_per_category.png
│       ├── pr_curve_mask_page1.png
│       ├── pr_curve_mask_page2.png
│       ├── pr_curve_box_page1.png
│       ├── pr_curve_box_page2.png
│       ├── f1_vs_threshold_mask.png
│       ├── f1_vs_threshold_box.png
│       └── confusion_heatmap_mask.png
│
└── Training and Validation/
    ├── plot_parts_metrics.py           ← Training chart generator from CSV
    ├── training_stats/
    │   └── metrics.csv                 ← Raw training log
    └── parts_plots/                    ← Generated training charts
        ├── combined_overview.png
        ├── val_map_overview.png
        ├── val_map75_mar.png
        ├── val_precision_recall_f1.png
        ├── val_per_category_ap.png
        ├── val_ema_vs_live.png
        ├── val_category_radar.png
        ├── val_loss_curve.png
        ├── train_loss_total.png
        ├── train_loss_components.png
        ├── train_loss_auxiliary.png
        └── train_cardinality_error.png
```

---

## 🧪 Testing — Test-Set Evaluation

### Script: `Testing/evaluate_rfdetr_carparts.py`

Runs the trained RF-DETR model against the Car Parts test split and computes full COCO-standard AP metrics for both instance segmentation (mask) and bounding box detection across all 19 classes, using `pycocotools.cocoeval.COCOeval`.

---

### 🔬 Evaluation Methodology

#### Images are fed to the model unresized

RF-DETR's post-processing interpolates the mask *logits* to the size of the image it was handed and only then thresholds them, so `det.mask` is returned in the input image's coordinate space. Test images are therefore passed to the model at their native size and the returned masks are used exactly as produced — **no resampling and no re-projection**. The mask's shape is asserted against the image dimensions, so a mismatch raises rather than being silently rescaled.

#### Bounding box derivation

Boxes are the **tight bounding box of the predicted mask**, not the model's box head output, so mask AP and box AP describe the same predicted region.

#### Class order is verified, never assumed

RF-DETR emits a 0-based class index in the order of the dataset it was *trained* on. The script replicates RF-DETR's own `cat2label` mapping — `{cat_id: i for i, cat_id in enumerate(sorted(coco.cats.keys()))}`, **unfiltered**, so a Roboflow dummy supercategory at id 0 counts and would shift every real class by one — and aborts if the annotation file's order disagrees with the hardcoded 19-class list. Continuing past a disagreement would relabel every prediction while still producing a plausible-looking table.

#### Ground-truth `area` is recomputed from the segmentation

COCOeval bins ground truth into size ranges using each annotation's `area` field. In this export, **all 4,591 annotations store the bounding-box area** rather than the polygon area — the stored value is larger than the true mask area for every single instance, by a median factor of **1.62×** (p95 = 3.45×). Instances are therefore binned as if they were much bigger than they are, and APs / APm / APl end up describing the wrong instances.

The script recomputes `area` from the decoded mask before evaluating. **566 of 4,591 annotations (12.3%) move to a different size bin** as a result:

| Bin | Using the stored `area` | Using the decoded mask | Change |
|-----|------------------------|------------------------|--------|
| Small | 9 | **44** | +35 |
| Medium | 595 | **1,091** | +496 |
| Large | 3,987 | **3,456** | −531 |

Without the correction, `APs` would be averaged over **9** instances instead of 44 — which is why an earlier version of this evaluation reported `APs` 3.5 rather than 8.8. Neither figure is reportable (see the note under the size-stratified results), but only the corrected one is computed over the right instances.

#### Area Range Configuration (APs / APm / APl)

Unlike the Car Damage evaluation (which uses the CarDD paper's non-standard ranges to match its baseline), the Car Parts evaluation uses **COCO standard area ranges**:

| Size Category | COCO Standard (used here) | CarDD (used for damage) |
|---------------|--------------------------|------------------------|
| **Small** | area < 32² | area < 128² |
| **Medium** | 32² ≤ area < 96² | 128² ≤ area < 256² |
| **Large** | area ≥ 96² | area ≥ 256² |

APs / APm / APl are therefore **not comparable** between the Car Damage and Car Parts evaluations.

For a segmentation result, COCOeval bins detections by their **mask** area. `pycocotools`' `loadRes` tests `if 'bbox' in anns[0]` *before* `elif 'segmentation' in anns[0]`, so a `bbox` key on a segm result silently switches the binning to *box* area while the ground truth stays binned by mask area. The segm results here therefore carry only `image_id`, `category_id`, `segmentation` and `score`.

#### Two thresholds, deliberately separate

| | Value | Purpose |
|---|---|---|
| `--threshold` | `0.001` | **Collection** threshold handed to COCOeval. Must stay low — COCOeval sweeps the score axis itself when building the PR curve, so raising it truncates the recall axis and *understates* AP. Not an operating point. |
| `REPORT_THRESHOLD` | `0.50` | **Reporting** threshold, used only for the P/R/F1 summary and the confusion matrix. A conventional value fixed in advance, so it is not tuned on the test set. |

AP is unaffected by `REPORT_THRESHOLD`.

#### Evaluator

All headline AP metrics come from **`pycocotools.cocoeval.COCOeval`** with default parameters: 10 IoU thresholds (0.50 : 0.05 : 0.95), 101-point interpolated precision, and a maximum of 100 detections per image.

---

### 🔄 Evaluation Flow

```
1. Load COCO annotations
   └─ Verify the category order against the hardcoded 19-class list, replicating
      RF-DETR's own cat2label mapping; abort on mismatch

2. Decode all GT masks from annotations
   ├─ Polygon → binary mask → RLE (native image coordinates)
   └─ Count and report any annotation dropped as undecodable or out-of-class

3. Load RF-DETR-Seg-Nano model checkpoint

4. For each test image (540 images):
   ├─ 4a. Resolve the image file path (with Roboflow filename fallback)
   ├─ 4b. Run inference at native size → masks in the image's coordinate space
   ├─ 4c. Derive the tight bounding box from each mask
   └─ 4d. Build COCO result entries (segm without a bbox key; bbox with one)
   Predictions discarded for an out-of-range class index are counted, not
   dropped silently, and reported after the loop

5. Re-open the annotations as a COCO object and recompute every GT 'area'
   from its segmentation
   └─ 566 of 4,591 annotations (12.3%) move to a different size bin

6. Compute the ground-truth support table (per class × COCO size bin, by mask area)

7. Run pycocotools COCOeval
   ├─ iouType = "segm" → Mask AP / AP50 / AP75 / APs / APm / APl
   └─ iouType = "bbox" → Box  AP / AP50 / AP75 / APs / APm / APl

8. Per-category evaluation (19 categories × 2 iouTypes)
   └─ Per-category AP, AP50, AP75 + PR curve data

9. Supplementary metrics at REPORT_THRESHOLD = 0.50
   ├─ Precision / Recall / F1 (per-class greedy matching @ IoU 0.50)
   └─ Confusion matrix (class-agnostic greedy matching @ IoU 0.50)

10. Generate 12 evaluation charts + save full results JSON
```

---

### 📈 Results

**Dataset:** Car Parts test split — **540 images, 4,591 annotations, 19 categories**
All 4,591 annotations were scored: 0 undecodable, 0 out-of-class, 0 images failed.
**Evaluator:** `pycocotools.cocoeval.COCOeval`, native image coordinates
**Area ranges:** COCO standard (small < 32², medium 32²–96², large ≥ 96²)
**Prediction pass:** 271.2 s for 540 images (~0.50 s/image) on an NVIDIA RTX 5070 Laptop GPU. This is wall-clock for the whole loop — image load, forward pass, and RLE encoding of 53,155 masks — not an inference benchmark. It is also power-state dependent: the identical run on battery rather than mains took about 40% longer, with every accuracy metric unchanged.

#### Overall Metrics

| Metric | Mask | Box | Δ (Box − Mask) |
|--------|------|-----|----------------|
| **AP** (IoU 0.50:0.95) | **62.1** | **64.5** | +2.4 |
| **AP50** | **85.0** | **86.4** | +1.4 |
| **AP75** | **68.0** | **71.3** | +3.3 |

#### Size-Stratified AP (COCO Standard Area Ranges)

| Metric | Mask | Box | GT instances |
|--------|------|-----|--------------|
| **APs** (small, area < 32²) | 8.8 | 12.1 | **44** ⚠️ |
| **APm** (medium, 32² ≤ area < 96²) | 33.2 | 49.2 | 1,091 |
| **APl** (large, area ≥ 96²) | 67.6 | 67.3 | 3,456 |

> ⚠️ **`APs` is reported for completeness only and should not be read as a measurement of small-object performance.** The entire test split contains just **44 small instances across all 19 classes**, and 17 of the 19 classes have fewer than 10. An AP averaged over so few annotations describes those particular annotations, not the model. `APm` (1,091 instances) and `APl` (3,456) are well supported.
>
> The cause is the dataset, not the model: car parts are large objects photographed close-up, so under COCO's 32² threshold almost nothing qualifies as "small". Of 4,591 instances, **75% are large** and only **1% are small**. This is also why the Car Damage evaluation uses CarDD's much larger thresholds — size bins have to suit the objects being measured.

#### Ground-Truth Support (per size bin)

| Bin | Definition | Instances | Share |
|-----|-----------|-----------|-------|
| Small | area < 32² | 44 | 1.0% |
| Medium | 32² ≤ area < 96² | 1,091 | 23.8% |
| Large | area ≥ 96² | 3,456 | 75.3% |
| **Total** | | **4,591** | |

Every one of the 19 classes has at least 10 instances overall, so the per-category APs below are all adequately supported.

#### Precision / Recall / F1

| Mode | Threshold | Precision | Recall | F1 | TP | FP | FN |
|------|-----------|-----------|--------|-----|----|----|-----|
| Mask | 0.50 | 0.8790 | 0.8532 | **0.8659** | 3,917 | 539 | 674 |
| Box | 0.50 | 0.8815 | 0.8556 | **0.8684** | 3,928 | 528 | 663 |

Measured at the fixed `REPORT_THRESHOLD = 0.50`, chosen in advance. For reference, the F1-maximising threshold *selected on this test set* is 0.4694, giving F1 0.8667 — a gain of 0.0008 over the fixed value, so there is no meaningful tuning advantage and the fixed-threshold numbers can be quoted directly.

#### Confusion (mask, IoU 0.50, score ≥ 0.50)

| Outcome | Count |
|---------|-------|
| Correctly classified | 3,914 |
| Matched but mislabelled | 41 |
| Missed (no prediction) | 636 |
| **Total ground truth** | **4,591** |
| Background false positives | 501 |

> Only 41 of 4,591 instances are matched by a prediction of the wrong class, so cross-part confusion is rare even between visually similar neighbours (Front_Door / Rear_Door, Front_Bumper / Rear_Bumper). The dominant error mode is **missed detections** (636). The matrix uses class-agnostic greedy matching, so a prediction can claim a ground-truth instance of a different class — which is what makes off-diagonal cells possible at all.

#### Per-Category Mask AP (sorted high → low)

| Class | Mask AP | Mask AP50 | Mask AP75 | GT instances |
|-------|---------|-----------|-----------|--------------|
| Front_Windshield_Glass | **89.8** | 98.5 | 97.8 | 329 |
| Hood_Bonnet | **87.0** | 97.7 | 95.0 | 329 |
| Front_Bumper | **84.7** | 97.2 | 91.8 | 316 |
| Front_Door | **80.1** | 96.5 | 92.4 | 372 |
| Diggi_Back_Door_Glass | **78.8** | 95.4 | 88.5 | 141 |
| Rear_Bumper | **75.4** | 93.5 | 83.9 | 138 |
| Rear_Door | **72.7** | 92.6 | 84.8 | 329 |
| Headlight | **69.8** | 93.7 | 79.6 | 558 |
| Diggi_Back_Door | **65.9** | 82.9 | 67.3 | 97 |
| Taillight | **64.9** | 89.3 | 72.4 | 275 |
| Fender | **59.4** | 88.8 | 69.2 | 152 |
| tyre | **58.8** | 84.7 | 67.6 | 421 |
| Side_Mirror | **56.6** | 87.1 | 67.8 | 315 |
| Rear_Door_Glass | **53.7** | 85.7 | 56.2 | 146 |
| Front_Door_Glass | **51.0** | 82.3 | 52.5 | 146 |
| Grill | **49.1** | 73.4 | 60.1 | 152 |
| Quarter_Panel | **36.1** | 58.0 | 37.2 | 136 |
| Roof | **27.4** | 63.2 | 19.3 | 134 |
| Running_Board | **19.6** | 54.7 | 7.9 | 105 |

> **Note:** Roof and Running_Board score lowest, and the AP50 → AP75 collapse (63.2 → 19.3 and 54.7 → 7.9) shows why: the model finds them but cannot delineate them precisely. Both are elongated, low-contrast regions that are rarely fully visible and blend into adjacent panels, so the boundary is genuinely ambiguous. Their support (134 and 105 instances) is adequate, so these are real results, not small-sample noise.

#### Per-Category Box AP (sorted high → low)

| Class | Box AP | Box AP50 | Box AP75 |
|-------|--------|----------|----------|
| Front_Windshield_Glass | **89.3** | 98.5 | 97.8 |
| Hood_Bonnet | **86.2** | 97.9 | 95.9 |
| Front_Bumper | **86.1** | 97.2 | 95.0 |
| Diggi_Back_Door_Glass | **80.5** | 95.5 | 87.1 |
| Front_Door | **79.4** | 96.7 | 91.5 |
| Rear_Bumper | **79.2** | 95.0 | 85.1 |
| Rear_Door | **76.6** | 93.4 | 88.3 |
| Headlight | **71.4** | 94.1 | 81.8 |
| Diggi_Back_Door | **67.8** | 87.6 | 69.9 |
| Taillight | **67.0** | 88.1 | 74.8 |
| Fender | **60.9** | 89.2 | 72.6 |
| tyre | **56.4** | 83.2 | 63.0 |
| Front_Door_Glass | **54.5** | 83.4 | 60.2 |
| Rear_Door_Glass | **54.2** | 86.8 | 62.2 |
| Grill | **53.6** | 76.0 | 63.8 |
| Side_Mirror | **50.9** | 87.5 | 55.7 |
| Running_Board | **44.5** | 67.6 | 48.1 |
| Quarter_Panel | **36.6** | 58.4 | 38.4 |
| Roof | **30.7** | 65.9 | 22.9 |

### Usage

```bash
python Testing/evaluate_rfdetr_carparts.py \
    --images_dir  /path/to/test/images \
    --annotations /path/to/_annotations.coco.json \
    --checkpoint  /path/to/checkpoint_best_total.pth \
    --resolution  960 \
    --threshold   0.001 \
    --output_json Testing/plots_parts/rfdetr_carparts_results.json \
    --plots_dir   Testing/plots_parts
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--images_dir` | — | Path to test image directory |
| `--annotations` | — | Path to COCO-format annotations JSON (image list, ground truth and category order) |
| `--checkpoint` | — | Path to `.pth` model checkpoint |
| `--resolution` | `960` | RF-DETR inference resolution |
| `--threshold` | `0.001` | Collection threshold for COCOeval. Keep it low — raising it truncates the recall axis and understates AP |
| `--output_json` | `rfdetr_carparts_results.json` | Output JSON with all metrics |
| `--plots_dir` | `./plots_parts` | Output directory for charts |

`REPORT_THRESHOLD = 0.50` is a constant in the script header, not a CLI flag. It affects only the P/R/F1 summary and the confusion matrix; AP is independent of it.

### Generated Charts (12 total)

| # | File | Description |
|---|------|-------------|
| 1–2 | `pr_curve_mask_page1/2.png` | Per-category Precision–Recall curves — mask (2 pages, 10+9 categories). Solid = IoU 0.50, dashed = IoU 0.55–0.95. |
| 3–4 | `pr_curve_box_page1/2.png` | Per-category Precision–Recall curves — box |
| 5 | `f1_vs_threshold_mask.png` | F1 / Precision / Recall vs confidence threshold (mask), with the fixed operating point and the test-selected oracle both marked |
| 6 | `f1_vs_threshold_box.png` | As above, for box |
| 7 | `ap_per_category_mask.png` | Horizontal bar chart — per-class mask AP (sorted descending) with mean line |
| 8 | `ap_per_category_box.png` | Horizontal bar chart — per-class box AP (sorted descending) with mean line |
| 9 | `mask_vs_box_overall.png` | Grouped bar — Mask vs Box for AP / AP50 / AP75 with Δ annotations |
| 10 | `mask_vs_box_per_category.png` | Grouped bar — Mask vs Box per category with Δ annotations |
| 11 | `ap50_ap75_bar.png` | AP50 vs AP75 per category — the gap indicates mask shape precision |
| 12 | `confusion_heatmap_mask.png` | Confusion matrix (mask @ IoU 0.50). Rows sum to 1; final column is ground truth claimed by no prediction; background false positives reported beneath |

---

## 📉 Training & Validation

### Script: `Training and Validation/plot_parts_metrics.py`

Reads the training log CSV and produces 12 charts tracking loss curves, mAP progression, EMA tracking, and per-category AP over all training epochs for all 19 part classes.

> **These charts are not test-set results.** `val/*` columns are the split RF-DETR scored at the end of each epoch, and `train/*` are training-time losses. Neither is the held-out test set, and the `val/*` numbers are computed with a different evaluator configuration from the results above — the two must not be quoted interchangeably. Every generated figure carries a provenance line stating this.

### Usage

```bash
python "Training and Validation/plot_parts_metrics.py" \
    --csv "Training and Validation/training_stats/metrics.csv" \
    --output_dir "Training and Validation/parts_plots"
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--csv` | `metrics.csv` | Path to training metrics CSV |
| `--output_dir` | `./parts_plots` | Output directory for charts |

### Generated Charts (12 total)

| # | File | Description |
|---|------|-------------|
| 1 | `val_map_overview.png` | Val Box mAP@50/50:95 + Segm mAP@50/50:95 over epochs |
| 2 | `val_map75_mar.png` | Val mAP@75 and Mean Average Recall over epochs |
| 3 | `val_precision_recall_f1.png` | Val Precision / Recall / F1 over epochs |
| 4 | `val_per_category_ap.png` | Per-part-class AP over epochs (4×5 grid, 19 parts) |
| 5 | `train_loss_total.png` | Total training loss over steps |
| 6 | `train_loss_components.png` | CE / BBox L1 / GIoU / Mask CE / Mask Dice per step |
| 7 | `train_loss_auxiliary.png` | Auxiliary decoder layer losses (layers 0–2 + enc) |
| 8 | `val_ema_vs_live.png` | EMA mAP vs live mAP (box + segm) |
| 9 | `val_category_radar.png` | Two-panel radar: per-part AP at final epoch (Parts 1–10, 11–19) |
| 10 | `val_loss_curve.png` | Validation loss over epochs |
| 11 | `train_cardinality_error.png` | Cardinality error per decoder layer over steps |
| 12 | `combined_overview.png` | 2×2 dashboard: loss / mAP / P-R-F1 / per-part horizontal bar |

> This run logs one row per epoch (20 rows). A rolling mean is applied only when the series is long enough for it to change anything; with 20 points it is not, so the loss charts show a single unsmoothed line rather than an identical curve drawn twice under a "smoothed" label.

---

## ⚙️ Requirements

`requirements.txt` lives at the repository root. The `Usage` blocks above run
from this folder, so install from the root first:

```bash
pip install -r ../../requirements.txt
```

---

## 📖 References

- **RF-DETR:** [Roboflow RF-DETR — Real-Time Detection Transformer](https://github.com/roboflow/rf-detr)
- **Evaluation standard:** COCO-style AP (IoU = 0.50:0.95, 101-point interpolation, maxDets 100) computed with `pycocotools.cocoeval.COCOeval` using COCO standard area ranges.
- **Dataset annotation format:** COCO JSON with polygon segmentation masks
