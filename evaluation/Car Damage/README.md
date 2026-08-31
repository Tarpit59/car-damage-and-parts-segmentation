# Car Damage — Model Evaluation

Evaluation of the **RF-DETR-Seg-Medium** model trained on the [CarDD](https://cardd-ustc.github.io/) dataset for 6-class car damage instance segmentation. The model is benchmarked against the **DCN+ (ResNet-101)** baseline reported in the original CarDD paper (Wang et al., IEEE TITS 2023).

---

## 🤖 Model & Dataset Download

> **🔗 [Google Drive — Models & Datasets](https://drive.google.com/drive/folders/1KxSQLXaXxa2vOlMYNeq6UL5Ua1hEDirn?usp=drive_link)**

| Item | Description |
|------|-------------|
| `checkpoint_best_total.pth` | Trained RF-DETR-Seg-Medium damage model |
| CarDD test set | 374 images · 785 annotations · 6 damage classes |
| `metrics.csv` | Raw training log |

---

## 📦 Dataset Details

- **Source:** The [CarDD (Car Damage Detection)](https://cardd-ustc.github.io/) dataset — a publicly available benchmark designed for vision-based car damage detection.
- **Training resolution:** images were resized to **1152 × 1152** for training.
- **Evaluation resolution:** the **original, un-resized** CarDD images are used (see [Evaluation Methodology](#-evaluation-methodology)). The test split is the official one — 374 images out of CarDD's 4,000 (2,816 train / 810 val / 374 test), the same split the paper's DCN+ row was measured on.
- **Format:** COCO JSON with polygon segmentation masks.

### Class Distribution

| Class | Train Images | Train Instances | Valid Images | Valid Instances | Test Images | Test Instances |
|---------------|-------------|----------------|-------------|----------------|------------|----------------|
| crack | 434 | 651 | 122 | 177 | 48 | 70 |
| dent | 1,242 | 1,806 | 352 | 501 | 157 | 236 |
| glass shatter | 469 | 475 | 134 | 135 | 71 | 71 |
| lamp broken | 489 | 494 | 139 | 141 | 65 | 69 |
| scratch | 1,507 | 2,560 | 431 | 728 | 183 | 307 |
| tire flat | 219 | 225 | 59 | 62 | 31 | 32 |
| **TOTAL** | **2,816** | **6,211** | **810** | **1,744** | **374** | **785** |

---

## 🏷️ Damage Classes

| Class | Description |
|-------|-------------|
| `crack` | Surface cracks on body panels |
| `dent` | Deformation / dents on panels |
| `glass shatter` | Shattered windshield or window glass |
| `lamp broken` | Broken headlight or tail light |
| `scratch` | Paint scratches or abrasions |
| `tire flat` | Flat / deflated tyre |

---

## 📂 Folder Structure

```
Car Damage/
├── Testing/
│   ├── evaluate_rfdetr_cardd_full.py   ← Full test-set evaluation script (main)
│   └── plots/                          ← Main evaluation results (15 charts + JSON)
│       ├── results.json
│       ├── results_coco_segm.json
│       ├── results_coco_bbox.json
│       ├── overall_radar.png
│       ├── pr_curve_per_category_mask.png
│       ├── pr_curve_per_category_box.png
│       ├── f1_vs_threshold_mask.png
│       ├── f1_vs_threshold_box.png
│       ├── ap_comparison_bar_mask.png
│       ├── ap_comparison_bar_box.png
│       ├── precision_recall_f1_summary.png
│       ├── iou_threshold_ap_curve.png
│       ├── confusion_heatmap_mask.png
│       ├── confusion_heatmap_box.png
│       ├── rfdetr_mask_vs_box_overall.png
│       ├── rfdetr_mask_vs_box_per_category.png
│       ├── comparison_overall_mask_box.png
│       └── comparison_per_category_mask_box.png
│
└── Training and Validation/
    ├── plot_training_metrics.py        ← Training chart generator from CSV
    ├── training_stats/
    │   └── metrics.csv                 ← Raw training log
    └── training_plots/                 ← Generated training charts
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

### Script: `Testing/evaluate_rfdetr_cardd_full.py`

Runs the trained RF-DETR model against the CarDD test split and computes full COCO-standard AP metrics for both instance segmentation (mask) and object detection (bounding box), then compares them with the DCN+ (ResNet-101) baseline from the CarDD paper.

---

### 🔬 Evaluation Methodology

#### Images are fed to the model unresized

The model was **trained** on images resized to 1152 × 1152. Evaluation does **not** repeat that resize.

RF-DETR's post-processing (`rfdetr/models/postprocess.py`) bilinearly interpolates the mask *logits* to the size of the image it was handed and only then thresholds them, so `det.mask` is returned in the input image's coordinate space. Feeding the **original** CarDD image therefore produces a mask already in original coordinates, with **no projection step at all**. The mask's shape is asserted against the image dimensions and a mismatch raises, so pointing `--images_dir` at a resized copy fails loudly instead of silently rescaling.

This is also the more accurate route. The alternative — infer on a resized copy, then `cv2.resize` the *already-binarised* mask back down — resamples a binary image and re-thresholds it, which is strictly lossier than interpolating the logits once. The loss falls on the classes whose boundaries are only a few pixels wide — `crack` and `scratch` — rather than on large, blob-like damage.

Because the model squashes whatever it is given to a square internally, giving it the native image is not only lossless at the boundary but also removes an entire class of coordinate-space bug: predictions and ground truth are in the same frame by construction rather than by a conversion that has to be trusted.

#### Bounding box derivation

Boxes are the **tight bounding box of the predicted mask**, not the model's box head output. This guarantees pixel-exact consistency between a mask and its box, so mask AP and box AP describe the same predicted region.

#### One annotation file drives the evaluation

`--orig_annotations` (the original, un-resized CarDD test annotations) is **required** and is the single source of the image list, the scored ground truth, the image IDs and the true image dimensions. Because one file supplies all of these, the set of images evaluated cannot drift out of step with the ground truth scored.

`--annotations` is **optional** and is never scored. Its only job is to verify the model's label order. RF-DETR emits a 0-based class index in the order of the dataset it was *trained* on, and that order cannot be recovered from the original CarDD file — the two genuinely differ:

```
training export order : crack, dent, glass shatter, lamp broken, scratch, tire flat
original CarDD order  : dent, scratch, crack, glass shatter, lamp broken, tire flat
```

Passing the training export makes the script replicate RF-DETR's own `cat2label` mapping (`{cat_id: i for i, cat_id in enumerate(sorted(coco.cats.keys()))}`, unfiltered — a Roboflow dummy supercategory at id 0 counts and shifts every real class by one) and abort if it disagrees with the hardcoded class list. Prediction → ground-truth category mapping is done **by name**, never positionally, so the differing orders are handled correctly.

#### Two thresholds, deliberately separate

| | Value | Purpose |
|---|---|---|
| `--threshold` | `0.001` | **Collection** threshold handed to COCOeval. Must stay low — COCOeval sweeps the score axis itself when building the PR curve, so raising it truncates the recall axis and *understates* AP. Not an operating point. |
| `REPORT_THRESHOLD` | `0.50` | **Reporting** threshold, used only for the P/R/F1 summary and the confusion matrix. A conventional value fixed in advance, so it is not tuned on the test set. |

Quoting a single-threshold precision at 0.001 would measure precision over ~48 detections per ground-truth instance and report a near-zero F1 for a healthy model, which is why the two are kept apart. AP is unaffected by `REPORT_THRESHOLD`.

#### Area Range Configuration (APs / APm / APl)

The CarDD paper uses **non-standard area ranges** for size-stratified evaluation (Table IV):

| Size Category | CarDD Paper (used here) | COCO Standard |
|---------------|------------------------|---------------|
| **Small** | area < 128² | area < 32² |
| **Medium** | 128² ≤ area < 256² | 32² ≤ area < 96² |
| **Large** | area ≥ 256² | area ≥ 96² |

`COCOeval.params.areaRng` is configured to match, so the reported APs / APm / APl are **directly comparable** with the paper's DCN+ numbers. COCO defaults would produce different values that cannot be fairly compared.

For a segmentation result, COCOeval bins detections by their **mask** area. `pycocotools`' `loadRes` tests `if 'bbox' in anns[0]` *before* `elif 'segmentation' in anns[0]`, so a `bbox` key on a segm result silently switches the binning to *box* area while the ground truth stays binned by mask area — computing size-stratified AP over two partitions that do not correspond. The segm results here therefore carry only `image_id`, `category_id`, `segmentation` and `score`, and a runtime assertion after `loadRes` re-raises if a detection's stored `area` ever disagrees with its decoded mask area. (AP, AP50 and AP75 use the "all" area range and are unaffected by this, which is what makes the defect invisible in the headline number.)

#### Evaluator

All headline AP metrics come from **`pycocotools.cocoeval.COCOeval`** with the area ranges above and its standard protocol: 10 IoU thresholds (0.50 : 0.05 : 0.95), 101-point interpolated precision, and a maximum of 100 detections per image.

---

### 🔄 Evaluation Flow

```
1. Load ORIGINAL COCO annotations  (--orig_annotations, required)
   ├─ Single source of: image list · GT masks/boxes · image IDs · dimensions
   └─ Guard: aborts if the stored GT 'area' fields are box areas, not mask areas

2. Build the model class-index → name map from the hardcoded class list
   └─ Optional: verify it against the training export (--annotations), replicating
      RF-DETR's own cat2label mapping; abort on disagreement

3. Map class name → original category_id  (by name, never positionally)
   └─ Abort if any class the model can predict is absent from the scored annotations

4. Decode all GT masks from the original annotations
   └─ Polygon → binary mask → RLE (native image coordinates)

5. Compute the ground-truth support table (per class × CarDD size bin, by mask area)

6. Load RF-DETR-Seg-Medium model checkpoint

7. For each test image (374 images):
   ├─ 7a. Resolve the image file (counted as failed if it cannot be found)
   ├─ 7b. Run inference on the ORIGINAL image
   │     └─ asserts the file's dimensions match the annotations, and that the
   │        returned mask has the same shape as the image
   ├─ 7c. Derive the tight bounding box from each mask
   └─ 7d. Build COCO result entries (segm without a bbox key; bbox with one)
   After the loop: abort if more than 10% of images could not be read, rather
   than reporting metrics computed over the remainder

8. Run pycocotools COCOeval
   ├─ iouType = "segm" → Mask AP / AP50 / AP75 / APs / APm / APl
   └─ iouType = "bbox" → Box  AP / AP50 / AP75 / APs / APm / APl

9. Per-category evaluation (6 categories × 2 iouTypes)
   └─ Per-category AP, AP50, AP75 + PR curve data

10. Supplementary metrics at REPORT_THRESHOLD = 0.50
    ├─ Precision / Recall / F1 (per-class greedy matching @ IoU 0.50)
    └─ Confusion matrix (class-agnostic greedy matching @ IoU 0.50)

11. Generate 15 evaluation charts + save full results JSON
```

---

### 📈 Results

**Dataset:** CarDD test split — **374 images, 785 annotations** (0 images failed)
**Evaluator:** `pycocotools.cocoeval.COCOeval`, native original image coordinates
**Area ranges:** CarDD non-standard (small < 128², medium 128²–256², large ≥ 256²)
**Prediction pass:** 97.8 s for 374 images (~0.26 s/image) on an NVIDIA RTX 5070 Laptop GPU. This is wall-clock for the whole loop — image load, forward pass, and RLE encoding of 37,326 masks — not an inference benchmark. It is also power-state dependent: the identical run on battery rather than mains took about 40% longer, with every accuracy metric unchanged.

#### Overall Metrics vs DCN+ (ResNet-101)

| Metric | RF-DETR (Ours) | DCN+ (Paper) | Δ |
|--------|---------------|-------------|---|
| **Mask AP** (IoU 0.50:0.95) | **59.5** | 57.0 | **+2.5** |
| **Mask AP50** | **78.9** | 77.7 | **+1.2** |
| **Mask AP75** | **59.7** | 58.4 | **+1.3** |
| **Box AP** (IoU 0.50:0.95) | **63.1** | 60.6 | **+2.5** |
| **Box AP50** | 78.6 | 78.8 | −0.2 |
| **Box AP75** | 64.7 | 64.8 | −0.1 |

> RF-DETR exceeds DCN+ (ResNet-101) on mask AP, AP50 and AP75, and on box AP. Box AP50 and AP75 differ by ≤ 0.2, which is within rounding and is best read as a tie. Both models are evaluated with `pycocotools.cocoeval.COCOeval` under the CarDD paper's area range configuration, on the same official test split.

#### Size-Stratified AP vs DCN+ (CarDD Area Ranges)

Reference values are the **DCN+ ResNet-101** row of Table IV in the CarDD paper, which reports APS / APM / APL alongside the overall metrics. The **GT instances** column gives the number of ground-truth instances each figure is averaged over.

| Metric | RF-DETR Mask | DCN+ Mask | Δ Mask | RF-DETR Box | DCN+ Box | Δ Box | GT instances |
|--------|-------------|-----------|--------|-------------|----------|-------|--------------|
| **APs** (small < 128²) | **39.4** | 34.6 | **+4.8** | **44.8** | 37.1 | **+7.7** | 260 |
| **APm** (medium 128²–256²) | **53.9** | 44.0 | **+9.9** | **58.0** | 48.0 | **+10.0** | 261 |
| **APl** (large ≥ 256²) | **72.1** | 71.6 | **+0.5** | 65.6 | 66.0 | −0.4 | 264 |

> **Note on area ranges:** CarDD's thresholds (128², 256²) are much larger than COCO's (32², 96²), because car damage instances in real-world images are generally larger than typical COCO objects.
>
> **Note on the comparison:** RF-DETR leads on every mask size bin, with the largest margin on medium instances (+9.9). The three bins are near-evenly populated (260 / 261 / 264 of 785), so all three figures are well supported. On box, the small and medium gains are larger still (+7.7, +10.0) while large is a tie.

#### Ground-Truth Support (per class × size bin)

Instance counts behind the numbers above, measured from decoded mask area. This table is the reason the per-class results look the way they do:

| Class | Mask AP | All | Small | Medium | Large | Dominant size |
|-------|---------|-----|-------|--------|-------|---------------|
| crack | 20.7 | 70 | **64** | 5 | 1 | small (91%) |
| scratch | 34.3 | 307 | **128** | 123 | 56 | small (42%) |
| dent | 37.4 | 236 | 53 | **109** | 74 | medium (46%) |
| lamp broken | 75.9 | 69 | 13 | 15 | **41** | large (59%) |
| tire flat | 93.7 | 32 | 1 | 3 | **28** | large (88%) |
| glass shatter | 94.8 | 71 | 1 | 6 | **64** | large (90%) |
| **TOTAL** | | **785** | **260** | **261** | **264** | |

> Ordered by AP, the dominant size runs small → small → medium → large → large → large without exception: the weakest classes are the ones made of the smallest objects. Cracks are hairline features (91% fall in the small bin) and score lowest; glass shatter and tire flat are large, well-defined regions and score highest.
>
> **Individual cells in this table must not be used to derive per-class size-stratified AP.** Six of them rest on ≤ 6 instances — `crack` large = **1**, `glass shatter` small = **1**, `tire flat` small = **1**, `tire flat` medium = 3, `crack` medium = 5, `glass shatter` medium = 6. An AP over one annotation is a coin flip. The *aggregate* APs / APm / APl above are well supported and are the figures to quote.

#### Per-Category Mask AP

| Class | RF-DETR | DCN+ | Δ | GT instances |
|-------|---------|------|---|--------------|
| crack | **20.7** | 16.6 | **+4.1** | 70 |
| dent | 37.4 | 40.5 | −3.1 | 236 |
| glass shatter | **94.8** | 89.6 | **+5.2** | 71 |
| lamp broken | **75.9** | 70.8 | **+5.1** | 69 |
| scratch | 34.3 | 34.3 | 0.0 | 307 |
| tire flat | **93.7** | 90.0 | **+3.7** | 32 |

#### Per-Category Box AP

| Class | RF-DETR | DCN+ | Δ | GT instances |
|-------|---------|------|---|--------------|
| crack | 29.4 | 29.6 | −0.2 | 70 |
| dent | 39.5 | 42.2 | −2.7 | 236 |
| glass shatter | **95.4** | 90.1 | **+5.3** | 71 |
| lamp broken | **77.6** | 69.5 | **+8.1** | 69 |
| scratch | 41.4 | 42.3 | −0.9 | 307 |
| tire flat | **95.1** | 90.2 | **+4.9** | 32 |

#### Mask vs Box AP (RF-DETR Internal Comparison)

| Metric | Mask | Box | Δ (Box − Mask) |
|--------|------|-----|----------------|
| AP | 59.5 | 63.1 | +3.6 |
| AP50 | 78.9 | 78.6 | −0.3 |
| AP75 | 59.7 | 64.7 | +5.0 |

> Box and mask AP50 are effectively identical: at a loose IoU the model localises the damage equally well either way. The gap opens at AP75 (+5.0), which is where mask *boundary* precision starts to matter — the cost of segmenting a damage region exactly, rather than merely bounding it.

#### Precision / Recall / F1

| Mode | Threshold | Precision | Recall | F1 | TP | FP | FN |
|------|-----------|-----------|--------|-----|----|----|-----|
| Mask | 0.50 | 0.7835 | 0.6178 | **0.6909** | 485 | 134 | 300 |
| Box | 0.50 | 0.7803 | 0.6153 | **0.6880** | 483 | 136 | 302 |

These are measured at the fixed `REPORT_THRESHOLD = 0.50`, chosen in advance. For reference, the F1-maximising threshold *selected on this test set* is 0.5306, giving F1 0.6925 — a gain of 0.0016 over the fixed value, so there is no meaningful tuning advantage to be had and the fixed-threshold numbers can be quoted directly.

#### Confusion (mask, IoU 0.50, score ≥ 0.50)

| Outcome | Count |
|---------|-------|
| Correctly classified | 485 |
| Matched but mislabelled | 10 |
| Missed (no prediction) | 290 |
| **Total ground truth** | **785** |
| Background false positives | 124 |

> Class confusion is rare — only 10 of 785 instances are matched by a prediction of the wrong class. The dominant error mode is **missed detections** (290), not misclassification. The confusion matrix is built with class-agnostic greedy matching, so a prediction can claim a ground-truth instance of a different class; this is what makes off-diagonal cells possible.

### Usage

```bash
python Testing/evaluate_rfdetr_cardd_full.py \
    --images_dir       /path/to/CarDD_COCO_original/test \
    --orig_annotations /path/to/CarDD_COCO_original/test/_annotations.coco.json \
    --annotations      /path/to/training_export/test/_annotations.coco.json \
    --checkpoint       /path/to/checkpoint_best_total.pth \
    --resolution  960 \
    --threshold   0.001 \
    --output_json Testing/plots/results.json \
    --plots_dir   Testing/plots
```

> `--images_dir` must point at the **original, un-resized** images. Pointing it at a resized copy raises rather than silently rescaling.

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--images_dir` | — | Path to the **original, un-resized** test image directory |
| `--orig_annotations` | `None` | **Required.** Original CarDD test annotation JSON — supplies the image list, scored ground truth, image IDs and true dimensions |
| `--annotations` | `None` | *Optional.* Annotation JSON of the dataset the model was **trained** on. Not scored, contributes no images; used only to verify the class-index order. Omit it and the hardcoded class list is taken on trust |
| `--checkpoint` | — | Path to `.pth` model checkpoint |
| `--resolution` | `960` | RF-DETR inference resolution |
| `--threshold` | `0.001` | Collection threshold for COCOeval. Keep it low — raising it truncates the recall axis and understates AP |
| `--output_json` | `results.json` | Output JSON with all metrics and chart data |
| `--plots_dir` | `./plots` | Output directory for charts |

`REPORT_THRESHOLD = 0.50` is a constant in the script header, not a CLI flag. It affects only the P/R/F1 summary and the confusion matrices; AP is independent of it.

### Generated Charts (15 total)

| # | File | Description |
|---|------|-------------|
| 1 | `pr_curve_per_category_mask.png` | Per-category Precision–Recall curves (mask). Solid = IoU 0.50, dashed = IoU 0.55–0.95. |
| 2 | `pr_curve_per_category_box.png` | Per-category Precision–Recall curves (box) |
| 3 | `f1_vs_threshold_mask.png` | F1 / Precision / Recall vs confidence threshold (mask), with the fixed operating point and the test-selected oracle both marked |
| 4 | `f1_vs_threshold_box.png` | As above, for box |
| 5 | `ap_comparison_bar_mask.png` | Per-category bar chart: RF-DETR vs DCN+ (mask AP) with Δ annotations |
| 6 | `ap_comparison_bar_box.png` | Per-category bar chart: RF-DETR vs DCN+ (box AP) with Δ annotations |
| 7 | `overall_radar.png` | Radar chart: AP / AP50 / AP75 for mask & box — RF-DETR vs DCN+ |
| 8 | `precision_recall_f1_summary.png` | Summary bar: P / R / F1 at the fixed operating threshold, with the oracle drawn as a dashed outline behind it |
| 9 | `iou_threshold_ap_curve.png` | Mean AP across categories vs IoU threshold, with DCN+'s published AP50 / AP75 as reference markers |
| 10 | `confusion_heatmap_mask.png` | Confusion matrix (mask @ IoU 0.50). Rows sum to 1; final column is ground truth claimed by no prediction; background false positives reported beneath |
| 11 | `confusion_heatmap_box.png` | As above, for box |
| 12 | `rfdetr_mask_vs_box_overall.png` | RF-DETR only: Mask AP vs Box AP grouped bar (AP / AP50 / AP75) |
| 13 | `rfdetr_mask_vs_box_per_category.png` | RF-DETR only: Mask AP vs Box AP per category |
| 14 | `comparison_overall_mask_box.png` | 4-way comparison: RF-DETR Mask / Box vs DCN+ Mask / Box (AP / AP50 / AP75) |
| 15 | `comparison_per_category_mask_box.png` | 4-way per-category: RF-DETR Mask / Box vs DCN+ Mask / Box |

---

## 📉 Training & Validation

### Script: `Training and Validation/plot_training_metrics.py`

Reads the training log CSV and produces 12 charts tracking loss curves, mAP progression, and per-category AP over all training epochs.

> **These charts are not test-set results.** `val/*` columns are the split RF-DETR scored at the end of each epoch, and `train/*` are training-time losses. Neither is the held-out test set, and the `val/*` numbers are computed with a different evaluator configuration from the results above — the two must not be quoted interchangeably. Every generated figure carries a provenance line stating this.

### Usage

```bash
python "Training and Validation/plot_training_metrics.py" \
    --csv "Training and Validation/training_stats/metrics.csv" \
    --output_dir "Training and Validation/training_plots"
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--csv` | `metrics.csv` | Path to training metrics CSV |
| `--output_dir` | `./training_plots` | Output directory for charts |

### Generated Charts (12 total)

| # | File | Description |
|---|------|-------------|
| 1 | `val_map_overview.png` | Val Box mAP@50/50:95 + Segm mAP@50/50:95 over epochs |
| 2 | `val_map75_mar.png` | Val mAP@75 and Mean Average Recall over epochs |
| 3 | `val_precision_recall_f1.png` | Val Precision / Recall / F1 over epochs |
| 4 | `val_per_category_ap.png` | Per-damage-class AP over epochs (2×3 grid) |
| 5 | `train_loss_total.png` | Total training loss over steps |
| 6 | `train_loss_components.png` | CE / BBox L1 / GIoU / Mask CE / Mask Dice per step |
| 7 | `train_loss_auxiliary.png` | Auxiliary decoder layer losses (layers 0–3 + enc) |
| 8 | `val_ema_vs_live.png` | EMA mAP vs live mAP (box + segm) |
| 9 | `val_category_radar.png` | Radar chart: per-category AP at final epoch |
| 10 | `val_loss_curve.png` | Validation loss over epochs |
| 11 | `train_cardinality_error.png` | Cardinality error per decoder layer over steps |
| 12 | `combined_overview.png` | 2×2 dashboard: loss / mAP / P-R-F1 / per-cat AP bar |

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

- **CarDD Dataset & DCN+ Baseline:** Wang, X., Li, W., & Wu, Z. (2023). *CarDD: A New Dataset for Vision-Based Car Damage Detection*. IEEE Transactions on Intelligent Transportation Systems, 24(7), 7202–7214. [DOI: 10.1109/TITS.2023.3258480](https://doi.org/10.1109/TITS.2023.3258480) — [Project Page](https://cardd-ustc.github.io/) — [arXiv:2211.00945](https://arxiv.org/abs/2211.00945)
- **RF-DETR:** [Roboflow RF-DETR — Real-Time Detection Transformer](https://github.com/roboflow/rf-detr)
- **Evaluation standard:** COCO-style AP (IoU = 0.50:0.95, 101-point interpolation, maxDets 100) computed with `pycocotools.cocoeval.COCOeval` using CarDD-specific area ranges for size-stratified metrics.
