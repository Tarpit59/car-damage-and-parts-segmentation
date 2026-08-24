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

- **Source:** The [CarDD (Car Damage Detection)](https://cardd-ustc.github.io/) dataset is used — a publicly available benchmark specifically designed for vision-based car damage detection.
- **Resolution:** All images are resized to **1152 × 1152** pixels before training and evaluation.
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
│   └── plots/                          ← Main evaluation results (14 charts + JSON)
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

Runs the trained RF-DETR model against the CarDD test split and computes full COCO-standard AP metrics for both instance segmentation (mask) and object detection (bounding box). All results are evaluated in **original image coordinates** using `pycocotools.cocoeval.COCOeval` and compared with the DCN+ (ResNet-101) baseline from the CarDD paper.

---

### 🔬 Evaluation Methodology

#### Why Not Evaluate Directly on Resized Images?

The RF-DETR model is trained on images resized to 1152 × 1152 pixels and internally infers at the configured resolution (960 px). However, the CarDD ground-truth annotations are defined in the **original image coordinate space** (varying dimensions). If predictions from the model's output resolution were directly compared against these ground-truth annotations using COCO evaluation, the IoU computation would operate across mismatched coordinate spaces — producing **meaningless metrics**.

Therefore, all model predictions (masks and bounding boxes) are **projected back to original image coordinates** before running `pycocotools.cocoeval.COCOeval`. This ensures that IoU, AP, and all derived metrics are computed in the same coordinate space as the CarDD paper's reported baselines.

#### Coordinate Projection Pipeline

- **Mask projection:** The model's output mask (at its internal resolution) is resized to the original image dimensions using **bilinear interpolation** followed by a **0.5 threshold** binarisation (`cv2.resize` + threshold). The resulting binary mask is then RLE-encoded for COCO evaluation.
- **Bounding box derivation:** Rather than scaling the model's bounding box coordinates (which can accumulate rounding errors), the bounding box is derived as the **tight bounding box of the projected mask** in original image coordinates. This ensures pixel-exact consistency between the mask and its corresponding bounding box.

#### Dual-Annotation Approach

The evaluation uses two separate COCO annotation files:

1. **Resized annotations** (`--annotations`): Provides the image file list and the model's class-index mapping (the model was trained on resized data, so its class ordering matches the resized file's category IDs).
2. **Original annotations** (`--orig_annotations`): Provides the ground-truth masks and bounding boxes in **original image coordinates**, along with original image IDs and dimensions used for coordinate projection.

Image matching between datasets is handled by a **3-step filename fallback**: exact basename match → Roboflow filename decoding (stripping `.rf.<hash>` suffixes) → stem-only matching.

#### Area Range Configuration (APs / APm / APl)

The CarDD paper uses **non-standard area ranges** for size-stratified evaluation (Table IV), which differ from the default COCO definition:

| Size Category | CarDD Paper (used here) | COCO Standard |
|---------------|------------------------|---------------|
| **Small** | area < 128² | area < 32² |
| **Medium** | 128² ≤ area < 256² | 32² ≤ area < 96² |
| **Large** | area ≥ 256² | area ≥ 96² |

Our evaluation script configures `COCOeval.params.areaRng` to match the CarDD paper's non-standard ranges, ensuring that the reported APs / APm / APl values are **directly comparable** with the DCN+ baseline numbers from the paper. This is critical: using COCO default area ranges would produce different APs / APm / APl values that cannot be fairly compared against the CarDD paper's results.

#### Evaluator

All headline AP metrics are computed using **`pycocotools.cocoeval.COCOeval`** — the standard COCO evaluation library — with the area range configuration described above. The evaluator uses 101-point precision interpolation across 10 IoU thresholds (0.50 : 0.05 : 0.95), consistent with the COCO evaluation protocol.

---

### 🔄 Evaluation Flow

```
1. Load RESIZED COCO annotations
   └─ Provides image file list + model class-index → name mapping

2. Load ORIGINAL COCO annotations
   └─ Provides GT masks/boxes in original coordinates, image IDs, image dimensions

3. Build cross-dataset mappings
   ├─ model_idx_to_name: class-index (0-based) → class name (from resized dataset)
   ├─ fname_to_orig_id: filename → original image_id
   └─ name_to_orig_cat_id: class name → original category_id

4. Decode all GT masks from original annotations
   └─ Polygon → binary mask → RLE (in original coordinates)

5. Load RF-DETR-Seg-Medium model checkpoint

6. For each test image (374 images):
   ├─ 6a. Resolve original image_id via 3-step filename matching
   │       (exact → Roboflow decode → stem match)
   ├─ 6b. Get original image dimensions (H × W)
   ├─ 6c. Run model inference → raw masks at model resolution
   ├─ 6d. Project masks to original coordinates (bilinear + 0.5 threshold)
   ├─ 6e. Derive bounding boxes from projected masks (tight bbox)
   └─ 6f. Build COCO-format result entries (segm + bbox)

7. Run pycocotools COCOeval
   ├─ iouType = "segm" → Mask AP / AP50 / AP75 / APs / APm / APl
   └─ iouType = "bbox" → Box AP / AP50 / AP75 / APs / APm / APl

8. Per-category evaluation (6 categories × 2 iouTypes)
   └─ Extract per-category AP, AP50, AP75 + PR curve data

9. Compute threshold-dependent metrics
   └─ F1 / Precision / Recall vs confidence threshold sweep (50 steps)

10. Generate 14 evaluation charts + save full results JSON
```

---

### 📈 Results

**Dataset:** CarDD test split — **374 images, 785 annotations**
**Evaluator:** `pycocotools.cocoeval.COCOeval` in original image coordinates
**Area ranges:** CarDD non-standard (small < 128², medium 128²–256², large ≥ 256²)

#### Overall Metrics vs DCN+ (ResNet-101)

| Metric | RF-DETR (Ours) | DCN+ (Paper) | Δ |
|--------|---------------|-------------|---|
| **Mask AP** (IoU 0.50:0.95) | **59.9** | 57.0 | **+2.9** |
| **Mask AP50** | **79.8** | 77.7 | **+2.1** |
| **Mask AP75** | **60.1** | 58.4 | **+1.7** |
| **Box AP** (IoU 0.50:0.95) | **63.3** | 60.6 | **+2.7** |
| **Box AP50** | **79.4** | 78.8 | **+0.6** |
| **Box AP75** | **65.3** | 64.8 | **+0.5** |

> RF-DETR achieves higher AP than DCN+ (ResNet-101) across all overall metrics (AP, AP50, AP75) for both mask and box evaluation. Both models are evaluated using `pycocotools.cocoeval.COCOeval` with the CarDD paper's area range configuration. Predictions are projected to original image coordinates before evaluation — a standard step when the model's inference resolution differs from the annotation coordinate space.

#### Size-Stratified AP vs DCN+ (CarDD Area Ranges)

These values use the CarDD paper's non-standard area ranges (small < 128², medium 128²–256², large ≥ 256²), **not** COCO defaults. Our evaluation configures `COCOeval.params.areaRng` to match these CarDD-specific ranges for fair comparison with the DCN+ baseline.
The DCN+ reference values below are taken from the **DCN+ ResNet-101** row in Table IV of the CarDD paper.

| Metric | RF-DETR Mask | DCN+ Mask | Δ Mask | RF-DETR Box | DCN+ Box | Δ Box |
|--------|-------------|-----------|--------|-------------|----------|-------|
| **APs** (small < 128²) | **41.2** | 34.6 | **+6.6** | **44.0** | 37.1 | **+6.9** |
| **APm** (medium 128²–256²) | **54.4** | 44.0 | **+10.4** | **57.5** | 48.0 | **+9.5** |
| **APl** (large ≥ 256²) | 62.7 | **71.6** | -8.9 | 65.5 | **66.0** | -0.5 |

> **Note on area ranges:** The CarDD paper defines area ranges as small < 128², medium 128²–256², large ≥ 256², which are significantly larger thresholds than the COCO standard (small < 32², medium 32²–96², large ≥ 96²). This is because car damage instances in real-world images are generally larger than typical COCO objects.
>
> **Note on size-stratified comparison:** RF-DETR improves over the DCN+ ResNet-101 baseline on small and medium-sized damage instances (APs: +6.6 mask, +6.9 box; APm: +10.4 mask, +9.5 box). DCN+ remains stronger on large instances (APl: -8.9 mask, -0.5 box), while RF-DETR still leads on the overall AP, AP50, and AP75 metrics shown above. The small/medium gains may be partly attributable to the transformer architecture's global attention mechanism, which captures long-range context effectively for fine-grained damage detection.

#### Per-Category Mask AP

| Class | RF-DETR | DCN+ | Δ |
|-------|---------|------|---|
| crack | 24.5 | 16.6 | **+7.9** |
| dent | 36.8 | 40.5 | −3.7 |
| glass shatter | **93.4** | 89.6 | **+3.8** |
| lamp broken | **76.3** | 70.8 | **+5.5** |
| scratch | 34.8 | 34.3 | +0.5 |
| tire flat | **93.7** | 90.0 | **+3.7** |

#### Per-Category Box AP

| Class | RF-DETR | DCN+ | Δ |
|-------|---------|------|---|
| crack | 31.3 | 29.6 | +1.7 |
| dent | 39.6 | 42.2 | −2.6 |
| glass shatter | **95.2** | 90.1 | **+5.1** |
| lamp broken | **77.0** | 69.5 | **+7.5** |
| scratch | 42.1 | 42.3 | −0.2 |
| tire flat | **94.5** | 90.2 | **+4.3** |

#### Mask vs Box AP (RF-DETR Internal Comparison)

| Metric | Mask | Box | Δ (Box − Mask) |
|--------|------|-----|----------------|
| AP | 59.9 | 63.3 | +3.4 |
| AP50 | 79.8 | 79.4 | −0.4 |
| AP75 | 60.1 | 65.3 | +5.2 |

#### Best Threshold (F1)

| Mode | Threshold | Precision | Recall | F1 |
|------|-----------|-----------|--------|-----|
| Mask | 0.408 | 0.7034 | 0.6828 | **0.693** |
| Box | 0.449 | 0.7387 | 0.6446 | **0.6884** |

### Usage

```bash
python Testing/evaluate_rfdetr_cardd_full.py \
    --images_dir  /path/to/test/images \
    --annotations /path/to/resized/_annotations.coco.json \
    --orig_annotations /path/to/original/_annotations.coco.json \
    --checkpoint  /path/to/checkpoint_best_total.pth \
    --resolution  960 \
    --threshold   0.001 \
    --output_json Testing/plots/results.json \
    --plots_dir   Testing/plots
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--images_dir` | — | Path to test image directory |
| `--annotations` | — | Path to **resized** COCO annotation JSON (provides image file list and model class-index mapping) |
| `--orig_annotations` | `None` | Path to **original** (un-resized) CarDD test annotation JSON for pycocotools evaluation and coordinate projection. If omitted, falls back to `--annotations`. |
| `--checkpoint` | — | Path to `.pth` model checkpoint |
| `--resolution` | `960` | RF-DETR inference resolution |
| `--threshold` | `0.001` | Minimum confidence threshold (low value = evaluate all detections) |
| `--output_json` | `results.json` | Output JSON with all metrics and chart data |
| `--plots_dir` | `./plots` | Output directory for charts |

### Generated Charts (14 total)

| # | File | Description |
|---|------|-------------|
| 1 | `pr_curve_per_category_mask.png` | Per-category Precision–Recall curves (mask). Solid = IoU 0.50, dashed = IoU 0.55–0.95. |
| 2 | `pr_curve_per_category_box.png` | Per-category Precision–Recall curves (box) |
| 3 | `f1_vs_threshold_mask.png` | F1 / Precision / Recall vs confidence threshold (mask) |
| 4 | `f1_vs_threshold_box.png` | F1 / Precision / Recall vs confidence threshold (box) |
| 5 | `ap_comparison_bar_mask.png` | Per-category bar chart: RF-DETR vs DCN+ (mask AP) with Δ annotations |
| 6 | `ap_comparison_bar_box.png` | Per-category bar chart: RF-DETR vs DCN+ (box AP) with Δ annotations |
| 7 | `overall_radar.png` | Radar chart: AP / AP50 / AP75 for mask & box — RF-DETR vs DCN+ |
| 8 | `precision_recall_f1_summary.png` | Summary bar: P / R / F1 at best confidence threshold (mask + box) |
| 9 | `confusion_heatmap_mask.png` | Detection heatmap: TP rate per GT class (mask @ IoU=0.50) |
| 10 | `confusion_heatmap_box.png` | Detection heatmap: TP rate per GT class (box @ IoU=0.50) |
| 11 | `rfdetr_mask_vs_box_overall.png` | RF-DETR only: Mask AP vs Box AP grouped bar (AP / AP50 / AP75) |
| 12 | `rfdetr_mask_vs_box_per_category.png` | RF-DETR only: Mask AP vs Box AP per category |
| 13 | `comparison_overall_mask_box.png` | 4-way comparison: RF-DETR Mask / Box vs DCN+ Mask / Box (AP / AP50 / AP75) |
| 14 | `comparison_per_category_mask_box.png` | 4-way per-category: RF-DETR Mask / Box vs DCN+ Mask / Box |

---

## 📉 Training & Validation

### Script: `Training and Validation/plot_training_metrics.py`

Reads the training log CSV and produces 12 publication-quality charts tracking loss curves, mAP progression, and per-category AP over all training epochs.

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
| 5 | `train_loss_total.png` | Total training loss over steps (raw + smoothed) |
| 6 | `train_loss_components.png` | CE / BBox L1 / GIoU / Mask CE / Mask Dice per step |
| 7 | `train_loss_auxiliary.png` | Auxiliary decoder layer losses (layers 0–3 + enc) |
| 8 | `val_ema_vs_live.png` | EMA mAP vs live mAP (box + segm) |
| 9 | `val_category_radar.png` | Radar chart: per-category AP at final epoch |
| 10 | `val_loss_curve.png` | Validation loss over epochs |
| 11 | `train_cardinality_error.png` | Cardinality error per decoder layer over steps |
| 12 | `combined_overview.png` | 2×2 dashboard: loss / mAP / P-R-F1 / per-cat AP bar |

---

## ⚙️ Requirements

```bash
pip install -r requirements.txt
```

---

## 📖 References

- **CarDD Dataset & DCN+ Baseline:** Wang, X., Li, W., & Wu, Z. (2023). *CarDD: A New Dataset for Vision-Based Car Damage Detection*. IEEE Transactions on Intelligent Transportation Systems, 24(7), 7202–7214. [DOI: 10.1109/TITS.2023.3258480](https://doi.org/10.1109/TITS.2023.3258480) — [Project Page](https://cardd-ustc.github.io/)
- **RF-DETR:** [Roboflow RF-DETR — Real-Time Detection Transformer](https://github.com/roboflow/rf-detr)
- **Evaluation standard:** COCO-style AP (IoU=0.50:0.95, 101-point interpolation) computed using `pycocotools.cocoeval.COCOeval` with CarDD-specific area ranges for size-stratified metrics.
