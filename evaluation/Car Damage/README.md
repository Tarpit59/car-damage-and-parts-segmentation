# Car Damage — Model Evaluation

Evaluation of the **RF-DETR-Seg-Medium** model trained on the [CarDD](https://cardd-ustc.github.io/) dataset for 6-class car damage instance segmentation. The model is compared against the **DCN+ (ResNet-101)** baseline from the original CarDD paper.

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
│   ├── evaluate_rfdetr_cardd_full.py   ← Full test-set evaluation script
│   └── plots/                          ← Generated evaluation charts + results JSON
│       ├── results_full.json
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

Runs the trained RF-DETR model against the CarDD test split and computes full COCO-style AP metrics for both instance segmentation (mask) and object detection (bounding box). All results are compared with the DCN+ (ResNet-101) paper baseline.

### 📈 Results

**Dataset:** CarDD test split — **374 images, 785 annotations**

#### Overall Metrics vs DCN+ (ResNet-101)

| Metric | RF-DETR (Ours) | DCN+ (Paper) | Δ |
|--------|---------------|-------------|---|
| **Mask AP** (IoU 0.50:0.95) | **60.0** | 57.0 | **+3.0** |
| **Mask AP50** | **79.6** | 77.7 | **+1.9** |
| **Mask AP75** | **60.2** | 58.4 | **+1.8** |
| **Box AP** (IoU 0.50:0.95) | **63.2** | 60.6 | **+2.6** |
| **Box AP50** | **79.4** | 78.8 | **+0.6** |
| **Box AP75** | **65.0** | 64.8 | **+0.2** |

> RF-DETR outperforms DCN+ ResNet-101 across all AP metrics.

#### Per-Category Mask AP

| Class | RF-DETR | DCN+ | Δ |
|-------|---------|------|---|
| crack | 23.9 | 16.6 | **+7.3** |
| dent | 36.6 | 40.5 | −3.9 |
| glass shatter | **93.6** | 89.6 | **+4.0** |
| lamp broken | **76.4** | 70.8 | **+5.6** |
| scratch | 34.7 | 34.3 | +0.4 |
| tire flat | **94.5** | 90.0 | **+4.5** |

#### Per-Category Box AP

| Class | RF-DETR | DCN+ | Δ |
|-------|---------|------|---|
| crack | 30.4 | 29.6 | +0.8 |
| dent | 39.4 | 42.2 | −2.8 |
| glass shatter | **95.9** | 90.1 | **+5.8** |
| lamp broken | **76.8** | 69.5 | **+7.3** |
| scratch | 41.7 | 42.3 | −0.6 |
| tire flat | **95.0** | 90.2 | **+4.8** |

#### Best Threshold (F1)

| Mode | Threshold | Precision | Recall | F1 |
|------|-----------|-----------|--------|----|
| Mask | 0.408 | 0.702 | 0.682 | **0.692** |
| Box | 0.449 | 0.736 | 0.642 | **0.686** |

### Usage

```bash
python Testing/evaluate_rfdetr_cardd_full.py \
    --images_dir  /path/to/test/images \
    --annotations /path/to/_annotations.coco.json \
    --checkpoint  /path/to/checkpoint_best_total.pth \
    --resolution  960 \
    --threshold   0.001 \
    --output_json Testing/plots/results_full.json \
    --plots_dir   Testing/plots
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--images_dir` | — | Path to test image directory |
| `--annotations` | — | Path to COCO-format `_annotations.coco.json` |
| `--checkpoint` | — | Path to `.pth` model checkpoint |
| `--resolution` | `960` | RF-DETR inference resolution |
| `--threshold` | `0.001` | Minimum confidence (low value = evaluate all detections) |
| `--output_json` | `results_full.json` | Output JSON with all metrics |
| `--plots_dir` | `./plots` | Output directory for charts |

### Generated Charts (15 total)

| # | File | Description |
|---|------|-------------|
| 1 | `pr_curve_per_category_mask.png` | PR curves per damage class (mask) |
| 2 | `pr_curve_per_category_box.png` | PR curves per damage class (box) |
| 3 | `f1_vs_threshold_mask.png` | F1 / Precision / Recall vs confidence threshold (mask) |
| 4 | `f1_vs_threshold_box.png` | F1 / Precision / Recall vs confidence threshold (box) |
| 5 | `ap_comparison_bar_mask.png` | Per-category bar: RF-DETR vs DCN+ (mask AP) |
| 6 | `ap_comparison_bar_box.png` | Per-category bar: RF-DETR vs DCN+ (box AP) |
| 7 | `overall_radar.png` | Radar chart: AP / AP50 / AP75 for mask & box |
| 8 | `precision_recall_f1_summary.png` | Summary bar: P / R / F1 at best threshold |
| 9 | `iou_threshold_ap_curve.png` | AP vs IoU threshold (0.50 → 0.95) |
| 10 | `confusion_heatmap_mask.png` | TP-rate heatmap per GT class (mask) |
| 11 | `confusion_heatmap_box.png` | TP-rate heatmap per GT class (box) |
| 12 | `rfdetr_mask_vs_box_overall.png` | RF-DETR Mask AP vs Box AP (overall) |
| 13 | `rfdetr_mask_vs_box_per_category.png` | RF-DETR Mask AP vs Box AP (per category) |
| 14 | `comparison_overall_mask_box.png` | 4-way: RF-DETR Mask / Box vs DCN+ Mask / Box |
| 15 | `comparison_per_category_mask_box.png` | 4-way per-category comparison |

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

- **CarDD Dataset & DCN+ Baseline:** [CarDD: A New Dataset for Vision-Based Car Damage Detection (2023)](https://cardd-ustc.github.io/)
- **RF-DETR:** [Roboflow RF-DETR](https://github.com/roboflow/rf-detr)
- **Evaluation metric standard:** COCO-style AP (IoU=0.50:0.95, 101-point interpolation) computed using a custom evaluation script (not pycocotools).
