# Car Parts — Model Evaluation

Evaluation of the **RF-DETR-Seg-Medium** model trained for **19-class car part instance segmentation**. Unlike the damage model, there is no published external baseline — all charts are self-contained RF-DETR metrics.

---

## 🤖 Model & Dataset Download

> **🔗 [Google Drive — Models & Datasets](https://drive.google.com/drive/folders/1KxSQLXaXxa2vOlMYNeq6UL5Ua1hEDirn?usp=drive_link)**

| Item | Description |
|------|-------------|
| `checkpoint_best_total.pth` | Trained RF-DETR-Seg-Medium car parts model |
| Car Parts test set | 540 images · 4,591 annotations · 19 part classes |
| `metrics.csv` | Raw training log |

---

## 📦 Dataset Details

- **Source:** Multiple publicly available car part datasets were combined and unified into a single dataset covering 19 part classes.
- **Resolution:** All images are resized to **1152 × 1152** pixels before training and evaluation.
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
│   └── plots_parts/                    ← Generated evaluation charts + results JSON
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
│       ├── iou_ap_curve.png
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

Runs the trained RF-DETR model against the Car Parts test split and computes full COCO-style AP metrics for both instance segmentation (mask) and bounding box (box) detection across all 19 classes.

### 📈 Results

**Dataset:** Car Parts test split — **540 images, 4,591 annotations, 19 categories**

#### Overall Metrics

| Metric | Mask | Box |
|--------|------|-----|
| **AP** (IoU 0.50:0.95) | **62.3** | **64.3** |
| **AP50** | **85.4** | **86.6** |
| **AP75** | **68.2** | **71.0** |

#### Best Threshold (F1)

| Mode | Threshold | Precision | Recall | F1 |
|------|-----------|-----------|--------|----|
| Mask | 0.469 | 0.865 | 0.868 | **0.867** |
| Box | 0.469 | 0.867 | 0.870 | **0.868** |

#### Per-Category Mask AP (sorted high → low)

| Class | Mask AP | Mask AP50 | Mask AP75 |
|-------|---------|-----------|-----------|
| Front_Windshield_Glass | **90.1** | 99.0 | 98.2 |
| Hood_Bonnet | **87.3** | 98.0 | 95.6 |
| Front_Bumper | **85.0** | 97.5 | 92.2 |
| Front_Door | **80.4** | 96.9 | 92.9 |
| Diggi_Back_Door_Glass | **79.1** | 95.9 | 88.8 |
| Rear_Bumper | **75.7** | 93.9 | 84.1 |
| Rear_Door | **73.0** | 93.2 | 85.0 |
| Headlight | **70.0** | 94.1 | 79.9 |
| Diggi_Back_Door | **66.2** | 83.5 | 67.7 |
| Taillight | **65.0** | 89.9 | 72.5 |
| Fender | **59.5** | 89.2 | 69.5 |
| tyre | **59.0** | 85.0 | 68.0 |
| Rear_Door_Glass | **53.9** | 86.1 | 56.6 |
| Front_Door_Glass | **51.2** | 82.8 | 52.6 |
| Grill | **49.2** | 73.8 | 60.3 |
| Side_Mirror | **56.8** | 87.4 | 68.1 |
| Quarter_Panel | **36.0** | 58.1 | 37.0 |
| Roof | **27.3** | 63.3 | 18.9 |
| Running_Board | **19.5** | 54.9 | 7.6 |

> **Note:** Roof and Running_Board have lower AP because they are rarely fully visible and overlap significantly with adjacent panels.

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
| `--annotations` | — | Path to COCO-format annotations JSON |
| `--checkpoint` | — | Path to `.pth` model checkpoint |
| `--resolution` | `960` | RF-DETR inference resolution |
| `--threshold` | `0.001` | Minimum confidence (low = evaluate all detections) |
| `--output_json` | `rfdetr_carparts_results.json` | Output JSON with all metrics |
| `--plots_dir` | `./plots_parts` | Output directory for charts |

### Generated Charts (13 total)

| # | File | Description |
|---|------|-------------|
| 1–2 | `pr_curve_mask_page1/2.png` | PR curves per class — mask (2 pages, 10+9 categories) |
| 3–4 | `pr_curve_box_page1/2.png` | PR curves per class — box |
| 5 | `f1_vs_threshold_mask.png` | F1 / P / R vs confidence threshold (mask) |
| 6 | `f1_vs_threshold_box.png` | F1 / P / R vs confidence threshold (box) |
| 7 | `ap_per_category_mask.png` | Horizontal bar — per-class mask AP (sorted) |
| 8 | `ap_per_category_box.png` | Horizontal bar — per-class box AP (sorted) |
| 9 | `mask_vs_box_overall.png` | Grouped bar — Mask vs Box for AP / AP50 / AP75 |
| 10 | `mask_vs_box_per_category.png` | Grouped bar — Mask vs Box per category |
| 11 | `iou_ap_curve.png` | Mean AP vs IoU threshold (0.50 → 0.95) |
| 12 | `ap50_ap75_bar.png` | AP50 vs AP75 per category — gap shows mask precision |
| 13 | `confusion_heatmap_mask.png` | TP-rate heatmap per GT class (mask @ IoU=0.50) |

---

## 📉 Training & Validation

### Script: `Training and Validation/plot_parts_metrics.py`

Reads the training log CSV and produces 12 publication-quality charts tracking loss curves, mAP progression, EMA tracking, and per-category AP over all training epochs for all 19 part classes.

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
| 5 | `train_loss_total.png` | Total training loss over steps (raw + smoothed) |
| 6 | `train_loss_components.png` | CE / BBox L1 / GIoU / Mask CE / Mask Dice per step |
| 7 | `train_loss_auxiliary.png` | Auxiliary decoder layer losses (layers 0–2 + enc) |
| 8 | `val_ema_vs_live.png` | EMA mAP vs live mAP (box + segm) |
| 9 | `val_category_radar.png` | Two-panel radar: per-part AP at final epoch (Parts 1–10, 11–19) |
| 10 | `val_loss_curve.png` | Validation loss over epochs |
| 11 | `train_cardinality_error.png` | Cardinality error per decoder layer over steps |
| 12 | `combined_overview.png` | 2×2 dashboard: loss / mAP / P-R-F1 / per-part horizontal bar |

---

## ⚙️ Requirements

```bash
pip install -r requirements.txt
```

---

## 📖 References

- **RF-DETR:** [Roboflow RF-DETR — Real-Time Detection Transformer](https://github.com/roboflow/rf-detr)
- **Evaluation metric standard:** COCO-style AP (IoU=0.50:0.95, 101-point interpolation) computed using a custom evaluation script (not pycocotools).
- **Dataset annotation format:** COCO JSON with polygon segmentation masks
