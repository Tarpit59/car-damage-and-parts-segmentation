# Car Parts — Model Evaluation

Evaluation of the **RF-DETR-Seg-Nano** model trained for **19-class car part instance segmentation**. Unlike the damage model, there is no published external baseline — all charts are self-contained RF-DETR metrics.

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

Runs the trained RF-DETR model against the Car Parts test split and computes full COCO-standard AP metrics for both instance segmentation (mask) and bounding box (box) detection across all 19 classes. All metrics are evaluated using `pycocotools.cocoeval.COCOeval`.

---

### 🔬 Evaluation Methodology

#### Coordinate Projection

The RF-DETR model infers at a configured resolution (960 px) which may differ from the original image dimensions. To ensure correct IoU computation, all model predictions are **projected back to original image coordinates** before evaluation:

- **Mask projection:** The model's output mask is resized to original image dimensions using **bilinear interpolation** followed by a **0.5 threshold** binarisation. The resulting binary mask is RLE-encoded for COCO evaluation.
- **Bounding box derivation:** Bounding boxes are derived as the **tight bounding box of the projected mask** rather than directly scaling the model's box predictions, ensuring pixel-exact consistency between mask and box.

#### Area Range Configuration (APs / APm / APl)

Unlike the Car Damage evaluation (which uses the CarDD paper's non-standard area ranges), the Car Parts evaluation uses **COCO standard area ranges**:

| Size Category | COCO Standard (used here) | CarDD (used for damage) |
|---------------|--------------------------|------------------------|
| **Small** | area < 32² | area < 128² |
| **Medium** | 32² ≤ area < 96² | 128² ≤ area < 256² |
| **Large** | area ≥ 96² | area ≥ 256² |

This means APs / APm / APl values are **not directly comparable** between the Car Damage and Car Parts evaluations due to the different area range definitions. The COCO standard ranges classify objects at much smaller thresholds, which explains the lower APs / APm values for car parts — most car parts are large-scale objects that fall entirely in the "large" category under COCO's definition.

#### Evaluator

All headline AP metrics are computed using **`pycocotools.cocoeval.COCOeval`** — the standard COCO evaluation library — with default parameters (10 IoU thresholds from 0.50 to 0.95, 101-point precision interpolation, max 100 detections per image).

---

### 🔄 Evaluation Flow

```
1. Load COCO annotations
   └─ Provides image file list, GT masks/boxes, category mappings

2. Decode all GT masks from annotations
   └─ Polygon → binary mask → RLE (in original coordinates)

3. Load RF-DETR-Seg-Nano model checkpoint

4. For each test image (540 images):
   ├─ 4a. Resolve image file path (with Roboflow filename fallback)
   ├─ 4b. Run model inference → raw masks at model resolution
   ├─ 4c. Project masks to original coordinates (bilinear + 0.5 threshold)
   ├─ 4d. Derive bounding boxes from projected masks (tight bbox)
   └─ 4e. Build COCO-format result entries (segm + bbox)

5. Run pycocotools COCOeval
   ├─ iouType = "segm" → Mask AP / AP50 / AP75 / APs / APm / APl
   └─ iouType = "bbox" → Box AP / AP50 / AP75 / APs / APm / APl

6. Per-category evaluation (19 categories × 2 iouTypes)
   └─ Extract per-category AP, AP50, AP75 + PR curve data

7. Compute threshold-dependent metrics
   └─ F1 / Precision / Recall vs confidence threshold sweep (50 steps)

8. Generate 12 evaluation charts + save full results JSON
```

---

### 📈 Results

**Dataset:** Car Parts test split — **540 images, 4,591 annotations, 19 categories**
**Evaluator:** `pycocotools.cocoeval.COCOeval` in original image coordinates
**Area ranges:** COCO standard (small < 32², medium 32²–96², large ≥ 96²)

#### Overall Metrics

| Metric | Mask | Box | Δ (Box − Mask) |
|--------|------|-----|----------------|
| **AP** (IoU 0.50:0.95) | **62.1** | **64.1** | +2.0 |
| **AP50** | **85.0** | **86.2** | +1.2 |
| **AP75** | **68.0** | **70.8** | +2.8 |

#### Size-Stratified AP (COCO Standard Area Ranges)

| Metric | Mask | Box |
|--------|------|-----|
| **APs** (small, area < 32²) | 3.5 | 12.9 |
| **APm** (medium, 32² ≤ area < 96²) | 17.3 | 26.2 |
| **APl** (large, area ≥ 96²) | 66.0 | 66.4 |

> **Note:** Under COCO standard area ranges, most car parts are classified as "large" objects (area ≥ 96² = 9,216 px²). The low APs and APm values reflect the rarity of very small car part instances in the dataset, not poor model performance on small objects.

#### Best Threshold (F1)

| Mode | Threshold | Precision | Recall | F1 |
|------|-----------|-----------|--------|-----|
| Mask | 0.469 | 0.8650 | 0.8684 | **0.8667** |
| Box | 0.469 | 0.8666 | 0.8700 | **0.8683** |

#### Per-Category Mask AP (sorted high → low)

| Class | Mask AP | Mask AP50 | Mask AP75 |
|-------|---------|-----------|-----------| 
| Front_Windshield_Glass | **89.8** | 98.5 | 97.8 |
| Hood_Bonnet | **87.0** | 97.7 | 95.0 |
| Front_Bumper | **84.7** | 97.2 | 91.8 |
| Front_Door | **80.1** | 96.5 | 92.4 |
| Diggi_Back_Door_Glass | **78.8** | 95.4 | 88.5 |
| Rear_Bumper | **75.4** | 93.5 | 83.9 |
| Rear_Door | **72.7** | 92.6 | 84.8 |
| Headlight | **69.8** | 93.7 | 79.6 |
| Diggi_Back_Door | **65.9** | 82.9 | 67.3 |
| Taillight | **64.9** | 89.3 | 72.4 |
| Fender | **59.4** | 88.8 | 69.2 |
| tyre | **58.8** | 84.7 | 67.6 |
| Side_Mirror | **56.6** | 87.1 | 67.8 |
| Rear_Door_Glass | **53.7** | 85.7 | 56.2 |
| Front_Door_Glass | **51.0** | 82.3 | 52.5 |
| Grill | **49.1** | 73.4 | 60.1 |
| Quarter_Panel | **36.1** | 58.0 | 37.2 |
| Roof | **27.4** | 63.2 | 19.3 |
| Running_Board | **19.6** | 54.7 | 7.9 |

> **Note:** Roof and Running_Board have lower AP because they are rarely fully visible and overlap significantly with adjacent panels.

#### Per-Category Box AP (sorted high → low)

| Class | Box AP | Box AP50 | Box AP75 |
|-------|--------|----------|----------|
| Front_Windshield_Glass | **88.8** | 98.5 | 97.8 |
| Hood_Bonnet | **86.0** | 97.9 | 96.2 |
| Front_Bumper | **85.8** | 97.2 | 95.0 |
| Diggi_Back_Door_Glass | **80.2** | 93.3 | 88.5 |
| Rear_Bumper | **79.3** | 95.0 | 85.1 |
| Front_Door | **78.9** | 96.7 | 91.3 |
| Rear_Door | **75.9** | 93.4 | 87.1 |
| Headlight | **70.8** | 94.2 | 81.4 |
| Diggi_Back_Door | **67.5** | 87.0 | 69.9 |
| Taillight | **66.2** | 87.5 | 75.1 |
| Fender | **59.8** | 89.2 | 66.4 |
| tyre | **56.1** | 83.3 | 63.4 |
| Front_Door_Glass | **54.2** | 83.4 | 59.2 |
| Rear_Door_Glass | **54.3** | 86.5 | 61.8 |
| Grill | **53.6** | 76.0 | 62.9 |
| Side_Mirror | **49.2** | 87.4 | 51.5 |
| Running_Board | **44.2** | 67.6 | 49.1 |
| Quarter_Panel | **36.2** | 58.4 | 39.0 |
| Roof | **30.8** | 65.9 | 23.9 |

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

### Generated Charts (12 total)

| # | File | Description |
|---|------|-------------|
| 1–2 | `pr_curve_mask_page1/2.png` | Per-category Precision–Recall curves — mask (2 pages, 10+9 categories). Solid = IoU 0.50, dashed = IoU 0.55–0.95. |
| 3–4 | `pr_curve_box_page1/2.png` | Per-category Precision–Recall curves — box |
| 5 | `f1_vs_threshold_mask.png` | F1 / Precision / Recall vs confidence threshold (mask) |
| 6 | `f1_vs_threshold_box.png` | F1 / Precision / Recall vs confidence threshold (box) |
| 7 | `ap_per_category_mask.png` | Horizontal bar chart — per-class mask AP (sorted descending) with mean line |
| 8 | `ap_per_category_box.png` | Horizontal bar chart — per-class box AP (sorted descending) with mean line |
| 9 | `mask_vs_box_overall.png` | Grouped bar — Mask vs Box for AP / AP50 / AP75 with Δ annotations |
| 10 | `mask_vs_box_per_category.png` | Grouped bar — Mask vs Box per category with Δ annotations |
| 11 | `ap50_ap75_bar.png` | AP50 vs AP75 per category — gap (↓) indicates mask shape precision |
| 12 | `confusion_heatmap_mask.png` | TP-rate heatmap per GT class (mask @ IoU=0.50) |

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
- **Evaluation standard:** COCO-style AP (IoU=0.50:0.95, 101-point interpolation) computed using `pycocotools.cocoeval.COCOeval` with COCO standard area ranges.
- **Dataset annotation format:** COCO JSON with polygon segmentation masks
