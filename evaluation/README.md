# Evaluation — AutoScan AI

This folder contains all evaluation scripts and generated plots for the two RF-DETR segmentation models that power the AutoScan AI pipeline. Both models are evaluated using **`pycocotools.cocoeval.COCOeval`** — the standard COCO evaluation library — with predictions projected to **original image coordinates** for fair metric computation.

---

## 📁 Folder Structure

```
evaluation/
├── Car Damage/
│   ├── Testing/                  ← Test-set evaluation (RF-DETR vs DCN+ baseline)
│   │   ├── evaluate_rfdetr_cardd_full.py
│   │   └── plots/               ← Main results: 14 charts + results JSON + COCO result JSONs
│   └── Training and Validation/ ← Training metrics visualisation from CSV
│       ├── plot_training_metrics.py
│       ├── training_plots/      ← 12 generated training charts
│       └── training_stats/
│           └── metrics.csv      ← Raw training log (epochs + steps)
│
└── Car Parts/
    ├── Testing/                  ← Test-set evaluation (RF-DETR standalone)
    │   ├── evaluate_rfdetr_carparts.py
    │   └── plots_parts/          ← Main results: 12 charts + results JSON
    └── Training and Validation/  ← Training metrics visualisation from CSV
        ├── plot_parts_metrics.py
        ├── training_stats/
        │   └── metrics.csv       ← Raw training log (epochs + steps)
        └── parts_plots/          ← 12 generated training charts
```

---

## 🤖 Models & Datasets

All trained model checkpoints and datasets are hosted on Google Drive:

> **🔗 [Download Models & Datasets](https://drive.google.com/drive/folders/1KxSQLXaXxa2vOlMYNeq6UL5Ua1hEDirn?usp=drive_link)**

The Drive contains:
- **Car Damage model** checkpoint (`checkpoint_best_total.pth`) — RF-DETR-Seg-Medium
- **Car Parts model** checkpoint (`checkpoint_best_total.pth`) — RF-DETR-Seg-Nano
- **CarDD dataset** (test split — 374 images, 785 annotations, 6 damage classes)
- **Car Parts dataset** (test split — 540 images, 4,591 annotations, 19 part classes)

---

## 🔬 Evaluation Methodology

Both evaluation scripts follow the same core methodology:

1. **Coordinate Projection:** Model predictions (masks + bounding boxes) are projected from the model's internal resolution back to **original image coordinates** using bilinear interpolation + 0.5 threshold binarisation.
2. **Evaluator:** All AP metrics are computed using `pycocotools.cocoeval.COCOeval` — the standard COCO evaluation library — ensuring reproducibility and comparability.
3. **Bounding Box Derivation:** Bounding boxes are derived as tight bounding boxes of projected masks (not from the model's box head) for pixel-exact mask–box consistency.

**Key difference — Area Ranges for APs / APm / APl:**

| Size Category | Car Damage (CarDD non-standard) | Car Parts (COCO standard) |
|---------------|-------------------------------|--------------------------|
| **Small** | area < 128² | area < 32² |
| **Medium** | 128² ≤ area < 256² | 32² ≤ area < 96² |
| **Large** | area ≥ 256² | area ≥ 96² |

The Car Damage evaluation uses CarDD-specific area ranges to match the paper's DCN+ baseline. The Car Parts evaluation uses COCO standard ranges. APs / APm / APl values are **not cross-comparable** between the two models due to this difference.

See each sub-folder's README for detailed methodology, evaluation flow, and full results.

---

## 📊 Quick Results Summary

### Car Damage Model (RF-DETR-Seg-Medium, trained on CarDD)

**Evaluator:** `pycocotools.cocoeval.COCOeval` | **Area ranges:** CarDD non-standard (128², 256²)

| Metric | RF-DETR (Ours) | DCN+ ResNet-101 (Paper) | Δ |
|--------|---------------|-------------------------|---|
| Mask AP | **59.9** | 57.0 | **+2.9** |
| Mask AP50 | **79.8** | 77.7 | **+2.1** |
| Mask AP75 | **60.1** | 58.4 | **+1.7** |
| Box AP | **63.3** | 60.6 | **+2.7** |
| Box AP50 | **79.4** | 78.8 | **+0.6** |
| Box AP75 | **65.3** | 64.8 | **+0.5** |

| Size-Stratified AP | RF-DETR Mask | DCN+ Mask | Δ Mask | RF-DETR Box | DCN+ Box | Δ Box |
|-------------------|-------------|-----------|--------|-------------|----------|-------|
| APs (small < 128²) | **41.2** | 34.6 | **+6.6** | **44.0** | 37.1 | **+6.9** |
| APm (medium 128²–256²) | **54.4** | 44.0 | **+10.4** | **57.5** | 48.0 | **+9.5** |
| APl (large ≥ 256²) | 62.7 | **71.6** | -8.9 | 65.5 | **66.0** | -0.5 |

DCN+ size-stratified values are taken from the **DCN+ ResNet-101** row in Table IV of the CarDD paper.

Best F1 @ threshold 0.408 → **Precision: 0.7034 · Recall: 0.6828 · F1: 0.693** (Mask)

### Car Parts Model (RF-DETR-Seg-Nano, 19 classes)

**Evaluator:** `pycocotools.cocoeval.COCOeval` | **Area ranges:** COCO standard (32², 96²)

| Metric | Mask | Box |
|--------|------|-----|
| AP | **62.1** | **64.1** |
| AP50 | **85.0** | **86.2** |
| AP75 | **68.0** | **70.8** |

| Size-Stratified AP | Mask | Box |
|-------------------|------|-----|
| APs (small < 32²) | 3.5 | 12.9 |
| APm (medium 32²–96²) | 17.3 | 26.2 |
| APl (large ≥ 96²) | 66.0 | 66.4 |

Best F1 @ threshold 0.469 → **Precision: 0.8650 · Recall: 0.8684 · F1: 0.8667** (Mask)

---

## 📂 Sub-folder READMEs

See each sub-folder for full details:

- [`Car Damage/README.md`](Car%20Damage/README.md) — Car damage model evaluation (methodology, results, charts, evaluation flow)
- [`Car Parts/README.md`](Car%20Parts/README.md) — Car parts model evaluation (methodology, results, charts, evaluation flow)

---

## ⚙️ Common Requirements

```bash
pip install -r requirements.txt
```
