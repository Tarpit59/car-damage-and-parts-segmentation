# Evaluation — AutoScan AI

This folder contains all evaluation scripts and generated plots for the two RF-DETR segmentation models that power the AutoScan AI pipeline.

---

## 📁 Folder Structure

```
evaluation/
├── Car Damage/
│   ├── Testing/                  ← Test-set evaluation (RF-DETR vs DCN+ baseline)
│   │   ├── evaluate_rfdetr_cardd_full.py
│   │   └── plots/                ← 15 generated PNG charts + results_full.json
│   └── Training and Validation/  ← Training metrics visualisation from CSV
│       ├── plot_training_metrics.py
│       ├── training_plots/       ← 12 generated training charts
│       └── training_stats/
│           └── metrics.csv       ← Raw training log (epochs + steps)
│
└── Car Parts/
    ├── Testing/                  ← Test-set evaluation (RF-DETR standalone)
    │   ├── evaluate_rfdetr_carparts.py
    │   └── plots_parts/          ← 13 generated PNG charts + rfdetr_carparts_results.json
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
- **Car Damage model** checkpoint (`checkpoint_best_total.pth`)
- **Car Parts model** checkpoint (`checkpoint_best_total.pth`)
- **CarDD dataset** (test split — 374 images, 785 annotations, 6 damage classes)
- **Car Parts dataset** (test split — 540 images, 4,591 annotations, 19 part classes)

---

## 📊 Quick Results Summary

### Car Damage Model (RF-DETR-Seg-Medium, trained on CarDD)

| Metric | RF-DETR (Ours) | DCN+ ResNet-101 (Paper) | Δ |
|--------|---------------|-------------------------|---|
| Mask AP | **60.0** | 57.0 | **+3.0** |
| Mask AP50 | **79.6** | 77.7 | **+1.9** |
| Mask AP75 | **60.2** | 58.4 | **+1.8** |
| Box AP | **63.2** | 60.6 | **+2.6** |
| Box AP50 | **79.4** | 78.8 | **+0.6** |
| Box AP75 | **65.0** | 64.8 | **+0.2** |

Best F1 @ threshold 0.408 → **Precision: 0.702 · Recall: 0.682 · F1: 0.692** (Mask)

### Car Parts Model (RF-DETR-Seg-Medium, 19 classes)

| Metric | Value |
|--------|-------|
| Mask AP | **62.3** |
| Mask AP50 | **85.4** |
| Mask AP75 | **68.2** |
| Box AP | **64.3** |
| Box AP50 | **86.6** |
| Box AP75 | **71.0** |

Best F1 @ threshold 0.469 → **Precision: 0.865 · Recall: 0.868 · F1: 0.867** (Mask)

---

## 📂 Sub-folder READMEs

See each sub-folder for full details:

- [`Car Damage/README.md`](Car%20Damage/README.md) — Car damage model evaluation
- [`Car Parts/README.md`](Car%20Parts/README.md) — Car parts model evaluation

---

## ⚙️ Common Requirements

```bash
pip install -r requirements.txt
```
