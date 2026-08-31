# Evaluation — AutoScan AI

This folder contains all evaluation scripts and generated plots for the two RF-DETR segmentation models that power the AutoScan AI pipeline. Both models are evaluated with **`pycocotools.cocoeval.COCOeval`** — the standard COCO evaluation library — on predictions produced in **native image coordinates**, with no resizing or re-projection at any stage.

---

## 📁 Folder Structure

```
evaluation/
├── Car Damage/
│   ├── Testing/                  ← Test-set evaluation (RF-DETR vs DCN+ baseline)
│   │   ├── evaluate_rfdetr_cardd_full.py
│   │   └── plots/               ← Main results: 15 charts + results JSON + COCO result JSONs
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

1. **No coordinate projection.** RF-DETR's post-processing interpolates the mask *logits* to the size of the image it was given and thresholds them there, so masks come back in the input image's coordinate space. Test images are therefore fed to the model **unresized**, and the returned masks are used exactly as produced. Nothing is resampled, so no accuracy is lost at the mask boundary. The mask shape is asserted against the image dimensions; a mismatch raises rather than being silently rescaled.
2. **Evaluator.** All AP metrics come from `pycocotools.cocoeval.COCOeval` with its standard protocol: 10 IoU thresholds (0.50 : 0.05 : 0.95), 101-point interpolated precision, and a maximum of 100 detections per image.
3. **Bounding boxes** are the tight bounding box of the predicted mask, so mask and box results describe the same pixels.
4. **Two thresholds, deliberately separate.**
   - `--threshold 0.001` is the **collection** threshold handed to COCOeval. It must stay low: COCOeval sweeps the score axis itself when it builds the precision–recall curve, so raising it truncates the recall axis and *understates* AP. It is not an operating point.
   - `REPORT_THRESHOLD = 0.50` is the **reporting** threshold, used only for the supplementary precision / recall / F1 summary and the confusion matrix. It is a conventional value fixed in advance, not tuned on the test set.

**Key difference — Area Ranges for APs / APm / APl:**

| Size Category | Car Damage (CarDD non-standard) | Car Parts (COCO standard) |
|---------------|-------------------------------|--------------------------|
| **Small** | area < 128² | area < 32² |
| **Medium** | 128² ≤ area < 256² | 32² ≤ area < 96² |
| **Large** | area ≥ 256² | area ≥ 96² |

The Car Damage evaluation uses CarDD-specific area ranges so that APs / APm / APl are directly comparable with the paper's DCN+ baseline. The Car Parts evaluation uses COCO standard ranges. APs / APm / APl are therefore **not cross-comparable** between the two models.

Every reported AP is accompanied by the number of ground-truth instances it was averaged over, so a reader can tell a well-supported result from one resting on a handful of annotations.

See each sub-folder's README for detailed methodology, evaluation flow, and full results.

---

## 📊 Quick Results Summary

### Car Damage Model (RF-DETR-Seg-Medium, trained on CarDD)

**Evaluator:** `pycocotools.cocoeval.COCOeval` | **Area ranges:** CarDD non-standard (128², 256²)
**Test split:** 374 images · 785 annotations (the official CarDD test split, 0 images failed)

| Metric | RF-DETR (Ours) | DCN+ ResNet-101 (Paper) | Δ |
|--------|---------------|-------------------------|---|
| Mask AP | **59.5** | 57.0 | **+2.5** |
| Mask AP50 | **78.9** | 77.7 | **+1.2** |
| Mask AP75 | **59.7** | 58.4 | **+1.3** |
| Box AP | **63.1** | 60.6 | **+2.5** |
| Box AP50 | 78.6 | 78.8 | −0.2 |
| Box AP75 | 64.7 | 64.8 | −0.1 |

| Size-Stratified AP | RF-DETR Mask | DCN+ Mask | Δ Mask | RF-DETR Box | DCN+ Box | Δ Box | GT instances |
|-------------------|-------------|-----------|--------|-------------|----------|-------|--------------|
| APs (small < 128²) | **39.4** | 34.6 | **+4.8** | **44.8** | 37.1 | **+7.7** | 260 |
| APm (medium 128²–256²) | **53.9** | 44.0 | **+9.9** | **58.0** | 48.0 | **+10.0** | 261 |
| APl (large ≥ 256²) | **72.1** | 71.6 | **+0.5** | 65.6 | 66.0 | −0.4 | 264 |

DCN+ values are the **DCN+ ResNet-101** row of Table IV in the CarDD paper, which reports APS / APM / APL alongside AP / AP50 / AP75.

**RF-DETR exceeds the DCN+ baseline on all six mask metrics.** On box, it leads on AP, APs and APm; AP50, AP75 and APl differ by ≤ 0.4, which is within rounding and should be read as a tie.

At the fixed reporting threshold 0.50 → **Precision 0.7835 · Recall 0.6178 · F1 0.6909** (mask; TP 485, FP 134, FN 300).

### Car Parts Model (RF-DETR-Seg-Nano, 19 classes)

**Evaluator:** `pycocotools.cocoeval.COCOeval` | **Area ranges:** COCO standard (32², 96²)
**Test split:** 540 images · 4,591 annotations (all scored, 0 undecodable, 0 out-of-class, 0 images failed)

| Metric | Mask | Box |
|--------|------|-----|
| AP | **62.1** | **64.5** |
| AP50 | **85.0** | **86.4** |
| AP75 | **68.0** | **71.3** |

| Size-Stratified AP | Mask | Box | GT instances |
|-------------------|------|-----|--------------|
| APs (small < 32²) | 8.8 | 12.1 | **44** ⚠️ |
| APm (medium 32²–96²) | 33.2 | 49.2 | 1,091 |
| APl (large ≥ 96²) | 67.6 | 67.3 | 3,456 |

> ⚠️ **`APs` for car parts is not a meaningful measurement.** The test split contains only **44 small instances across all 19 classes**, and 17 of the 19 classes have fewer than 10. That figure describes a handful of annotations rather than the model's behaviour on small objects, and it is reported here only for completeness. `APm` and `APl` are well supported. Car parts are large objects photographed close-up, so almost nothing falls below COCO's 32² threshold.

At the fixed reporting threshold 0.50 → **Precision 0.8790 · Recall 0.8532 · F1 0.8659** (mask; TP 3,917, FP 539, FN 674).

---

## 📂 Sub-folder READMEs

See each sub-folder for full details:

- [`Car Damage/README.md`](Car%20Damage/README.md) — Car damage model evaluation (methodology, results, charts, evaluation flow)
- [`Car Parts/README.md`](Car%20Parts/README.md) — Car parts model evaluation (methodology, results, charts, evaluation flow)

---

## ⚙️ Common Requirements

`requirements.txt` lives at the repository root, so install from there:

```bash
cd <repo root>
pip install -r requirements.txt
```
