"""
plot_parts_metrics.py
======================
Generates training + validation charts from the RF-DETR CarDD Parts
Segmentation training CSV (19-class car-part model).

CHARTS GENERATED (saved to ./parts_plots/)
──────────────────────────────────────────────
  1.  val_map_overview.png            Val mAP50 / mAP50-95 / segm_mAP50 / segm_mAP50-95 over epochs
  2.  val_map75_mar.png               Val mAP75 + mAR over epochs
  3.  val_precision_recall_f1.png     Val Precision / Recall / F1 over epochs
  4.  val_per_category_ap.png         Per-category AP (all 19 parts) over epochs  [4×5 grid]
  5.  train_loss_total.png            Total training loss over steps
  6.  train_loss_components.png       Main loss components (ce / bbox / giou / mask_ce / mask_dice)
  7.  train_loss_auxiliary.png        Auxiliary decoder losses (layers 0-2 + enc)
  8.  val_ema_vs_live.png             EMA mAP50 vs live mAP50 (box + segm)
  9.  val_category_radar.png          Two-panel radar: per-category AP at last epoch
  10. val_loss_curve.png              Validation loss over epochs
  11. train_cardinality_error.png     Cardinality error over steps
  12. combined_overview.png           2×2 dashboard: loss / mAP / P-R-F1 / horizontal per-cat AP bar

USAGE
─────
  python plot_parts_metrics.py --csv metrics.csv --output_dir ./parts_plots
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# A Windows console defaults to cp1252, which cannot encode the arrows, box
# characters and check marks used throughout the messages below: the very first
# print raises UnicodeEncodeError and the script dies before doing any work.
# Force UTF-8 on the streams that support it.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ── CONSTANTS ─────────────────────────────────────────────────────────────────
# CSV column suffix → display label
CLASSES = [
    "Diggi_Back_Door",
    "Diggi_Back_Door_Glass",
    "Fender",
    "Front_Bumper",
    "Front_Door",
    "Front_Door_Glass",
    "Front_Windshield_Glass",
    "Grill",
    "Headlight",
    "Hood_Bonnet",
    "Quarter_Panel",
    "Rear_Bumper",
    "Rear_Door",
    "Rear_Door_Glass",
    "Roof",
    "Running_Board",
    "Side_Mirror",
    "Taillight",
    "tyre",
]

# Human-readable labels for axes / titles
CLASS_LABELS = {
    "Diggi_Back_Door":        "Diggi Back Door",
    "Diggi_Back_Door_Glass":  "Diggi Door Glass",
    "Fender":                 "Fender",
    "Front_Bumper":           "Front Bumper",
    "Front_Door":             "Front Door",
    "Front_Door_Glass":       "Front Door Glass",
    "Front_Windshield_Glass": "Front Windshield",
    "Grill":                  "Grill",
    "Headlight":              "Headlight",
    "Hood_Bonnet":            "Hood/Bonnet",
    "Quarter_Panel":          "Quarter Panel",
    "Rear_Bumper":            "Rear Bumper",
    "Rear_Door":              "Rear Door",
    "Rear_Door_Glass":        "Rear Door Glass",
    "Roof":                   "Roof",
    "Running_Board":          "Running Board",
    "Side_Mirror":            "Side Mirror",
    "Taillight":              "Taillight",
    "tyre":                   "Tyre",
}

# 19 visually distinct colours
CAT_COLORS = {
    "Diggi_Back_Door":        "#E63946",
    "Diggi_Back_Door_Glass":  "#FF6B6B",
    "Fender":                 "#F4A261",
    "Front_Bumper":           "#E9C46A",
    "Front_Door":             "#2A9D8F",
    "Front_Door_Glass":       "#48CAE4",
    "Front_Windshield_Glass": "#90E0EF",
    "Grill":                  "#457B9D",
    "Headlight":              "#1D3557",
    "Hood_Bonnet":            "#A371F7",
    "Quarter_Panel":          "#D2A8FF",
    "Rear_Bumper":            "#3FB950",
    "Rear_Door":              "#2EA043",
    "Rear_Door_Glass":        "#56CFE1",
    "Roof":                   "#6A0572",
    "Running_Board":          "#C77DFF",
    "Side_Mirror":            "#F78166",
    "Taillight":              "#F4D35E",
    "tyre":                   "#8B949E",
}

# General palette
C_MAP50    = "#58A6FF"
C_MAP5095  = "#1A73E8"
C_SEG50    = "#3FB950"
C_SEG5095  = "#2EA043"
C_MAP75    = "#F78166"
C_MAR      = "#D2A8FF"
C_PREC     = "#58A6FF"
C_REC      = "#3FB950"
C_F1       = "#F78166"
C_LOSS     = "#E63946"
C_LR       = "#F4A261"
C_EMA      = "#A371F7"
C_VAL_LOSS = "#F85149"


# ── THEME ─────────────────────────────────────────────────────────────────────
def _style():
    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linestyle":    "--",
        "figure.facecolor":  "#0D1117",
        "axes.facecolor":    "#161B22",
        "axes.labelcolor":   "#C9D1D9",
        "xtick.color":       "#C9D1D9",
        "ytick.color":       "#C9D1D9",
        "text.color":        "#C9D1D9",
        "axes.titlecolor":   "#E6EDF3",
        "legend.facecolor":  "#21262D",
        "legend.edgecolor":  "#30363D",
        "grid.color":        "#21262D",
        "axes.edgecolor":    "#30363D",
        "lines.linewidth":   2.0,
    })


def _best_epoch_line(ax, epochs, values, color):
    """Mark the epoch with the best (max) value — dashed line + clearly readable annotation."""
    idx  = int(np.argmax(np.array(values)))
    ep_x = epochs.iloc[idx] if hasattr(epochs, "iloc") else epochs[idx]
    val  = values.iloc[idx]  if hasattr(values,  "iloc") else values[idx]
    ax.axvline(ep_x, color=color, ls=":", alpha=0.60, lw=1.5)
    ax.scatter([ep_x], [val], color=color, s=60, zorder=6)
    ax.annotate(
        f"{val:.2f}  ep{int(ep_x)}",
        xy=(ep_x, val),
        xytext=(8, -14),
        textcoords="offset points",
        fontsize=8, color=color, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#0D1117",
                  edgecolor=color, alpha=0.85, lw=0.8),
    )


def _savefig(fig, path, tight=True):
    if tight:
        fig.tight_layout()
    _stamp_source(fig)
    fig.savefig(path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [PLOT] Saved → {path}")


# ── SMOOTHING ─────────────────────────────────────────────────────────────────
def _smooth_window(n, divisor=20, min_window=3):
    """
    Rolling-mean window for a series of n points, or None when a rolling mean
    would change nothing.

    The window used to be `n // divisor` floored at 1, which is 1 for any
    series shorter than 40 points -- and a rolling mean of width 1 is the
    identity. RF-DETR logs one train row per epoch, so this run has 20 of them
    and the window was always 1: the "smoothed" curve was the raw curve drawn a
    second time in a heavier stroke, beneath a legend entry advertising
    smoothing that never happened. A reader takes the heavy line for a trend
    and the faint one for noise around it, when they are the same numbers.
    Returning None instead lets the caller draw a single honest line.
    """
    w = max(1, int(n) // divisor)
    return w if w >= min_window else None


def _roll(series, w):
    """Rolling mean when w is a real window, otherwise the series untouched."""
    return series.rolling(w, center=True, min_periods=1).mean() if w else series


def _plot_series(ax, x, raw, color, label, w, lw=2.5, fill=True,
                 fill_alpha=0.10, raw_lw=1.2, raw_alpha=0.30):
    """
    Plot `raw`, overlaying a rolling mean only when `w` is a real window.
    Returns the series drawn as the primary line.
    """
    if w:
        ax.plot(x, raw, color=color, lw=raw_lw, alpha=raw_alpha, label="Raw")
        series = _roll(raw, w)
        lbl = f"{label} (rolling mean, w={w})"
    else:
        series = raw
        lbl = label
    ax.plot(x, series, color=color, lw=lw, label=lbl)
    if fill:
        ax.fill_between(x, series, alpha=fill_alpha, color=color)
    return series


# ── PROVENANCE ────────────────────────────────────────────────────────────────
# Every figure here comes from the training CSV: train/* are training-time
# losses, val/* are the split RF-DETR scored at the end of each epoch. Neither
# is the held-out test set, and the val/* numbers are NOT the test-set results
# produced by evaluate_rfdetr_*.py -- those use a different evaluator
# configuration and must never be quoted interchangeably with these. Each
# figure carries the line below so a chart lifted out of this directory and
# dropped into a report still states which split it came from.
SPLIT_NOTE = ("source: training metrics CSV - train/* = training loss, "
              "val/* = per-epoch validation split; neither is the held-out "
              "test set")


def _stamp_source(fig):
    fig.text(0.005, 0.002, SPLIT_NOTE, fontsize=7, color="#8B949E",
             ha="left", va="bottom")



# ── DATA LOADING ──────────────────────────────────────────────────────────────
def load_data(csv_path):
    df = pd.read_csv(csv_path)

    # Split into val rows (have val/mAP_50) and train rows (have train/loss)
    val_mask   = df["val/mAP_50"].notna()
    train_mask = df["train/loss"].notna()

    val_df   = df[val_mask].copy().reset_index(drop=True)
    train_df = df[train_mask].copy().reset_index(drop=True)

    # Use epoch as x-axis for val (shift 0-based CSV → 1-based display), step for train
    val_df["epoch_f"]  = val_df["epoch"].astype(int) + 1
    train_df["step_f"] = train_df["step"].astype(float)

    return val_df, train_df


# ══════════════════════════════════════════════════════════════════════════════
# CHART 1 ─ Val mAP Overview
# ══════════════════════════════════════════════════════════════════════════════
def plot_val_map_overview(val_df, out):
    _style()
    fig, ax = plt.subplots(figsize=(12, 5.5))

    ep = val_df["epoch_f"]

    ax.plot(ep, val_df["val/mAP_50"]        * 100, color=C_MAP50,   lw=2.5, label="Box mAP@50")
    ax.plot(ep, val_df["val/mAP_50_95"]     * 100, color=C_MAP5095, lw=2.5, label="Box mAP@50:95", ls="--")
    ax.plot(ep, val_df["val/segm_mAP_50"]   * 100, color=C_SEG50,   lw=2.5, label="Segm mAP@50")
    ax.plot(ep, val_df["val/segm_mAP_50_95"]* 100, color=C_SEG5095, lw=2.5, label="Segm mAP@50:95", ls="--")

    ax.fill_between(ep, val_df["val/mAP_50"]*100,       alpha=0.06, color=C_MAP50)
    ax.fill_between(ep, val_df["val/segm_mAP_50"]*100,  alpha=0.06, color=C_SEG50)

    _best_epoch_line(ax, ep, val_df["val/mAP_50"]*100,      C_MAP50)
    _best_epoch_line(ax, ep, val_df["val/segm_mAP_50"]*100, C_SEG50)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("mAP (%)", fontsize=12)
    ax.set_title("Validation mAP over Training  ·  Box & Segmentation",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, ncol=2)
    ax.set_ylim(0)
    ax.set_xticks(ep)
    _savefig(fig, out / "val_map_overview.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 2 ─ Val mAP75 + mAR
# ══════════════════════════════════════════════════════════════════════════════
def plot_val_map75_mar(val_df, out):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ep = val_df["epoch_f"]

    # Left: mAP75
    ax = axes[0]
    ax.plot(ep, val_df["val/mAP_75"]*100, color=C_MAP75, lw=2.5, label="Box mAP@75")
    ax.fill_between(ep, val_df["val/mAP_75"]*100, alpha=0.10, color=C_MAP75)
    _best_epoch_line(ax, ep, val_df["val/mAP_75"]*100, C_MAP75)
    ax.set_xlabel("Epoch"); ax.set_ylabel("mAP@75 (%)"); ax.set_ylim(0)
    ax.set_title("Validation mAP@75 (Box)", fontsize=12, fontweight="bold")
    ax.set_xticks(ep)
    ax.legend(fontsize=10)

    # Right: mAR
    ax = axes[1]
    ax.plot(ep, val_df["val/mAR"]*100,     color=C_MAR,  lw=2.5, label="Box mAR")
    ax.plot(ep, val_df["val/ema_mAR"]*100, color=C_EMA,  lw=2.0, label="EMA mAR", ls="--", alpha=0.8)
    ax.fill_between(ep, val_df["val/mAR"]*100, alpha=0.10, color=C_MAR)
    _best_epoch_line(ax, ep, val_df["val/mAR"]*100, C_MAR)
    ax.set_xlabel("Epoch"); ax.set_ylabel("mAR (%)"); ax.set_ylim(0)
    ax.set_title("Validation mAR (Box)", fontsize=12, fontweight="bold")
    ax.set_xticks(ep)
    ax.legend(fontsize=10)

    fig.suptitle("Validation mAP@75  &  Mean Average Recall",
                 fontsize=14, fontweight="bold", color="#E6EDF3", y=1.02)
    _savefig(fig, out / "val_map75_mar.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 3 ─ Val Precision / Recall / F1
# ══════════════════════════════════════════════════════════════════════════════
def plot_val_prf(val_df, out):
    _style()
    fig, ax = plt.subplots(figsize=(12, 5.5))

    ep = val_df["epoch_f"]
    p  = val_df["val/precision"]
    r  = val_df["val/recall"]
    f1 = val_df["val/F1"]

    ax.plot(ep, p,  color=C_PREC, lw=2.5, label="Precision")
    ax.plot(ep, r,  color=C_REC,  lw=2.5, label="Recall")
    ax.plot(ep, f1, color=C_F1,   lw=2.5, label="F1")

    ax.fill_between(ep, f1, alpha=0.10, color=C_F1)

    _best_epoch_line(ax, ep, f1.values, C_F1)
    _best_epoch_line(ax, ep, p.values,  C_PREC)
    _best_epoch_line(ax, ep, r.values,  C_REC)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title("Validation Precision / Recall / F1 over Training",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xticks(ep)
    _savefig(fig, out / "val_precision_recall_f1.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 4 ─ Per-Category AP over Epochs  (4 rows × 5 cols for 19 parts)
# ══════════════════════════════════════════════════════════════════════════════
def plot_per_category_ap(val_df, out):
    _style()
    NCOLS, NROWS = 5, 4          # 20 cells, 19 used + 1 hidden
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(22, 16))
    fig.suptitle("Validation AP per Car Part over Training",
                 fontsize=15, fontweight="bold", color="#E6EDF3", y=1.01)

    ep = val_df["epoch_f"]

    for i, cat in enumerate(CLASSES):
        ax  = axes[i // NCOLS][i % NCOLS]
        col = CAT_COLORS[cat]
        col_name = f"val/AP/{cat}"

        if col_name not in val_df.columns:
            ax.set_visible(False)
            continue

        vals = val_df[col_name] * 100
        ax.plot(ep, vals, color=col, lw=2.2)
        ax.fill_between(ep, vals, alpha=0.14, color=col)
        _best_epoch_line(ax, ep, vals, col)

        ax.set_title(CLASS_LABELS[cat], fontsize=9.5, fontweight="bold", color=col)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("AP (%)", fontsize=8)
        ax.set_ylim(0)
        ax.set_xticks(ep)
        ax.tick_params(axis='x', labelsize=7)

        # Final value annotation
        ax.text(0.97, 0.06, f"Final: {vals.iloc[-1]:.1f}%",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color=col, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#21262D",
                          edgecolor=col, alpha=0.7))

    # Hide the last unused cell
    axes[NROWS-1][NCOLS-1].set_visible(False)
    _savefig(fig, out / "val_per_category_ap.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 5 ─ Training Total Loss
# ══════════════════════════════════════════════════════════════════════════════
def plot_train_loss_total(train_df, out):
    _style()
    fig, ax = plt.subplots(figsize=(12, 5.5))

    st = train_df["step_f"]
    w  = _smooth_window(len(train_df))
    _plot_series(ax, st, train_df["train/loss"], C_LOSS, "Total loss", w)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Total Loss", fontsize=12)
    ax.set_title("Training Total Loss over Steps",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    _savefig(fig, out / "train_loss_total.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 6 ─ Training Loss Components
# ══════════════════════════════════════════════════════════════════════════════
def plot_train_loss_components(train_df, out):
    _style()

    components = {
        "CE (class)":    ("train/loss_ce",        "#58A6FF"),
        "BBox L1":       ("train/loss_bbox",       "#F4A261"),
        "GIoU":          ("train/loss_giou",       "#2A9D8F"),
        "Mask CE":       ("train/loss_mask_ce",    "#E63946"),
        "Mask Dice":     ("train/loss_mask_dice",  "#A371F7"),
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Training Loss Components over Steps",
                 fontsize=14, fontweight="bold", color="#E6EDF3", y=1.01)

    st = train_df["step_f"]
    w  = _smooth_window(len(train_df))

    for idx, (label, (col_name, color)) in enumerate(components.items()):
        ax = axes[idx // 3][idx % 3]
        if col_name not in train_df.columns:
            ax.set_visible(False)
            continue

        _plot_series(ax, st, train_df[col_name], color, label, w,
                     lw=2.2, fill_alpha=0.12)

        ax.set_title(label, fontsize=11, fontweight="bold", color=color)
        ax.set_xlabel("Step", fontsize=9)
        ax.set_ylabel("Loss", fontsize=9)

    # 6th panel: all smoothed on same axis
    ax = axes[1][2]
    for label, (col_name, color) in components.items():
        if col_name in train_df.columns:
            sm = _roll(train_df[col_name], w)
            ax.plot(st, sm, color=color, lw=1.8, label=label)
    ax.set_title("All Components (overlay)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.legend(fontsize=8, ncol=2)

    _savefig(fig, out / "train_loss_components.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 7 ─ Auxiliary Decoder Loss Layers
# ══════════════════════════════════════════════════════════════════════════════
def plot_train_loss_auxiliary(train_df, out):
    _style()

    aux_groups = {
        "CE (auxiliary)":        [f"train/loss_ce_{i}" for i in range(3)] + ["train/loss_ce_enc"],
        "BBox L1 (auxiliary)":   [f"train/loss_bbox_{i}" for i in range(3)] + ["train/loss_bbox_enc"],
        "GIoU (auxiliary)":      [f"train/loss_giou_{i}" for i in range(3)] + ["train/loss_giou_enc"],
        "Mask CE (auxiliary)":   [f"train/loss_mask_ce_{i}" for i in range(3)] + ["train/loss_mask_ce_enc"],
        "Mask Dice (auxiliary)": [f"train/loss_mask_dice_{i}" for i in range(3)] + ["train/loss_mask_dice_enc"],
    }
    layer_colors = ["#58A6FF", "#F4A261", "#2A9D8F", "#A371F7", "#F78166"]
    layer_labels = ["Layer 0", "Layer 1", "Layer 2", "Enc"]

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle("Auxiliary Decoder Layer Losses over Steps",
                 fontsize=14, fontweight="bold", color="#E6EDF3", y=1.01)

    st = train_df["step_f"]
    w  = _smooth_window(len(train_df))

    for idx, (group_label, cols) in enumerate(aux_groups.items()):
        ax = axes[idx // 2][idx % 2]
        for ci, (col, lbl) in enumerate(zip(cols, layer_labels)):
            if col in train_df.columns:
                sm = _roll(train_df[col], w)
                ax.plot(st, sm, color=layer_colors[ci], lw=1.8, label=lbl)

        ax.set_title(group_label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Step", fontsize=9); ax.set_ylabel("Loss", fontsize=9)
        ax.legend(fontsize=8, ncol=3)

    # Hide the last empty panel
    axes[2][1].set_visible(False)
    _savefig(fig, out / "train_loss_auxiliary.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 9 ─ EMA vs Live mAP
# ══════════════════════════════════════════════════════════════════════════════
def plot_ema_vs_live(val_df, out):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ep = val_df["epoch_f"]

    # Box mAP
    ax = axes[0]
    ax.plot(ep, val_df["val/mAP_50"]*100,     color=C_MAP50, lw=2.5, label="Live mAP@50")
    ax.plot(ep, val_df["val/ema_mAP_50"]*100, color=C_EMA,   lw=2.0, ls="--", label="EMA mAP@50")
    ax.plot(ep, val_df["val/mAP_50_95"]*100,     color=C_MAP5095, lw=2.5, label="Live mAP@50:95")
    ax.plot(ep, val_df["val/ema_mAP_50_95"]*100, color="#D2A8FF", lw=2.0, ls="--", label="EMA mAP@50:95")
    ax.fill_between(ep, val_df["val/mAP_50"]*100, val_df["val/ema_mAP_50"]*100,
                    alpha=0.08, color=C_EMA)
    ax.set_title("Box mAP: Live vs EMA", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("mAP (%)"); ax.set_ylim(0)
    ax.set_xticks(ep)
    ax.legend(fontsize=9, ncol=2)

    # Segm mAP
    ax = axes[1]
    ax.plot(ep, val_df["val/segm_mAP_50"]*100,     color=C_SEG50,  lw=2.5, label="Live Segm mAP@50")
    ax.plot(ep, val_df["val/ema_segm_mAP_50"]*100, color=C_EMA,    lw=2.0, ls="--", label="EMA Segm mAP@50")
    ax.plot(ep, val_df["val/segm_mAP_50_95"]*100,     color=C_SEG5095, lw=2.5, label="Live Segm mAP@50:95")
    ax.plot(ep, val_df["val/ema_segm_mAP_50_95"]*100, color="#D2A8FF", lw=2.0, ls="--", label="EMA Segm mAP@50:95")
    ax.fill_between(ep, val_df["val/segm_mAP_50"]*100, val_df["val/ema_segm_mAP_50"]*100,
                    alpha=0.08, color=C_EMA)
    ax.set_title("Segm mAP: Live vs EMA", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("mAP (%)"); ax.set_ylim(0)
    ax.set_xticks(ep)
    ax.legend(fontsize=9, ncol=2)

    fig.suptitle("EMA  vs  Live Validation mAP  (Box & Segmentation)",
                 fontsize=14, fontweight="bold", color="#E6EDF3", y=1.02)
    _savefig(fig, out / "val_ema_vs_live.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 10 ─ Per-Category AP Radar — two panels (19 classes split ~10+9)
# ══════════════════════════════════════════════════════════════════════════════
def plot_category_radar(val_df, out):
    _style()

    last    = val_df.iloc[-1]
    last_ep = int(val_df["epoch_f"].iloc[-1])
    all_vals = [last[f"val/AP/{c}"] * 100 for c in CLASSES]

    # Split into two halves
    splits = [CLASSES[:10], CLASSES[10:]]
    split_vals = [all_vals[:10], all_vals[10:]]
    split_titles = ["Parts 1 – 10", "Parts 11 – 19"]

    fig, axs = plt.subplots(1, 2, figsize=(18, 9),
                            subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#0D1117")
    fig.suptitle(f"Per-Category AP Radar  ·  Epoch {last_ep}",
                 fontsize=14, fontweight="bold", color="#E6EDF3", y=1.02)

    for ax, cats, vals, title in zip(axs, splits, split_vals, split_titles):
        ax.set_facecolor("#161B22")
        n      = len(cats)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        angles_c = angles + angles[:1]
        vals_c   = vals  + vals[:1]

        # Background shading per segment
        for i, cat in enumerate(cats):
            ax.fill([angles[i], angles[(i+1) % n], angles[(i+1) % n], angles[i]],
                    [0, 0, 100, 100], color=CAT_COLORS[cat], alpha=0.04)

        ax.plot(angles_c, vals_c, color="#1A73E8", lw=2.5)
        ax.fill(angles_c, vals_c, color="#1A73E8", alpha=0.18)

        for i, cat in enumerate(cats):
            ax.scatter([angles[i]], [vals[i]], color=CAT_COLORS[cat], s=80, zorder=6)
            ax.annotate(f"{vals[i]:.1f}%",
                        xy=(angles[i], vals[i]),
                        xytext=(0, 6), textcoords="offset points",
                        fontsize=8, color=CAT_COLORS[cat],
                        ha="center", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.15", facecolor="#0D1117",
                                  edgecolor=CAT_COLORS[cat], alpha=0.75, lw=0.6))

        short_labels = [CLASS_LABELS[c] for c in cats]
        ax.set_thetagrids(np.degrees(angles), short_labels,
                          fontsize=9, color="#C9D1D9")
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(["20", "40", "60", "80", "100"], color="#8B949E", fontsize=7)
        ax.grid(color="#21262D", linestyle="--", alpha=0.5)
        ax.spines["polar"].set_color("#30363D")
        ax.set_title(title, fontsize=12, fontweight="bold",
                     color="#E6EDF3", pad=18)

    _savefig(fig, out / "val_category_radar.png", tight=False)


# ══════════════════════════════════════════════════════════════════════════════
# CHART 11 ─ Validation Loss
# ══════════════════════════════════════════════════════════════════════════════
def plot_val_loss(val_df, out):
    _style()
    fig, ax = plt.subplots(figsize=(12, 5.5))

    ep = val_df["epoch_f"]
    ax.plot(ep, val_df["val/loss"], color=C_VAL_LOSS, lw=2.5, label="Val Loss")
    ax.fill_between(ep, val_df["val/loss"], alpha=0.12, color=C_VAL_LOSS)

    # Mark minimum with a dot only
    idx = val_df["val/loss"].idxmin()
    min_val = val_df["val/loss"].iloc[idx]
    min_ep  = ep.iloc[idx]
    ax.scatter([min_ep], [min_val], color=C_VAL_LOSS, s=80, zorder=5,
               label=f"Min: {min_val:.2f} (ep {int(min_ep)})")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Validation Loss", fontsize=12)
    ax.set_title("Validation Loss over Training", fontsize=14, fontweight="bold")
    ax.set_xticks(ep)
    ax.legend(fontsize=11)
    _savefig(fig, out / "val_loss_curve.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 12 ─ Cardinality Error
# ══════════════════════════════════════════════════════════════════════════════
def plot_cardinality_error(train_df, out):
    _style()
    fig, ax = plt.subplots(figsize=(12, 5.0))

    st = train_df["step_f"]
    w  = _smooth_window(len(train_df))

    layers = {
        "Main":    ("train/cardinality_error",   "#58A6FF"),
        "Layer 0": ("train/cardinality_error_0", "#F4A261"),
        "Layer 1": ("train/cardinality_error_1", "#2A9D8F"),
        "Layer 2": ("train/cardinality_error_2", "#A371F7"),
        "Enc":     ("train/cardinality_error_enc","#3FB950"),
    }

    for lbl, (col, color) in layers.items():
        if col in train_df.columns:
            sm = _roll(train_df[col], w)
            ax.plot(st, sm, color=color, lw=1.8, label=lbl)

    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Cardinality Error", fontsize=12)
    ax.set_title("Cardinality Error over Training (per Decoder Layer)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, ncol=3)
    _savefig(fig, out / "train_cardinality_error.png")


# ══════════════════════════════════════════════════════════════════════════════
# CHART 13 ─ Combined 2×2 Dashboard
# ══════════════════════════════════════════════════════════════════════════════
def plot_combined_dashboard(val_df, train_df, out):
    _style()
    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor("#0D1117")
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.30)

    ep = val_df["epoch_f"]
    st = train_df["step_f"]
    w  = _smooth_window(len(train_df))

    # ─── Top-left: Training loss ───────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#161B22")
    _plot_series(ax1, st, train_df["train/loss"], C_LOSS, "Total loss", w,
                 raw_lw=1.0)
    ax1.set_title("Training Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Step"); ax1.set_ylabel("Loss")
    ax1.legend(fontsize=9)

    # ─── Top-right: Val mAP ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#161B22")
    ax2.plot(ep, val_df["val/mAP_50"]*100,        color=C_MAP50,   lw=2.2, label="Box mAP@50")
    ax2.plot(ep, val_df["val/mAP_50_95"]*100,      color=C_MAP5095, lw=2.2, label="Box mAP@50:95", ls="--")
    ax2.plot(ep, val_df["val/segm_mAP_50"]*100,    color=C_SEG50,   lw=2.2, label="Segm mAP@50")
    ax2.plot(ep, val_df["val/segm_mAP_50_95"]*100, color=C_SEG5095, lw=2.2, label="Segm mAP@50:95", ls="--")
    ax2.fill_between(ep, val_df["val/mAP_50"]*100, alpha=0.06, color=C_MAP50)
    ax2.set_title("Validation mAP", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("mAP (%)"); ax2.set_ylim(0)
    ax2.set_xticks(ep)
    ax2.legend(fontsize=8, ncol=2)

    # ─── Bottom-left: Precision / Recall / F1 ─────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#161B22")
    ax3.plot(ep, val_df["val/precision"], color=C_PREC, lw=2.2, label="Precision")
    ax3.plot(ep, val_df["val/recall"],    color=C_REC,  lw=2.2, label="Recall")
    ax3.plot(ep, val_df["val/F1"],        color=C_F1,   lw=2.2, label="F1")
    ax3.fill_between(ep, val_df["val/F1"], alpha=0.10, color=C_F1)
    ax3.set_title("Precision / Recall / F1", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("Score"); ax3.set_ylim(0, 1.05)
    ax3.set_xticks(ep)
    ax3.legend(fontsize=9)

    # ─── Bottom-right: Per-category AP (last epoch — horizontal bar) ─────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#161B22")
    last    = val_df.iloc[-1]
    last_ep = int(val_df["epoch_f"].iloc[-1])
    vals    = [last[f"val/AP/{c}"] * 100 for c in CLASSES]
    labels  = [CLASS_LABELS[c] for c in CLASSES]
    colors  = [CAT_COLORS[c] for c in CLASSES]
    y_pos   = np.arange(len(CLASSES))

    bars = ax4.barh(y_pos, vals, color=colors, edgecolor="#30363D", alpha=0.88)
    for bar, val in zip(bars, vals):
        ax4.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}", va="center", ha="left", fontsize=7.5,
                 color=bar.get_facecolor(), fontweight="bold")

    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(labels, fontsize=7.5)
    ax4.set_xlabel("AP (%)"); ax4.set_xlim(0, 110)
    ax4.invert_yaxis()
    ax4.set_title(f"Per-Part AP  ·  Epoch {last_ep}", fontsize=12, fontweight="bold")

    fig.suptitle("RF-DETR-Seg  ·  Parts Segmentation Training Dashboard",
                 fontsize=16, fontweight="bold", color="#E6EDF3", y=1.01)
    _savefig(fig, out / "combined_overview.png", tight=False)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",        default="metrics.csv",
                   help="Path to the training metrics CSV")
    p.add_argument("--output_dir", default="./parts_plots",
                   help="Directory to save charts")
    return p.parse_args()


def main():
    args  = parse_args()
    out   = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading CSV: {args.csv}")
    val_df, train_df = load_data(args.csv)
    print(f"[INFO] Val rows  : {len(val_df)}  |  Train rows: {len(train_df)}")
    print(f"[INFO] Epochs    : {int(val_df['epoch_f'].min())} → {int(val_df['epoch_f'].max())}")
    print(f"[INFO] Steps     : {int(train_df['step_f'].min())} → {int(train_df['step_f'].max())}")
    print(f"[INFO] Output dir: {out}/\n")

    charts = [
        ("Chart 1  – Val mAP Overview",          lambda: plot_val_map_overview(val_df, out)),
        ("Chart 2  – Val mAP75 & mAR",           lambda: plot_val_map75_mar(val_df, out)),
        ("Chart 3  – Val Precision/Recall/F1",   lambda: plot_val_prf(val_df, out)),
        ("Chart 4  – Val Per-Category AP",        lambda: plot_per_category_ap(val_df, out)),
        ("Chart 5  – Train Total Loss",           lambda: plot_train_loss_total(train_df, out)),
        ("Chart 6  – Train Loss Components",      lambda: plot_train_loss_components(train_df, out)),
        ("Chart 7  – Train Auxiliary Losses",     lambda: plot_train_loss_auxiliary(train_df, out)),
        ("Chart 8  – EMA vs Live mAP",            lambda: plot_ema_vs_live(val_df, out)),
        ("Chart 9  – Per-Category Radar",         lambda: plot_category_radar(val_df, out)),
        ("Chart 10 – Validation Loss",            lambda: plot_val_loss(val_df, out)),
        ("Chart 11 – Cardinality Error",          lambda: plot_cardinality_error(train_df, out)),
        ("Chart 12 – Combined Dashboard",         lambda: plot_combined_dashboard(val_df, train_df, out)),
    ]

    for name, fn in charts:
        print(f"[INFO] Generating {name} …")
        fn()

    print(f"\n[INFO] ✓ All {len(charts)} charts saved to: {out}/")
    print("[INFO] Files:")
    for f in sorted(out.glob("*.png")):
        print(f"        {f.name}")


if __name__ == "__main__":
    main()