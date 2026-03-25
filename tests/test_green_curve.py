"""
Test anomaly detection on the green curve extracted from the original image.

Demonstrates the 3-stage detection pipeline:
  Stage 1: Preprocessing (data loading)
  Stage 2: Algorithm detection (GSR spectral analysis)
  Stage 3: Post-processing (amplitude + frequency rules)

Also compares with a simple amplitude-only baseline.
"""

import json
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import median_filter
import sys
import os

sys.path.append(os.getcwd())

from data.preprocessor import Preprocessor
from models.gsr import GSR
from core.postprocess import PostProcessor
from config import AmplitudeConfig, DirectionConfig, FrequencyConfig, PostProcessingConfig

np.random.seed(42)
OUTPUT_DIR = Path("test_results")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# Data Loading
# ============================================================================

def load_extracted_data():
    """Load data extracted from the original image."""
    path = Path("test_results/green_curve_extracted.csv")
    if not path.exists():
        raise FileNotFoundError("Run extract_green_curve.py first.")

    df = pl.read_csv(str(path))
    metric = df['request_count'].to_numpy()
    t = np.arange(len(df))

    # Ground truth: spike = value significantly above rolling baseline
    baseline = median_filter(metric, size=61)
    failure_mask = metric > (baseline * 1.3)

    return df, metric, failure_mask, t, baseline


# ============================================================================
# Detection Methods
# ============================================================================

def run_gsr_detection(df, train_len):
    """Stage 1+2: Preprocess then GSR algorithm detection."""
    # --- Stage 1: Preprocessing (ratio-to-baseline) ---
    preprocessor = Preprocessor(mode="ratio", baseline_window=61)
    preprocessor.fit(df.head(train_len))
    preprocessed = preprocessor.transform(df)

    # --- Stage 2: GSR scoring ---
    input_dim = len(df.columns) - 1
    config = {
        "batch_size": 256,
        "gsr_auto_tune": True,
        "gsr_tune_window_sizes": [4, 8, 12, 16],
        "gsr_tune_dev_weights": [1.0, 2.0, 5.0],
        "gsr_tune_spec_weights": [0.0, 0.1, 0.3],
    }
    model = GSR("test_gsr", config, input_dim)
    model.fit(preprocessed.head(train_len))

    res = model.predict(preprocessed)
    if res.is_ok():
        scores = res.unwrap()
        pad = len(df) - len(scores)
        if pad > 0:
            scores = np.pad(scores, (pad, 0), mode='constant', constant_values=0)
        threshold = model.auto_threshold if model.auto_threshold is not None else 2.0
        return scores, threshold
    return np.zeros(len(df)), 2.0


def run_postprocessor(anomalies, df, feature_cols):
    """Stage 3: Post-processing filters false positives from algorithm output.

    - Amplitude (relative): confirm value > local_baseline * 1.3
    - Direction: suppress downward deviations (only upward = anomaly)
    """
    config = PostProcessingConfig(
        enabled=True,
        amplitude=AmplitudeConfig(
            enabled=True,
            relative_threshold=1.3,
            baseline_window=61,
        ),
        frequency=FrequencyConfig(enabled=False),
        direction=DirectionConfig(enabled=True, direction="up"),
    )
    pp = PostProcessor(config)
    return pp.process(anomalies, df, feature_cols)


# ============================================================================
# Metrics
# ============================================================================

def calc_metrics(truth, detected):
    tp = np.sum(truth & detected)
    fp = np.sum(~truth & detected)
    fn = np.sum(truth & ~detected)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    return p, r, f1


# ============================================================================
# Visualization
# ============================================================================

def _style_ax(ax, grid_color='#e5e7eb', text_color='#1f2937'):
    """Apply common axis styling."""
    ax.set_facecolor('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(grid_color)
    ax.spines['bottom'].set_color(grid_color)
    ax.tick_params(colors=text_color, labelsize=9)
    ax.grid(True, alpha=0.25, color=grid_color, linewidth=0.5)


def _shade_gt(ax, t, mask, color='#22c55e', alpha=0.10, label='Ground Truth'):
    """Add green shading for ground-truth anomaly regions."""
    if not np.any(mask):
        return
    diff = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    for i, (s, e) in enumerate(zip(starts, ends)):
        ax.axvspan(t[s], t[min(e, len(t) - 1)], color=color, alpha=alpha,
                   label=label if i == 0 else "")


def _draw_detection_row(ax, t, metric, baseline, failure_mask, detected,
                        title, metrics_label, show_fp=False, show_fn=False,
                        vline_positions=None):
    """Draw a detection row as a curve with marked detection points.

    Args:
        vline_positions: Optional array of x-indices where red vertical
            dashed lines should be drawn (for cross-panel FP alignment).
            If None, lines are drawn at this panel's own FP positions.
    """
    C_LINE = '#2563eb'
    C_BASELINE = '#9ca3af'
    C_GT = '#22c55e'
    C_TP = '#3b82f6'
    C_FP = '#ef4444'
    C_FN = '#8b5cf6'

    _style_ax(ax)

    # Metric curve + baseline
    ax.plot(t, metric, color=C_LINE, linewidth=0.8, alpha=0.7)
    ax.plot(t, baseline, color=C_BASELINE, linewidth=1.0, linestyle='--', alpha=0.5)
    _shade_gt(ax, t, failure_mask, alpha=0.08)

    fp_idx = np.where(detected & ~failure_mask)[0]
    fn_idx = np.where(~detected & failure_mask)[0]
    tp_idx = np.where(detected & failure_mask)[0]

    # FP drop lines: from metric value down to x-axis + time label
    vlines = vline_positions if vline_positions is not None else fp_idx
    fp_set = set(fp_idx.tolist())
    for idx in vlines:
        y_val = metric[idx]
        ls = '-' if idx in fp_set else ':'
        ax.plot([t[idx], t[idx]], [0, y_val], color=C_FP, alpha=0.5,
                linewidth=0.9, linestyle=ls)
        # Time label on x-axis
        hours = 12 + idx / 60  # data starts at 12:00, 1 point per minute
        if hours >= 24:
            hours -= 24
        h, m = int(hours), int((hours % 1) * 60)
        ax.text(t[idx], -0.02 * metric.max(), f'{h:02d}:{m:02d}',
                color=C_FP, fontsize=6, fontweight='bold',
                ha='center', va='top', rotation=45)

    # TP as blue dots
    if len(tp_idx) > 0:
        ax.scatter(t[tp_idx], metric[tp_idx], color=C_TP, s=22, zorder=5,
                   edgecolors='white', linewidths=0.4, label=f'TP ({len(tp_idx)})')

    # FP as red dots with annotation arrows
    if show_fp and len(fp_idx) > 0:
        ax.scatter(t[fp_idx], metric[fp_idx], color=C_FP, s=40, zorder=6,
                   edgecolors='white', linewidths=0.4, label=f'FP ({len(fp_idx)})')
        # Annotate FP clusters
        y_range = metric.max() - metric.min()
        clusters = []
        cluster_start = fp_idx[0]
        for i in range(1, len(fp_idx)):
            if fp_idx[i] - fp_idx[i - 1] > 10:
                clusters.append((cluster_start, fp_idx[i - 1]))
                cluster_start = fp_idx[i]
        clusters.append((cluster_start, fp_idx[-1]))
        for cs, ce in clusters:
            cx = int((cs + ce) / 2)
            cy = metric[cx]
            ax.annotate('FP', xy=(cx, cy),
                        xytext=(cx, cy + y_range * 0.35),
                        fontsize=9, fontweight='bold', color=C_FP,
                        ha='center', va='bottom',
                        arrowprops=dict(arrowstyle='->', color=C_FP, lw=1.8))

    # FN as purple triangles
    if show_fn and len(fn_idx) > 0:
        ax.scatter(t[fn_idx], metric[fn_idx], color=C_FN, s=40, zorder=6,
                   marker='v', label=f'FN ({len(fn_idx)})')

    p, r, f1 = calc_metrics(failure_mask, detected)
    ax.set_title(f"{title}  —  P={p:.2f}  R={r:.2f}  F1={f1:.2f}   {metrics_label}",
                 fontsize=10, fontweight='bold', loc='left', color='#1f2937')
    ax.set_ylabel('Request Count', color='#1f2937', fontsize=9)
    ax.legend(loc='upper right', fontsize=7, framealpha=0.9, edgecolor='#e5e7eb',
              ncol=4, handletextpad=0.4, columnspacing=1.0)


def plot_results(t, metric, failure_mask, baseline,
                 gsr_scores, gsr_threshold,
                 gsr_detected, pp_detected):
    C_LINE = '#2563eb'
    C_BASELINE = '#9ca3af'
    C_SCORE = '#3b82f6'
    C_THRESH = '#ef4444'
    C_ANOMALY = '#ef4444'
    C_GT = '#22c55e'
    C_GRID = '#e5e7eb'
    C_TEXT = '#1f2937'

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
    })

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True,
                             gridspec_kw={'height_ratios': [3, 1.5, 2, 2],
                                          'hspace': 0.35})
    fig.patch.set_facecolor('white')

    time_ticks = [0, 120, 240, 360, 480, 600, 720, 840, 960, 1080, 1200, 1320]
    time_labels = ['12:00', '14:00', '16:00', '18:00', '20:00', '22:00',
                   '00:00', '02:00', '04:00', '06:00', '08:00', '10:00']

    # ── Panel 1: Raw data + anomaly markers ──
    ax = axes[0]
    _style_ax(ax)
    ax.plot(t, metric, color=C_LINE, linewidth=0.9, alpha=0.85, label='Request Count')
    ax.plot(t, baseline, color=C_BASELINE, linewidth=1.2, linestyle='--', alpha=0.7,
            label='Baseline')
    _shade_gt(ax, t, failure_mask)

    # Final PP detections as dots on the curve
    pp_idx = np.where(pp_detected)[0]
    if len(pp_idx) > 0:
        ax.scatter(t[pp_idx], metric[pp_idx], color=C_ANOMALY, s=18, zorder=5,
                   edgecolors='white', linewidths=0.4, label='Final Detection')

    ax.set_ylabel("Request Count", color=C_TEXT)
    ax.set_title("Green Curve — Anomaly Detection Pipeline",
                 color=C_TEXT, fontsize=14, fontweight='bold', pad=12)
    ax.legend(loc='upper right', framealpha=0.9, edgecolor=C_GRID, fontsize=8)

    # ── Panel 2: GSR Scores ──
    ax = axes[1]
    _style_ax(ax)
    ax.fill_between(t, 0, gsr_scores, color=C_SCORE, alpha=0.15)
    ax.plot(t, gsr_scores, color=C_SCORE, linewidth=0.8, label='GSR Score')
    ax.axhline(y=gsr_threshold, color=C_THRESH, linestyle='--', linewidth=1.2, alpha=0.8,
               label=f'Threshold ({gsr_threshold:.2f})')

    above_idx = np.where(gsr_scores > gsr_threshold)[0]
    if len(above_idx) > 0:
        ax.scatter(t[above_idx], gsr_scores[above_idx], color=C_ANOMALY, s=12, zorder=5,
                   edgecolors='white', linewidths=0.3)

    p_gsr, r_gsr, f1_gsr = calc_metrics(failure_mask, gsr_detected)
    ax.set_ylabel("Score", color=C_TEXT)
    ax.set_title(f"Stage 2: GSR Algorithm (margin=0.9)  —  P={p_gsr:.2f}  R={r_gsr:.2f}  F1={f1_gsr:.2f}",
                 color=C_TEXT, fontsize=10, fontweight='bold', loc='left')
    ax.legend(loc='upper right', framealpha=0.9, edgecolor=C_GRID, fontsize=8)

    # ── Mark FP positions on Panel 1 & 2 ──
    fp_before_idx = np.where(gsr_detected & ~failure_mask)[0]
    for panel_ax in [axes[0], axes[1]]:
        for idx in fp_before_idx:
            panel_ax.axvline(t[idx], color=C_ANOMALY, alpha=0.4,
                             linewidth=0.8, linestyle='--')

    # ── Panel 3: Before post-processing (GSR raw detections) ──
    n_fp_before = len(fp_before_idx)
    _draw_detection_row(
        axes[2], t, metric, baseline, failure_mask, gsr_detected,
        title="Before Post-Processing",
        metrics_label=f"({n_fp_before} false positives)" if n_fp_before > 0 else "",
        show_fp=True, show_fn=True,
    )

    # ── Panel 4: After post-processing (same vlines as panel 3 for comparison) ──
    n_fp_after = int(np.sum(pp_detected & ~failure_mask))
    filtered = n_fp_before - n_fp_after
    _draw_detection_row(
        axes[3], t, metric, baseline, failure_mask, pp_detected,
        title="After Post-Processing",
        metrics_label=f"(filtered {filtered} FP)" if filtered > 0 else "",
        show_fp=True, show_fn=True,
        vline_positions=fp_before_idx,
    )
    axes[3].set_xlabel("Time", color=C_TEXT, fontsize=11)
    axes[3].set_xticks(time_ticks)
    axes[3].set_xticklabels(time_labels)

    plt.savefig(OUTPUT_DIR / "green_curve_detection.png", dpi=180, facecolor='white',
                bbox_inches='tight', pad_inches=0.3)
    plt.close()
    plt.rcParams.update(plt.rcParamsDefault)
    print(f"   Plot saved to {OUTPUT_DIR / 'green_curve_detection.png'}")


# ============================================================================
# ECharts Data Export
# ============================================================================

def _idx_to_time(idx):
    """Convert data index to HH:MM string (data starts at 12:00, 1pt/min)."""
    hours = 12 + idx / 60
    if hours >= 24:
        hours -= 24
    return f'{int(hours):02d}:{int((hours % 1) * 60):02d}'


def export_echart_data(t, metric, baseline, failure_mask,
                       gsr_scores, gsr_threshold,
                       gsr_detected, pp_detected):
    """Export all chart data to JSON for the ECharts HTML page."""
    time_labels = [_idx_to_time(i) for i in range(len(t))]

    # Ground truth regions (start, end pairs)
    diff = np.diff(np.concatenate(([0], failure_mask.astype(int), [0])))
    gt_regions = list(zip(
        np.where(diff == 1)[0].tolist(),
        (np.where(diff == -1)[0] - 1).tolist()
    ))

    fp_before = np.where(gsr_detected & ~failure_mask)[0].tolist()
    tp_before = np.where(gsr_detected & failure_mask)[0].tolist()
    tp_after = np.where(pp_detected & failure_mask)[0].tolist()
    fp_after = np.where(pp_detected & ~failure_mask)[0].tolist()

    p_gsr, r_gsr, f1_gsr = calc_metrics(failure_mask, gsr_detected)
    p_pp, r_pp, f1_pp = calc_metrics(failure_mask, pp_detected)

    data = {
        'timeLabels': time_labels,
        'metric': metric.tolist(),
        'baseline': baseline.tolist(),
        'gsrScores': gsr_scores.tolist(),
        'gsrThreshold': float(gsr_threshold),
        'gtRegions': gt_regions,
        'fpBefore': fp_before,
        'tpBefore': tp_before,
        'fpAfter': fp_after,
        'tpAfter': tp_after,
        'ppDetected': np.where(pp_detected)[0].tolist(),
        'metricsGsr': {'p': round(p_gsr, 2), 'r': round(r_gsr, 2), 'f1': round(f1_gsr, 2)},
        'metricsPp': {'p': round(p_pp, 2), 'r': round(r_pp, 2), 'f1': round(f1_pp, 2)},
        'nFpBefore': len(fp_before),
        'nFiltered': len(fp_before) - len(fp_after),
    }

    out_path = OUTPUT_DIR / "green_curve_data.json"
    with open(out_path, 'w') as f:
        json.dump(data, f)
    print(f"   ECharts data saved to {out_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 60)
    print("Green Curve: 3-Stage Detection Pipeline")
    print("=" * 60)

    # Stage 1: Load data
    print("\n[Stage 1] Loading extracted data...")
    df, metric, failure_mask, t, baseline = load_extracted_data()
    feature_cols = [c for c in df.columns if c != 'timestamp']
    print(f"   Points: {len(df)}, Anomaly points: {np.sum(failure_mask)}")
    print(f"   Baseline: {baseline.mean():.0f}, Max spike: {metric.max():.0f}")

    # Stage 2: GSR algorithm detection
    print("\n[Stage 2] Running GSR detection...")
    train_len = 300  # ~5 hours of normal baseline
    gsr_scores, gsr_threshold = run_gsr_detection(df, train_len)
    gsr_detected = gsr_scores > gsr_threshold

    p, r, f1 = calc_metrics(failure_mask, gsr_detected)
    print(f"   GSR-only: P={p:.4f} R={r:.4f} F1={f1:.4f}")

    # Stage 3: Post-processing
    print("\n[Stage 3] Applying post-processing rules...")
    pp_detected = run_postprocessor(gsr_detected, df, feature_cols)

    p, r, f1 = calc_metrics(failure_mask, pp_detected)
    print(f"   After PP:  P={p:.4f} R={r:.4f} F1={f1:.4f}")

    # Also show what pure amplitude detection gives us
    print("\n[Comparison] Pure amplitude detection (no algorithm)...")
    amp_threshold = baseline.mean() * 1.2
    amp_only = metric > amp_threshold
    p, r, f1 = calc_metrics(failure_mask, amp_only)
    print(f"   Amp-only:  P={p:.4f} R={r:.4f} F1={f1:.4f}")

    # Visualization
    print("\n[Visualization]")
    plot_results(t, metric, failure_mask, baseline,
                 gsr_scores, gsr_threshold,
                 gsr_detected, pp_detected)

    # Export data for ECharts HTML
    export_echart_data(t, metric, baseline, failure_mask,
                       gsr_scores, gsr_threshold,
                       gsr_detected, pp_detected)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
