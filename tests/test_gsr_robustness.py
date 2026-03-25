"""
GSR Algorithm Robustness Test Suite

Generates synthetic test scenarios to validate the GSR algorithm across:
  1. Single-dim spike detection (baseline)
  2. Multi-dim data: anomaly in one feature, noise in others
  3. Multi-dim data: anomaly in ALL features simultaneously
  4. Gradual drift (slow level shift)
  5. High noise environment
  6. Low noise (near-constant baseline)
  7. Varying anomaly magnitudes (1.5x, 2x, 4x baseline)
  8. Short vs long anomaly segments
  9. Multiple anomaly clusters
  10. Seasonal pattern with anomalies

Each scenario generates a Polars DataFrame with 'timestamp' and feature columns,
plus a ground-truth boolean mask. The pipeline (Preprocessor → GSR) is run and
F1 / Precision / Recall are reported.
"""

import sys
import os
sys.path.append(os.getcwd())

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path

from data.preprocessor import Preprocessor
from models.gsr import GSR
from core.postprocess import PostProcessor
from config import PostProcessingConfig, AmplitudeConfig, FrequencyConfig, DirectionConfig

np.random.seed(42)
OUTPUT_DIR = Path("test_results")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# Helpers
# ============================================================================

def calc_metrics(truth: np.ndarray, detected: np.ndarray):
    tp = np.sum(truth & detected)
    fp = np.sum(~truth & detected)
    fn = np.sum(truth & ~detected)
    # Special case: no ground-truth anomalies (e.g. downward-only scenario).
    # Perfect behaviour = zero detections → P=1, R=1, F1=1.
    if not np.any(truth):
        if not np.any(detected):
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 1.0, 0.0  # all detections are FP
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def make_df(columns: dict, n: int) -> pl.DataFrame:
    """Build a Polars DataFrame with a timestamp column."""
    columns["timestamp"] = np.arange(n)
    return pl.DataFrame(columns)


def inject_anomalies(values: np.ndarray, anomaly_ranges: list, multiplier: float):
    """Multiply values in anomaly_ranges by multiplier."""
    out = values.copy()
    for (s, e) in anomaly_ranges:
        out[s:e] *= multiplier
    return out


def inject_anomalies_offset(values: np.ndarray, anomaly_ranges: list, offset: float):
    """Add offset to values in anomaly_ranges (use negative for downward)."""
    out = values.copy()
    for (s, e) in anomaly_ranges:
        out[s:e] += offset
    return out


def make_ground_truth(n: int, anomaly_ranges: list) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for (s, e) in anomaly_ranges:
        mask[s:e] = True
    return mask


def run_pipeline(df: pl.DataFrame, train_len: int, baseline_window: int = 61):
    """Run Preprocessor + GSR and return (scores, threshold, detected, model, baseline)."""
    pp = Preprocessor(mode="ratio", baseline_window=baseline_window)
    pp.fit(df.head(train_len))
    preprocessed = pp.transform(df)

    input_dim = len([c for c in df.columns if c != "timestamp"])
    config = {
        "batch_size": 256,
        "gsr_auto_tune": True,
        "gsr_tune_window_sizes": [4, 8, 12, 16],
        "gsr_tune_dev_weights": [1.0, 2.0, 5.0],
        "gsr_tune_spec_weights": [0.0, 0.1, 0.3],
    }
    model = GSR("test", config, input_dim)
    model.fit(preprocessed.head(train_len))

    res = model.predict(preprocessed)
    scores = res.unwrap()
    pad = len(df) - len(scores)
    if pad > 0:
        scores = np.pad(scores, (pad, 0), mode="constant", constant_values=0)

    threshold = model.auto_threshold if model.auto_threshold is not None else 2.0
    detected = scores > threshold

    # Post-processing: direction filter (suppress downward spikes)
    feat_cols = [c for c in df.columns if c != "timestamp"]
    pp_config = PostProcessingConfig(
        enabled=True,
        amplitude=AmplitudeConfig(enabled=False),
        frequency=FrequencyConfig(enabled=False),
        direction=DirectionConfig(enabled=True, direction="up"),
    )
    post_processor = PostProcessor(pp_config)
    detected = post_processor.process(detected, df, feat_cols)

    # Extract baseline for plotting (first non-timestamp feature)
    prep_feat_cols = [c for c in preprocessed.columns if c != "timestamp"]
    baseline = preprocessed[prep_feat_cols[0]].to_numpy()

    return scores, threshold, detected, model, baseline


def plot_scenario_detail(name, df, gt, train_len, scores, threshold,
                         detected, model, baseline, ax_row):
    """Plot a single scenario across 3 axes (like the green curve plot)."""
    # Colors
    C_LINE = '#2563eb'
    C_BASELINE = '#9ca3af'
    C_SCORE = '#3b82f6'
    C_THRESH = '#ef4444'
    C_ANOMALY = '#ef4444'
    C_GT = '#22c55e'
    C_GRID = '#e5e7eb'
    C_TEXT = '#1f2937'

    feat_cols = [c for c in df.columns if c != "timestamp"]
    metric = df[feat_cols[0]].to_numpy()
    t = np.arange(len(metric))

    p, r, f1 = calc_metrics(gt, detected)
    tag = "PASS" if f1 >= 0.9 else ("WARN" if f1 >= 0.7 else "FAIL")
    tag_color = "#22c55e" if f1 >= 0.9 else ("#f59e0b" if f1 >= 0.7 else "#ef4444")

    # ── Panel 1: Raw data + anomaly markers ──
    ax = ax_row[0]
    ax.set_facecolor('white')
    ax.plot(t, metric, color=C_LINE, linewidth=0.7, alpha=0.85, label=feat_cols[0])

    # Training region shading
    ax.axvspan(0, train_len, color='#dbeafe', alpha=0.3, label='Training')

    # Ground truth shading
    if np.any(gt):
        diff = np.diff(np.concatenate(([0], gt.astype(int), [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        for i, (s, e) in enumerate(zip(starts, ends)):
            ax.axvspan(s, min(e, len(t)-1), color=C_GT, alpha=0.12,
                       label='Ground Truth' if i == 0 else "")

    # Red dots on detected anomaly points
    anom_idx = np.where(detected)[0]
    if len(anom_idx) > 0:
        ax.scatter(t[anom_idx], metric[anom_idx], color=C_ANOMALY, s=10, zorder=5,
                   edgecolors='white', linewidths=0.3, label='Detected')

    ax.set_title(f"{name}  —  F1={f1:.2f} [{tag}]",
                 color=tag_color, fontsize=10, fontweight='bold', loc='left')
    ax.legend(loc='upper right', fontsize=6, framealpha=0.8, edgecolor=C_GRID)
    ax.grid(True, alpha=0.2, color=C_GRID, linewidth=0.5)
    ax.tick_params(labelsize=7, colors=C_TEXT)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(C_GRID)
    ax.spines['bottom'].set_color(C_GRID)

    # ── Panel 2: GSR Scores + threshold ──
    ax = ax_row[1]
    ax.set_facecolor('white')
    ax.fill_between(t, 0, scores, color=C_SCORE, alpha=0.12)
    ax.plot(t, scores, color=C_SCORE, linewidth=0.6, label='GSR Score')
    ax.axhline(y=threshold, color=C_THRESH, linestyle='--', linewidth=1.0, alpha=0.8,
               label=f'Threshold ({threshold:.3f})')

    above_idx = np.where(scores > threshold)[0]
    if len(above_idx) > 0:
        ax.scatter(t[above_idx], scores[above_idx], color=C_ANOMALY, s=6, zorder=5,
                   edgecolors='white', linewidths=0.2)

    info = f"ws={model.window_size}  dw={model.deviation_weight:.0f}  sw={model.spectral_weight:.1f}  ac1={model.data_ac1:.2f}"
    ax.set_title(f"Scores  |  {info}", color=C_TEXT, fontsize=8, loc='left')
    ax.legend(loc='upper right', fontsize=6, framealpha=0.8, edgecolor=C_GRID)
    ax.grid(True, alpha=0.2, color=C_GRID, linewidth=0.5)
    ax.tick_params(labelsize=7, colors=C_TEXT)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(C_GRID)
    ax.spines['bottom'].set_color(C_GRID)

    # ── Panel 3: Detection comparison ──
    ax = ax_row[2]
    ax.set_facecolor('white')

    gt_y = np.where(gt, 1.0, 0.0)
    ax.bar(t, gt_y, width=1.0, color=C_GT, alpha=0.35, label='Ground Truth')

    det_y = np.where(detected, 0.7, 0.0)
    ax.bar(t, det_y, width=1.0, color=C_ANOMALY, alpha=0.5, label='Detected')

    fp_idx = np.where(detected & ~gt)[0]
    fn_idx = np.where(~detected & gt)[0]
    if len(fp_idx) > 0:
        ax.scatter(t[fp_idx], np.full(len(fp_idx), 0.85), marker='x', color='#f97316',
                   s=15, zorder=5, linewidths=1.0, label=f'FP ({len(fp_idx)})')
    if len(fn_idx) > 0:
        ax.scatter(t[fn_idx], np.full(len(fn_idx), 0.85), marker='v', color='#6366f1',
                   s=15, zorder=5, label=f'FN ({len(fn_idx)})')

    ax.set_title(f"P={p:.2f}  R={r:.2f}  F1={f1:.2f}", color=C_TEXT, fontsize=8, loc='left')
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0, 0.5, 1.0])
    ax.legend(loc='upper right', fontsize=6, ncol=4, framealpha=0.8, edgecolor=C_GRID)
    ax.grid(True, alpha=0.2, color=C_GRID, linewidth=0.5)
    ax.tick_params(labelsize=7, colors=C_TEXT)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(C_GRID)
    ax.spines['bottom'].set_color(C_GRID)


# ============================================================================
# Scenario Generators
# ============================================================================

def scenario_1d_spike():
    """S1: Single-dim, clear spikes (baseline scenario)."""
    n = 1000
    base = 100.0
    noise_std = 2.0
    values = base + np.random.randn(n) * noise_std
    anomaly_ranges = [(400, 420), (600, 610), (800, 805)]
    values = inject_anomalies(values, anomaly_ranges, 2.0)
    gt = make_ground_truth(n, anomaly_ranges)
    df = make_df({"metric": values}, n)
    return "S1: 1D Spike (2x)", df, gt, 200


def scenario_multidim_single_feature():
    """S2: 3-dim data, anomaly only in feature_1."""
    n = 1000
    base = 100.0
    f1 = base + np.random.randn(n) * 2.0
    f2 = base + np.random.randn(n) * 5.0  # noisy but normal
    f3 = base + np.random.randn(n) * 3.0  # normal
    anomaly_ranges = [(500, 530)]
    f1 = inject_anomalies(f1, anomaly_ranges, 2.0)
    gt = make_ground_truth(n, anomaly_ranges)
    df = make_df({"feature_1": f1, "feature_2": f2, "feature_3": f3}, n)
    return "S2: 3D, anomaly in 1 feature", df, gt, 200


def scenario_multidim_all_features():
    """S3: 3-dim data, anomaly in ALL features."""
    n = 1000
    base = 100.0
    f1 = base + np.random.randn(n) * 2.0
    f2 = base + np.random.randn(n) * 2.0
    f3 = base + np.random.randn(n) * 2.0
    anomaly_ranges = [(500, 530)]
    f1 = inject_anomalies(f1, anomaly_ranges, 2.0)
    f2 = inject_anomalies(f2, anomaly_ranges, 1.8)
    f3 = inject_anomalies(f3, anomaly_ranges, 2.2)
    gt = make_ground_truth(n, anomaly_ranges)
    df = make_df({"feature_1": f1, "feature_2": f2, "feature_3": f3}, n)
    return "S3: 3D, anomaly in ALL features", df, gt, 200


def scenario_gradual_drift():
    """S4: Gradual level shift (ramp up over 50 points to 2x)."""
    n = 1000
    base = 100.0
    values = base + np.random.randn(n) * 2.0
    # Gradual ramp from 1x to 2x over 50 points, then stay at 2x for 30 points
    ramp = np.linspace(1.0, 2.0, 50)
    values[500:550] *= ramp
    values[550:580] *= 2.0
    gt = make_ground_truth(n, [(500, 580)])
    df = make_df({"metric": values}, n)
    return "S4: Gradual Drift (ramp to 2x)", df, gt, 200


def scenario_high_noise():
    """S5: High noise (std = 15% of baseline), spikes at 2x."""
    n = 1000
    base = 100.0
    noise_std = 15.0  # 15% of baseline
    values = base + np.random.randn(n) * noise_std
    anomaly_ranges = [(500, 530)]
    values = inject_anomalies(values, anomaly_ranges, 2.0)
    gt = make_ground_truth(n, anomaly_ranges)
    df = make_df({"metric": values}, n)
    return "S5: High Noise (15% std), 2x spike", df, gt, 200


def scenario_low_noise():
    """S6: Very low noise (std = 0.1% of baseline), spikes at 1.5x."""
    n = 1000
    base = 100.0
    noise_std = 0.1
    values = base + np.random.randn(n) * noise_std
    anomaly_ranges = [(500, 530)]
    values = inject_anomalies(values, anomaly_ranges, 1.5)
    gt = make_ground_truth(n, anomaly_ranges)
    df = make_df({"metric": values}, n)
    return "S6: Low Noise (0.1% std), 1.5x spike", df, gt, 200


def scenario_varying_magnitudes():
    """S7: Three anomaly clusters with different magnitudes (1.5x, 2x, 4x)."""
    n = 1200
    base = 100.0
    values = base + np.random.randn(n) * 2.0
    values[400:420] *= 1.5
    values[600:620] *= 2.0
    values[900:920] *= 4.0
    gt = make_ground_truth(n, [(400, 420), (600, 620), (900, 920)])
    df = make_df({"metric": values}, n)
    return "S7: Varying Magnitudes (1.5x, 2x, 4x)", df, gt, 200


def scenario_short_anomaly():
    """S8: Very short anomaly (3 points only)."""
    n = 1000
    base = 100.0
    values = base + np.random.randn(n) * 2.0
    anomaly_ranges = [(500, 503)]
    values = inject_anomalies(values, anomaly_ranges, 3.0)
    gt = make_ground_truth(n, anomaly_ranges)
    df = make_df({"metric": values}, n)
    return "S8: Short Anomaly (3 points, 3x)", df, gt, 200


def scenario_long_anomaly():
    """S9: Long anomaly segment (200 points)."""
    n = 1200
    base = 100.0
    values = base + np.random.randn(n) * 2.0
    anomaly_ranges = [(500, 700)]
    values = inject_anomalies(values, anomaly_ranges, 2.0)
    gt = make_ground_truth(n, anomaly_ranges)
    df = make_df({"metric": values}, n)
    return "S9: Long Anomaly (200 points, 2x)", df, gt, 200


def scenario_multiple_clusters():
    """S10: Many small anomaly clusters spread across the series."""
    n = 1500
    base = 100.0
    values = base + np.random.randn(n) * 2.0
    anomaly_ranges = [(300, 310), (500, 510), (700, 715), (900, 905),
                      (1100, 1120), (1300, 1310)]
    values = inject_anomalies(values, anomaly_ranges, 2.5)
    gt = make_ground_truth(n, anomaly_ranges)
    df = make_df({"metric": values}, n)
    return "S10: Multiple Clusters (6 groups)", df, gt, 200


def scenario_seasonal():
    """S11: Seasonal pattern (sinusoidal) with anomalies."""
    n = 1200
    base = 100.0
    seasonal = 20.0 * np.sin(2 * np.pi * np.arange(n) / 200)  # period=200
    values = base + seasonal + np.random.randn(n) * 2.0
    anomaly_ranges = [(500, 520), (900, 915)]
    values = inject_anomalies(values, anomaly_ranges, 2.0)
    gt = make_ground_truth(n, anomaly_ranges)
    df = make_df({"metric": values}, n)
    return "S11: Seasonal + Anomaly (2x)", df, gt, 200


def scenario_multidim_dilution():
    """S12: 5-dim data, anomaly in 1 feature, 4 noisy irrelevant features.
    Tests whether noisy features dilute the anomaly signal."""
    n = 1000
    base = 100.0
    f_anom = base + np.random.randn(n) * 2.0
    anomaly_ranges = [(500, 530)]
    f_anom = inject_anomalies(f_anom, anomaly_ranges, 2.0)
    gt = make_ground_truth(n, anomaly_ranges)

    cols = {"anomaly_feature": f_anom}
    for i in range(4):
        cols[f"noise_{i}"] = base + np.random.randn(n) * 10.0  # high noise, no anomaly
    df = make_df(cols, n)
    return "S12: 5D Dilution (1 signal + 4 noise)", df, gt, 200


def scenario_small_spike():
    """S13: Very small anomaly (1.3x baseline) — near detection limit."""
    n = 1000
    base = 100.0
    values = base + np.random.randn(n) * 2.0
    anomaly_ranges = [(500, 530)]
    values = inject_anomalies(values, anomaly_ranges, 1.3)
    gt = make_ground_truth(n, anomaly_ranges)
    df = make_df({"metric": values}, n)
    return "S13: Small Spike (1.3x) — near limit", df, gt, 200


def scenario_downward_only():
    """S14: Downward-only spikes — should NOT be detected.
    Tests that the direction filter suppresses all downward anomalies.
    GT is empty (no upward anomalies), so perfect score = no detections."""
    n = 1000
    base = 100.0
    values = base + np.random.randn(n) * 2.0
    # Inject downward spikes (multiply by 0.5 = halve the value)
    values = inject_anomalies(values, [(400, 420)], 0.5)
    values = inject_anomalies(values, [(600, 620)], 0.3)
    # GT is empty — downward spikes are NOT anomalies
    gt = np.zeros(n, dtype=bool)
    df = make_df({"metric": values}, n)
    return "S14: Downward Only (should suppress)", df, gt, 200


def scenario_mixed_direction():
    """S15: Mixed upward + downward spikes — only upward should be detected.
    Tests that the pipeline detects upward anomalies while suppressing
    downward ones in the same series."""
    n = 1000
    base = 100.0
    values = base + np.random.randn(n) * 2.0
    # Upward spike (anomaly)
    up_ranges = [(400, 420)]
    values = inject_anomalies(values, up_ranges, 2.0)
    # Downward spike (NOT anomaly)
    values = inject_anomalies(values, [(600, 620)], 0.3)
    # GT only marks the upward spike
    gt = make_ground_truth(n, up_ranges)
    df = make_df({"metric": values}, n)
    return "S15: Mixed Up+Down (detect up only)", df, gt, 200


def scenario_adjacent_up_down():
    """S16: Upward spike immediately followed by a downward dip.
    Common pattern in real data (e.g., traffic spike then recovery dip).
    Only the upward part should be detected."""
    n = 1000
    base = 100.0
    values = base + np.random.randn(n) * 2.0
    # Upward spike
    up_ranges = [(500, 520)]
    values = inject_anomalies(values, up_ranges, 2.5)
    # Immediate downward dip after spike
    values = inject_anomalies_offset(values, [(520, 540)], -50.0)
    # GT only marks the upward spike
    gt = make_ground_truth(n, up_ranges)
    df = make_df({"metric": values}, n)
    return "S16: Up Spike + Down Dip (detect up only)", df, gt, 200


# ============================================================================
# Main
# ============================================================================

ALL_SCENARIOS = [
    scenario_1d_spike,
    scenario_multidim_single_feature,
    scenario_multidim_all_features,
    scenario_gradual_drift,
    scenario_high_noise,
    scenario_low_noise,
    scenario_varying_magnitudes,
    scenario_short_anomaly,
    scenario_long_anomaly,
    scenario_multiple_clusters,
    scenario_seasonal,
    scenario_multidim_dilution,
    scenario_small_spike,
    scenario_downward_only,
    scenario_mixed_direction,
    scenario_adjacent_up_down,
]


def main():
    print("=" * 70)
    print("GSR + PostProcessor (direction=up) Robustness Test Suite")
    print("=" * 70)

    results = []

    for scenario_fn in ALL_SCENARIOS:
        name, df, gt, train_len = scenario_fn()
        n_features = len([c for c in df.columns if c != "timestamp"])
        n_anomalies = int(gt.sum())

        try:
            scores, threshold, detected, model, baseline = run_pipeline(df, train_len)
            p, r, f1 = calc_metrics(gt, detected)
            ws = model.window_size
            dw = model.deviation_weight
            sw = model.spectral_weight

            results.append({
                "name": name,
                "dims": n_features,
                "anomalies": n_anomalies,
                "P": p,
                "R": r,
                "F1": f1,
                "threshold": threshold,
                "ws": ws,
                "dw": dw,
                "sw": sw,
                "status": "OK",
                "_df": df,
                "_gt": gt,
                "_train_len": train_len,
                "_scores": scores,
                "_detected": detected,
                "_model": model,
                "_baseline": baseline,
            })
        except Exception as e:
            results.append({
                "name": name,
                "dims": n_features,
                "anomalies": n_anomalies,
                "P": 0, "R": 0, "F1": 0,
                "threshold": 0, "ws": 0, "dw": 0, "sw": 0,
                "status": f"ERROR: {e}",
            })

    # ── Print results table ──
    print(f"\n{'Scenario':<42} {'Dim':>3} {'Anom':>4}  {'P':>5} {'R':>5} {'F1':>5}  "
          f"{'Thresh':>7} {'WS':>3} {'DW':>4} {'SW':>4}  {'Status'}")
    print("-" * 120)

    pass_count = 0
    warn_count = 0
    fail_count = 0

    for r in results:
        if r["status"] != "OK":
            tag = "ERROR"
            fail_count += 1
        elif r["F1"] >= 0.9:
            tag = "PASS"
            pass_count += 1
        elif r["F1"] >= 0.7:
            tag = "WARN"
            warn_count += 1
        else:
            tag = "FAIL"
            fail_count += 1

        print(f"{r['name']:<42} {r['dims']:>3} {r['anomalies']:>4}  "
              f"{r['P']:>5.2f} {r['R']:>5.2f} {r['F1']:>5.2f}  "
              f"{r['threshold']:>7.3f} {r['ws']:>3} {r['dw']:>4.1f} {r['sw']:>4.1f}  "
              f"{tag}")

    print("-" * 120)
    print(f"Summary: {pass_count} PASS, {warn_count} WARN, {fail_count} FAIL "
          f"(out of {len(results)} scenarios)")

    # ── Visualization 1: summary bar chart ──
    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    names = [r["name"] for r in results]
    f1s = [r["F1"] for r in results]
    colors = ["#22c55e" if f >= 0.9 else "#f59e0b" if f >= 0.7 else "#ef4444" for f in f1s]

    bars = ax.barh(range(len(names)), f1s, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("F1 Score", fontsize=11)
    ax.set_title("GSR Robustness Test — F1 Scores by Scenario",
                 fontsize=13, fontweight="bold", pad=10)
    ax.axvline(x=0.9, color="#22c55e", linestyle="--", alpha=0.5, label="PASS (≥0.9)")
    ax.axvline(x=0.7, color="#f59e0b", linestyle="--", alpha=0.5, label="WARN (≥0.7)")
    ax.set_xlim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8)
    ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.2)

    for i, (bar, f1_val) in enumerate(zip(bars, f1s)):
        ax.text(f1_val + 0.01, i, f"{f1_val:.2f}", va="center", fontsize=8,
                color="#1f2937")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gsr_robustness.png", dpi=180, facecolor="white",
                bbox_inches="tight")
    plt.close()
    print(f"\nSummary chart saved to {OUTPUT_DIR / 'gsr_robustness.png'}")

    # ── Visualization 2: per-scenario detail plots ──
    detail_results = [r for r in results if r["status"] == "OK"]
    n_scenarios = len(detail_results)
    if n_scenarios > 0:
        # 3 panels per scenario, arrange in pages of 4 scenarios
        per_page = 4
        n_pages = (n_scenarios + per_page - 1) // per_page

        for page in range(n_pages):
            start = page * per_page
            end = min(start + per_page, n_scenarios)
            page_results = detail_results[start:end]
            n_rows = len(page_results)

            fig, axes = plt.subplots(n_rows, 3, figsize=(20, 4.5 * n_rows),
                                     gridspec_kw={'width_ratios': [3, 2, 2],
                                                  'hspace': 0.45, 'wspace': 0.25})
            fig.patch.set_facecolor('white')

            if n_rows == 1:
                axes = axes.reshape(1, -1)

            for i, r in enumerate(page_results):
                plot_scenario_detail(
                    r["name"], r["_df"], r["_gt"], r["_train_len"],
                    r["_scores"], r["threshold"], r["_detected"],
                    r["_model"], r["_baseline"], axes[i])

            fig.suptitle(f"GSR Robustness — Scenario Details (Page {page+1}/{n_pages})",
                         fontsize=14, fontweight='bold', y=1.0, color='#1f2937')
            plt.savefig(OUTPUT_DIR / f"gsr_robustness_detail_{page+1}.png",
                        dpi=150, facecolor="white", bbox_inches="tight")
            plt.close()
            print(f"Detail chart saved to {OUTPUT_DIR / f'gsr_robustness_detail_{page+1}.png'}")

    print(f"\nAll charts saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
