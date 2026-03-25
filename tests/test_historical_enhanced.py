"""
Comprehensive test suite for Enhanced Historical Threshold Model.

Tests 6 synthetic scenarios + 1 real dataset (green_curve) with detailed metrics.
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── CJK font configuration (macOS) ──────────────────────────────────────────
# Try fonts in priority order; fall back to default if none are available
_CJK_FONTS = ['PingFang SC', 'Heiti TC', 'STHeiti', 'Arial Unicode MS', 'Noto Sans CJK SC']
_available = [f for f in matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
              if any(name.lower() in f.lower() for name in _CJK_FONTS)]
if _available:
    matplotlib.rcParams['font.sans-serif'] = _CJK_FONTS + ['DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

from models.historical import HistoricalThresholdModel
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


class ScenarioGenerator:
    """Generate synthetic time series scenarios for testing."""
    
    @staticmethod
    def s1_stationary_noise_burst(n_train=1000, n_test=500, seed=42):
        """S1: Stationary noise + sudden burst anomalies."""
        np.random.seed(seed)
        
        # Training: stable baseline with noise
        train = np.random.normal(50, 5, n_train)
        
        # Test: similar baseline with sudden bursts
        test = np.random.normal(50, 5, n_test)
        labels = np.zeros(n_test, dtype=int)
        
        # Inject 3 burst anomalies
        burst_indices = [100, 250, 400]
        for idx in burst_indices:
            test[idx:idx+10] += np.random.uniform(30, 50, 10)
            labels[idx:idx+10] = 1
            
        return train, test, labels
    
    @staticmethod
    def s2_stable_baseline_spikes(n_train=1000, n_test=500, seed=43):
        """S2: Stable baseline + occasional spikes (core use case)."""
        np.random.seed(seed)
        
        # Training: stable baseline around 50
        train = 50 + np.random.normal(0, 3, n_train)
        
        # Test: same stable baseline with spike anomalies
        test = 50 + np.random.normal(0, 3, n_test)
        labels = np.zeros(n_test, dtype=int)
        
        # Spike anomalies that clearly exceed historical max
        test[150:160] += 25
        labels[150:160] = 1
        
        test[300:315] += 35
        labels[300:315] = 1
        
        test[420:430] += 30
        labels[420:430] = 1
        
        return train, test, labels
    
    @staticmethod
    def s3_periodic_with_spikes(n_train=1000, n_test=500, seed=44):
        """S3: Periodic pattern + occasional training spikes (test smoothing)."""
        np.random.seed(seed)
        
        # Training: periodic with random spikes (to test smoothing)
        t_train = np.arange(n_train)
        train = 40 + 10 * np.sin(2 * np.pi * t_train / 50) + np.random.normal(0, 2, n_train)
        
        # Add fewer, smaller training spikes (more realistic)
        spike_idx = np.random.choice(n_train, 8, replace=False)
        train[spike_idx] += np.random.uniform(15, 25, 8)
        
        # Test: similar periodic with true anomalies
        t_test = np.arange(n_test)
        test = 40 + 10 * np.sin(2 * np.pi * t_test / 50) + np.random.normal(0, 2, n_test)
        labels = np.zeros(n_test, dtype=int)
        
        # True anomalies (much higher than training peaks + noise spikes)
        test[200:220] += 40
        labels[200:220] = 1
        
        test[350:365] += 50
        labels[350:365] = 1
        
        return train, test, labels
    
    @staticmethod
    def s4_multifeature_volatility(n_train=1000, n_test=500, seed=45):
        """S4: Multi-feature with different volatilities."""
        np.random.seed(seed)
        
        # Feature 1: Low volatility
        f1_train = np.random.normal(50, 2, n_train)
        f1_test = np.random.normal(50, 2, n_test)
        
        # Feature 2: High volatility
        f2_train = np.random.normal(30, 10, n_train)
        f2_test = np.random.normal(30, 10, n_test)
        
        train = np.column_stack([f1_train, f2_train])
        test = np.column_stack([f1_test, f2_test])
        labels = np.zeros(n_test, dtype=int)
        
        # Anomaly: F1 slightly exceeds (should be caught - low volatility)
        f1_test[100:120] += 15
        labels[100:120] = 1
        
        # Anomaly: F2 significantly exceeds (should be caught despite high volatility)
        f2_test[300:320] += 50
        labels[300:320] = 1
        
        test = np.column_stack([f1_test, f2_test])
        
        return train, test, labels
    
    @staticmethod
    def s5_narrow_range_sensitive(n_train=1000, n_test=500, seed=46):
        """S5: Narrow range baseline (test sensitivity)."""
        np.random.seed(seed)
        
        # Training: very stable, narrow range
        train = 100 + np.random.normal(0, 1, n_train)
        
        # Test: same narrow range with small anomalies
        test = 100 + np.random.normal(0, 1, n_test)
        labels = np.zeros(n_test, dtype=int)
        
        # Small anomalies (but clear relative to narrow baseline)
        test[180:200] += 8
        labels[180:200] = 1
        
        test[350:370] += 12
        labels[350:370] = 1
        
        return train, test, labels
    
    @staticmethod
    def s6_low_snr(n_train=1000, n_test=500, seed=47):
        """S6: Low signal-to-noise ratio scenario."""
        np.random.seed(seed)
        
        # Training: high noise, low signal
        train = 100 + np.random.normal(0, 20, n_train)
        
        # Test: similar noise + anomalies
        test = 100 + np.random.normal(0, 20, n_test)
        labels = np.zeros(n_test, dtype=int)
        
        # Anomalies need to be strong to be detectable
        test[150:170] += 80
        labels[150:170] = 1
        
        test[350:370] += 100
        labels[350:370] = 1
        
        return train, test, labels

    @staticmethod
    def s7_training_contamination(n_train=1000, n_test=500,
                                   contamination_levels=None, seed=48,
                                   n_seeds=30):
        """S7: Progressive training contamination analysis (Monte Carlo).

        For each contamination level, n_seeds different burst placements are
        generated so callers can average metrics and compute std-dev bands.

        Returns:
            contamination_levels (list[int])         — burst counts tested
            test (np.ndarray)                         — fixed test set
            labels (np.ndarray)                       — fixed ground truth
            trains (dict[int, list[np.ndarray]])      — n_seeds train arrays
                                                        per contamination level
        """
        if contamination_levels is None:
            contamination_levels = list(range(0, 16)) + [20, 30, 50]

        rng_base = np.random.RandomState(seed)

        # Fixed test set (same for every run)
        test   = 50 + rng_base.normal(0, 3, n_test)
        labels = np.zeros(n_test, dtype=int)
        test[150:160] += 30;  labels[150:160] = 1
        test[350:360] += 35;  labels[350:360] = 1

        # Fixed clean base training data
        base_train = 50 + rng_base.normal(0, 3, n_train)

        trains: dict = {}
        for n_bursts in contamination_levels:
            variants = []
            for s in range(n_seeds):
                train = base_train.copy()
                if n_bursts > 0:
                    rng_b = np.random.RandomState(seed * 1000 + n_bursts * 100 + s)
                    burst_idx = rng_b.choice(n_train, n_bursts, replace=False)
                    train[burst_idx] += rng_b.uniform(25, 40, n_bursts)
                variants.append(train)
            trains[n_bursts] = variants

        return contamination_levels, test, labels, trains


def evaluate_scenario(name, train_data, test_data, true_labels, config=None, verbose=False):
    """
    Evaluate model on a single scenario.
    
    Returns:
        dict with metrics and predictions
    """
    # Create model with OPTIMIZED parameters (FINAL - Iteration 2 Best)
    if config is None:
        config = {
            'historical_smoothing_enabled': True,
            'historical_threshold_strategy': 'fusion',
            'historical_tiered_scoring': True,
            'historical_smoothing_method': 'median',
            'historical_smoothing_window_min': 3,
            'historical_smoothing_window_max': 9,
            'historical_robust_max_percentile': 95.0,  # Balanced spike filtering
            'historical_stationary_margin': 1.10,      # Balanced threshold margin
            'historical_mad_multiplier': 3.5,
            'historical_trending_k_base': 1.2,
            'historical_trending_k_max': 4.0,
            'historical_ac1_low_threshold': 0.2,
            'historical_ac1_high_threshold': 0.6,
            'historical_score_tier1': 0.1,
            'historical_score_tier2': 0.3,
            'historical_score_tier1_weight': 0.5,
            'historical_score_tier3_weight': 1.5
        }
    
    model = HistoricalThresholdModel(name=f"Test_{name}", config=config)
    
    # Convert to DataFrame
    if train_data.ndim == 1:
        train_df = pd.DataFrame({'value': train_data})
        test_df = pd.DataFrame({'value': test_data})
    else:
        train_df = pd.DataFrame(train_data, columns=[f'f{i}' for i in range(train_data.shape[1])])
        test_df = pd.DataFrame(test_data, columns=[f'f{i}' for i in range(test_data.shape[1])])
    
    # Train
    result = model.fit(train_df)
    if result.is_err():
        return {'error': str(result.err_value)}
    
    # Predict
    result = model.predict(test_df)
    if result.is_err():
        return {'error': str(result.err_value)}
    
    scores = result.unwrap()

    # Use auto-tuned detection threshold from training distribution (like GSR's auto_threshold)
    detection_threshold = model.auto_detection_threshold
    predictions = (scores > detection_threshold).astype(int)
    
    # Metrics
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    
    cm = confusion_matrix(true_labels, predictions)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    return {
        'name': name,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        'scores': scores,
        'predictions': predictions,
        'train_data': train_data if isinstance(train_data, np.ndarray) else train_data.values,
        'test_data': test_data if isinstance(test_data, np.ndarray) else test_data.values,
        'true_labels': true_labels,
        'model': model,
        'auto_detection_threshold': detection_threshold,
    }


def plot_scenario_results(results_list, save_dir='test_results'):
    """Generate visualization plots for all scenarios."""
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    for res in results_list:
        if 'error' in res:
            continue
            
        name = res['name']
        train_data = res['train_data']
        test_data = res['test_data']
        true_labels = res['true_labels']
        predictions = res['predictions']
        scores = res['scores']
        model = res['model']
        
        # Handle single vs multi-feature
        if train_data.ndim == 1:
            n_features = 1
            train_data = train_data.reshape(-1, 1)
            test_data = test_data.reshape(-1, 1)
        else:
            n_features = train_data.shape[1]
        
        n_rows = n_features + 1
        height_ratios = [3] * n_features + [2]
        fig, axes = plt.subplots(
            n_rows, 1,
            figsize=(15, 3.5 * n_features + 3),
            sharex=True,
            gridspec_kw={'height_ratios': height_ratios, 'hspace': 0.08}
        )
        if n_rows == 2:
            axes = list(axes)

        train_time = np.arange(len(train_data))
        test_time  = np.arange(len(train_data), len(train_data) + len(test_data))
        split_x    = len(train_data)          # x coordinate of train/test boundary

        precision  = res.get('precision', 0)
        recall     = res.get('recall', 0)
        f1         = res.get('f1', 0)
        det_thresh = model.auto_detection_threshold

        true_anom_idx = np.where(true_labels == 1)[0]
        pred_anom_idx = np.where(predictions == 1)[0]

        # Plot each feature signal
        for i in range(n_features):
            ax = axes[i]
            ax.set_facecolor('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Train / test shading
            ax.axvspan(train_time[0], split_x, color='#d1fae5', alpha=0.35, label='Train region')
            ax.axvline(split_x, color='#6b7280', lw=1.2, linestyle='--')

            ax.plot(train_time, train_data[:, i], color='#22c55e', lw=0.9, alpha=0.8, label='Train (Normal)')
            ax.plot(test_time,  test_data[:, i],  color='#2563eb', lw=0.9, alpha=0.8, label='Test')

            feature_name = f'f{i}' if n_features > 1 else 'value'
            if feature_name in model.thresholds:
                threshold = model.thresholds[feature_name]
                ax.axhline(y=threshold, color='#ef4444', linestyle='--', linewidth=1.8,
                           label=f'Threshold ({threshold:.2f})')
                if feature_name in model.feature_stats:
                    stats = model.feature_stats[feature_name]
                    ax.axhline(y=stats['median'], color='#f59e0b', linestyle=':',
                               linewidth=1.3, alpha=0.8, label=f"Median ({stats['median']:.2f})")

            # True anomaly background spans
            if len(true_anom_idx):
                for idx in true_anom_idx:
                    ax.axvspan(test_time[idx] - 0.5, test_time[idx] + 0.5,
                               color='#fca5a5', alpha=0.35)
                ax.scatter(test_time[true_anom_idx], test_data[true_anom_idx, i],
                           color='#ef4444', s=25, alpha=0.7, label='True Anomaly', marker='x', zorder=4)

            if len(pred_anom_idx):
                ax.scatter(test_time[pred_anom_idx], test_data[pred_anom_idx, i],
                           color='#8b5cf6', s=45, alpha=0.5, label='Predicted',
                           marker='o', facecolors='none', linewidths=1.5, zorder=5)

            feat_label = f'Feature {i}' if n_features > 1 else 'Signal'
            ax.set_title(f"{name}  —  {feat_label}   "
                         f"P={precision:.3f}  R={recall:.3f}  F1={f1:.3f}",
                         fontsize=10, fontweight='bold', pad=4)
            ax.set_ylabel('Value', fontsize=9)
            ax.legend(loc='upper left', fontsize=7.5, ncol=4, framealpha=0.85)
            ax.grid(True, alpha=0.25)

        # Score panel
        ax_score = axes[-1]
        ax_score.set_facecolor('white')
        ax_score.spines['top'].set_visible(False)
        ax_score.spines['right'].set_visible(False)

        ax_score.axvspan(train_time[0], split_x, color='#d1fae5', alpha=0.25)
        ax_score.axvline(split_x, color='#6b7280', lw=1.2, linestyle='--')

        # Pad scores with NaN for training region so x-axis aligns with signal panels
        all_time   = np.concatenate([train_time, test_time])
        padded_scores = np.full(len(all_time), np.nan)
        padded_scores[len(train_data):] = scores

        ax_score.fill_between(all_time, padded_scores, alpha=0.25, color='#8b5cf6')
        ax_score.plot(all_time, padded_scores, color='#8b5cf6', lw=1.0, alpha=0.9, label='Anomaly score')
        ax_score.axhline(y=det_thresh, color='#ef4444', linestyle='--', linewidth=1.8,
                         label=f'Detection threshold ({det_thresh:.2f})')

        # TP / FP / FN markers on score panel
        tp_idx = np.where((predictions == 1) & (true_labels == 1))[0]
        fp_idx = np.where((predictions == 1) & (true_labels == 0))[0]
        fn_idx = np.where((predictions == 0) & (true_labels == 1))[0]
        if len(tp_idx): ax_score.scatter(test_time[tp_idx], scores[tp_idx],
                                          color='#2563eb', s=35, zorder=5, label=f'TP ({len(tp_idx)})')
        if len(fp_idx): ax_score.scatter(test_time[fp_idx], scores[fp_idx],
                                          color='#ef4444', s=35, zorder=5, label=f'FP ({len(fp_idx)})')
        if len(fn_idx): ax_score.scatter(test_time[fn_idx], scores[fn_idx],
                                          color='#f59e0b', s=40, zorder=5, marker='v', label=f'FN ({len(fn_idx)})')

        ax_score.set_title(f"{name}  —  Anomaly Score", fontsize=10, fontweight='bold', pad=4)
        ax_score.set_xlabel('Time step', fontsize=9)
        ax_score.set_ylabel('Score (normalized)', fontsize=9)
        ax_score.legend(loc='upper left', fontsize=7.5, ncol=5, framealpha=0.85)
        ax_score.grid(True, alpha=0.25)

        fig.align_ylabels(axes)
        plt.savefig(save_path / f"{name}.png", dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"Saved plot: {save_path / f'{name}.png'}")


def test_green_curve():
    """Test on the real green_curve dataset.
    
    Data: timestamp + request_count, 1-min intervals (~22 hours).
    Ground truth: value > rolling_median(61) * 1.3 (same as test_green_curve.py)
    Training: first 300 points (~5 hours of normal baseline).
    """
    # Try both locations
    for candidate in [
        Path("test_results/green_curve_extracted.csv"),
        Path("green_curve_extracted.csv"),
    ]:
        if candidate.exists():
            green_curve_path = candidate
            break
    else:
        print("⚠️  Green curve dataset not found")
        return None

    df = pd.read_csv(green_curve_path)
    metric = df['request_count'].values.astype(float)

    # Derive ground truth: spike = value > rolling_median * 1.3
    baseline = pd.Series(metric).rolling(window=61, center=True, min_periods=1).median().values
    labels_all = (metric > baseline * 1.3).astype(int)

    # Ratio preprocessing: normalize by rolling median → normal ≈ 1.0, spike >> 1.0
    # This removes baseline drift so the historical threshold sees stationary data.
    # Rolling median window 61 (same as test_green_curve.py) computed on full series.
    ratio = metric / (baseline + 1e-8)

    # Split: 300 train, rest test (matching test_green_curve.py convention)
    train_len = 300
    train_data = ratio[:train_len].reshape(-1, 1)
    test_data  = ratio[train_len:].reshape(-1, 1)
    true_labels = labels_all[train_len:]

    print(f"   Points: total={len(metric)}, train={train_len}, test={len(test_data)}")
    print(f"   Ratio range: train=[{ratio[:train_len].min():.3f}, {ratio[:train_len].max():.3f}]")
    print(f"   Anomalies in test: {true_labels.sum()} ({true_labels.mean()*100:.1f}%)")

    # For ratio-preprocessed data: training ratio is in ~[0.94, 1.05]
    # Ground truth: spike = value > baseline * 1.3, so ratio > 1.3
    # With MAD≈0.025: median + 12×MAD = 1.0 + 0.3 = 1.3, matching GT definition
    # stationary_margin also provides a ceiling: p99(1.044)×1.28 ≈ 1.336
    config = {
        'historical_smoothing_method': 'median',
        'historical_smoothing_window_min': 3,
        'historical_smoothing_window_max': 9,
        'historical_robust_max_percentile': 99.0,
        'historical_stationary_margin': 1.28,
        'historical_mad_multiplier': 12.0,  # 1.0 + 12×0.025 = 1.30 ≈ GT threshold
        'historical_trending_k_base': 1.2,
        'historical_trending_k_max': 4.0,
        'historical_ac1_low_threshold': 0.2,
        'historical_ac1_high_threshold': 0.6,
        'historical_score_tier1': 0.05,
        'historical_score_tier2': 0.15,
        'historical_score_tier1_weight': 0.3,
        'historical_score_tier3_weight': 2.0,
    }

    result = evaluate_scenario('GreenCurve', train_data, test_data, true_labels, config)
    if result and 'error' not in result:
        result['baseline'] = baseline[train_len:]
        result['metric'] = metric[train_len:]
    return result


def test_training_contamination(save_dir='test_results/historical_enhanced'):
    """S7: Run progressive training contamination analysis and generate plot."""
    print(f"\n{'─' * 80}")
    print("Testing: S7_Training_Contamination")
    print(f"{'─' * 80}")

    gen = ScenarioGenerator()
    n_seeds = 30
    levels, test, labels, trains = gen.s7_training_contamination(n_seeds=n_seeds)

    n_train = len(trains[levels[0]][0])
    print(f"   Train size={n_train}, Test size={len(test)}, Seeds per level={n_seeds}")
    print(f"   Contamination levels: {levels}")
    print(f"   True anomalies in test: {labels.sum()} ({labels.mean()*100:.1f}%)")

    # ── Evaluate: for each level, run all seeds and aggregate ───────────────
    rows = []
    for n_bursts in levels:
        contamination_pct = n_bursts / n_train * 100
        seed_f1s, seed_ps, seed_rs, seed_fdts = [], [], [], []
        first_result = None  # keep one full result for example panels

        for train in trains[n_bursts]:
            result = evaluate_scenario(
                f'Contamination_{n_bursts}', train, test, labels
            )
            if 'error' in result:
                continue
            model = result['model']
            fdt = list(model.feature_detection_thresholds.values())[0] \
                  if model.feature_detection_thresholds else float('nan')
            seed_f1s.append(result['f1'])
            seed_ps.append(result['precision'])
            seed_rs.append(result['recall'])
            seed_fdts.append(fdt)
            if first_result is None:
                first_result = result
                first_model  = model

        if not seed_f1s or first_result is None:
            print(f"   ERROR at {n_bursts} bursts: all seeds failed")
            rows.append({'n_bursts': n_bursts, 'contamination_pct': contamination_pct,
                         'p': 0, 'p_std': 0, 'r': 0, 'r_std': 0,
                         'f1': 0, 'f1_std': 0,
                         'feat_det_thresh': float('nan'), 'feat_det_std': float('nan'),
                         'threshold': float('nan'),
                         'tp': 0, 'fp': 0, 'fn': len(labels),
                         'scores': np.zeros(len(test)),
                         'predictions': np.zeros(len(test), dtype=int),
                         'train': trains[n_bursts][0]})
            continue

        f1_mean  = float(np.mean(seed_f1s));   f1_std  = float(np.std(seed_f1s))
        p_mean   = float(np.mean(seed_ps));    p_std   = float(np.std(seed_ps))
        r_mean   = float(np.mean(seed_rs));    r_std   = float(np.std(seed_rs))
        fdt_mean = float(np.mean(seed_fdts));  fdt_std = float(np.std(seed_fdts))
        thr      = list(first_model.thresholds.values())[0] \
                   if first_model.thresholds else float('nan')

        print(f"   n_bursts={n_bursts:>2} ({contamination_pct:.1f}%)  "
              f"feat_det={fdt_mean:.3f}±{fdt_std:.3f}  "
              f"F1={f1_mean:.3f}±{f1_std:.3f}  "
              f"P={p_mean:.3f}  R={r_mean:.3f}")

        rows.append({
            'n_bursts': n_bursts,
            'contamination_pct': contamination_pct,
            'p': p_mean, 'p_std': p_std,
            'r': r_mean, 'r_std': r_std,
            'f1': f1_mean, 'f1_std': f1_std,
            'threshold': thr,
            'feat_det_thresh': fdt_mean,
            'feat_det_std': fdt_std,
            'det_thresh': first_model.auto_detection_threshold,
            'tp': first_result['tp'], 'fp': first_result['fp'],
            'fn': first_result['fn'],
            'scores': first_result['scores'],
            'predictions': first_result['predictions'],
            'train': trains[n_bursts][0],
            # keep full arrays for violin/box potential
            '_f1s': seed_f1s, '_fdts': seed_fdts,
        })

    # ── Visualization ───────────────────────────────────────────────────────
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    x          = np.array([r['n_bursts'] for r in rows])
    f1s        = np.array([r['f1']       for r in rows])
    f1_stds    = np.array([r.get('f1_std', 0)  for r in rows])
    precs      = np.array([r['p']        for r in rows])
    p_stds     = np.array([r.get('p_std', 0)   for r in rows])
    recs       = np.array([r['r']        for r in rows])
    r_stds     = np.array([r.get('r_std', 0)   for r in rows])
    feat_dets  = np.array([r.get('feat_det_thresh', float('nan')) for r in rows])
    fdt_stds   = np.array([r.get('feat_det_std',    0)            for r in rows])
    n_train    = len(trains[levels[0]][0])

    C_F1  = '#2563eb'; C_P = '#16a34a'; C_R = '#d97706'
    C_THR = '#ef4444'; C_FD = '#8b5cf6'; C_GRID = '#e5e7eb'
    C_SAFE = '#dcfce7'; C_FAIL = '#fee2e2'

    # ── Determine key transition points on MEAN curve ────────────────────
    last_safe_row  = next((r for r in reversed(rows) if r['f1'] >= 0.8), rows[0])
    first_fail_row = next((r for r in rows if r['f1'] < 0.8), rows[-1])
    crit_fdt = (last_safe_row.get('feat_det_thresh', float('nan')) +
                first_fail_row.get('feat_det_thresh', float('nan'))) / 2

    # 4 example levels: clean / last_safe / first_fail / worst
    example_nbs_ordered = [
        rows[0]['n_bursts'],
        last_safe_row['n_bursts'],
        first_fail_row['n_bursts'],
        rows[-1]['n_bursts'],
    ]
    seen_ex = set()
    example_nbs = [nb for nb in example_nbs_ordered
                   if nb not in seen_ex and not seen_ex.add(nb)]
    ex_rows = {r['n_bursts']: r for r in rows if r['n_bursts'] in example_nbs}
    n_ex = len(example_nbs)

    # ── Figure layout ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(max(18, 5.5 * n_ex), 16), facecolor='white')
    gs  = fig.add_gridspec(
        4, n_ex,
        height_ratios=[1.8, 1.4, 2.0, 1.5],
        hspace=0.50, wspace=0.30,
        left=0.07, right=0.97, top=0.93, bottom=0.05,
    )
    fig.suptitle(
        f'S7 – 训练集污染临界点分析  (Monte Carlo N={n_seeds} 次/等级)\n'
        'Critical Contamination Threshold: when do training bursts break detection?',
        fontsize=13, fontweight='bold', color='#1f2937',
    )

    # ── Row 0: Mean ± 1σ F1/P/R curves (full width) ──────────────────────
    ax_full = fig.add_subplot(gs[0, :])
    ax_full.set_facecolor('white')
    ax_full.spines['top'].set_visible(False); ax_full.spines['right'].set_visible(False)
    ax_full.grid(True, alpha=0.3, color=C_GRID)

    ls_nb = last_safe_row['n_bursts']
    ff_nb = first_fail_row['n_bursts']
    ax_full.axvspan(x[0] - 0.5, ls_nb + 0.5,  color=C_SAFE,    alpha=0.5,  label='安全区')
    ax_full.axvspan(ls_nb + 0.5, ff_nb + 0.5,  color='#fef9c3', alpha=0.8,  label='临界区')
    ax_full.axvspan(ff_nb + 0.5, x[-1] + 0.5,  color=C_FAIL,   alpha=0.35, label='失效区')

    def _plot_band(ax, xv, mean, std, color, marker, ls, label):
        ax.fill_between(xv, np.clip(mean - std, 0, 1), np.clip(mean + std, 0, 1),
                        color=color, alpha=0.18)
        ax.plot(xv, mean, marker + ls, color=color, lw=2.2, ms=7, zorder=4, label=label)

    _plot_band(ax_full, x, f1s,  f1_stds, C_F1, 'o', '-',  f'F1 均值 ± 1σ  (N={n_seeds})')
    _plot_band(ax_full, x, precs, p_stds, C_P,  's', '--', 'Precision 均值 ± 1σ')
    _plot_band(ax_full, x, recs,  r_stds, C_R,  '^', '--', 'Recall 均值 ± 1σ')
    ax_full.axhline(0.8, color='#6b7280', lw=1.2, linestyle=':', label='PASS 线 (0.8)')
    ax_full.set_ylim(-0.08, 1.22)

    ax_full.axvline(ls_nb, color='#16a34a', lw=2, linestyle='--', alpha=0.8)
    ls_pct = ls_nb / n_train * 100
    ax_full.text(ls_nb, 1.13,
                 f'最后安全点\n{ls_nb} bursts ({ls_pct:.1f}%)\nF1={last_safe_row["f1"]:.3f}',
                 ha='center', va='top', fontsize=8.5, color='#166534',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor=C_SAFE, alpha=0.9))

    ax_full.axvline(ff_nb, color=C_THR, lw=2, linestyle='--', alpha=0.8)
    ff_pct = ff_nb / n_train * 100
    ax_full.text(ff_nb, 0.55,
                 f'首次失效点\n{ff_nb} bursts ({ff_pct:.1f}%)\nF1={first_fail_row["f1"]:.3f}',
                 ha='center', va='center', fontsize=8.5, color='#991b1b',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor=C_FAIL, alpha=0.9))

    ax_full2 = ax_full.twiny()
    ax_full2.set_xlim(ax_full.get_xlim())
    tick_x = [r['n_bursts'] for r in rows if r['n_bursts'] <= 20] + [rows[-1]['n_bursts']]
    ax_full2.set_xticks(tick_x)
    ax_full2.set_xticklabels([f"{nb/n_train*100:.1f}%" for nb in tick_x], fontsize=7.5)
    ax_full2.set_xlabel('污染率 (burst数/训练集)', fontsize=9, color='#4b5563')
    ax_full2.spines['top'].set_visible(True); ax_full2.spines['right'].set_visible(False)

    ax_full.set_xlabel('训练集中 burst 异常数量', fontsize=10)
    ax_full.set_ylabel('指标值', fontsize=10)
    ax_full.set_title(f'Monte Carlo 均值曲线 ± 1σ 置信带 — 临界点: '
                      f'{ls_nb} bursts({ls_pct:.1f}%) → {ff_nb} bursts({ff_pct:.1f}%)',
                      fontweight='bold', fontsize=11)
    ax_full.legend(fontsize=9, loc='lower left', ncol=6, framealpha=0.9)

    # ── Row 1 left: feat_det_thresh mean ± 1σ ────────────────────────────
    ax_fd = fig.add_subplot(gs[1, :n_ex // 2 + (1 if n_ex % 2 else 0)])
    ax_fd.set_facecolor('white')
    ax_fd.spines['top'].set_visible(False); ax_fd.spines['right'].set_visible(False)
    ax_fd.grid(True, alpha=0.3, color=C_GRID)

    ax_fd.axvspan(x[0] - 0.5, ls_nb + 0.5, color=C_SAFE,    alpha=0.4)
    ax_fd.axvspan(ls_nb + 0.5, ff_nb + 0.5, color='#fef9c3', alpha=0.7)
    ax_fd.axvspan(ff_nb + 0.5, x[-1] + 0.5, color=C_FAIL,   alpha=0.25)

    ax_fd.fill_between(x,
                       np.clip(feat_dets - fdt_stds, 0, None),
                       feat_dets + fdt_stds,
                       color=C_FD, alpha=0.20)
    ax_fd.plot(x, feat_dets, 's-', color=C_FD, lw=2, ms=7, zorder=4,
               label='feat_det_thresh 均值 ± 1σ')
    ax_fd.axhline(crit_fdt, color=C_THR, lw=1.5, linestyle='--',
                  label=f'临界值 ≈ {crit_fdt:.3f}')
    ax_fd.axvline(ls_nb, color='#16a34a', lw=1.8, linestyle='--', alpha=0.7)
    ax_fd.axvline(ff_nb, color=C_THR,    lw=1.8, linestyle='--', alpha=0.7)
    ax_fd.set_xlabel('训练集中 burst 异常数量', fontsize=10)
    ax_fd.set_ylabel('feat_det_thresh', fontsize=10, color=C_FD)
    ax_fd.tick_params(axis='y', labelcolor=C_FD)
    ax_fd.set_title('归一化基准 (feat_det_thresh) 演化 — 跨越临界值后检测失效',
                    fontweight='bold', fontsize=10)
    ax_fd.legend(fontsize=8.5, loc='upper left')

    # Text box: root cause explanation
    ax_fd.text(0.98, 0.45,
               '当 feat_det_thresh > 临界值\n测试异常的归一化分数 < 1.0\n→ 漏检 (FN)\n\n'
               '主阈值 (threshold) 不受影响\n因为 MAD/百分位统计鲁棒',
               transform=ax_fd.transAxes, fontsize=8, va='center', ha='right',
               color='#374151',
               bbox=dict(boxstyle='round,pad=0.35', facecolor='#ede9fe', alpha=0.92))

    # Transition summary table (right half of row 1)
    ax_tbl = fig.add_subplot(gs[1, n_ex // 2 + (1 if n_ex % 2 else 0):])
    ax_tbl.axis('off')
    tbl_rows_data = []
    for r in rows:
        nb = r['n_bursts']
        if nb > 20 and nb not in (rows[-1]['n_bursts'],):
            continue
        mark = ''
        if nb == ls_nb: mark = '← 最后安全'
        if nb == ff_nb: mark = '← 首次失效'
        tbl_rows_data.append([
            str(nb),
            f"{nb/n_train*100:.1f}%",
            f"{r.get('feat_det_thresh', float('nan')):.4f}",
            f"{r['f1']:.3f}",
            f"{r['tp']}/{r['tp']+r['fn']}",
            mark,
        ])
    col_labels = ['Bursts', '污染率', 'feat_det', 'F1', 'TP/Total', '']
    tbl = ax_tbl.table(
        cellText=tbl_rows_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.2)
    # Color rows
    for i, r_data in enumerate(tbl_rows_data):
        nb_val = int(r_data[0])
        color = '#f0fdf4' if nb_val <= ls_nb else ('#fef3c7' if nb_val == ff_nb else '#fff1f2')
        for j in range(len(col_labels)):
            tbl[i + 1, j].set_facecolor(color)
    ax_tbl.set_title('逐级检测结果汇总', fontweight='bold', fontsize=10, pad=8)

    # ── Rows 2-3: 4 example columns (signal + score) ─────────────────────
    test_t = np.arange(len(test))
    ex_labels = ['干净训练', '最后安全', '首次失效', '严重污染']

    for col_idx, nb in enumerate(example_nbs):
        if nb not in ex_rows:
            continue
        rd = ex_rows[nb]
        ax_sig = fig.add_subplot(gs[2, col_idx])
        ax_scr = fig.add_subplot(gs[3, col_idx])

        for ax in (ax_sig, ax_scr):
            ax.set_facecolor('white')
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.22, color=C_GRID)

        preds     = rd['predictions']
        scores    = rd['scores']
        tp_idx    = np.where((preds == 1) & (labels == 1))[0]
        fp_idx    = np.where((preds == 1) & (labels == 0))[0]
        fn_idx    = np.where((preds == 0) & (labels == 1))[0]
        train_arr = rd['train']
        n_tr      = len(train_arr)
        train_t   = np.arange(n_tr)
        score_t   = test_t + n_tr

        # Signal panel
        ax_sig.axvspan(0, n_tr, color='#d1fae5', alpha=0.22)
        ax_sig.plot(train_t, train_arr, color='#22c55e', lw=0.65, alpha=0.6, label='训练')
        if nb > 0:
            rng_b = np.random.RandomState(48 + nb * 7)
            b_idx = rng_b.choice(n_tr, nb, replace=False)
            ax_sig.scatter(b_idx, train_arr[b_idx], color='#f97316', s=28, zorder=5,
                           marker='^', label=f'污染({nb})')
        ax_sig.plot(score_t, test, color='#2563eb', lw=0.7, alpha=0.85, label='测试')
        for s, e in [(150, 160), (350, 360)]:
            ax_sig.axvspan(n_tr + s, n_tr + e, color='#fca5a5', alpha=0.3)
        ax_sig.axvline(n_tr, color='#6b7280', lw=1.2, linestyle='--', alpha=0.6)
        ax_sig.axhline(rd['threshold'], color=C_THR, lw=1.3, linestyle='--',
                       label=f'thr={rd["threshold"]:.1f}')

        pct = rd.get('contamination_pct', nb / n_train * 100)
        f1v = rd['f1']
        tag = ex_labels[col_idx] if col_idx < len(ex_labels) else ''
        status_color = '#166534' if f1v >= 0.8 else ('#92400e' if f1v > 0 else '#991b1b')
        ax_sig.set_title(
            f'[{tag}]  {nb} bursts ({pct:.1f}%)\nF1={f1v:.3f}  '
            f'TP={rd["tp"]}  FP={rd["fp"]}  FN={rd["fn"]}',
            fontsize=9, fontweight='bold', color=status_color,
        )
        ax_sig.set_ylabel('数值', fontsize=8)
        ax_sig.legend(fontsize=6.5, loc='upper right', ncol=2, framealpha=0.85)

        # Score panel
        ax_scr.axvspan(0, n_tr, color='#d1fae5', alpha=0.22)
        ax_scr.fill_between(score_t, scores, alpha=0.22, color=C_FD)
        ax_scr.plot(score_t, scores, color=C_FD, lw=0.9)
        ax_scr.axhline(1.0, color=C_THR, lw=1.8, linestyle='--', label='阈值=1.0')
        ax_scr.axvline(n_tr, color='#6b7280', lw=1.2, linestyle='--', alpha=0.6)
        fdt = rd.get('feat_det_thresh', float('nan'))
        ax_scr.text(0.02, 0.93, f'feat_det={fdt:.4f}',
                    transform=ax_scr.transAxes, fontsize=7, color=C_FD,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#ede9fe', alpha=0.88))
        # Mark peak scores at anomaly locations
        peak_score = max(scores[150:160].max(), scores[350:360].max()) if len(scores) > 360 else float('nan')
        ax_scr.text(0.98, 0.93, f'异常峰值={peak_score:.3f}',
                    transform=ax_scr.transAxes, fontsize=7, color='#374151', ha='right',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#f1f5f9', alpha=0.88))
        if len(tp_idx):
            ax_scr.scatter(score_t[tp_idx], scores[tp_idx], color='#2563eb',
                           s=28, zorder=5, label=f'TP({len(tp_idx)})')
        if len(fp_idx):
            ax_scr.scatter(score_t[fp_idx], scores[fp_idx], color=C_THR,
                           s=28, zorder=5, label=f'FP({len(fp_idx)})')
        if len(fn_idx):
            ax_scr.scatter(score_t[fn_idx], scores[fn_idx], color='#f59e0b',
                           s=32, zorder=5, marker='v', label=f'FN({len(fn_idx)})')
        ax_scr.set_xlabel('时间步', fontsize=8)
        ax_scr.set_ylabel('异常分数', fontsize=8)
        ax_scr.legend(fontsize=6.5, loc='upper right', ncol=2, framealpha=0.85)
        ax_scr.set_xlim(ax_sig.get_xlim())

    out_path = Path(save_dir) / 'S7_Training_Contamination.png'
    plt.savefig(str(out_path), dpi=130, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   Saved plot: {out_path}")

    # Return summary result for the summary table
    clean_row = rows[0] if rows else {}
    return {
        'name': 'S7_Training_Contamination',
        'precision': clean_row.get('p', 0),
        'recall':    clean_row.get('r', 0),
        'f1':        clean_row.get('f1', 0),
        'tp': clean_row.get('tp', 0), 'fp': clean_row.get('fp', 0),
        'tn': 0, 'fn': clean_row.get('fn', 0),
        'rows': rows,
    }


def main():
    """Run all tests and generate report."""
    print("=" * 80)
    print("ENHANCED HISTORICAL THRESHOLD MODEL - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    
    gen = ScenarioGenerator()
    
    scenarios = [
        ("S1_Stationary_Noise_Burst", gen.s1_stationary_noise_burst),
        ("S2_Stable_Baseline_Spikes", gen.s2_stable_baseline_spikes),
        ("S3_Periodic_Spikes", gen.s3_periodic_with_spikes),
        ("S4_MultiFeature_Volatility", gen.s4_multifeature_volatility),
        ("S5_Narrow_Range_Sensitive", gen.s5_narrow_range_sensitive),
        ("S6_Low_SNR", gen.s6_low_snr),
    ]
    
    results_list = []
    
    # Run synthetic scenarios
    for name, generator in scenarios:
        print(f"\n{'─' * 80}")
        print(f"Testing: {name}")
        print(f"{'─' * 80}")
        
        train, test, labels = generator()
        result = evaluate_scenario(name, train, test, labels)
        
        if 'error' in result:
            print(f"❌ ERROR: {result['error']}")
        else:
            print(f"✅ Precision: {result['precision']:.3f}")
            print(f"✅ Recall:    {result['recall']:.3f}")
            print(f"✅ F1-Score:  {result['f1']:.3f}")
            print(f"   TP={result['tp']}, FP={result['fp']}, TN={result['tn']}, FN={result['fn']}")
        
        results_list.append(result)
    
    # Test on green curve
    print(f"\n{'─' * 80}")
    print(f"Testing: Green Curve Dataset")
    print(f"{'─' * 80}")
    
    gc_result = test_green_curve()
    if gc_result:
        if 'error' in gc_result:
            print(f"❌ ERROR: {gc_result['error']}")
        else:
            print(f"✅ Precision: {gc_result['precision']:.3f}")
            print(f"✅ Recall:    {gc_result['recall']:.3f}")
            print(f"✅ F1-Score:  {gc_result['f1']:.3f}")
            print(f"   TP={gc_result['tp']}, FP={gc_result['fp']}, TN={gc_result['tn']}, FN={gc_result['fn']}")
        results_list.append(gc_result)

    # S7: Training contamination analysis (has its own plot; exclude from standard plot)
    s7_result = test_training_contamination()

    # Generate plots (only standard results that have train_data)
    print(f"\n{'═' * 80}")
    print("Generating visualization plots...")
    print(f"{'═' * 80}")

    plot_scenario_results(results_list, save_dir='test_results/historical_enhanced')

    # Append S7 to results_list AFTER plot_scenario_results
    if s7_result:
        results_list.append(s7_result)
    
    # Summary table
    print(f"\n{'═' * 80}")
    print("SUMMARY TABLE")
    print(f"{'═' * 80}")
    print(f"{'Scenario':<35} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Status':>10}")
    print(f"{'-' * 80}")
    
    for res in results_list:
        if 'error' in res:
            print(f"{res.get('name', 'Unknown'):<35} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'FAIL':>10}")
        else:
            status = 'PASS' if res['f1'] >= 0.8 else 'WARN' if res['f1'] >= 0.6 else 'FAIL'
            print(f"{res['name']:<35} {res['precision']:>10.3f} {res['recall']:>10.3f} {res['f1']:>10.3f} {status:>10}")
    
    print(f"{'═' * 80}\n")
    
    # Overall pass rate
    pass_count = sum(1 for r in results_list if 'error' not in r and r['f1'] >= 0.8)
    total_count = len(results_list)
    print(f"Overall Pass Rate: {pass_count}/{total_count} ({pass_count/total_count*100:.1f}%)")
    print(f"\nPlots saved to: test_results/historical_enhanced/")


if __name__ == "__main__":
    main()
