import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import logging
import sys
import os

# Ensure we can import from local modules
sys.path.append(os.getcwd())

from models.gsr import GSR
from utils.errors import Result

# Setup
np.random.seed(42)
torch.manual_seed(42)
OUTPUT_DIR = Path("simulation_results_2")
OUTPUT_DIR.mkdir(exist_ok=True)
logging.basicConfig(level=logging.ERROR) # Reduce log noise

def get_gsr_scores(df, train_len, config_overrides=None):
    """
    Use GSR (Global Spectral Residual) to score the series.
    """
    print(f"Training/Scoring with GSR (Train len: {train_len})...")
    input_dim = len(df.columns) - 1 # exclude timestamp
    
    config = {
        "window_size": 64, 
        "batch_size": 256,
        "gsr_judgement_window_size": 3, 
        "gsr_use_local_avg": True,
        "gsr_preprocessing": "none",
        "gsr_z_threshold": 3.0
    }
    
    if config_overrides:
        config.update(config_overrides)
    
    model = GSR("sim_gsr", config, input_dim)
    
    # Fit on the training (normal) portion
    train_data = df.head(train_len)
    model.fit(train_data)
    
    # Predict on whole series
    res = model.predict(df)
    if res.is_ok():
        scores = res.unwrap()
        # Pad the scores to match input length (window_size - 1)
        pad_width = len(df) - len(scores)
        if pad_width > 0:
            scores = np.pad(scores, (pad_width, 0), mode='constant', constant_values=0)
        return scores
    else:
        print(f"GSR Error: {res.err_value}")
        return np.zeros(len(df))

def save_and_plot(name, df, metrics, title, failure_mask, scores):
    # Save CSV (add scores and failure label)
    df_save = df.clone()
    df_save = df_save.with_columns([
        pl.Series("gsr_score", scores),
        pl.Series("is_failure", failure_mask.astype(int))
    ])
    df_save.write_csv(OUTPUT_DIR / f"{name}.csv")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    t = df['timestamp'].to_numpy()
    
    # Top Plot: Metric + Failures
    primary_metric = metrics[0]
    y_values = df[primary_metric].to_numpy()
    
    ax1.plot(t, y_values, label='Metric Value', color='#1f77b4', linewidth=1.5, alpha=0.9)
    
    # Highlight failure regions
    if np.any(failure_mask):
        # Find contiguous regions
        diff = np.diff(np.concatenate(([0], failure_mask.astype(int), [0])))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        for start, end in zip(starts, ends):
            ax1.axvspan(t[start], t[end-1], color='red', alpha=0.15, label='Anomaly Region' if start == starts[0] else "")
            # Add visual border to anomaly region
            ax1.axvline(x=t[start], color='red', linestyle=':', alpha=0.5)
            ax1.axvline(x=t[end-1], color='red', linestyle=':', alpha=0.5)
    
    ax1.set_title(title, fontsize=12, fontweight='bold')
    ax1.set_ylabel(primary_metric.replace('_', ' ').title(), fontsize=10)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Bottom Plot: GSR Score
    viz_threshold = 3.0
    
    ax2.plot(t, scores, label='GSR Anomaly Score', color='#2ca02c', linewidth=1.5)
    ax2.fill_between(t, 0, scores, where=(scores > viz_threshold), color='#d62728', alpha=0.3, interpolate=True)
    ax2.axhline(y=viz_threshold, color='gray', linestyle='--', alpha=0.5, label='Visual Threshold')
    
    ax2.set_ylabel("Anomaly Score", fontsize=10)
    ax2.set_xlabel("Time (step)", fontsize=10)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}.png", dpi=100)
    plt.close()
    print(f"Generated {name}: CSV and Plot saved to {OUTPUT_DIR}")

def simulate_hardware_failure():
    """
    Scenario 1: Hardware - "Heartbeat Loss / Component Failure"
    Real-world case: A cooling fan or disk spindle fails completely. The normal periodic signal 
    disappears entirely during the fault.
    
    Why it's hard: Mean value stays similar, no obvious spike or drop.
    Simple thresholds see nothing wrong.
    
    GSR Advantage: Detects the loss of periodic component (spectral peak disappears).
    """
    print("\n--- Simulating Hardware Failure (Subtle Glitches) ---")
    np.random.seed(42)
    length = 1000
    t = np.arange(length)
    
    # Normal: Complex periodic signal (e.g., sensor reading)
    # Fundamental frequency + harmonics to look like the image
    freq = 2 * np.pi / 30.0
    signal = 10 * np.sin(freq * t) + 3 * np.sin(2 * freq * t) + 2 * np.sin(3 * freq * t)
    
    # Add trend and noise
    trend = 0.01 * t
    noise = np.random.normal(0, 0.2, length)
    
    metric = 50 + trend + signal + noise
    failure_mask = np.zeros(length, dtype=bool)
    
    # Anomaly: Sparse, subtle glitches in the periodic signal
    # We introduce anomalies at specific points, not a continuous block
    # These represent "missed beats" or "sensor glitches"
    anomaly_indices = [520, 585, 650, 715, 780, 845]
    
    for idx in anomaly_indices:
        # Subtle glitch: 30% drop in value (Partial sensor failure/dip)
        # Instead of 0.001, we just dampen the signal significantly
        # This is harder because it might look like a normal low trough
        metric[idx] = metric[idx] * 0.7
        failure_mask[idx] = True
        
    df = pl.DataFrame({
        'timestamp': t,
        'disk_io_ops': metric,
    })
    
    # Config for Hardware: 
    # Optimized for subtle point anomaly detection (partial failure/dip)
    # The dip values (e.g. 40) overlap with normal troughs (e.g. 38), so raw value detection fails.
    # We use 'diff' preprocessing to detect the SHARP DROP (rate of change) instead.
    # Window=32 is small enough to keep the sharp edge distinct.
    scores = get_gsr_scores(df, train_len=200, config_overrides={
        "window_size": 32,  # Smaller window for local sharp features
        "gsr_preprocessing": "diff",  # Highlight sharp edges (the dip)
        "gsr_z_threshold": 1.0,  # Low threshold for edges
        "gsr_saliency_weight": 20.0,
        "gsr_energy_weight": 0.0,
        "gsr_z_mode": "absolute"
    })
    
    save_and_plot("1_hardware_failure", df, ['disk_io_ops'], 
                 "Hardware: Sparse Sensor Glitches (Hidden Anomalies)",
                 failure_mask, scores)

def simulate_compute_failure():
    """
    Scenario 2: Compute - "Mutex Locking / Shape Change"
    """
    print("\n--- Simulating Compute Failure (Mutex Locking/Shape Change) ---")
    length = 1000
    t = np.arange(length)
    
    # Normal: Smooth, organic sine-like usage (User load)
    # 3 mixed sine waves to look "organic" but smooth
    s1 = 15 * np.sin(t / 50.0)
    s2 = 5 * np.sin(t / 15.0)
    s3 = 2 * np.sin(t / 5.0)
    organic_signal = 50 + s1 + s2 + s3
    noise = np.random.normal(0, 1.5, length)
    
    metric = organic_signal + noise
    failure_mask = np.zeros(length, dtype=bool)
    
    # Anomaly: Mutex Locking (Square Wave) at 500-700
    s, e = 500, 700
    failure_mask[s:e] = True
    
    # Square wave with same amplitude range (approx 30 to 70)
    square_wave = 20 * np.sign(np.sin(t[s:e] / 20.0)) 
    
    # Add a little noise
    metric[s:e] = 50 + square_wave + np.random.normal(0, 0.5, e-s)
    
    df = pl.DataFrame({
        'timestamp': t,
        'cpu_utilization': metric,
    })
    
    # Standard config works well for shape changes
    scores = get_gsr_scores(df, train_len=300)
    
    save_and_plot("2_compute_failure", df, ['cpu_utilization'], 
                 "Compute: Mutex Locking (Organic vs Rigid Shape)",
                 failure_mask, scores)

def simulate_network_failure():
    """
    Scenario 3: Network - "Jitter Storm / Texture Change"
    """
    print("\n--- Simulating Network Failure (Jitter Storm) ---")
    length = 1000
    t = np.arange(length)
    
    # Normal: Stable baseline with occasional Gaussian noise
    base = 20.0
    metric = base + np.random.normal(0, 0.5, length)
    
    failure_mask = np.zeros(length, dtype=bool)
    
    # Anomaly: Jitter Storm at 600-800
    s, e = 600, 800
    failure_mask[s:e] = True
    
    # Increase variance and frequency of noise (High entropy)
    jitter = np.random.uniform(-3, 3, e-s) + np.random.normal(0, 1.0, e-s)
    metric[s:e] = base + jitter
    
    df = pl.DataFrame({
        'timestamp': t,
        'network_latency_ms': metric,
    })
    
    # Standard config works well for variance changes
    scores = get_gsr_scores(df, train_len=300)
    
    save_and_plot("3_network_failure", df, ['network_latency_ms'], 
                 "Network: Jitter Storm (Texture/Variance Change)",
                 failure_mask, scores)

def simulate_security_failure():
    """
    Scenario 4: Security - "Stealth Beacon / Hidden Periodicity"
    Real-world case: C2 beaconing malware sends heartbeat every N seconds.
    
    Why it's hard: Beacon amplitude is small compared to background noise.
    
    GSR Advantage: Strict periodicity creates a spectral line even with low amplitude.
    """
    print("\n--- Simulating Security Failure (Stealth Beacon) ---")
    length = 1000
    t = np.arange(length)
    
    # Normal: White noise (stationary, no trend) - cleaner baseline
    np.random.seed(42)
    noise_level = 2.0  # Lower noise for cleaner normal region
    metric = 100 + np.random.normal(0, noise_level, length)
    
    failure_mask = np.zeros(length, dtype=bool)
    
    # Anomaly: Periodic Beacon at 500-800
    s, e = 500, 800
    failure_mask[s:e] = True
    
    # Add a continuous sine wave beacon (creates stronger spectral line than impulses)
    beacon_period = 15
    beacon_freq = 2 * np.pi / beacon_period
    beacon_amp = 10.0  # Strong amplitude for clear detection
    
    t_beacon = np.arange(e - s)
    # Make it additive only (0 to beacon_amp) so it doesn't drop below baseline
    # "Breathing" pattern: sin wave normalized to [0, 1] then scaled
    beacon_signal = beacon_amp * (0.5 * (1 + np.sin(t_beacon * beacon_freq)))
    metric[s:e] += beacon_signal
    
    df = pl.DataFrame({
        'timestamp': t,
        'request_count': metric,
    })
    
    # Config for Security:
    # Use 'positive' z_mode (default) to detect beacon addition
    # Higher Z-threshold to suppress noise in 200-300 region
    scores = get_gsr_scores(df, train_len=400, config_overrides={
        "gsr_z_threshold": 2.0, # Lower threshold slightly for smoother signal
        "gsr_saliency_weight": 2.5,
        "gsr_energy_weight": 1.0,
        "gsr_z_mode": "positive"
    })
    
    save_and_plot("4_security_failure", df, ['request_count'], 
                 "Security: Stealth Beacon (Hidden Periodicity)",
                 failure_mask, scores)

if __name__ == "__main__":
    print("Generating complex simulation scenarios to demonstrate GSR advantages...")
    simulate_hardware_failure()
    simulate_compute_failure()
    simulate_network_failure()
    simulate_security_failure()
    print("\nAll simulations completed. Results in simulation_results_2/")
