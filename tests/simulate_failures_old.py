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

# Configuration: Set to True to load existing CSV data instead of generating new data
USE_EXISTING_DATA = False

def generate_realistic_series(length, base_val=50, noise_level=2, trend_strength=0.05):
    """
    Generates a realistic time series with:
    - Composite seasonality (slow daily-like + fast hourly-like cycles)
    - Random drift/walk (non-stationary trend)
    - Base Gaussian noise + Occasional spikes (heavy tails)
    """
    t = np.arange(length)
    
    # 1. Composite Seasonality
    # Main cycle (e.g., daily)
    s1 = 5 * np.sin(t / 50.0) 
    # Fast cycle (e.g., hourly load fluctuations)
    s2 = 2 * np.sin(t / 15.0 + 0.5)
    seasonality = s1 + s2
    
    # 2. Random Trend (Random Walk)
    # Accumulate small random steps to create a wandering baseline
    drift = np.cumsum(np.random.normal(0, trend_strength, length))
    # Center the drift so it doesn't wander too far off base_val immediately
    drift = drift - drift.mean()
    
    # 3. Noise & Spikes
    noise = np.random.normal(0, noise_level, length)
    
    # Add occasional random spikes (1% chance)
    num_spikes = int(0.01 * length)
    spike_indices = np.random.choice(length, num_spikes, replace=False)
    # Spikes are exponential magnitude
    spikes = np.zeros(length)
    spikes[spike_indices] = np.random.exponential(noise_level * 3, num_spikes) * np.random.choice([-1, 1], num_spikes)
    
    return base_val + seasonality + drift + noise + spikes

def get_gsr_scores(df, train_len):
    """
    Use GSR (Global Spectral Residual) to score the series.
    """
    print("Training/Scoring with GSR...")
    # Config for the model
    input_dim = len(df.columns) - 1 # exclude timestamp
    # 4: 32， 1
    # 3: 16, 1
    # 2: 8, 1
    # 1: 4, 1
    config = {
        "window_size": 32, # Optimal balance for periodic patterns and burst detection
        "batch_size": 256,
        "gsr_judgement_window_size": 1,
        "gsr_use_local_avg": False, # Use global mean for stable periodic baseline
    }
    
    model = GSR("sim_gsr", config, input_dim)
    
    # Fit is no-op for GSR but good practice
    train_data = df.head(train_len)
    model.fit(train_data)
    
    # Predict
    res = model.predict(df)
    if res.is_ok():
        scores = res.unwrap()
        # Pad the scores to match input length (window_size - 1)
        # TimeSeriesDataset behavior: N samples -> N - W + 1 results
        pad_width = len(df) - len(scores)
        if pad_width > 0:
            scores = np.pad(scores, (pad_width, 0), mode='constant', constant_values=0)
        return scores
    else:
        print(f"GSR Error: {res.err_value}")
        return np.zeros(len(df))

def save_and_plot(name, df, metrics, title, failure_mask, scores):
    # Save CSV (add scores)
    df_save = df.clone()
    df_save = df_save.with_columns(pl.Series("gsr_score", scores))
    df_save.write_csv(OUTPUT_DIR / f"{name}.csv")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    t = df['timestamp'].to_numpy()
    
    # Top Plot: Metric + Failures
    # We only plot the first metric in the list as the primary one
    primary_metric = metrics[0]
    y_values = df[primary_metric].to_numpy()
    
    ax1.plot(t, y_values, label='Original value', color='blue', linewidth=1)
    
    # Mark failures
    if np.any(failure_mask):
        ax1.scatter(t[failure_mask], y_values[failure_mask], color='red', s=10, zorder=5, label='Failure event')
    
    ax1.set_title(title)
    ax1.set_ylabel(primary_metric)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Bottom Plot: GSR Score
    # Clip scores for visualization to distinct normal vs anomaly
    # Z-scores can be large, but we clip to keep plot readable.
    scores_plot = np.clip(scores, -5, 50) 
    ax2.plot(t, scores_plot, label='GSR Score (clipped -5 to 50)', color='salmon', linewidth=1)
    ax2.set_ylabel("Anomaly Score (Z-Score)")
    ax2.set_xlabel("Time (s)")
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}.png")
    plt.close()
    print(f"Generated {name}: CSV and Plot saved to {OUTPUT_DIR}")

def simulate_hardware_failure():
    """
    (1) Hardware failures: Disk I/O with periodic patterns + burst stalls.
    Realistic scenario: Database server disk showing daily/hourly access patterns
    with occasional disk stalls (bad sectors/mechanical issues).
    
    Key features:
    - Normal: Moderate periodic pattern (daily + hourly cycles) - visible but not overwhelming
    - Anomaly: Strong burst faults (disk stalls) at specific points
    - Challenge: Distinguish periodic patterns from burst anomalies
    """
    csv_path = OUTPUT_DIR / "1_hardware_failure.csv"
    
    if USE_EXISTING_DATA and csv_path.exists():
        print(f"Loading existing data from {csv_path}...")
        df = pl.read_csv(csv_path)
        if 'gsr_score' in df.columns:
            df = df.drop('gsr_score')
        
        # Reconstruct failure mask from data characteristics
        length = len(df)
        failure_mask = np.zeros(length, dtype=bool)
        burst_events = [(250, 8), (380, 10), (510, 12), (640, 9), (770, 11), (880, 8)]
        for burst_start, burst_duration in burst_events:
            burst_end = min(burst_start + burst_duration, length)
            failure_mask[burst_start:burst_end] = True
    else:
        length = 1000
        t = np.arange(length)
        
        # === PERIODIC BASELINE (Normal Operation) ===
        # Moderate periodic patterns - visible but not overwhelming
        # Daily cycle: slower oscillation (period ~200 samples) - peak/off-peak hours
        daily_cycle = 3.0 * np.sin(2 * np.pi * t / 200.0)
        # Hourly cycle: faster oscillation (period ~50 samples) - batch jobs
        hourly_cycle = 1.5 * np.sin(2 * np.pi * t / 50.0)
        
        # Combine periodic components for realistic I/O pattern
        periodic_pattern = daily_cycle + hourly_cycle
        
        # Base latency with periodic pattern (baseline ~12ms)
        disk_latency = 12.0 + periodic_pattern
        
        # Add normal operational noise
        disk_latency += np.random.normal(0, 0.8, length)
        
        availability = np.ones(length)
        
        # === BURST ANOMALIES (Disk Stalls) ===
        # Simulate disk stalls/retries as sudden burst events
        # These represent bad sectors, mechanical delays, or retry storms
        
        burst_events = [
            (250, 8, 70),   # (start_time, duration, magnitude)
            (380, 10, 85),
            (510, 12, 90),
            (640, 9, 75),
            (770, 11, 95),
            (880, 8, 80),
        ]
        
        failure_mask = np.zeros(length, dtype=bool)
        
        for burst_start, burst_duration, burst_magnitude in burst_events:
            burst_end = min(burst_start + burst_duration, length)
            failure_mask[burst_start:burst_end] = True
            
            # Burst profile: sudden spike with exponential decay
            burst_len = burst_end - burst_start
            burst_profile = burst_magnitude * np.exp(-np.linspace(0, 2.5, burst_len))
            disk_latency[burst_start:burst_end] += burst_profile
            
            # Add high-frequency jitter during burst (mechanical instability)
            disk_latency[burst_start:burst_end] += np.random.normal(0, 10, burst_len)
        
        df = pl.DataFrame({
            'timestamp': t,
            'disk_latency_ms': disk_latency,
            'system_availability': availability * 10 
        })
    
    scores = get_gsr_scores(df, train_len=200)
    
    save_and_plot("1_hardware_failure", df, ['disk_latency_ms'], 
                 "Hardware: Disk I/O (Periodic + Burst Stalls)",
                 failure_mask, scores)

def simulate_compute_failure():
    """
    (2) Compute failures: Memory Leak with GC pauses.
    Simulation: "CrashLoopBackOff". Sawtooth pattern but with variable slope and GC drops.
    """
    csv_path = OUTPUT_DIR / "2_compute_failure.csv"
    
    if USE_EXISTING_DATA and csv_path.exists():
        print(f"Loading existing data from {csv_path}...")
        df = pl.read_csv(csv_path)
        if 'gsr_score' in df.columns:
            df = df.drop('gsr_score')
        
        # Reconstruct failure mask (memory leak periods)
        length = len(df)
        failure_mask = (df['timestamp'] >= 300) & (df['timestamp'] < 700)
        failure_mask = failure_mask.to_numpy()
    else:
        length = 1000
        t = np.arange(length)
        
        memory_usage = generate_realistic_series(length, base_val=30, noise_level=1.5, trend_strength=0.05)
        restarts = np.zeros(length)
        failure_mask = np.zeros(length, dtype=bool)
        
        start_idx = 300
        current_idx = start_idx
        
        while current_idx < length:
            # Variable duration and slope for each leak cycle
            duration = np.random.randint(60, 120)
            end_ramp = min(current_idx + duration, length)
            ramp_len = end_ramp - current_idx
            
            # Immediate spike at leak onset for early detection
            # Then continue with accelerating leak
            leak_curve = np.zeros(ramp_len)
            # Initial jump: 10-15% immediate increase (stronger signal)
            initial_spike_len = min(5, ramp_len)
            leak_curve[0:initial_spike_len] = np.linspace(10, 15, initial_spike_len)
            # Then accelerating leak from that point (steeper curve for early visibility)
            if ramp_len > initial_spike_len:
                remaining = np.power(np.linspace(0, 1, ramp_len-initial_spike_len), 1.1) * 55
                leak_curve[initial_spike_len:] = 15 + remaining
            
            memory_usage[current_idx:end_ramp] += leak_curve
            
            # Simulate GC pauses (sudden small drops during leak)
            if ramp_len > 20:
                gc_points = np.random.choice(np.arange(current_idx+10, end_ramp-5), size=2, replace=False)
                for gc_idx in gc_points:
                    # Drop memory by 10-15 temporarily
                    memory_usage[gc_idx:gc_idx+3] -= 15
            
            # Add moderate jitter to leak (preserve signal clarity)
            memory_usage[current_idx:end_ramp] += np.random.normal(0, 3, ramp_len)
            
            failure_mask[current_idx:end_ramp] = True
            
            # Crash/Restart
            if end_ramp < length:
                restarts[end_ramp] = 1 
                failure_mask[end_ramp] = True
                
                # Restart recovery
                startup_len = 15
                end_startup = min(end_ramp + startup_len, length)
                memory_usage[end_ramp:end_startup] = 20 + np.random.normal(0, 2, end_startup-end_ramp)
            
            current_idx = end_ramp + 20
        
        df = pl.DataFrame({
            'timestamp': t,
            'memory_usage_pct': memory_usage,
            'restart_events': restarts * 100 
        })
    
    scores = get_gsr_scores(df, train_len=250)
    
    save_and_plot("2_compute_failure", df, ['memory_usage_pct'], 
                 "Compute Failure: Realistic Memory Leak & CrashLoop",
                 failure_mask, scores)

def simulate_network_failure():
    """
    (3) Network failures: Latency spikes (Heavy tail) & Packet Loss.
    Simulation: Bursty congestion events rather than constant noise.
    """
    csv_path = OUTPUT_DIR / "3_network_failure.csv"
    
    if USE_EXISTING_DATA and csv_path.exists():
        print(f"Loading existing data from {csv_path}...")
        df = pl.read_csv(csv_path)
        if 'gsr_score' in df.columns:
            df = df.drop('gsr_score')
        
        # Reconstruct failure mask (congestion bursts)
        length = len(df)
        bursts = [(420, 480), (520, 580), (620, 680)]
        failure_mask = np.zeros(length, dtype=bool)
        for b_start, b_end in bursts:
            failure_mask[b_start:b_end] = True
    else:
        length = 1000
        t = np.arange(length)
        
        latency = generate_realistic_series(length, base_val=25, noise_level=3)
        # Lower noise_level to prevent extreme random spikes (e.g., t=355 spike)
        throughput = generate_realistic_series(length, base_val=500, noise_level=12, trend_strength=0.2)
        
        # Failure event
        start = 400
        end = 700
        
        # Bursty Congestion: 3 distinct bursts of bad connectivity
        bursts = [(420, 480), (520, 580), (620, 680)]
        
        failure_mask = np.zeros(length, dtype=bool)
        
        for b_start, b_end in bursts:
            b_len = b_end - b_start
            failure_mask[b_start:b_end] = True
            
            # Latency: Massive spikes + jitter
            latency[b_start:b_end] += 100 
            # Lognormal spikes (heavy tail)
            spikes = np.random.lognormal(mean=4, sigma=1, size=b_len) 
            latency[b_start:b_end] += spikes
            
            # Throughput: Drop and instability
            throughput[b_start:b_end] *= 0.3
            throughput[b_start:b_end] += np.random.normal(0, 40, b_len)
        
        df = pl.DataFrame({
            'timestamp': t,
            'latency_ms': latency,
            'throughput_qps': throughput
        })
    
    scores = get_gsr_scores(df, train_len=350)
    
    save_and_plot("3_network_failure", df, ['latency_ms'], 
                 "Network Failure: Bursty Congestion (Latency Spikes)",
                 failure_mask, scores)

def simulate_security_failure():
    """
    (4) Security: Volumetric DDoS with wave pattern.
    """
    csv_path = OUTPUT_DIR / "4_security_failure.csv"
    
    if USE_EXISTING_DATA and csv_path.exists():
        print(f"Loading existing data from {csv_path}...")
        df = pl.read_csv(csv_path)
        if 'gsr_score' in df.columns:
            df = df.drop('gsr_score')
        
        # Reconstruct failure mask (DDoS attack period)
        length = len(df)
        t = df['timestamp'].to_numpy()
        failure_mask = (t >= 500) & (t < 800)
    else:
        length = 1000
        t = np.arange(length)
        
        requests = generate_realistic_series(length, base_val=100, noise_level=15)
        cpu_load = generate_realistic_series(length, base_val=25, noise_level=3)
        
        start = 500
        end = 800
        attack_len = end - start
        
        # Attack Pattern: Fast rise, then oscillating waves
        # Sigmoid rise
        x = np.linspace(-6, 6, attack_len)
        sigmoid = 1 / (1 + np.exp(-x))
        
        # Oscillating waves (attackers pulsing traffic)
        waves = 0.2 * np.sin(np.linspace(0, 4*np.pi, attack_len)) + 1
        
        attack_profile = sigmoid * waves * 1500 # Max +1500 QPS
        
        requests[start:end] += attack_profile
        requests[start:end] += np.random.normal(0, 100, attack_len) # High noise
        
        # CPU Saturation with pegging at 100%
        cpu_load[start:end] = 95 + 5 * waves + np.random.normal(0, 5, attack_len)
        cpu_load = np.clip(cpu_load, 0, 100)
        
        df = pl.DataFrame({
            'timestamp': t,
            'http_requests_qps': requests,
            'cpu_load_pct': cpu_load
        })
        
        failure_mask = (t >= start) & (t < end)
    
    scores = get_gsr_scores(df, train_len=450)
    
    save_and_plot("4_security_failure", df, ['http_requests_qps'], 
                 "Security Failure: Volumetric DDoS (Waves)",
                 failure_mask, scores)

if __name__ == "__main__":
    if USE_EXISTING_DATA:
        print("Re-scoring existing data with current GSR parameters...")
    else:
        print("Generating new simulation data with GSR...")
    
    simulate_hardware_failure()
    simulate_compute_failure()
    simulate_network_failure()
    simulate_security_failure()
    print("Done.")
