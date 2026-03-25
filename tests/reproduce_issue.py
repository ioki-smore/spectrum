
import numpy as np
import polars as pl
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.getcwd()))

from models.gsr_ae import GSR_AE
from utils.logger import get_logger

# Mock config
class MockConfig:
    def __init__(self):
        self.params = {
            "gsr_ae_window_size": 64,
            "gsr_ae_latent_dim": 5,
            "gsr_ae_sigma": 3.0,
            "gsr_ae_lr": 1e-3,
            "gsr_ae_epochs": 5,
            "gsr_ae_amp_check_window": 5,  # The fix parameter
            "gsr_ae_suppression_factor": 0.1,
            "gsr_ae_amp_threshold": 5.0,
            "gsr_ae_th_lo_q": 0.15
        }
    
    def get(self, key, default=None):
        return self.params.get(key, default)

def run_test():
    logger = get_logger("repro_smearing")
    config = MockConfig()
    
    # 1. Create Model
    model = GSR_AE("test_gsr_ae", config, input_dim=1)
    
    # 2. Generate Training Data (Normal Sin Wave)
    t = np.linspace(0, 100 * np.pi, 2000)
    values = np.sin(t) + np.random.normal(0, 0.1, 2000)
    train_df = pl.DataFrame({"feature_0": values})
    
    # Train
    print("Training model...")
    model.fit(train_df)
    
    # 3. Generate Test Data with Burst
    # Normal -> Burst -> Normal
    t_test = np.linspace(0, 20 * np.pi, 500)
    values_test = np.sin(t_test) + np.random.normal(0, 0.1, 500)
    
    # Inject burst at index 200-210 (length 10)
    # Amplitude 10.0 (High)
    burst_start = 200
    burst_end = 210
    values_test[burst_start:burst_end] += 10.0
    
    test_df = pl.DataFrame({"feature_0": values_test})
    
    # 4. Predict
    print("Running prediction...")
    res = model.predict(test_df)
    scores = res.unwrap()
    
    # 5. Analyze "Smearing"
    # Window size 64.
    # The burst ends at 210.
    # We care about windows starting after the burst has passed the "recent" check.
    # Current time T (end of window).
    # Burst is at [200, 210].
    # If T = 220. Window is [157, 220]. Contains burst [200, 210].
    # Burst is NOT in last 5 (216-220).
    # So T=220 should be SUPPRESSED (Low Score).
    
    # Let's inspect scores around the burst
    # We expect high scores during the burst (T around 210)
    # We expect LOW scores immediately after burst exits the amp_check_window
    
    # The output scores correspond to windows.
    # predict returns one score per window.
    # The dataset indices are: 0..N-W.
    # Index i corresponds to window [i, i+W]. The "time" of the prediction is usually associated with i+W.
    
    print("\nAnalyzing scores around burst end...")
    # Map score index to end-of-window index
    # score[i] corresponds to window ending at i + window_size
    
    # Critical region: When window end passes 210.
    # Burst ends at 210.
    # At T=211. Window [148, 211]. Last 5: [207, 211]. 207-210 is high. is_high_amp=True.
    # At T=215. Window [152, 215]. Last 5: [211, 215]. Normal. is_high_amp=False.
    # So from T=216 onwards, we should see suppression.
    
    window_size = 64
    check_window = 5
    
    # Loop through window end times
    for t_end in range(burst_end - 5, burst_end + 20):
        # Calculate score index
        score_idx = t_end - window_size
        if score_idx < 0: continue
        
        score = scores[score_idx]
        
        # Check if burst is in check window
        # Check window range: [t_end - check_window, t_end]
        # Burst range: [burst_start, burst_end]
        # Overlap?
        check_start_t = t_end - check_window
        
        overlap = max(0, min(t_end, burst_end) - max(check_start_t, burst_start))
        in_check = overlap > 0
        
        status = "IN CHECK" if in_check else "CLEAN"
        print(f"T={t_end} (idx={score_idx}): Score={score:.4f} [{status}]")

if __name__ == "__main__":
    run_test()
