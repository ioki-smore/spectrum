
import torch
import numpy as np

def simulate_logic():
    print("Simulating GSR_AE Smearing Logic...")
    
    # Parameters
    window_size = 64
    amp_check_window = 5
    amp_threshold = 5.0
    th_lo = 100.0
    th_hi = 300.0
    suppression_factor = 0.1
    
    # Simulate a batch of windows
    # Case 1: Normal window
    # Case 2: Window containing a burst in the PAST (indices 0-10) but normal recently
    # Case 3: Window containing a burst RECENTLY (indices 50-60)
    
    batch_size = 3
    # Shape: (Batch, Window, Features=1)
    batch_x = torch.zeros(batch_size, window_size, 1)
    
    # 1. Normal (Low amplitude noise)
    batch_x[0] = torch.randn(window_size, 1) * 0.1
    
    # 2. Past Burst (Smearing Candidate)
    # Burst at start of window
    batch_x[1] = torch.randn(window_size, 1) * 0.1
    batch_x[1, 0:10, :] = 10.0 # High amplitude burst
    
    # 3. Recent Burst (True Anomaly)
    batch_x[2] = torch.randn(window_size, 1) * 0.1
    batch_x[2, -10:, :] = 10.0 # High amplitude burst
    
    # Simulate Scores (Reconstruction Error)
    # We assume reconstruction error is proportional to the input "energy" or deviation
    # Normal -> Low score
    # Burst -> High score (MSE will be large for the burst parts)
    
    # Let's say MSE is roughly sum of squares of values (simplified)
    # Normal: 0.1^2 * 64 ~= 0.64
    # Burst: 10^2 * 10 = 1000
    
    score_global = torch.tensor([10.0, 1000.0, 1000.0])
    
    print(f"Original Scores: {score_global}")
    
    # Logic from GSR_AE.predict
    
    # 1. Amplitude Check (Recent)
    check_feats = batch_x
    if amp_check_window > 0 and amp_check_window < window_size:
        check_feats = batch_x[:, -amp_check_window:, :]
        
    # Check max amplitude in checked region
    max_amp_per_feat, _ = torch.max(torch.abs(check_feats), dim=1)
    max_amp_sample, _ = torch.max(max_amp_per_feat, dim=1)
    
    print(f"Max Amp (Recent {amp_check_window}): {max_amp_sample}")
    
    is_high_amp = max_amp_sample > amp_threshold
    print(f"Is High Amp (> {amp_threshold}): {is_high_amp}")
    
    is_low_score = score_global < th_lo
    
    # Burst Detection
    burst_mask = is_low_score & is_high_amp
    if torch.any(burst_mask):
        print("Burst detected (Low Score + High Amp). Boosting score.")
        score_global[burst_mask] = th_hi * 2.0
        
    # Noise Suppression
    is_normal_amp = ~is_high_amp
    if torch.any(is_normal_amp):
        print("Normal Amp detected. Suppressing score.")
        score_global[is_normal_amp] *= suppression_factor
        
    print(f"Final Scores: {score_global}")
    print(f"Threshold Hi: {th_hi}")
    
    is_anomaly = score_global > th_hi
    print(f"Is Anomaly: {is_anomaly}")
    
    # Analysis
    # Case 2 (Past Burst):
    # - Original Score: 1000 (High)
    # - Recent Amp: Low (Normal) -> is_high_amp = False
    # - Suppression: 1000 * 0.1 = 100.
    # - 100 < 300 (th_hi). So it is NOT an anomaly.
    # Result: CLEAN.
    
    # BUT, what if the burst was BIGGER?
    # Say burst is 50.0 amplitude.
    # Score: 50^2 * 10 = 25000.
    # Suppressed: 2500.
    # 2500 > 300. ANOMALY.
    
    # So strictly applying a factor might not be enough if the anomaly is huge.
    
    print("\n--- TEST WITH HUGE BURST ---")
    score_huge = torch.tensor([25000.0])
    is_high_amp_huge = torch.tensor([False]) # Past burst
    
    if ~is_high_amp_huge:
        score_huge *= suppression_factor
        
    print(f"Huge Score: 25000 -> {score_huge.item()}")
    print(f"Is Anomaly: {score_huge.item() > th_hi}")

if __name__ == "__main__":
    simulate_logic()
