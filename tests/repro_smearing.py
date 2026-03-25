
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from pathlib import Path
import shutil
import logging

from config import AppConfig, ModelsConfig, DataConfig
from core.pipeline import Pipeline
from utils.errors import Result

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("repro")

def generate_data(file_path: Path, n_points: int, start_ts: int, anomaly_start: int = -1, anomaly_len: int = 0):
    """
    Generate synthetic data with 5 features:
    - Normal: Sine waves + noise (Amplitudes 1-3)
    - Anomaly: Large burst in all features (Amplitude +20)
    """
    t = np.arange(n_points)
    n_features = 5
    
    data_dict = {
        "timestamp": start_ts + t * 1000
    }
    
    for i in range(n_features):
        # Different frequencies and phases
        freq = 0.1 + i * 0.05
        phase = i * np.pi / 4
        # Amplitude around 2.0
        signal = 2.0 * np.sin(t * freq + phase) + np.random.normal(0, 0.2, n_points)
        
        if anomaly_start >= 0 and anomaly_len > 0:
            end = min(anomaly_start + anomaly_len, n_points)
            # Huge burst
            signal[anomaly_start : end] += 20.0
            
        data_dict[f"feature_{i}"] = signal
        
    data_dict["label"] = [0]*n_points
    
    df = pl.DataFrame(data_dict)
    df.write_csv(file_path)
    return df

def run_repro():
    work_dir = Path("repro_workspace")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir()
    
    data_dir = work_dir / "data" / "source" / "test_interval"
    data_dir.mkdir(parents=True)
    
    models_dir = work_dir / "models"
    models_dir.mkdir()
    
    # 1. Generate Training Data (Pure Normal, 1 day = 86400 points)
    logger.info("Generating training data (1 day)...")
    base_ts = 1700000000000
    train_df = generate_data(
        data_dir / "train-test_interval.csv", 
        n_points=86400, 
        start_ts=base_ts,
        anomaly_start=-1
    )
    
    # 2. Generate Test Data (Normal -> Fault -> Normal)
    # Start after train data
    test_start_ts = train_df['timestamp'].max() + 1000
    logger.info("Generating test data...")
    
    # 1000 normal, 100 fault, 2000 normal
    # Fault from index 1000 to 1100
    test_df = generate_data(
        data_dir / "test-test_interval.csv", 
        n_points=3100, 
        start_ts=test_start_ts,
        anomaly_start=1000, 
        anomaly_len=100
    )
    
    # 3. Configure Pipeline
    config = AppConfig()
    config.data.source_path = str(work_dir / "data" / "source")
    config.models.save_path = str(models_dir)
    config.detection.summary_file = str(work_dir / "results" / "summary.csv")
    config.training.data_window = 1 
    
    # Enable LSTM and GSR_AE
    config.models.enabled_models = ["lstm", "gsr_ae"]
    config.models.window_size = 64
    config.models.batch_size = 256
    config.models.epochs = 2
    
    pipeline = Pipeline("test_interval", config)
    
    # 4. Train
    logger.info("Training pipeline...")
    res = pipeline.train(force=True)
    if res.is_err():
        logger.error(f"Training failed: {res.err_value}")
        return
    
    # Manually commit to end of training data so detect() only sees test data
    train_end_ts = train_df['timestamp'].max()
    pipeline.loader.commit(train_end_ts)
        
    # 5. Detect
    logger.info("Running detection...")
    res = pipeline.detect()
    if res.is_err():
        logger.error(f"Detection failed: {res.err_value}")
        return
        
    events = res.unwrap()
    logger.info(f"Detected {len(events)} events.")
    
    # 6. Analyze Details
    summary_path = Path(config.detection.summary_file)
    details_path = summary_path.parent / "test_interval_details.csv"
    
    if not details_path.exists():
        logger.error("Details file not found.")
        return
        
    # Read with truncation to avoid schema errors if any
    try:
        # Try reading with flexible options
        res_df = pl.read_csv(details_path, infer_schema_length=0, truncate_ragged_lines=True)
        
        # Helper to convert boolean string to bool using comparison
        def parse_bool_col(col_name):
            # Check for 'true' case-insensitive
            return pl.col(col_name).str.to_lowercase().eq("true")

        cols_to_convert = [c for c in res_df.columns if 'is_anomaly' in c]
        
        exprs = [pl.col('timestamp').cast(pl.Int64)]
        for c in cols_to_convert:
            exprs.append(parse_bool_col(c).alias(c))
            
        res_df = res_df.with_columns(exprs)
    except Exception as e:
        logger.error(f"Failed to read details CSV: {e}")
        # Try reading as plain text to debug
        with open(details_path, 'r') as f:
            lines = f.readlines()
            logger.info(f"File lines: {len(lines)}")
            logger.info(f"Head:\n{''.join(lines[:5])}")
        return
    
    # Filter for test period
    res_df = res_df.filter(pl.col('timestamp') >= test_start_ts)
    
    # Fault range (timestamps)
    fault_start_ts = test_start_ts + 1000 * 1000
    fault_end_ts = test_start_ts + 1100 * 1000
    
    # Check anomalies AFTER fault_end_ts
    # Smearing should last window_size (64 steps)
    
    window_ms = 64 * 1000
    post_fault_start = fault_end_ts
    post_fault_end = fault_end_ts + 500 * 1000 # Look at 500s after fault
    
    post_fault_df = res_df.filter(
        (pl.col('timestamp') > post_fault_start) & 
        (pl.col('timestamp') <= post_fault_end)
    )
    
    anomalies = post_fault_df.filter(pl.col('is_anomaly'))
    
    # Define "smearing region" vs "clean region"
    smear_end_ts = fault_end_ts + window_ms
    
    smear_anomalies = anomalies.filter(pl.col('timestamp') <= smear_end_ts)
    clean_anomalies = anomalies.filter(pl.col('timestamp') > smear_end_ts)
    
    print("\n--- Post-Fault Detection Analysis ---")
    print(f"Fault Interval: {fault_start_ts} - {fault_end_ts}")
    print(f"Smearing Window End: {smear_end_ts}")
    
    print(f"\nTotal Anomalies in 500s post-fault: {len(anomalies)}")
    print(f"Anomalies in Smearing Window (expected): {len(smear_anomalies)}")
    print(f"Anomalies AFTER Smearing Window (UNEXPECTED): {len(clean_anomalies)}")
    
    if len(smear_anomalies) > 0:
        print("\nSmearing Anomalies samples (Head):")
        cols = ['timestamp', 'score', 'is_anomaly']
        if 'LSTM_test_interval_score' in smear_anomalies.columns:
            cols.append('LSTM_test_interval_score')
        if 'GSR_AE_test_interval_score' in smear_anomalies.columns:
            cols.append('GSR_AE_test_interval_score')
            
        print(smear_anomalies.select(cols).head(10))
        print("\nSmearing Anomalies samples (Tail):")
        print(smear_anomalies.select(cols).tail(10))

    if len(clean_anomalies) > 0:
        print("\nUnexpected False Positives samples:")
        print(clean_anomalies.head(10))

if __name__ == "__main__":
    run_repro()
