import os
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from config import AppConfig, DataConfig, ModelsConfig, TrainingConfig, DetectionConfig, LoggingConfig
from core.pipeline import Pipeline
from core.reporting import ReportHandler
from utils.logger import get_logger

logger = get_logger("simulation")


def setup_environment(base_dir: Path):
    """Create necessary directories."""
    data_dir = base_dir / "data" / "source" / "1min"
    models_dir = base_dir / "models"
    results_dir = base_dir / "results"

    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    return data_dir, models_dir, results_dir


def generate_data(data_dir: Path, days: int = 7, anomaly: bool = False, start_time: int = None):
    """Generate synthetic time-series data, split into daily files."""
    logger.info(f"Generating {'anomaly' if anomaly else 'normal'} data for {days} days...")

    if start_time is None:
        start_time = int(datetime.now().timestamp() * 1000) - (days * 24 * 3600 * 1000)

    interval_ms = 60 * 1000  # 1 min
    points_per_day = 24 * 60

    # Base signal parameters
    total_points = days * points_per_day
    t_full = np.linspace(0, days * 2 * np.pi, total_points)

    for day in range(days):
        day_start_time = start_time + day * 24 * 3600 * 1000

        # Slicing time array for this day
        day_start_idx = day * points_per_day
        day_end_idx = (day + 1) * points_per_day

        t_day = t_full[day_start_idx:day_end_idx]

        timestamps = [day_start_time + i * interval_ms for i in range(points_per_day)]
        values = np.sin(t_day) + np.random.normal(0, 0.1, points_per_day)

        # Inject anomaly only in the last day if requested
        if anomaly and day == days - 1:
            # Inject spike in the last hour
            spike_start = points_per_day - 60
            values[spike_start:spike_start + 10] += 5.0  # Huge spike

        df = pl.DataFrame({"timestamp": timestamps, "value": values, "label": [0] * points_per_day})

        # Save as CSV (daily file)
        # Using a naming convention that ensures sort order: data_{timestamp}.csv
        filename = f"data_{day_start_time}.csv"
        df.write_csv(data_dir / filename)

    return None


def run_simulation():
    base_dir = Path("simulation_workspace")
    if base_dir.exists():
        shutil.rmtree(base_dir)

    data_dir, models_dir, results_dir = setup_environment(base_dir)
    summary_file = results_dir / "summary.csv"

    # 1. Configuration
    config = AppConfig(data=DataConfig(source_path=str(base_dir / "data" / "source")),
        models=ModelsConfig(save_path=str(models_dir)), training=TrainingConfig(data_window=3),
        # Shorten for simulation
        detection=DetectionConfig(summary_file=str(summary_file), top_k=3), logging=LoggingConfig())

    pipeline = Pipeline("1min", config)
    report_handler = ReportHandler(config)

    print("\n" + "=" * 50)
    print("STEP 1: INITIAL TRAINING")
    print("=" * 50)

    # Generate 5 days of normal data (more than 3 day window)
    start_time = int((datetime.now() - timedelta(days=6)).timestamp() * 1000)
    generate_data(data_dir, days=5, start_time=start_time)

    # Train
    logger.info("Triggering training...")
    res = pipeline.train()
    if res.is_ok():
        print("✅ Training Successful")
    else:
        print(f"❌ Training Failed: {res.err_value}")
        return

    print("\n" + "=" * 50)
    print("STEP 2: DETECTION (ANOMALY)")
    print("=" * 50)

    # Generate 1 hour of anomaly data (continuing from previous)
    last_ts = pipeline.loader.last_timestamp
    # Data loader might skip to latest, so let's just make sure we are ahead
    # generate_data generates 'days', let's just generate a small file
    # Mocking generation of short duration
    anomaly_start = last_ts + 60000
    timestamps = [anomaly_start + i * 60000 for i in range(60)]
    values = np.sin(np.linspace(0, 1, 60)) + np.random.normal(0, 0.1, 60)
    # Inject spike
    values[30:40] += 8.0

    df_anomaly = pl.DataFrame({"timestamp": timestamps, "value": values, "label": [0] * 60})
    df_anomaly.write_csv(data_dir / "anomaly.csv")

    logger.info("Triggering detection...")
    res = pipeline.detect()

    found_anomaly = False
    if res.is_ok():
        events = res.unwrap()
        print(f"✅ Detection completed. Found {len(events)} events.")
        if len(events) > 0:
            found_anomaly = True
            report_handler.append(events)  # Save to summary.csv
            for event in events:
                print(f"   -> Anomaly Event: {event['start_time']} - {event['end_time']}")
    else:
        print(f"❌ Detection Failed: {res.err_value}")
        return

    if not found_anomaly:
        print("⚠️ No anomaly found! Adjust simulation parameters.")
        return

    print("\n" + "=" * 50)
    print("STEP 3: USER FEEDBACK")
    print("=" * 50)

    # Simulate User Feedback
    # Read summary file, mark first event as false alarm
    # We use pandas for easy CSV manipulation here to mimic external tool/user
    print("Simulating user marking event as False Alarm...")
    df_summary = pd.read_csv(summary_file)
    print(f"Current Summary:\n{df_summary[['interval', 'start_time', 'is_false_alarm', 'processed']]}")

    # Mark as false alarm
    df_summary.loc[0, 'is_false_alarm'] = True
    df_summary.loc[0, 'processed'] = False  # Ensure it is picked up
    df_summary.to_csv(summary_file, index=False)
    print("Feedback saved.")

    print("\n" + "=" * 50)
    print("STEP 4: PROCESS FEEDBACK (INCREMENTAL TRAIN)")
    print("=" * 50)

    # Read pending
    pending_df = report_handler.read_pending_feedback()
    if pending_df is None or len(pending_df) == 0:
        print("❌ No pending feedback found.")
        return

    print(f"Found {len(pending_df)} pending feedback items.")

    for row in pending_df.iter_rows(named=True):
        print(f"Processing feedback for {row['interval']} ({row['start_time']} - {row['end_time']})...")
        res = pipeline.incremental_train(row['start_time'], row['end_time'])
        if res.is_ok():
            print("✅ Incremental Training Successful")
            report_handler.mark_processed([row['_idx']])
        else:
            print(f"❌ Incremental Training Failed: {res.err_value}")

    # Verify Threshold Increase
    new_threshold = pipeline.state_manager.threshold
    print(f"New Threshold: {new_threshold}")

    # Check details csv exists
    details_csv = results_dir / "1min_details.csv"
    if details_csv.exists():
        print(f"✅ Details CSV generated at {details_csv}")
        df_details = pl.read_csv(details_csv)
        print("Details Columns:", df_details.columns)
        print("Sample Data:")
        print(df_details.head(3))
    else:
        print("❌ Details CSV not found!")

    print("\n" + "=" * 50)
    print("SIMULATION COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    run_simulation()
