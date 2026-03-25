"""
Analysis utilities for clustering anomalies and building events.
"""

from typing import List, Dict, Optional

import numpy as np
import polars as pl


def cluster_anomalies(mask: np.ndarray, timestamps: np.ndarray, max_gap_ms: int = 5 * 60 * 1000) -> List[
    Dict[str, int]]:
    """
    Cluster adjacent anomaly points into discrete segments.
    
    Logic:
    - Iterates through anomaly indices.
    - If the time gap between two consecutive anomaly points is <= `max_gap_ms`, they belong to the same segment.
    - If the gap is larger, a new segment starts.
    
    Args:
        mask: Boolean array where True indicates an anomaly.
        timestamps: Array of timestamps corresponding to the mask.
        max_gap_ms: Maximum time difference allowed to group anomalies.
        
    Returns:
        List of dicts with 'start_idx' and 'end_idx' for each segment.
    """
    if not np.any(mask):
        return []

    anomaly_indices = np.where(mask)[0]
    segments = []

    if len(anomaly_indices) == 0:
        return segments

    current_start = anomaly_indices[0]
    current_end = anomaly_indices[0]

    for i in range(1, len(anomaly_indices)):
        idx = anomaly_indices[i]
        prev_idx = anomaly_indices[i - 1]

        gap = timestamps[idx] - timestamps[prev_idx]

        if gap <= max_gap_ms:
            current_end = idx
        else:
            segments.append({'start_idx': current_start, 'end_idx': current_end})
            current_start = idx
            current_end = idx

    segments.append({'start_idx': current_start, 'end_idx': current_end})

    return segments


def build_events(interval: str, df: pl.DataFrame, scores: np.ndarray, anomalies: np.ndarray, feature_cols: List[str],
                 contributions: Optional[np.ndarray], top_k: int,
                 model_names: Optional[List[str]] = None) -> List[Dict]:
    """
    Construct detection events from raw anomaly outputs.
    
    Currently implements a simple strategy:
    - Merges ALL detected anomalies in the current window into a SINGLE event.
    - The event duration spans from the first detected anomaly to the last.
    - Computes top-k contributing features based on average contribution scores within the event window.
    
    Args:
        interval: The monitoring interval name.
        df: The source DataFrame containing timestamps.
        scores: Anomaly scores aligned with df (potentially shorter if windowing applied).
        anomalies: Boolean mask of anomalies.
        feature_cols: List of feature names corresponding to contributions columns.
        contributions: Matrix of feature contributions (same length as scores).
        top_k: Number of top contributing metrics to include in the report.
        model_names: List of active sub-model names that contributed to the consensus.
        
    Returns:
        List containing a single event dictionary (or empty list if no anomalies).
    """
    offset = len(df) - len(scores)
    timestamps = df['timestamp'][offset:].to_numpy()

    if not np.any(anomalies):
        return []

    # Merge first and last anomaly points into one event
    anomaly_indices = np.where(anomalies)[0]
    start_idx = anomaly_indices[0]
    end_idx = anomaly_indices[-1]

    start_ts = timestamps[start_idx]
    end_ts = timestamps[end_idx]

    top_k_str = ""
    if contributions is not None and len(contributions) == len(scores):
        seg_contribs = contributions[start_idx: end_idx + 1]
        avg_contribs = np.mean(seg_contribs, axis=0)

        # Normalize: convert to percentage of total absolute contribution
        abs_contribs = np.abs(avg_contribs)
        total = abs_contribs.sum()
        if total > 0:
            pct_contribs = abs_contribs / total * 100.0
        else:
            pct_contribs = np.zeros_like(abs_contribs)

        # Rank by normalized contribution
        top_idx = np.argsort(pct_contribs)[-top_k:][::-1]
        top_k_str = ";".join(
            f"{feature_cols[i]}({pct_contribs[i]:.1f}%)" for i in top_idx
        )

    model_str = "+".join(model_names) if model_names else "ensemble"

    return [{"interval": interval, "model": model_str, "start_time": start_ts, "end_time": end_ts,
             "top_k_metrics": top_k_str, "is_false_alarm": False, "processed": False}]
