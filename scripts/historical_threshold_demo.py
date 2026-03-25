import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_synthetic_data(n_train=1000, n_test=500):
    """Generate synthetic time series data with 2 features."""
    np.random.seed(42)
    
    # Feature 1: CPU-like (baseline 20-40, occasional normal spikes to 60)
    cpu_train = np.random.normal(30, 5, n_train) + np.sin(np.linspace(0, 10*np.pi, n_train)) * 10
    cpu_train[np.random.choice(n_train, 10, replace=False)] += np.random.uniform(10, 20, 10) # normal spikes
    
    # Feature 2: Memory-like (stable baseline around 50, slow drift)
    mem_train = np.random.normal(50, 2, n_train) + np.linspace(0, 10, n_train)
    
    train_df = pd.DataFrame({
        'cpu': np.clip(cpu_train, 0, 100),
        'memory': np.clip(mem_train, 0, 100)
    })
    
    # Test data with anomalies
    cpu_test = np.random.normal(30, 5, n_test) + np.sin(np.linspace(0, 5*np.pi, n_test)) * 10
    mem_test = np.random.normal(50, 2, n_test) + np.linspace(10, 15, n_test)
    
    # Inject anomalies
    # Anomaly 1: CPU spike exceeding historical max
    cpu_test[100:110] += 50
    
    # Anomaly 2: Memory leak exceeding historical max
    mem_test[300:400] += np.linspace(0, 40, 100)
    
    # Anomaly 3: Both spike
    cpu_test[450:460] += 40
    mem_test[450:460] += 30
    
    test_df = pd.DataFrame({
        'cpu': np.clip(cpu_test, 0, 100),
        'memory': np.clip(mem_test, 0, 100)
    })
    
    return train_df, test_df

class HistoricalThresholdDetector:
    def __init__(self, tolerance_pct=0.1, robust_max_percentile=99.0):
        """
        Args:
            tolerance_pct: How much percentage above the historical max is allowed (e.g., 0.1 for 10%)
            robust_max_percentile: Percentile to use as the 'max' to avoid extreme outliers in training
        """
        self.tolerance_pct = tolerance_pct
        self.robust_max_percentile = robust_max_percentile
        self.thresholds = {}
        self.raw_max = {}
        
    def fit(self, df):
        """Calculate thresholds based on training data."""
        for col in df.columns:
            # Use percentile to get a robust maximum, ignoring extreme single-point outliers in train
            hist_max = np.percentile(df[col].dropna(), self.robust_max_percentile)
            self.raw_max[col] = hist_max
            # Threshold is historical max + tolerance
            self.thresholds[col] = hist_max * (1 + self.tolerance_pct)
            print(f"Feature '{col}': Hist Max (p{self.robust_max_percentile}) = {hist_max:.2f}, Threshold = {self.thresholds[col]:.2f} (+{self.tolerance_pct*100}%)")
            
    def predict(self, df):
        """Detect anomalies in test data."""
        anomalies = pd.DataFrame(index=df.index)
        scores = pd.DataFrame(index=df.index)
        
        for col in df.columns:
            if col not in self.thresholds:
                continue
                
            threshold = self.thresholds[col]
            
            # True if value > threshold
            is_anomaly = df[col] > threshold
            anomalies[col] = is_anomaly
            
            # Score is how much it exceeds the threshold (0 if normal)
            excess = df[col] - threshold
            scores[col] = np.maximum(0, excess)
            
        # Overall anomaly if any feature is anomalous
        anomalies['overall'] = anomalies.any(axis=1)
        return anomalies, scores

def plot_results(train_df, test_df, detector, anomalies, save_path="historical_threshold_demo.png"):
    n_features = len(train_df.columns)
    fig, axes = plt.subplots(n_features, 1, figsize=(15, 4 * n_features), sharex=False)
    
    if n_features == 1:
        axes = [axes]
        
    train_time = np.arange(len(train_df))
    test_time = np.arange(len(train_df), len(train_df) + len(test_df))
    
    for i, col in enumerate(train_df.columns):
        ax = axes[i]
        
        # Plot training data
        ax.plot(train_time, train_df[col], color='#2ca02c', alpha=0.7, label='Train Data (Normal)') # Green curve
        
        # Plot test data
        ax.plot(test_time, test_df[col], color='#1f77b4', alpha=0.7, label='Test Data')
        
        # Plot threshold
        threshold = detector.thresholds[col]
        raw_max = detector.raw_max[col]
        
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold (+{detector.tolerance_pct*100}%)')
        ax.axhline(y=raw_max, color='orange', linestyle=':', linewidth=1.5, label=f'Historical Max (p{detector.robust_max_percentile})')
        
        # Highlight anomalies
        anomaly_idx = anomalies.index[anomalies[col]].tolist()
        if anomaly_idx:
            # Map test indices to full time scale
            plot_idx = [idx + len(train_df) for idx in anomaly_idx]
            ax.scatter(plot_idx, test_df[col].iloc[anomaly_idx], color='red', s=50, label='Anomaly Detected', zorder=5)
            
        ax.set_title(f"Feature: {col}")
        ax.set_ylabel("Value")
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    print("1. Generating synthetic data...")
    train_df, test_df = generate_synthetic_data()
    
    print("\n2. Training Historical Threshold Detector...")
    # 5% tolerance above historical 99th percentile
    detector = HistoricalThresholdDetector(tolerance_pct=0.05, robust_max_percentile=99.0)
    detector.fit(train_df)
    
    print("\n3. Detecting anomalies on test data...")
    anomalies, scores = detector.predict(test_df)
    
    print(f"\nFound anomalies in {anomalies['overall'].sum()} out of {len(test_df)} test points.")
    
    print("\n4. Generating visualization...")
    plot_path = "historical_threshold_demo.png"
    plot_results(train_df, test_df, detector, anomalies, save_path=plot_path)
