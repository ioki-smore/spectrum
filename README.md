# Spectrum Anomaly Detection System

A robust, configurable anomaly detection system for time-series metrics, featuring ensemble models, dynamic thresholding, and incremental learning.

## Project Structure

```
spectrum/
├── config/                 # Configuration files
│   └── config.yaml        # Main configuration
├── core/                   # Core logic
│   └── manager.py         # Orchestrates training, detection, and feedback
├── data/                   # Data handling
│   ├── dataset/           # PyTorch Datasets
│   │   └── timeseries.py  # Sliding window dataset
│   ├── loader.py          # Data loading, validation, and state persistence
│   └── processor.py       # Data normalization (StandardScaler/MinMaxScaler)
├── deploy/                 # Deployment configurations
│   ├── spectrum.service   # Linux systemd service
│   ├── com.spectrum.service.plist  # macOS launchd service
│   ├── install.sh         # Linux installation script
│   └── uninstall.sh       # Linux uninstallation script
├── logs/                   # Application logs (with rotation)
├── models/                 # Anomaly detection models
│   ├── base.py            # Abstract base model class
│   ├── usad.py            # USAD autoencoder model
│   ├── lstm.py            # LSTM autoencoder model
│   ├── sr.py              # Spectral Residual model
│   └── ensemble.py        # Ensemble with soft voting
├── results/                # Detection outputs
├── saved_models/           # Serialized models, processors, and version history
│   └── versions/          # Model version backups for rollback
├── tests/                  # Unit tests
├── utils/                  # Utilities
│   ├── device.py          # PyTorch device handling
│   ├── logger.py          # Logging with rotation and retention
│   └── thresholding.py    # POT (Peak Over Threshold) dynamic thresholding
├── main.py                 # Main service entry point with CLI
├── config.py               # Typed configuration (dataclasses)
└── pyproject.toml          # Project metadata and dependencies
```

## Features

- **Ensemble Models**: Combines USAD, LSTM Autoencoder, and Spectral Residual using soft voting with Z-score normalization.
- **Dynamic Thresholding**: Uses Peak Over Threshold (POT) for robust anomaly detection.
- **Multi-Interval Support**: Handles data with different sampling intervals (e.g., 5s, 10s, 1min) independently.
- **Automated Scheduling**: 
  - Periodic detection pipeline (configurable interval).
  - Feedback processing loop for incremental learning.
- **Incremental Learning**: Automatically fine-tunes models on false alarms marked by users.
- **Model Version Management**: Automatic backups before training with rollback support.
- **State Persistence**: Tracks processed data across service restarts.
- **Log Rotation**: Configurable log retention (default 15 days).

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd spectrum
   ```

2. **Install dependencies**:
   ```bash
   pip install .
   ```
   Or manually:
   ```bash
   pip install torch polars numpy pyyaml joblib scikit-learn apscheduler
   ```

## Configuration

Edit `config/config.yaml` to adjust settings:

```yaml
training:
  start_time: "01:00"
  data_window: "7d"
  
detection:
  interval_minutes: 5
  summary_file: "results/summary.csv"
  
data:
  source_path: "data/source"

models:
  save_path: "saved_models"
```

## Usage

### Running as a Service

To start the main orchestration service:

```bash
python main.py start
# OR via CLI
python main.py train --interval 1min
python main.py detect --interval all
```

### Command Line Interface (CLI)

Manual triggers for testing or ad-hoc operations:

- **Train models**:
  ```bash
  python main.py train --interval 1min
  python main.py train --interval all
  ```

- **Run detection**:
  ```bash
  python main.py detect --interval 5min
  ```

### Deployment (Systemd)

1. Copy the service file:
   ```bash
   sudo cp deploy/anomaly-detector.service /etc/systemd/system/
   ```
2. Reload daemon and enable:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable anomaly-detector
   sudo systemctl start anomaly-detector
   ```

## Development

### Running Tests

To run the unit test suite:

```bash
python -m unittest discover tests
```

## Data Format

The system expects CSV files in `data/source/{interval}/` (e.g., `data/source/1min/`).
Expected columns:
- `timestamp`: Unix timestamp or Datetime
- `value` (or other feature columns): Numeric metric values
- `label` (optional): 0 for normal, 1 for anomaly (used for validation/metrics, not unsupervised training)

## Outputs

- **Summary**: `results/summary.csv` contains high-level detection results.
- **Details**: `results/{interval}_details.csv` contains point-wise anomaly scores and labels.
