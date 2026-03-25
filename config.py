from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

from utils.errors import Result, ErrorCode, Ok, Err
from utils.logger import get_logger, configure_logging

logger = get_logger("config")


# ============================================================================
# Post-Processing Configuration
# ============================================================================

@dataclass
class AmplitudeConfig:
    """Configuration for amplitude-based anomaly confirmation.

    Confirms algorithm detections whose amplitude exceeds a threshold.
    Supports two modes (relative takes precedence when > 0):

      - **Absolute**: ``|value| > threshold``
      - **Relative**: ``value > local_baseline * relative_threshold``

    Attributes:
        enabled: Whether amplitude filtering is active.
        threshold: Absolute value threshold (used when
            ``relative_threshold`` is 0).
        relative_threshold: Multiplier on a local rolling-median
            baseline.  When > 0, takes precedence over ``threshold``.
        baseline_window: Rolling-median window for the local baseline
            (only used when ``relative_threshold > 0``).
        features: Optional list of feature columns to check.
            Empty means all feature columns.
    """
    enabled: bool = True
    threshold: float = 100.0
    relative_threshold: float = 0.0
    baseline_window: int = 61
    features: List[str] = field(default_factory=list)


@dataclass
class FrequencyConfig:
    """Configuration for frequency-based anomaly filtering."""
    enabled: bool = True
    window_size: int = 10
    threshold: float = 0.7
    features: List[str] = field(default_factory=list)


@dataclass
class DirectionConfig:
    """Configuration for direction-based anomaly filtering.

    Filters anomalies based on the direction of deviation from a local
    baseline (rolling median).  For example, setting ``direction='up'``
    will suppress downward spikes, keeping only upward ones.

    Attributes:
        enabled: Whether direction filtering is active.
        direction: Which direction counts as anomalous.
            ``'up'``   – only upward deviations (value > baseline).
            ``'down'`` – only downward deviations (value < baseline).
            ``'both'`` – no direction filtering (default).
        baseline_window: Rolling-median window used to estimate the
            local baseline for direction comparison.
        features: Optional list of feature columns to check.  If empty,
            all feature columns are checked.
    """
    enabled: bool = False
    direction: str = "up"
    baseline_window: int = 51
    features: List[str] = field(default_factory=list)


@dataclass
class PostProcessingConfig:
    """Configuration for post-processing rules."""
    enabled: bool = True
    amplitude: AmplitudeConfig = field(default_factory=AmplitudeConfig)
    frequency: FrequencyConfig = field(default_factory=FrequencyConfig)
    direction: DirectionConfig = field(default_factory=DirectionConfig)


# ============================================================================
# Core Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    start_time: str = "01:00"
    # Days of training data required
    data_window: int = 7
    # Training interval in minutes (default: 12 hours)
    interval: int = 720
    # Feedback processing interval in minutes
    feedback_interval: int = 5


@dataclass
class DetectionConfig:
    # Interval between detections in minutes
    interval: int = 20
    summary_file: str = "results/summary.csv"
    # Number of top anomalies to display
    top_k: int = 3
    # Timeout for detection job in seconds (default: None = 90% of interval)
    timeout: int = None


@dataclass
class DataConfig:
    source_path: str = "data/source"


@dataclass
class ModelsConfig:
    save_path: str = "saved_models"

    # Enabled models: ["usad", "lstm", "sr", "historical"] - empty means all
    enabled_models: List[str] = field(default_factory=lambda: ["usad", "lstm", "sr"])

    # Historical Threshold Model - Adaptive Multi-Strategy (OPTIMIZED)
    historical_smoothing_method: str = "median"
    historical_smoothing_window_min: int = 3
    historical_smoothing_window_max: int = 9
    historical_robust_max_percentile: float = 95.0  # Balanced spike filtering
    historical_stationary_margin: float = 1.10      # Balanced threshold margin
    historical_mad_multiplier: float = 3.5
    historical_trending_k_base: float = 1.2
    historical_trending_k_max: float = 4.0
    historical_ac1_low_threshold: float = 0.2
    historical_ac1_high_threshold: float = 0.6
    historical_score_tier1: float = 0.1
    historical_score_tier2: float = 0.3
    historical_score_tier1_weight: float = 0.5
    historical_score_tier3_weight: float = 1.5

    # Common parameters
    window_size: int = 64
    epochs: int = 10
    batch_size: int = 128

    # USAD parameters
    usad_latent_size: int = 10
    usad_error_check_window: int = 5

    # LSTM parameters
    lstm_hidden_dim: int = 32
    lstm_num_layers: int = 1

    # SR parameters
    sr_filter_size: int = 3

    # POT threshold parameters
    pot_risk: float = 1e-4
    pot_level: float = 0.98

    # Extra parameters for extensibility
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Dictionary-style access for backward compatibility."""
        # Handle old parameter names
        if key == "latent_size":
            return self.usad_latent_size
        if key == "lstm_layers":
            return self.lstm_num_layers
        if hasattr(self, key):
            return getattr(self, key)
        return self.extra_params.get(key, default)


@dataclass
class LoggingConfig:
    log_dir: str = "logs"
    log_file: str = "rue_ai.log"
    # Rotation interval: 'midnight', 'H', 'D', etc.
    rotation: str = "midnight"
    # Days to keep old logs
    retention: int = 7
    # Max total size of all logs in MB (0 = no limit)
    max_total_size: int = 100
    # DEBUG, INFO, WARNING, ERROR
    level: str = "INFO"


@dataclass
class AppConfig:
    training: TrainingConfig = field(default_factory=TrainingConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    postprocessing: PostProcessingConfig = field(default_factory=PostProcessingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    config_path: str = "config/config.yaml"

    @staticmethod
    def _load_section(cls, config_section):
        if not config_section:
            return cls()
        
        # Special handling for nested configs like PostProcessingConfig
        if cls is PostProcessingConfig:
            amp_conf = AmplitudeConfig(**config_section.get('amplitude', {}))
            freq_conf = FrequencyConfig(**config_section.get('frequency', {}))
            dir_conf = DirectionConfig(**config_section.get('direction', {}))
            return cls(
                enabled=config_section.get('enabled', True),
                amplitude=amp_conf,
                frequency=freq_conf,
                direction=dir_conf,
            )

        valid_keys = cls.__annotations__.keys()
        filtered_data = {k: v for k, v in config_section.items() if k in valid_keys}
        # Extra parameters for extensibility
        if cls is ModelsConfig:
            extras = {k: v for k, v in config_section.items() if k not in valid_keys}
            instance = cls(**filtered_data)
            instance.extra_params = extras
            return instance

        return cls(**filtered_data)

    @classmethod
    def load(cls, config_path: str = "config/config.yaml") -> Result['AppConfig']:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config file (relative to CWD or project root, or absolute)
            
        Returns:
            Result[AppConfig]: Ok(config) on success, Err(ErrorCode) on failure
        """
        path = Path(config_path)

        # Resolve path strategy:
        # 1. If absolute, use as is
        # 2. If relative, try CWD first
        # 3. If not found in CWD, try relative to project root (directory of this file)
        if not path.is_absolute():
            absolute_path = Path.cwd() / path
            if absolute_path.exists():
                path = absolute_path
            else:
                # Fallback to project root
                project_root = Path(__file__).resolve().parent
                absolute_path = project_root / path
                if absolute_path.exists():
                    path = absolute_path

        if not path.exists():
            return Err(ErrorCode.CONFIG_MISSING)
        # NOTE: YAML errors will bubble up (as ValueError/YAMLError)
        with open(path, 'r') as f:
            config = yaml.safe_load(f) or {}
        # Convert path back to string for storage
        return Ok(cls(training=cls._load_section(TrainingConfig, config.get('training')),
                      detection=cls._load_section(DetectionConfig, config.get('detection')),
                      postprocessing=cls._load_section(PostProcessingConfig, config.get('postprocessing')),
                      data=cls._load_section(DataConfig, config.get('data')),
                      models=cls._load_section(ModelsConfig, config.get('models')),
                      logging=cls._load_section(LoggingConfig, config.get('logging')), config_path=str(path.resolve())))

    def validate(self) -> bool:
        """Validates critical configuration to ensure realistic settings."""
        valid = True

        # --- Data Config ---
        if not self.data.source_path:
            logger.error("Config Error: data.source_path is empty")
            valid = False

        # --- Models Config ---
        if not self.models.save_path:
            logger.error("Config Error: models.save_path is empty")
            valid = False

        if self.models.window_size <= 0:
            logger.error(f"Config Error: models.window_size must be > 0, got {self.models.window_size}")
            valid = False

        if self.models.epochs <= 0:
            logger.error(f"Config Error: models.epochs must be > 0, got {self.models.epochs}")
            valid = False

        if self.models.batch_size <= 0:
            logger.error(f"Config Error: models.batch_size must be > 0, got {self.models.batch_size}")
            valid = False

        if not (0 < self.models.pot_risk < 1):
            logger.error(f"Config Error: models.pot_risk must be in (0, 1), got {self.models.pot_risk}")
            valid = False

        if not (0 < self.models.pot_level < 1):
            logger.error(f"Config Error: models.pot_level must be in (0, 1), got {self.models.pot_level}")
            valid = False

        # --- Training Config ---
        if self.training.data_window <= 0:
            logger.error(f"Config Error: training.data_window must be > 0, got {self.training.data_window}")
            valid = False

        if self.training.interval <= 0:
            logger.error(f"Config Error: training.interval must be > 0, got {self.training.interval}")
            valid = False

        if self.training.feedback_interval <= 0:
            logger.error(f"Config Error: training.feedback_interval must be > 0, got {self.training.feedback_interval}")
            valid = False

        try:
            datetime.strptime(self.training.start_time, "%H:%M")
        except ValueError:
            logger.error(f"Config Error: training.start_time must be in 'HH:MM' format, got {self.training.start_time}")
            valid = False

        # --- Detection Config ---
        if self.detection.interval <= 0:
            logger.error(f"Config Error: detection.interval must be > 0, got {self.detection.interval}")
            valid = False

        if self.detection.top_k < 1:
            logger.error(f"Config Error: detection.top_k must be >= 1, got {self.detection.top_k}")
            valid = False

        if self.detection.timeout is not None and self.detection.timeout <= 0:
            logger.error(f"Config Error: detection.timeout must be > 0 if set, got {self.detection.timeout}")
            valid = False

        # --- Logging Config ---
        if self.logging.retention < 0:
            logger.error(f"Config Error: logging.retention must be >= 0, got {self.logging.retention}")
            valid = False

        if self.logging.max_total_size < 0:
            logger.error(f"Config Error: logging.max_total_size must be >= 0, got {self.logging.max_total_size}")
            valid = False

        return valid


def init_config(config_path: str = "config/config.yaml") -> Result[AppConfig]:
    """Load config, configure logging, and validate. Returns Result."""
    # 1. Load Config
    res = AppConfig.load(config_path)
    if res.is_err():
        return Err(res.err_value)

    config: AppConfig = res.unwrap()

    # 2. Configure Logging
    res = configure_logging(config.logging)
    if res.is_err():
        return Err(res.err_value)

    # 3. Validate Config
    if not config.validate():
        return Err(ErrorCode.CONFIG_INVALID)

    return Ok(config)
