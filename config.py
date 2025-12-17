import yaml
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from utils.logger import get_logger

logger = get_logger("config")

@dataclass
class TrainingConfig:
    start_time: str = "01:00"
    data_window: str = "7d"

@dataclass
class DetectionConfig:
    interval_minutes: int = 5
    summary_file: str = "results/summary.csv"
    top_k: int = 3

@dataclass
class DataConfig:
    source_path: str = "data/source"

@dataclass
class ModelsConfig:
    default_type: str = "ensemble"
    save_path: str = "saved_models"
    window_size: int = 64
    epochs: int = 10
    batch_size: int = 128
    
    # USAD
    latent_size: int = 10
    
    # LSTM
    lstm_hidden_dim: int = 32
    lstm_layers: int = 1
    
    # SR
    sr_filter_size: int = 3
    
    # POT
    pot_risk: float = 1e-4
    pot_level: float = 0.98

    # Capture additional model parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Backward compatibility for dictionary-style access"""
        if hasattr(self, key):
            return getattr(self, key)
        return self.extra_params.get(key, default)

@dataclass
class LoggingConfig:
    log_dir: str = "logs"
    log_file: str = "spectrum.log"
    retention_days: int = 15
    max_bytes: int = 10 * 1024 * 1024  # 10MB per file
    backup_count: int = 5  # Number of backup files to keep

@dataclass
class AppConfig:
    training: TrainingConfig = field(default_factory=TrainingConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    config_path: str = "config/config.yaml"

    @classmethod
    def load(cls, config_path: str = "config/config.yaml") -> 'AppConfig':
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found at {config_path}. Using defaults.")
            return cls(config_path=config_path)

        try:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f) or {}

            # Helper to safely populate dataclasses ignoring unknown keys if needed, 
            # or we can just strict map. For robustness, let's map known sections.
            
            def load_section(section_cls, section_data):
                if not section_data:
                    return section_cls()
                # Filter keys that exist in the dataclass
                valid_keys = section_cls.__annotations__.keys()
                filtered_data = {k: v for k, v in section_data.items() if k in valid_keys}
                
                # Handle extra params for ModelsConfig specifically
                if section_cls is ModelsConfig:
                    extras = {k: v for k, v in section_data.items() if k not in valid_keys}
                    instance = section_cls(**filtered_data)
                    instance.extra_params = extras
                    return instance
                
                return section_cls(**filtered_data)

            return cls(
                training=load_section(TrainingConfig, data.get('training')),
                detection=load_section(DetectionConfig, data.get('detection')),
                data=load_section(DataConfig, data.get('data')),
                models=load_section(ModelsConfig, data.get('models')),
                logging=load_section(LoggingConfig, data.get('logging')),
                config_path=config_path
            )

        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            return cls(config_path=config_path)

    def validate(self) -> bool:
        """Validates critical configuration."""
        # Dataclasses ensure types to some extent if strictly initialized, 
        # but here we just check if key values are reasonable.
        valid = True
        
        if not self.data.source_path:
            logger.error("Config Error: data.source_path is empty")
            valid = False
            
        if not self.models.save_path:
            logger.error("Config Error: models.save_path is empty")
            valid = False

        # Add more logic as needed
        return valid

    # Backward compatibility for config.training.get() pattern if used anywhere
    # But generally we should move to attribute access. 
    # The existing code uses config.training.get(...) because it was a dict.
    # We need to support that or refactor usage.
    # Refactoring usage is better for the "Optimize" goal. 
    # But for safety, let's add .get() to the nested classes or wrapper.
    # Actually, I implemented .get() in ModelsConfig. 
    # TrainingConfig, DetectionConfig, DataConfig also need .get() if existing code uses it.
    
    # Let's verify usage. 
    # core/manager.py uses config.training.get('data_window', '7d')
    # So yes, we need .get() on these sub-configs to avoid breaking everything immediately,
    # OR we update the code to use attributes. 
    # Updating code to use attributes is cleaner optimization.

# Global instance
config = AppConfig.load()

# Reconfigure logging with loaded config settings
from utils.logger import configure_logging
configure_logging(config.logging)

# Monkey patch .get methods for backward compatibility during transition if we miss any
def add_get_method(cls):
    def get(self, key: str, default: Any = None):
        return getattr(self, key, default)
    cls.get = get
    return cls

TrainingConfig = add_get_method(TrainingConfig)
DetectionConfig = add_get_method(DetectionConfig)
DataConfig = add_get_method(DataConfig)
LoggingConfig = add_get_method(LoggingConfig)
# ModelsConfig already has custom get

