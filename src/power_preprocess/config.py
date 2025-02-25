from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from typing import Dict, List

import yaml


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing.

    Attributes:
        parameters: Dictionary of preprocessing parameters
        numerical_features: List of numerical feature names
        target_zones: List of target zone column names
        time_features: List of time-based feature names
        catalog_name: Name of the Unity Catalog
        schema_name: Name of the schema in Unity Catalog
    """

    parameters: Dict[str, bool] = None
    numerical_features: List[str] = None
    target_zones: List[str] = None
    time_features: List[str] = None
    catalog_name: Optional[str] = None
    schema_name: Optional[str] = None

    def __post_init__(self) -> None:
        """Set default values if none provided."""
        self.parameters = self.parameters or {"handle_missing": True, "add_time_features": True, "normalize": True}
        self.numerical_features = self.numerical_features or [
            "Temperature",
            "Humidity",
            "Wind Speed",
            "general diffuse flows",
            "diffuse flows",
        ]
        self.target_zones = self.target_zones or [
            "Zone 1 Power Consumption",
            "Zone 2  Power Consumption",
            "Zone 3  Power Consumption",
        ]
        self.time_features = self.time_features or ["Hour", "DayOfWeek", "Month", "IsWeekend", "Hour_sin", "Hour_cos"]
        self.catalog_name = self.catalog_name
        self.schema_name = self.schema_name

    @classmethod
    def from_yaml(cls, config_path: str) -> "PreprocessingConfig":
        """Load configuration from a YAML file.

        Args:
            config_path: Path to the YAML configuration file

        Returns:
            PreprocessingConfig instance with loaded configuration

        Raises:
            FileNotFoundError: If the config file doesn't exist
            yaml.YAMLError: If the YAML file is malformed
        """
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
