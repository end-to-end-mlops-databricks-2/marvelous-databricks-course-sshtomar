from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import yaml
from pydantic import BaseModel, Field


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing.
    
    Attributes:
        parameters: Dictionary of preprocessing parameters
        numerical_features: List of numerical feature names
        target_zones: List of target zone column names
        time_features: List of time-based feature names
    """
    parameters: Dict[str, bool] = None
    numerical_features: List[str] = None
    target_zones: List[str] = None
    time_features: List[str] = None

    def __post_init__(self) -> None:
        """Set default values if none provided."""
        self.parameters = self.parameters or {
            "handle_missing": True,
            "add_time_features": True,
            "normalize": True
        }
        self.numerical_features = self.numerical_features or [
            'Temperature', 'Humidity', 'Wind Speed',
            'general diffuse flows', 'diffuse flows'
        ]
        self.target_zones = self.target_zones or [
            'Zone 1 Power Consumption',
            'Zone 2  Power Consumption',
            'Zone 3  Power Consumption'
        ]
        self.time_features = self.time_features or [
            'Hour', 'DayOfWeek', 'Month', 'IsWeekend',
            'Hour_sin', 'Hour_cos'
        ]


class PreprocessingConfig(BaseModel):
    """Configuration class for power consumption preprocessing.
    
    Attributes:
        numerical_features: List of numerical feature column names
        time_features: List of time-based feature column names
        target_zones: List of power consumption zone columns
        parameters: Dictionary of preprocessing parameters
        data_path: Path to the raw data file
    """
    
    numerical_features: List[str] = Field(
        default=[
            "Temperature",
            "Humidity",
            "Wind Speed",
            "general diffuse flows",
            "diffuse flows"
        ]
    )
    
    time_features: List[str] = Field(
        default=[
            "Hour",
            "DayOfWeek",
            "Month",
            "IsWeekend",
            "Hour_sin",
            "Hour_cos"
        ]
    )
    
    target_zones: List[str] = Field(
        default=[
            "Zone 1 Power Consumption",
            "Zone 2  Power Consumption",
            "Zone 3  Power Consumption"
        ]
    )
    
    parameters: Dict[str, Any] = Field(
        default={
            "handle_missing": True,
            "add_time_features": True,
            "normalize": True
        }
    )
    
    data_path: Optional[Path] = None

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