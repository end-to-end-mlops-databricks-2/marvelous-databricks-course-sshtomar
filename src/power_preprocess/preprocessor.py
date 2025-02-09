from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas import DataFrame


class PowerDataPreprocessor:
    """Preprocessor for the Tetouan City Power Consumption dataset.
    
    This class handles the preprocessing steps for the power consumption dataset,
    including loading, cleaning, feature engineering, and normalization.
    
    Attributes:
        raw_data: DataFrame containing the raw loaded data
        processed_data: DataFrame containing the processed data
    """

    def __init__(self) -> None:
        """Initialize the preprocessor."""
        self.raw_data: Optional[DataFrame] = None
        self.processed_data: Optional[DataFrame] = None

    def load_data(self, file_path: Union[str, Path]) -> DataFrame:
        """Load the power consumption dataset from a CSV file.
        
        Args:
            file_path: Path to the CSV file containing the dataset
            
        Returns:
            DataFrame containing the loaded data
            
        Raises:
            FileNotFoundError: If the specified file does not exist
        """
        self.raw_data = pd.read_csv(file_path)
        return self.raw_data

    def preprocess(
        self,
        handle_missing: bool = True,
        add_time_features: bool = True,
        normalize: bool = True
    ) -> DataFrame:
        """Execute the preprocessing pipeline on the loaded data.
        
        Args:
            handle_missing: Whether to handle missing values
            add_time_features: Whether to add time-based features
            normalize: Whether to normalize numerical features
            
        Returns:
            DataFrame containing the processed data
            
        Raises:
            ValueError: If data hasn't been loaded yet
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        data = self.raw_data.copy()
        
        # Convert DateTime column to datetime type
        data['DateTime'] = pd.to_datetime(data['DateTime'])
        
        if handle_missing:
            data = self._handle_missing_values(data)
            
        if add_time_features:
            data = self._add_time_features(data)
            
        if normalize:
            data = self._normalize_features(data)
            
        self.processed_data = data
        return self.processed_data

    def _handle_missing_values(self, data: DataFrame) -> DataFrame:
        """Handle missing values in the dataset.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with handled missing values
        """
        # Fill missing numerical values with median of their respective columns
        numerical_columns = data.select_dtypes(include=[np.number]).columns
        data[numerical_columns] = data[numerical_columns].fillna(data[numerical_columns].median())
        return data

    def _add_time_features(self, data: DataFrame) -> DataFrame:
        """Add time-based features to the dataset.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with additional time-based features
        """
        data['Hour'] = data['DateTime'].dt.hour
        data['DayOfWeek'] = data['DateTime'].dt.dayofweek
        data['Month'] = data['DateTime'].dt.month
        data['IsWeekend'] = data['DayOfWeek'].isin([5, 6]).astype(int)
        
        # Add cyclical time features
        data['Hour_sin'] = np.sin(2 * np.pi * data['Hour']/24)
        data['Hour_cos'] = np.cos(2 * np.pi * data['Hour']/24)
        
        return data

    def _normalize_features(self, data: DataFrame) -> DataFrame:
        """Normalize numerical features using min-max scaling.
        
        Args:
            data: Input DataFrame
            
        Returns:
            DataFrame with normalized numerical features
        """
        # Identify columns to normalize (exclude DateTime and engineered categorical features)
        cols_to_normalize = [
            'Temperature', 'Humidity', 'Wind Speed', 
            'general diffuse flows', 'diffuse flows',
            'Zone 1 Power Consumption', 'Zone 2  Power Consumption', 'Zone 3  Power Consumption'
        ]
        
        for col in cols_to_normalize:
            if col in data.columns:
                min_val = data[col].min()
                max_val = data[col].max()
                data[col] = (data[col] - min_val) / (max_val - min_val)
                
        return data

    def get_feature_target_split(
        self,
        target_zones: Optional[List[int]] = None
    ) -> Tuple[DataFrame, DataFrame]:
        """Split the processed data into features and target variables.
        
        Args:
            target_zones: List of zone numbers (1-3) to include as targets.
                        If None, includes all zones.
        
        Returns:
            Tuple containing:
                - DataFrame with feature columns
                - DataFrame with target columns
                
        Raises:
            ValueError: If processed data is not available
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Run preprocess() first.")

        if target_zones is None:
            target_zones = [1, 2, 3]

        target_cols = [f'Zone {z} Power Consumption' for z in target_zones]
        feature_cols = [col for col in self.processed_data.columns 
                       if col not in target_cols and col != 'DateTime']

        X = self.processed_data[feature_cols]
        y = self.processed_data[target_cols]

        return X, y 