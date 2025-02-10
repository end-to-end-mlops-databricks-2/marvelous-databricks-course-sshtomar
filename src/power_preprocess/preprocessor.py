from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas import DataFrame
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp

from power_preprocess.config import PreprocessingConfig


class PowerDataPreprocessor:
    """Preprocessor for the Tetouan City Power Consumption dataset.

    This class handles the preprocessing steps for the power consumption dataset,
    including loading, cleaning, feature engineering, normalization, and Databricks integration.

    Attributes:
        raw_data: DataFrame containing the raw loaded data
        processed_data: DataFrame containing the processed data
        spark: SparkSession for Databricks operations
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None) -> None:
        """Initialize the preprocessor.

        Args:
            config: Optional configuration object. If None, uses default values.
        """
        self.config = config or PreprocessingConfig()
        self.raw_data: Optional[DataFrame] = None
        self.processed_data: Optional[DataFrame] = None
        self.spark = SparkSession.builder.getOrCreate()

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

    def preprocess(self) -> DataFrame:
        """Execute the preprocessing pipeline on the loaded data using configuration settings.

        Returns:
            DataFrame containing the processed data

        Raises:
            ValueError: If data hasn't been loaded yet
        """
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        data = self.raw_data.copy()

        # Convert DateTime column to datetime type
        data["DateTime"] = pd.to_datetime(data["DateTime"])

        if self.config.parameters["handle_missing"]:
            data = self._handle_missing_values(data)

        if self.config.parameters["add_time_features"]:
            data = self._add_time_features(data)

        if self.config.parameters["normalize"]:
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
        data["Hour"] = data["DateTime"].dt.hour
        data["DayOfWeek"] = data["DateTime"].dt.dayofweek
        data["Month"] = data["DateTime"].dt.month
        data["IsWeekend"] = data["DayOfWeek"].isin([5, 6]).astype(int)

        # Add cyclical time features
        data["Hour_sin"] = np.sin(2 * np.pi * data["Hour"] / 24)
        data["Hour_cos"] = np.cos(2 * np.pi * data["Hour"] / 24)

        return data

    def _normalize_features(self, data: DataFrame) -> DataFrame:
        """Normalize numerical features using min-max scaling.

        Args:
            data: Input DataFrame

        Returns:
            DataFrame with normalized numerical features
        """
        # Normalize numerical features
        cols_to_normalize = self.config.numerical_features + self.config.target_zones

        for col in cols_to_normalize:
            if col in data.columns:
                min_val = data[col].min()
                max_val = data[col].max()
                data[col] = (data[col] - min_val) / (max_val - min_val)

        return data

    def get_feature_target_split(self, target_zones: Optional[List[str]] = None) -> Tuple[DataFrame, DataFrame]:
        """Split the processed data into features and target variables.

        Args:
            target_zones: List of zone names to include as targets.
                        If None, uses zones from config.

        Returns:
            Tuple containing:
                - DataFrame with feature columns
                - DataFrame with target columns

        Raises:
            ValueError: If processed data is not available
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Run preprocess() first.")

        target_cols = target_zones or self.config.target_zones
        feature_cols = self.config.numerical_features + self.config.time_features

        X = self.processed_data[feature_cols]
        y = self.processed_data[target_cols]

        return X, y

    def save_to_catalog(self, train_set: DataFrame, test_set: DataFrame) -> None:
        """Save the train and test sets into Databricks tables with timestamps.

        Args:
            train_set: Training dataset to be saved
            test_set: Test dataset to be saved

        Raises:
            ValueError: If catalog_name or schema_name is not configured
        """
        if not (self.config.catalog_name and self.config.schema_name):
            raise ValueError("Catalog and schema names must be configured")

        train_set_with_timestamp = self.spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = self.spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )

        test_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        )

    def enable_change_data_feed(self) -> None:
        """Enable Change Data Feed (CDF) for train and test set tables.

        This allows tracking of data changes in the Delta tables.

        Raises:
            ValueError: If catalog_name or schema_name is not configured
        """
        if not (self.config.catalog_name and self.config.schema_name):
            raise ValueError("Catalog and schema names must be configured")

        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )
