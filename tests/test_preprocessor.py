from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from pandas import DataFrame

from power_preprocess.preprocessor import PowerDataPreprocessor

if TYPE_CHECKING:
    pass


@pytest.fixture
def sample_data() -> DataFrame:
    """Create a sample DataFrame for testing."""
    return pd.DataFrame(
        {
            "DateTime": ["1/1/2017 0:00", "1/1/2017 0:10", "1/1/2017 0:20"],
            "Temperature": [6.559, 6.414, 6.313],
            "Humidity": [73.8, 74.5, 74.5],
            "Wind Speed": [0.083, 0.083, 0.08],
            "general diffuse flows": [0.051, 0.07, 0.062],
            "diffuse flows": [0.119, 0.085, 0.1],
            "Zone 1 Power Consumption": [34055.69, 29814.68, 29128.10],
            "Zone 2  Power Consumption": [16128.87, 19375.07, 19006.68],
            "Zone 3  Power Consumption": [20240.96, 20131.08, 19668.43],
        }
    )


def test_load_data(sample_data: DataFrame, tmp_path: str) -> None:
    """Test loading data from CSV file."""
    # Save sample data to temporary CSV
    file_path = tmp_path / "test_data.csv"
    sample_data.to_csv(file_path, index=False)

    preprocessor = PowerDataPreprocessor()
    loaded_data = preprocessor.load_data(file_path)

    assert loaded_data is not None
    assert len(loaded_data) == len(sample_data)
    assert all(loaded_data.columns == sample_data.columns)


def test_preprocess_pipeline(sample_data: DataFrame) -> None:
    """Test the complete preprocessing pipeline."""
    preprocessor = PowerDataPreprocessor()
    preprocessor.raw_data = sample_data

    processed_data = preprocessor.preprocess()

    assert processed_data is not None
    assert "Hour" in processed_data.columns
    assert "DayOfWeek" in processed_data.columns
    assert "IsWeekend" in processed_data.columns

    # Check if numerical features are normalized between 0 and 1
    numerical_cols = ["Temperature", "Humidity", "Wind Speed"]
    for col in numerical_cols:
        assert processed_data[col].min() >= 0
        assert processed_data[col].max() <= 1


def test_get_feature_target_split(sample_data: DataFrame) -> None:
    """Test splitting data into features and targets."""
    preprocessor = PowerDataPreprocessor()
    preprocessor.raw_data = sample_data
    preprocessor.preprocess()

    X, y = preprocessor.get_feature_target_split(target_zones=[1, 2])

    assert len(X) == len(y)
    assert "Zone 1 Power Consumption" in y.columns
    assert "Zone 2  Power Consumption" in y.columns
    assert "Zone 3  Power Consumption" not in y.columns


def test_handle_missing_values(sample_data: DataFrame) -> None:
    """Test handling of missing values."""
    # Add some missing values
    sample_data.loc[0, "Temperature"] = np.nan

    preprocessor = PowerDataPreprocessor()
    preprocessor.raw_data = sample_data
    processed_data = preprocessor.preprocess()

    assert not processed_data["Temperature"].isna().any()


def test_add_time_features(sample_data: DataFrame) -> None:
    """Test addition of time-based features."""
    preprocessor = PowerDataPreprocessor()
    preprocessor.raw_data = sample_data
    processed_data = preprocessor.preprocess(normalize=False)

    assert "Hour_sin" in processed_data.columns
    assert "Hour_cos" in processed_data.columns
    assert all(processed_data["Hour"].isin(range(24)))
    assert all(processed_data["DayOfWeek"].isin(range(7)))
