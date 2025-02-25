from typing import Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame


def calculate_power_statistics(data: DataFrame) -> DataFrame:
    """Calculate statistical measures for power consumption across zones.

    Args:
        data: DataFrame containing power consumption data

    Returns:
        DataFrame containing statistical measures for each zone
    """
    power_cols = [col for col in data.columns if "Power Consumption" in col]
    stats = []

    for col in power_cols:
        zone_stats = {
            "Zone": col.split("Zone")[1].split("Power")[0].strip(),
            "Mean": data[col].mean(),
            "Std": data[col].std(),
            "Min": data[col].min(),
            "Max": data[col].max(),
        }
        stats.append(zone_stats)

    return pd.DataFrame(stats)


def create_time_windows(data: DataFrame, window_size: int, target_offset: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding time windows for time series prediction.

    Args:
        data: Input DataFrame with time series data
        window_size: Number of time steps in each window
        target_offset: Number of steps ahead to predict

    Returns:
        Tuple containing:
            - Array of windowed input sequences
            - Array of target values
    """
    values = data.values
    X, y = [], []

    for i in range(len(values) - window_size - target_offset + 1):
        X.append(values[i : (i + window_size)])
        y.append(values[i + window_size + target_offset - 1])

    return np.array(X), np.array(y)


def split_by_time(
    data: DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """Split the dataset into train, validation, and test sets based on time.

    Args:
        data: Input DataFrame with datetime index
        train_ratio: Proportion of data to use for training
        val_ratio: Proportion of data to use for validation

    Returns:
        Tuple containing train, validation, and test DataFrames
    """
    n = len(data)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    train = data.iloc[:train_size]
    val = data.iloc[train_size : train_size + val_size]
    test = data.iloc[train_size + val_size :]

    return train, val, test
