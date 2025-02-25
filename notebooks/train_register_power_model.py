# Databricks notebook source
import mlflow
from pyspark.sql import SparkSession
from power_preprocess.models.forecaster import PowerForecastModel
from power_preprocess.preprocessor import PowerDataPreprocessor
from power_preprocess.config import PreprocessingConfig
from typing import Optional
import pandas as pd
from pathlib import Path
#import dbutils

# COMMAND ----------

# Set up MLflow to use Unity Catalog
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# Initialize Spark session
spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# Initialize preprocessor and model
config = {
    "parameters": {"handle_missing": True, "add_time_features": True, "normalize": True}
}
preprocessor = PowerDataPreprocessor(config=PreprocessingConfig(**config))
preprocessor.load_data("../data/raw/power_consumption.csv")
forecast_model = PowerForecastModel(preprocessor=preprocessor)

# COMMAND ----------

# Load and preprocess the data
data_path: str = "dbfs:/Volumes/mlops_dev/tomarshu/data/power_consumption.csv"
#if not dbutils.fs.ls(data_path):
#    raise FileNotFoundError(f"Data file not found at: {data_path}")

try:
    preprocessor.load_data(data_path)
except Exception as e:
    raise RuntimeError(f"Failed to load data: {str(e)}")

preprocessed_data: pd.DataFrame = preprocessor.preprocess()

# Split data into train/validation/test sets
train_data, val_data, test_data = preprocessor.split_data(
    train_ratio=0.7, 
    val_ratio=0.15, 
    random_split=False  # Use temporal split for time series
)

# Store splits in Unity Catalog
version = "v1"  # Increment this when data changes
preprocessor.save_to_catalog(train_data, val_data, test_data, version=version)

# COMMAND ----------

# Train the model
forecast_model.train(train_data=train_data, validation_data=val_data)

# COMMAND ----------

# Evaluate the model
metrics = forecast_model.evaluate(test_data=test_data)
print("Model Performance Metrics:")
for zone, zone_metrics in metrics.items():
    print(f"\n{zone}:")
    for metric_name, value in zone_metrics.items():
        print(f"{metric_name}: {value:.4f}")

# COMMAND ----------

# Now register the models AFTER evaluation
forecast_model.register_best_models(metric="rmse")

# COMMAND ----------

# Generate predictions
predictions = forecast_model.predict(test_data)
for zone, zone_predictions in predictions.items():
    print(f"\nPredictions for {zone}:")
    print(zone_predictions[:5])  # Show first 5 predictions

# COMMAND ---------- 