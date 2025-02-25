from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from pandas import DataFrame
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models.signature import infer_signature

from power_preprocess.preprocessor import PowerDataPreprocessor


class PowerForecastModel:
    """Time series forecasting model for power consumption prediction.

    This class handles the training, evaluation, and MLflow tracking of a LightGBM-based
    forecasting model for power consumption data.

    Attributes:
        preprocessor: PowerDataPreprocessor instance for data preparation
        models: Dictionary of trained models for each zone
        experiment_name: Name of the MLflow experiment
        run_id: Current MLflow run ID
    """

    def __init__(
        self,
        preprocessor: PowerDataPreprocessor,
        experiment_name: str = "/Users/tomarshubham24@gmail.com/power_consumption_forecast",
    ) -> None:
        """Initialize the forecasting model.

        Args:
            preprocessor: Instance of PowerDataPreprocessor for data preparation
            experiment_name: Name for the MLflow experiment
        """
        self.preprocessor = preprocessor
        self.models: Dict[str, LGBMRegressor] = {}
        self.experiment_name = experiment_name
        self.run_id: Optional[str] = None

        # Set up MLflow experiment
        mlflow.set_experiment(experiment_name)

    def train(
        self,
        train_data: DataFrame,
        target_zones: Optional[List[str]] = None,
        model_params: Optional[Dict] = None,
    ) -> None:
        """Train forecasting models for each target zone.

        Args:
            train_data: Training dataset
            target_zones: List of zones to train models for. If None, uses config defaults
            model_params: Dictionary of LightGBM parameters. If None, uses defaults

        Raises:
            ValueError: If target zones are not found in the data
        """
        default_params = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "random_state": 42,
        }
        params = model_params or default_params
        zones = target_zones or self.preprocessor.config.target_zones

        with mlflow.start_run() as run:
            self.run_id = run.info.run_id
            mlflow.log_params(params)

            X, y = self.preprocessor.get_feature_target_split(target_zones=zones)

            for zone in zones:
                model = LGBMRegressor(**params)
                model.fit(X, y[zone])
                self.models[zone] = model

                # Infer the model signature
                signature = infer_signature(X, y[zone])

                # Log the model with signature for each zone
                mlflow.sklearn.log_model(
                    model,
                    f"model_{zone.replace(' ', '-').lower()}",
                    registered_model_name=f"power_forecast_{zone.replace(' ', '-').lower()}",
                    signature=signature
                )

    def evaluate(
        self, test_data: DataFrame, target_zones: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate the trained models on test data.

        Args:
            test_data: Test dataset
            target_zones: List of zones to evaluate. If None, uses all trained models

        Returns:
            Dictionary of evaluation metrics for each zone

        Raises:
            ValueError: If no models have been trained yet
        """
        if not self.models:
            raise ValueError("No trained models found. Run train() first.")

        zones = target_zones or list(self.models.keys())
        X_test, y_test = self.preprocessor.get_feature_target_split(target_zones=zones)
        metrics: Dict[str, Dict[str, float]] = {}

        for zone in zones:
            if zone not in self.models:
                continue

            y_pred = self.models[zone].predict(X_test)
            zone_metrics = {
                "mae": mean_absolute_error(y_test[zone], y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test[zone], y_pred)),
                "r2": r2_score(y_test[zone], y_pred),
            }
            metrics[zone] = zone_metrics

            # Log metrics to MLflow
            if self.run_id:
                with mlflow.start_run(run_id=self.run_id):
                    for metric_name, value in zone_metrics.items():
                        mlflow.log_metric(f"{zone}_{metric_name}", value)

        return metrics

    def predict(
        self, data: DataFrame, target_zones: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """Generate predictions for the input data.

        Args:
            data: Input data for prediction
            target_zones: List of zones to generate predictions for. If None, uses all trained models

        Returns:
            Dictionary mapping zone names to their predictions

        Raises:
            ValueError: If no models have been trained yet
        """
        if not self.models:
            raise ValueError("No trained models found. Run train() first.")

        zones = target_zones or list(self.models.keys())
        X = self.preprocessor.get_feature_target_split(target_zones=zones)[0]
        predictions: Dict[str, np.ndarray] = {}

        for zone in zones:
            if zone in self.models:
                predictions[zone] = self.models[zone].predict(X)

        return predictions

    def register_best_models(self, metric: str = "rmse", target_zones: Optional[List[str]] = None) -> None:
        """Register the best models for each zone in Unity Catalog based on the specified metric.
        
        Args:
            metric: The metric to use for selecting the best model (e.g., "rmse", "mae")
            target_zones: List of zones to register models for. If None, uses all trained zones.
            
        Raises:
            ValueError: If no models have been trained yet or no metrics are available
        """
        if not hasattr(self, 'models') or not self.models:
            raise ValueError("No models have been trained yet. Call train() first.")
        
        if not hasattr(self, 'metrics') or not self.metrics:
            raise ValueError("No metrics available. Call evaluate() first.")
        
        zones_to_register = target_zones or self.preprocessor.config.target_zones
        
        for zone in zones_to_register:
            # Check if we have metrics for this zone
            if zone not in self.metrics:
                print(f"Warning: No metrics available for {zone}, skipping registration.")
                continue
            
            # Look for the specific metric in the zone's metrics
            zone_metrics = self.metrics[zone]
            if metric not in zone_metrics:
                print(f"Warning: Metric '{metric}' not available for {zone}, skipping registration.")
                continue
            
            try:
                # Get the model for this zone
                model = self.models.get(zone)
                if model is None:
                    print(f"Warning: No model found for {zone}, skipping registration.")
                    continue
                
                # Register the model with MLflow
                model_name = f"{zone.replace(' ', '_')}_forecaster"
                
                with mlflow.start_run(run_name=f"register_{model_name}"):
                    # Log metrics for this zone
                    for m_name, m_value in zone_metrics.items():
                        mlflow.log_metric(f"{zone}_{m_name}", m_value)
                    
                    # Register the model to Unity Catalog
                    registered_model_uri = mlflow.sklearn.log_model(
                        model,
                        artifact_path="model",
                        registered_model_name=f"models.power_forecasting.{model_name}"
                    )
                    
                    print(f"Registered model: {model_name}, URI: {registered_model_uri}")
            
            except Exception as e:
                print(f"Failed to register model for {zone}: {str(e)}")