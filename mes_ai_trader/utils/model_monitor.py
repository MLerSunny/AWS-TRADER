"""
Model monitoring utilities for tracking model performance and drift over time.

This module provides tools to:
1. Track model predictions vs. actual outcomes
2. Detect concept drift in input features
3. Monitor feature importance changes
4. Schedule model retraining when needed
"""
import json
import logging
import os
import time
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

# Import configuration
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import config

logger = logging.getLogger(__name__)

class DriftStatus(Enum):
    """Model or feature drift status."""
    NO_DRIFT = "no_drift"
    MILD_DRIFT = "mild_drift"
    SEVERE_DRIFT = "severe_drift"


class ModelMonitor:
    """
    Monitor model performance and detect drift.
    
    This class maintains a historical record of predictions, 
    outcomes, feature distributions, and performance metrics.
    """
    
    def __init__(
        self,
        model_name: str,
        feature_names: List[str],
        reference_data: Optional[pd.DataFrame] = None,
        metrics_dir: Optional[str] = None,
        retraining_threshold: float = 0.05,
    ):
        """
        Initialize the model monitor.
        
        Args:
            model_name: Name of the model to monitor
            feature_names: List of feature names
            reference_data: Baseline data for drift detection
            metrics_dir: Directory to store monitoring data
            retraining_threshold: Threshold for triggering retraining
        """
        self.model_name = model_name
        self.feature_names = feature_names
        self.retraining_threshold = retraining_threshold
        
        # Setup metrics directory
        self.metrics_dir = metrics_dir or Path(config.get("METRICS_DIR", "/tmp/model_metrics"))
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Path to the model metrics file
        self.metrics_file = Path(self.metrics_dir) / f"{model_name}_metrics.csv"
        self.shap_history_file = Path(self.metrics_dir) / f"{model_name}_shap_history.csv"
        
        # Initialize metrics dataframe
        if os.path.exists(self.metrics_file):
            self.metrics_df = pd.read_csv(self.metrics_file)
        else:
            self.metrics_df = pd.DataFrame(columns=[
                "timestamp", "accuracy", "precision", "recall", "f1_score", 
                "drift_status", "sample_size", "window_start", "window_end"
            ])
        
        # Initialize feature distribution reference data
        self.reference_data = reference_data
        
        # Feature importance history
        if os.path.exists(self.shap_history_file):
            self.shap_history = pd.read_csv(self.shap_history_file)
        else:
            self.shap_history = pd.DataFrame(
                columns=["timestamp"] + feature_names
            )
        
        # Current window metrics
        self.current_window = {
            "predictions": [],
            "actuals": [],
            "features": [],
            "start_time": datetime.now(),
        }
        
        # Scheduled retraining status
        self.retraining_needed = False
        self.retraining_reason = None
        
        logger.info(f"Initialized model monitor for {model_name}")
    
    def record_prediction(
        self,
        features: Union[Dict, List, np.ndarray, pd.DataFrame],
        prediction: Union[float, List[float], np.ndarray],
        actual: Optional[Union[float, int]] = None,
        timestamp: Optional[datetime] = None,
    ):
        """
        Record a prediction with its features and optional actual outcome.
        
        Args:
            features: Input features for the prediction
            prediction: Model's prediction
            actual: Actual outcome (if available)
            timestamp: Time of prediction (defaults to now)
        """
        # Standardize inputs
        if isinstance(features, pd.DataFrame):
            features = features.iloc[0].to_dict() if len(features) > 0 else {}
        elif isinstance(features, np.ndarray):
            features = {name: value for name, value in zip(self.feature_names, features.flatten())}
        
        if isinstance(prediction, np.ndarray):
            prediction = prediction.tolist()
            
        if isinstance(actual, np.ndarray):
            actual = actual.item() if actual.size == 1 else actual.tolist()
            
        # Record the prediction
        timestamp = timestamp or datetime.now()
        
        self.current_window["predictions"].append(prediction)
        self.current_window["actuals"].append(actual)
        self.current_window["features"].append(features)
        
        # Check if it's time to compute window metrics
        window_size = len(self.current_window["predictions"])
        if window_size >= 1000:  # Process metrics every 1000 predictions
            self.compute_window_metrics()
            
        # Periodically check for drift
        if window_size % 100 == 0:
            self.check_drift()
    
    def record_actual(self, prediction_index: int, actual_value: Union[float, int]):
        """
        Record an actual outcome for a previous prediction.
        
        Args:
            prediction_index: Index of the prediction in the current window
            actual_value: The actual outcome
        """
        if prediction_index < len(self.current_window["actuals"]):
            self.current_window["actuals"][prediction_index] = actual_value
    
    def compute_window_metrics(self):
        """
        Compute performance metrics for the current window and save them.
        """
        if not self.current_window["predictions"]:
            return  # No data to process
            
        # Create a dataframe of predictions and actuals
        window_data = pd.DataFrame({
            "prediction": self.current_window["predictions"],
            "actual": self.current_window["actuals"]
        })
        
        # Filter out rows where actuals are None
        window_data = window_data.dropna(subset=["actual"])
        
        if len(window_data) == 0:
            logger.warning("No actual values available to compute metrics")
            return
        
        # For classification models
        if all(isinstance(p, (list, tuple)) for p in window_data["prediction"]):
            # Get class with highest probability
            window_data["prediction_class"] = window_data["prediction"].apply(
                lambda x: np.argmax(x) if isinstance(x, (list, tuple)) else x
            )
        else:
            # For binary classification
            window_data["prediction_class"] = (window_data["prediction"] > 0.5).astype(int)
        
        # Compute metrics
        accuracy = np.mean(window_data["prediction_class"] == window_data["actual"])
        
        # Compute precision, recall, f1 for binary classification
        if set(window_data["actual"].unique()) == {0, 1} and set(window_data["prediction_class"].unique()) <= {0, 1}:
            from sklearn.metrics import precision_score, recall_score, f1_score
            
            precision = precision_score(
                window_data["actual"], 
                window_data["prediction_class"],
                zero_division=0
            )
            recall = recall_score(
                window_data["actual"], 
                window_data["prediction_class"],
                zero_division=0
            )
            f1 = f1_score(
                window_data["actual"], 
                window_data["prediction_class"],
                zero_division=0
            )
        else:
            # For non-binary classification or regression
            precision = recall = f1 = None
        
        # Get drift status
        drift_status = self.check_drift()
        
        # Record metrics
        now = datetime.now()
        metrics = {
            "timestamp": now,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "drift_status": drift_status.value,
            "sample_size": len(window_data),
            "window_start": self.current_window["start_time"],
            "window_end": now
        }
        
        # Add to metrics dataframe
        self.metrics_df = pd.concat([
            self.metrics_df, 
            pd.DataFrame([metrics])
        ], ignore_index=True)
        
        # Save metrics
        self.metrics_df.to_csv(self.metrics_file, index=False)
        
        # Reset window
        self.current_window = {
            "predictions": [],
            "actuals": [],
            "features": [],
            "start_time": now,
        }
        
        # Check if performance degradation requires retraining
        self.check_retraining_needed()
        
        logger.info(
            f"Computed metrics for {self.model_name}: "
            f"accuracy={accuracy:.3f}, "
            f"drift_status={drift_status.value}"
        )
    
    def check_drift(self) -> DriftStatus:
        """
        Check for concept drift in the feature distributions.
        
        Returns:
            DriftStatus: The detected drift status
        """
        if not self.reference_data is not None or not self.current_window["features"]:
            return DriftStatus.NO_DRIFT
        
        # Convert current window features to DataFrame
        current_features = pd.DataFrame(self.current_window["features"])
        
        # Only keep common columns
        common_features = list(set(current_features.columns) & set(self.reference_data.columns))
        
        if not common_features:
            logger.warning("No common features between reference and current data")
            return DriftStatus.NO_DRIFT
        
        # Count features with significant drift
        drifted_features = 0
        severe_drifted_features = 0
        
        for feature in common_features:
            # Skip non-numeric features
            if not pd.api.types.is_numeric_dtype(current_features[feature]) or \
               not pd.api.types.is_numeric_dtype(self.reference_data[feature]):
                continue
            
            # Get distributions
            ref_values = self.reference_data[feature].dropna().values
            current_values = current_features[feature].dropna().values
            
            if len(current_values) < 10:
                # Not enough data
                continue
            
            # Perform Kolmogorov-Smirnov test
            ks_statistic, p_value = stats.ks_2samp(ref_values, current_values)
            
            # Check drift magnitude
            if p_value < 0.01:
                severe_drifted_features += 1
                drifted_features += 1
            elif p_value < 0.05:
                drifted_features += 1
        
        # Determine drift status
        if severe_drifted_features >= len(common_features) * 0.3:
            return DriftStatus.SEVERE_DRIFT
        elif drifted_features >= len(common_features) * 0.2:
            return DriftStatus.MILD_DRIFT
        else:
            return DriftStatus.NO_DRIFT
    
    def update_shap_values(self, shap_values: Dict[str, float]):
        """
        Record SHAP values to track feature importance over time.
        
        Args:
            shap_values: Dictionary mapping feature names to SHAP values
        """
        # Record timestamp
        shap_values["timestamp"] = datetime.now()
        
        # Add to history
        self.shap_history = pd.concat([
            self.shap_history,
            pd.DataFrame([shap_values])
        ], ignore_index=True)
        
        # Save history
        self.shap_history.to_csv(self.shap_history_file, index=False)
        
        # Check for changes in feature importance ranking
        self.check_feature_importance_shift(shap_values)
    
    def check_feature_importance_shift(self, current_shap: Dict[str, float]):
        """
        Check if there's a significant shift in feature importance.
        
        Args:
            current_shap: Dictionary of latest SHAP values
        """
        if len(self.shap_history) < 5:
            return  # Need more history
        
        # Get historical SHAP values, excluding timestamp
        historical = self.shap_history.drop(columns=["timestamp"])
        
        if len(historical.columns) < 2:
            return  # Need at least 2 features
        
        # Compute average historical importance for each feature
        avg_historical = historical.mean().sort_values(ascending=False)
        
        # Get top 5 features historically
        top_historical = avg_historical.index[:5].tolist()
        
        # Get top 5 features from current SHAP values
        current_shap_series = pd.Series({k: v for k, v in current_shap.items() if k != "timestamp"})
        top_current = current_shap_series.sort_values(ascending=False).index[:5].tolist()
        
        # Check for significant changes in top features
        if len(set(top_historical) - set(top_current)) >= 2:
            logger.warning(
                f"Significant shift in feature importance detected for {self.model_name}. "
                f"Historical top features: {top_historical}, "
                f"Current top features: {top_current}"
            )
            
            self.retraining_needed = True
            self.retraining_reason = "feature_importance_shift"
    
    def check_retraining_needed(self) -> bool:
        """
        Check if model retraining is needed based on performance metrics.
        
        Returns:
            bool: True if retraining is needed
        """
        if len(self.metrics_df) < 5:
            return False  # Not enough history
            
        # Get recent metrics
        recent_metrics = self.metrics_df.sort_values("timestamp").tail(5)
        
        # Check for performance degradation
        if "accuracy" in recent_metrics.columns:
            baseline_accuracy = recent_metrics["accuracy"].iloc[0]
            current_accuracy = recent_metrics["accuracy"].iloc[-1]
            
            if baseline_accuracy - current_accuracy > self.retraining_threshold:
                logger.warning(
                    f"Performance degradation detected for {self.model_name}. "
                    f"Baseline accuracy: {baseline_accuracy:.3f}, "
                    f"Current accuracy: {current_accuracy:.3f}"
                )
                
                self.retraining_needed = True
                self.retraining_reason = "performance_degradation"
                return True
        
        # Check for severe drift
        if "drift_status" in recent_metrics.columns:
            severe_drift_count = sum(
                status == DriftStatus.SEVERE_DRIFT.value 
                for status in recent_metrics["drift_status"].tail(3)
            )
            
            if severe_drift_count >= 2:
                logger.warning(
                    f"Persistent severe drift detected for {self.model_name}."
                )
                
                self.retraining_needed = True
                self.retraining_reason = "severe_drift"
                return True
        
        return self.retraining_needed
    
    def schedule_retraining(self):
        """
        Schedule model retraining by writing to a retraining queue.
        """
        if not self.retraining_needed:
            return
            
        # Create retraining queue directory
        retraining_dir = Path(self.metrics_dir) / "retraining_queue"
        os.makedirs(retraining_dir, exist_ok=True)
        
        # Create retraining request file
        request_file = retraining_dir / f"{self.model_name}_{int(time.time())}.json"
        
        request_data = {
            "model_name": self.model_name,
            "reason": self.retraining_reason,
            "requested_at": datetime.now().isoformat(),
            "metrics_snapshot": self.metrics_df.tail(5).to_dict(orient="records"),
        }
        
        with open(request_file, "w") as f:
            json.dump(request_data, f, indent=2)
            
        logger.info(f"Scheduled retraining for {self.model_name} due to {self.retraining_reason}")
        
        # Reset flag
        self.retraining_needed = False
        self.retraining_reason = None 