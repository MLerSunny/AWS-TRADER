#!/usr/bin/env python
"""
LightGBM Training Script for MES Trader

This script pulls 30 days of features from the feature store, trains a LightGBM classifier
to predict price movement direction (up/down) in the next 30 seconds, saves the model,
and logs SHAP values for model interpretation.
"""
import logging
import os
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from functools import lru_cache

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from feast import FeatureStore
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
MARKET_ID = "MES"  # Micro E-mini S&P 500
FEATURE_REPO_PATH = "../feature_repo"
MODEL_OUTPUT_PATH = "lgbm_model.pkl"
SHAP_OUTPUT_PATH = "../../docs/shap_last.png"
PREDICTION_HORIZON = 30  # 30 seconds forward for prediction
TRAIN_DAYS = 30  # Use 30 days of data for training
CACHE_DIR = ".cache"

# Ensure output directories exist
os.makedirs(os.path.dirname(SHAP_OUTPUT_PATH), exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


def create_training_label(df, forward_periods=PREDICTION_HORIZON):
    """
    Create binary classification labels for price movement direction.
    
    Args:
        df (pd.DataFrame): DataFrame containing price information
        forward_periods (int): Number of seconds to look forward
    
    Returns:
        pd.Series: Binary labels (1 for price up, 0 for price down or unchanged)
    """
    # Forward shift the close price by the prediction horizon
    df['future_price'] = df['close'].shift(-forward_periods)
    
    # Calculate the return
    df['future_return'] = (df['future_price'] - df['close']) / df['close']
    
    # Create binary labels
    df['label'] = (df['future_return'] > 0).astype(int)
    
    return df['label']


@lru_cache(maxsize=1)
def get_feature_store():
    """Get and cache the feature store connection."""
    logger.info("Initializing feature store connection...")
    try:
        store = FeatureStore(repo_path=FEATURE_REPO_PATH)
        return store
    except Exception as e:
        logger.error(f"Failed to initialize feature store: {e}")
        raise


def fetch_training_data(use_cache=True):
    """
    Fetch 30 days of historical features from Feast feature store.
    
    Args:
        use_cache (bool): Whether to use cached data if available
        
    Returns:
        pd.DataFrame: Combined features with price data and labels
    """
    cache_file = os.path.join(CACHE_DIR, f"training_data_{TRAIN_DAYS}d.pkl")
    
    # Return cached data if it exists and is recent (< 1 day old)
    if use_cache and os.path.exists(cache_file):
        file_age = time.time() - os.path.getmtime(cache_file)
        if file_age < 86400:  # 24 hours in seconds
            logger.info(f"Loading cached training data from {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cached data: {e}")
                # Continue to fetch new data
    
    # If no cache or cache is stale, fetch new data
    try:
        # Get feature store
        store = get_feature_store()
        
        # Calculate date range for feature fetching
        end_date = datetime.now()
        start_date = end_date - timedelta(days=TRAIN_DAYS)
        
        logger.info(f"Fetching features from {start_date} to {end_date}...")
        
        # Create an entity DataFrame with timestamps for the past 30 days
        timestamps = pd.date_range(start=start_date, end=end_date, freq="1min")
        entities = pd.DataFrame({
            "market_id": [MARKET_ID] * len(timestamps),
            "event_timestamp": timestamps
        })
        
        # Define the features to fetch
        features = [
            "rsi_features:rsi_14",
            "rsi_features:rsi_14_trend",
            "rsi_features:rsi_14_overbought",
            "rsi_features:rsi_14_oversold",
            "atr_features:atr_14",
            "atr_features:atr_14_normalized",
            "atr_features:atr_14_percentile",
            "vwap_features:vwap",
            "vwap_features:price_to_vwap",
            "vwap_features:vwap_trend",
            "order_book_features:imbalance",
            "order_book_features:bid_volume",
            "order_book_features:ask_volume",
            "order_book_features:imbalance_ma",
            "margin_features:initial_margin",
            "margin_features:maintenance_margin",
            "margin_features:margin_change_1d",
            "margin_features:margin_to_price_ratio"
        ]
        
        # Get historical features
        feature_data = store.get_historical_features(
            entity_df=entities,
            features=features
        ).to_df()
        
        logger.info(f"Retrieved {len(feature_data)} feature records")
        
        # Clean up NaN values in feature data
        feature_data = handle_missing_values(feature_data)
        
        # Fetch OHLCV data (simplified - in a real scenario, this would come from a database)
        # For this example, we'll generate synthetic price data aligned with our features
        ohlcv_data = generate_synthetic_prices(feature_data.index, feature_data)
        
        # Combine features with price data
        combined_data = pd.concat([feature_data, ohlcv_data], axis=1)
        
        # Create labels for prediction
        combined_data['label'] = create_training_label(combined_data)
        
        # Drop rows with NaN (from the forward shift in label creation)
        combined_data = combined_data.dropna()
        
        logger.info(f"Final training dataset has {len(combined_data)} rows and {combined_data.shape[1]} columns")
        
        # Cache the data
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(combined_data, f)
            logger.info(f"Cached training data to {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to cache training data: {e}")
        
        return combined_data
        
    except Exception as e:
        logger.error(f"Error fetching training data: {e}")
        raise


def handle_missing_values(df):
    """
    Handle missing values in feature data.
    
    Args:
        df (pd.DataFrame): Feature DataFrame with potential NaN values
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    # Check for NaN values
    original_nan_count = df.isna().sum().sum()
    if original_nan_count:
        logger.info(f"Found {original_nan_count} NaN values in feature data")
    
    # For numeric features, forward-fill, then backward-fill
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(method='ffill')
    df[numeric_cols] = df[numeric_cols].fillna(method='bfill')
    
    # For any remaining NaNs, use column mean
    for col in numeric_cols:
        if df[col].isna().any():
            col_mean = df[col].mean()
            df[col] = df[col].fillna(col_mean)
    
    # For categorical/boolean features, fill with most frequent value
    cat_cols = df.select_dtypes(include=['object', 'bool']).columns
    for col in cat_cols:
        if df[col].isna().any():
            most_frequent = df[col].mode()[0]
            df[col] = df[col].fillna(most_frequent)
    
    # Verify all NaNs are handled
    remaining_nan_count = df.isna().sum().sum()
    if remaining_nan_count:
        logger.warning(f"Still have {remaining_nan_count} NaN values after cleaning")
    else:
        logger.info("All NaN values have been handled")
    
    return df


def generate_synthetic_prices(index, feature_data):
    """
    Generate synthetic price data based on feature information.
    In a real scenario, this would be replaced with actual price data.
    
    Args:
        index: DataFrame index to match
        feature_data: Feature data to align with
    
    Returns:
        pd.DataFrame: Synthetic OHLCV data
    """
    n = len(index)
    base_price = 4500  # Base price for MES
    
    # Use the order book imbalance as a signal for price trend
    if 'order_book_features:imbalance' in feature_data.columns:
        imbalance = feature_data['order_book_features:imbalance'].fillna(0)
    else:
        imbalance = np.random.normal(0, 0.1, n)
    
    # Ensure imbalance has no NaN values
    imbalance = imbalance.fillna(0)
    
    # Generate a random walk with drift influenced by imbalance
    daily_volatility = 0.01
    drift = imbalance * 0.5  # Scale imbalance effect
    
    # Handle potential NaN values in drift calculation
    drift = np.array(drift)
    np.nan_to_num(drift, copy=False, nan=0.0)
    
    price_changes = drift + np.random.normal(0, daily_volatility, n)
    
    # Calculate price series
    price_series = base_price * (1 + np.cumsum(price_changes))
    
    # Create OHLCV dataframe
    ohlcv = pd.DataFrame(index=index)
    ohlcv['open'] = price_series
    ohlcv['high'] = price_series * (1 + np.abs(np.random.normal(0, 0.001, n)))
    ohlcv['low'] = price_series * (1 - np.abs(np.random.normal(0, 0.001, n)))
    ohlcv['close'] = price_series * (1 + np.random.normal(0, 0.0005, n))
    ohlcv['volume'] = np.random.lognormal(4, 1, n)
    
    return ohlcv


def train_lightgbm_model(data):
    """
    Train a LightGBM classifier model.
    
    Args:
        data (pd.DataFrame): Training data with features and labels
    
    Returns:
        lgb.Booster: Trained LightGBM model
    """
    logger.info("Preparing training and validation datasets...")
    
    # Separate features and target
    y = data['label']
    X = data.drop(['label', 'future_price', 'future_return'], axis=1, errors='ignore')
    
    # Convert boolean features to integers
    for col in X.select_dtypes(include=['bool']).columns:
        X[col] = X[col].astype(int)
    
    # Check for and handle any remaining NaN values
    if X.isna().any().any():
        logger.warning("Found NaN values in features, applying cleaning")
        X = handle_missing_values(X)
    
    # Use time series split for financial data
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Prepare for Cross-Validation
    cv_scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Define parameters with early stopping
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': -1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': 42,
            'early_stopping_rounds': 50
        }
        
        # Train the model
        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50)]
        )
        
        # Evaluate the model
        y_pred_proba = model.predict(X_val)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_pred)
        
        # Calculate AUC if applicable
        try:
            auc = roc_auc_score(y_val, y_pred_proba)
            cv_scores.append((accuracy, auc))
            logger.info(f"Fold accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
        except Exception:
            cv_scores.append((accuracy, None))
            logger.info(f"Fold accuracy: {accuracy:.4f}")
    
    # Calculate average scores
    avg_accuracy = np.mean([score[0] for score in cv_scores])
    logger.info(f"Average cross-validation accuracy: {avg_accuracy:.4f}")
    
    # Final model training on all data
    logger.info("Training final model on all data...")
    train_data_all = lgb.Dataset(X, label=y)
    
    final_model = lgb.train(
        params,
        train_data_all,
        num_boost_round=500
    )
    
    # Use the last fold for evaluation and SHAP values
    _, X_final_val = X.iloc[train_idx], X.iloc[val_idx]
    _, y_final_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # Final evaluation
    final_pred_proba = final_model.predict(X_final_val)
    final_pred = (final_pred_proba > 0.5).astype(int)
    
    final_accuracy = accuracy_score(y_final_val, final_pred)
    logger.info(f"Final model validation accuracy: {final_accuracy:.4f}")
    logger.info(f"Classification report:\n{classification_report(y_final_val, final_pred)}")
    
    # Log confusion matrix
    cm = confusion_matrix(y_final_val, final_pred)
    logger.info(f"Confusion matrix:\n{cm}")
    
    return final_model, X_final_val


def generate_shap_values(model, X_val):
    """
    Generate and save SHAP values visualization.
    
    Args:
        model (lgb.Booster): Trained LightGBM model
        X_val (pd.DataFrame): Validation data features
    """
    logger.info("Generating SHAP values...")
    
    try:
        # Create explainer
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_val)
        
        # Create and save SHAP summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_val, plot_type="bar", show=False)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(SHAP_OUTPUT_PATH, dpi=300, bbox_inches='tight')
        logger.info(f"SHAP values plot saved to {SHAP_OUTPUT_PATH}")
        
        # Also save feature importance plot
        plt.figure(figsize=(12, 8))
        lgb.plot_importance(model, max_num_features=20)
        plt.tight_layout()
        plt.savefig(SHAP_OUTPUT_PATH.replace('shap_last.png', 'feature_importance.png'), dpi=300)
        logger.info("Feature importance plot saved")
        
        # Close the plots to free memory
        plt.close('all')
    except Exception as e:
        logger.error(f"Error generating SHAP values: {e}")
        # Continue with model saving even if SHAP fails


def save_model(model):
    """
    Save the trained model to disk.
    
    Args:
        model (lgb.Booster): Trained LightGBM model
    """
    try:
        # Save as pickle file
        with open(MODEL_OUTPUT_PATH, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {MODEL_OUTPUT_PATH}")
        
        # Also save as native LightGBM format
        lgb_path = MODEL_OUTPUT_PATH.replace('.pkl', '.lgb')
        model.save_model(lgb_path)
        logger.info(f"Model also saved in LightGBM format to {lgb_path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise


def main():
    """Main execution function."""
    start_time = time.time()
    try:
        logger.info("Starting LightGBM model training process...")
        
        # Fetch training data
        data = fetch_training_data()
        
        # Train model
        model, X_val = train_lightgbm_model(data)
        
        # Generate SHAP values
        generate_shap_values(model, X_val)
        
        # Save model
        save_model(model)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Training process completed successfully in {elapsed_time:.2f} seconds!")
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Error during model training after {elapsed_time:.2f} seconds: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main() 