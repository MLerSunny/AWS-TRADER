#!/usr/bin/env python
"""
Temporal Fusion Transformer (TFT) Training Script for MES Trader

This script trains a TFT model to predict 10%, 50%, and 90% quantiles for
the next 30 seconds of MES price returns using data from the feature store.
"""
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Union

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from feast import FeatureStore
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models import TemporalFusionTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
MARKET_ID = "MES"  # Micro E-mini S&P 500
FEATURE_REPO_PATH = "../feature_repo"
MODEL_OUTPUT_PATH = "tft_model.pt"
PREDICTION_HORIZON = 30  # 30 seconds forward for prediction
CONTEXT_LENGTH = 60  # Use 60 seconds of history
TRAIN_DAYS = 30  # Use 30 days of data for training
MAX_EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001
QUANTILES = [0.1, 0.5, 0.9]  # 10%, 50%, and 90% quantiles
DROPOUT = 0.1
HIDDEN_SIZE = 64
ATTENTION_HEAD_SIZE = 4
RANDOM_SEED = 42


def fetch_and_prepare_data():
    """
    Fetch data from the feature store and prepare it for TFT model training.
    
    Returns:
        pd.DataFrame: Prepared time series data with features and target
    """
    logger.info("Initializing feature store connection...")
    store = FeatureStore(repo_path=FEATURE_REPO_PATH)
    
    # Calculate date range for feature fetching
    end_date = datetime.now()
    start_date = end_date - timedelta(days=TRAIN_DAYS)
    
    logger.info(f"Fetching features from {start_date} to {end_date}...")
    
    # Create an entity DataFrame with timestamps for the past 30 days
    timestamps = pd.date_range(start=start_date, end=end_date, freq="1s")
    entities = pd.DataFrame({
        "market_id": [MARKET_ID] * len(timestamps),
        "event_timestamp": timestamps
    })
    
    # Define the features to fetch
    features = [
        "rsi_features:rsi_14",
        "rsi_features:rsi_14_trend",
        "atr_features:atr_14",
        "atr_features:atr_14_normalized",
        "vwap_features:vwap",
        "vwap_features:price_to_vwap",
        "vwap_features:vwap_trend",
        "order_book_features:imbalance",
        "order_book_features:bid_volume",
        "order_book_features:ask_volume",
        "order_book_features:imbalance_ma",
        "margin_features:initial_margin",
        "margin_features:margin_to_price_ratio"
    ]
    
    # Get historical features
    feature_data = store.get_historical_features(
        entity_df=entities,
        features=features
    ).to_df()
    
    logger.info(f"Retrieved {len(feature_data)} feature records")
    
    # Generate synthetic price data for demonstration
    # In a real application, this would be replaced with actual price data
    price_data = generate_synthetic_prices(feature_data.index, feature_data)
    
    # Combine features with price data
    data = pd.concat([feature_data, price_data], axis=1)
    
    # Calculate future returns (target)
    data["future_return"] = calculate_future_returns(data, PREDICTION_HORIZON)
    
    # Prepare the data for the TFT model
    data = prepare_for_tft(data)
    
    return data


def generate_synthetic_prices(index, feature_data):
    """
    Generate synthetic price data based on feature information.
    In a real scenario, this would be replaced with actual price data.
    
    Args:
        index: DataFrame index to match
        feature_data: Feature data to align with
    
    Returns:
        pd.DataFrame: Synthetic price data
    """
    n = len(index)
    base_price = 4500  # Base price for MES
    
    # Use the order book imbalance as a signal for price trend
    if 'order_book_features:imbalance' in feature_data.columns:
        imbalance = feature_data['order_book_features:imbalance'].fillna(0)
    else:
        imbalance = np.random.normal(0, 0.1, n)
    
    # Generate a random walk with drift influenced by imbalance
    daily_volatility = 0.01
    drift = imbalance * 0.5  # Scale imbalance effect
    price_changes = drift + np.random.normal(0, daily_volatility, n)
    
    # Calculate price series
    price_series = base_price * (1 + np.cumsum(price_changes))
    
    # Create price dataframe
    price_df = pd.DataFrame(index=index)
    price_df['price'] = price_series
    
    return price_df


def calculate_future_returns(df, horizon):
    """
    Calculate future returns for the given horizon.
    
    Args:
        df: DataFrame with price data
        horizon: Number of seconds to look forward
    
    Returns:
        pd.Series: Future returns
    """
    # Forward shift the price by the prediction horizon
    future_price = df['price'].shift(-horizon)
    
    # Calculate the return
    returns = (future_price - df['price']) / df['price']
    
    return returns


def prepare_for_tft(df):
    """
    Prepare the data for the TFT model.
    
    Args:
        df: DataFrame with features and target
    
    Returns:
        pd.DataFrame: Prepared data for TFT
    """
    # Drop NaNs from returns calculation
    df = df.dropna()
    
    # Create a time_idx column (required by TFT)
    # This is a continuous index starting from 0
    df = df.reset_index(drop=False)
    df = df.rename(columns={"event_timestamp": "timestamp"})
    df["time_idx"] = (df["timestamp"] - df["timestamp"].min()).dt.total_seconds().astype(int)
    
    # Convert boolean columns to integers
    for col in df.select_dtypes(include=['bool']).columns:
        df[col] = df[col].astype(int)
    
    # Ensure market_id is a string
    df["market_id"] = df["market_id"].astype(str)
    
    # For TFT, we need a group_id column
    df["group_id"] = df["market_id"]
    
    logger.info(f"Prepared data has {len(df)} rows and {df.shape[1]} columns")
    
    return df


def create_tft_datasets(data):
    """
    Create training and validation datasets for the TFT model.
    
    Args:
        data: Prepared DataFrame
    
    Returns:
        tuple: (training_data, validation_data)
    """
    # Define the max prediction length
    max_prediction_length = PREDICTION_HORIZON
    
    # Define the max encoder length (context)
    max_encoder_length = CONTEXT_LENGTH
    
    # Training cutoff (80% of data)
    training_cutoff = int(len(data) * 0.8)
    
    # Variables that are known in the future (none in this case)
    future_variables = []
    
    # Define the target and time-varying variables
    target = "future_return"
    time_varying_known_categoricals = []
    time_varying_known_reals = [
        "time_idx",
        "rsi_features:rsi_14",
        "rsi_features:rsi_14_trend",
        "atr_features:atr_14",
        "atr_features:atr_14_normalized",
        "vwap_features:vwap",
        "vwap_features:price_to_vwap",
        "vwap_features:vwap_trend",
        "order_book_features:imbalance",
        "order_book_features:bid_volume",
        "order_book_features:ask_volume",
        "order_book_features:imbalance_ma",
        "margin_features:initial_margin",
        "margin_features:margin_to_price_ratio",
        "price"
    ]
    time_varying_unknown_categoricals = []
    time_varying_unknown_reals = [target]
    
    # Static variables (constant for a given group)
    static_categoricals = ["group_id"]
    static_reals = []
    
    # Define variable groups
    variable_groups = {}
    
    # Create the training dataset
    training = TimeSeriesDataSet(
        data=data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target=target,
        group_ids=["group_id"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=static_categoricals,
        static_reals=static_reals,
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_categoricals=time_varying_unknown_categoricals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        variable_groups=variable_groups,
        target_normalizer=GroupNormalizer(
            groups=["group_id"], transformation="softplus"
        ),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    # Create the validation dataset
    validation = TimeSeriesDataSet.from_dataset(
        training, data, min_prediction_idx=training_cutoff + 1
    )
    
    # Create data loaders
    train_dataloader = training.to_dataloader(
        train=True, batch_size=BATCH_SIZE, num_workers=0, shuffle=True
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=BATCH_SIZE, num_workers=0, shuffle=False
    )
    
    return training, train_dataloader, val_dataloader


def train_tft_model(training, train_dataloader, val_dataloader):
    """
    Train the TFT model.
    
    Args:
        training: Training dataset
        train_dataloader: Training data loader
        val_dataloader: Validation data loader
    
    Returns:
        TemporalFusionTransformer: Trained model
    """
    logger.info("Creating and training TFT model...")
    
    # Set random seed for reproducibility
    pl.seed_everything(RANDOM_SEED)
    
    # Define the loss function for quantile regression
    loss = QuantileLoss(quantiles=QUANTILES)
    
    # Create the TFT model
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=LEARNING_RATE,
        hidden_size=HIDDEN_SIZE,
        attention_head_size=ATTENTION_HEAD_SIZE,
        dropout=DROPOUT,
        loss=loss,
        log_interval=10,
        reduce_on_plateau_patience=3,
    )
    
    # Define callbacks for training
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
    )
    
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename="tft-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        callbacks=[early_stop_callback, checkpoint_callback],
        gradient_clip_val=0.1,
        limit_train_batches=50,  # For faster training in this example
    )
    
    # Train the model
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    
    # Load the best model
    best_model_path = checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    
    return best_tft


def save_model(model):
    """
    Save the trained model.
    
    Args:
        model: Trained TFT model
    """
    logger.info(f"Saving model to {MODEL_OUTPUT_PATH}")
    torch.save(model.state_dict(), MODEL_OUTPUT_PATH)


def main():
    """Main execution function."""
    try:
        logger.info("Starting TFT model training process...")
        
        # Fetch and prepare data
        data = fetch_and_prepare_data()
        
        # Create TFT datasets
        training, train_dataloader, val_dataloader = create_tft_datasets(data)
        
        # Train the model
        model = train_tft_model(training, train_dataloader, val_dataloader)
        
        # Save the model
        save_model(model)
        
        logger.info("Training process completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during model training: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main() 