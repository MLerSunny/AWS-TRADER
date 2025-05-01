#!/usr/bin/env python
"""
Data generator for populating feature stores with sample data.
This is a utility script for development and testing purposes.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pyarrow as pa
import pyarrow.parquet as pq

# Ensure the output directory exists
OUTPUT_DIR = "sample_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Constants
MARKET_ID = "MES"  # Micro E-mini S&P 500
START_DATE = datetime.now() - timedelta(days=30)
END_DATE = datetime.now()
FREQ = "1min"  # 1-minute bars


def generate_timestamps(start_date, end_date, freq):
    """Generate a range of timestamps."""
    return pd.date_range(start=start_date, end=end_date, freq=freq)


def generate_rsi_data(timestamps):
    """Generate sample RSI data."""
    n = len(timestamps)
    
    # Create DataFrame
    df = pd.DataFrame({
        "market_id": [MARKET_ID] * n,
        "event_timestamp": timestamps,
        "created_timestamp": [datetime.now()] * n,
        
        # RSI typically ranges from 0 to 100
        "rsi_14": np.random.uniform(20, 80, n),
    })
    
    # Derive additional features
    df["rsi_14_trend"] = np.where(df["rsi_14"].diff() > 0, 1, 
                          np.where(df["rsi_14"].diff() < 0, -1, 0))
    df["rsi_14_overbought"] = df["rsi_14"] > 70
    df["rsi_14_oversold"] = df["rsi_14"] < 30
    
    return df


def generate_atr_data(timestamps):
    """Generate sample ATR data."""
    n = len(timestamps)
    price = 4500  # Base price for MES
    
    # Create DataFrame
    df = pd.DataFrame({
        "market_id": [MARKET_ID] * n,
        "event_timestamp": timestamps,
        "created_timestamp": [datetime.now()] * n,
        
        # ATR values (typically a few points for MES)
        "atr_14": np.random.uniform(5, 15, n),
    })
    
    # Derive additional features
    df["atr_14_normalized"] = df["atr_14"] / price * 100  # ATR as percentage of price
    df["atr_14_percentile"] = np.random.uniform(0, 1, n)  # Percentile rank
    
    return df


def generate_vwap_data(timestamps):
    """Generate sample VWAP data."""
    n = len(timestamps)
    base_price = 4500  # Base price for MES
    
    # Generate prices with some randomness
    prices = base_price + np.cumsum(np.random.normal(0, 3, n))
    
    # Create DataFrame
    df = pd.DataFrame({
        "market_id": [MARKET_ID] * n,
        "event_timestamp": timestamps,
        "created_timestamp": [datetime.now()] * n,
        
        # VWAP typically close to current price but with less volatility
        "vwap": prices - np.random.normal(0, 1, n),
    })
    
    # Current price (simulated)
    df["current_price"] = prices
    
    # Derive additional features
    df["price_to_vwap"] = df["current_price"] / df["vwap"]
    df["vwap_trend"] = np.where(df["vwap"].diff() > 0, 1, 
                      np.where(df["vwap"].diff() < 0, -1, 0))
    
    # Remove the temporary column
    df.drop("current_price", axis=1, inplace=True)
    
    return df


def generate_ob_imbalance_data(timestamps):
    """Generate sample order book imbalance data."""
    n = len(timestamps)
    
    # Create DataFrame
    df = pd.DataFrame({
        "market_id": [MARKET_ID] * n,
        "event_timestamp": timestamps,
        "created_timestamp": [datetime.now()] * n,
        
        # Bid and ask volumes (random but realistic for MES)
        "bid_volume": np.random.uniform(10, 100, n),
        "ask_volume": np.random.uniform(10, 100, n),
    })
    
    # Derive imbalance (-1 to 1)
    total_volume = df["bid_volume"] + df["ask_volume"]
    df["imbalance"] = (df["bid_volume"] - df["ask_volume"]) / total_volume
    
    # Moving average of imbalance (5-period simple MA)
    df["imbalance_ma"] = df["imbalance"].rolling(5, min_periods=1).mean()
    
    return df


def generate_margin_data(timestamps):
    """Generate sample margin data."""
    n = len(timestamps)
    contract_value = 4500 * 50  # MES is $50 * price
    
    # Create DataFrame
    df = pd.DataFrame({
        "market_id": [MARKET_ID] * n,
        "event_timestamp": timestamps,
        "created_timestamp": [datetime.now()] * n,
        
        # Margin requirements (realistic for MES)
        "initial_margin": [12000] * n,  # Example initial margin
        "maintenance_margin": [10000] * n,  # Example maintenance margin
    })
    
    # Add some variation over time
    time_variation = np.sin(np.linspace(0, 10, n)) * 500
    df["initial_margin"] = df["initial_margin"] + time_variation
    df["maintenance_margin"] = df["maintenance_margin"] + time_variation
    
    # Derive additional features
    df["margin_change_1d"] = df["initial_margin"].diff(24).fillna(0)  # Assumes 24 entries per day
    df["margin_to_price_ratio"] = df["initial_margin"] / contract_value * 100
    
    return df


def save_to_parquet(df, filename):
    """Save DataFrame to Parquet file."""
    full_path = os.path.join(OUTPUT_DIR, filename)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, full_path)
    print(f"Saved {len(df)} rows to {full_path}")


def main():
    """Generate and save all feature data."""
    timestamps = generate_timestamps(START_DATE, END_DATE, FREQ)
    
    # Generate and save each feature set
    save_to_parquet(generate_rsi_data(timestamps), "rsi_features.parquet")
    save_to_parquet(generate_atr_data(timestamps), "atr_features.parquet")
    save_to_parquet(generate_vwap_data(timestamps), "vwap_features.parquet")
    save_to_parquet(generate_ob_imbalance_data(timestamps), "ob_imbalance_features.parquet")
    save_to_parquet(generate_margin_data(timestamps), "margin_features.parquet")


if __name__ == "__main__":
    main() 