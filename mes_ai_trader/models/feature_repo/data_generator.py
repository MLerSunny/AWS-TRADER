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
import json
import time
import requests
from typing import Dict, List, Any

# Ensure the output directory exists
OUTPUT_DIR = "sample_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Constants
MARKET_ID = "MES"  # Micro E-mini S&P 500
START_DATE = datetime.now() - timedelta(days=30)
END_DATE = datetime.now()
FREQ = "1min"  # 1-minute bars

# Set environment variable for Feast
os.environ["FEAST_USAGE"] = "False"

# Configure constants
PRICING_API_URL = os.environ.get("PRICING_API_URL", "https://api.cmegroup.com/pricinginformation/v1/margins")
PRICING_API_KEY = os.environ.get("PRICING_API_KEY", "")
PRICING_REFRESH_INTERVAL_HOURS = int(os.environ.get("PRICING_REFRESH_INTERVAL_HOURS", "1"))


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


def retrieve_market_data(start_date: datetime.datetime, end_date: datetime.datetime) -> pd.DataFrame:
    """
    Retrieve market data for the given time range.
    This is a placeholder - in production, this would retrieve from a real data source.
    
    Args:
        start_date: Start date for data retrieval
        end_date: End date for data retrieval
        
    Returns:
        DataFrame with market data
    """
    # Generate synthetic data
    date_range = pd.date_range(start=start_date, end=end_date, freq="1min")
    
    # Create base dataframe with timestamps
    df = pd.DataFrame({"event_timestamp": date_range})
    
    # Add synthetic market data
    df["symbol"] = "MES"
    df["price"] = 4500 + np.random.normal(0, 20, size=len(date_range)).cumsum()
    df["volume"] = np.random.poisson(100, size=len(date_range))
    df["bid"] = df["price"] - np.random.normal(0.5, 0.1, size=len(date_range))
    df["ask"] = df["price"] + np.random.normal(0.5, 0.1, size=len(date_range))
    
    # Add derived features
    df["price_sma_5"] = df["price"].rolling(5).mean()
    df["price_sma_20"] = df["price"].rolling(20).mean()
    df["price_sma_50"] = df["price"].rolling(50).mean()
    df["price_rsi_14"] = calculate_rsi(df["price"], 14)
    
    # Fill NAs with forward fill, then backward fill for edge cases
    df = df.fillna(method="ffill").fillna(method="bfill")
    
    return df


def retrieve_cme_margin_requirements() -> Dict[str, Any]:
    """
    Retrieve CME margin requirements from the CME API.
    Refreshes hourly and caches the results.
    
    Returns:
        Dictionary with margin requirements for MES
    """
    cache_file = "/tmp/cme_margins.json"
    cache_time_file = "/tmp/cme_margins_time.txt"
    
    # Check if we have a recent cached version
    try:
        if os.path.exists(cache_time_file) and os.path.exists(cache_file):
            with open(cache_time_file, "r") as f:
                last_update = float(f.read().strip())
            
            # If cache is recent enough, use it
            if time.time() - last_update < (PRICING_REFRESH_INTERVAL_HOURS * 3600):
                with open(cache_file, "r") as f:
                    return json.load(f)
    except Exception as e:
        print(f"Error reading cache: {e}")
    
    # If no cache or expired, call the API
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {PRICING_API_KEY}" if PRICING_API_KEY else None
        }
        
        # Remove None values from headers
        headers = {k: v for k, v in headers.items() if v is not None}
        
        # Call CME API
        response = requests.get(
            f"{PRICING_API_URL}?productCodes=MES",
            headers=headers
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Cache the results
            with open(cache_file, "w") as f:
                json.dump(data, f)
            
            with open(cache_time_file, "w") as f:
                f.write(str(time.time()))
            
            return process_cme_margin_data(data)
        else:
            print(f"Error calling CME API: {response.status_code} - {response.text}")
            # Fall back to hardcoded values if API fails
            return get_fallback_margin_data()
    except Exception as e:
        print(f"Exception calling CME API: {e}")
        # Fall back to hardcoded values
        return get_fallback_margin_data()


def process_cme_margin_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process raw CME margin data into a format usable by our feature store.
    
    Args:
        data: Raw data from CME API
        
    Returns:
        Processed margin data
    """
    try:
        # Extract MES margin data
        mes_data = None
        for product in data.get("products", []):
            if product.get("productCode") == "MES":
                mes_data = product
                break
        
        if mes_data:
            # Extract relevant fields
            return {
                "symbol": "MES",
                "initial_margin": float(mes_data.get("initialMargin", 0)),
                "maintenance_margin": float(mes_data.get("maintenanceMargin", 0)),
                "effective_date": mes_data.get("effectiveDate", ""),
                "settlement_price": float(mes_data.get("settlementPrice", 0)),
                "last_updated": datetime.datetime.now().isoformat()
            }
        else:
            return get_fallback_margin_data()
    except Exception as e:
        print(f"Error processing CME data: {e}")
        return get_fallback_margin_data()


def get_fallback_margin_data() -> Dict[str, Any]:
    """
    Return fallback margin data if the API fails.
    
    Returns:
        Dictionary with fallback margin values
    """
    return {
        "symbol": "MES",
        "initial_margin": 1320.0,  # $1,320 per contract
        "maintenance_margin": 1200.0,  # $1,200 per contract
        "effective_date": datetime.datetime.now().strftime("%Y-%m-%d"),
        "settlement_price": 4500.0,
        "last_updated": datetime.datetime.now().isoformat(),
        "is_fallback": True
    }


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate the Relative Strength Index for a price series.
    
    Args:
        series: Price series
        period: RSI period
        
    Returns:
        Series containing RSI values
    """
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    
    ema_up = up.ewm(com=period-1, adjust=False).mean()
    ema_down = down.ewm(com=period-1, adjust=False).mean()
    
    rs = ema_up / ema_down
    rsi = 100 - (100 / (1 + rs))
    return rsi


def get_account_data() -> pd.DataFrame:
    """
    Get account data for feature store.
    
    Returns:
        DataFrame with account data
    """
    # Get margin requirements from CME
    margin_data = retrieve_cme_margin_requirements()
    
    # Create dataframe
    now = datetime.datetime.now()
    df = pd.DataFrame({
        "event_timestamp": [now],
        "account_id": ["main"],
        "equity": [50000.0],  # $50,000 account equity
        "margin_used": [12000.0],  # Example value
        "initial_margin_rate": [margin_data["initial_margin"]],
        "maintenance_margin_rate": [margin_data["maintenance_margin"]],
        "leverage": [5.0],  # 5x leverage
        "risk_limit_pct": [0.02],  # 2% risk per trade
    })
    
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
    save_to_parquet(get_account_data(), "margin_features.parquet")


if __name__ == "__main__":
    main() 