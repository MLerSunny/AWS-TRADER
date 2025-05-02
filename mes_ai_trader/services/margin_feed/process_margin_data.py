#!/usr/bin/env python
"""
Process margin data for Feature Store offline storage.
This script is used by the nightly pipeline to rebuild margin data parquet files
based on the latest margin data from the CME API.

The script:
1. Reads the latest margin data from S3
2. Formats it for Feature Store offline usage
3. Writes the parquet files to the correct location
"""
import os
import sys
import json
import glob
import logging
import datetime
import pandas as pd
import boto3
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
INPUT_PATH = '/opt/ml/processing/input/margin-data'
OUTPUT_PATH = '/opt/ml/processing/output/margin-features'
LATEST_FILE = 'margin-data-latest.json'
DATE_FORMAT = '%Y-%m-%d'
DEFAULT_SYMBOLS = ['MES']

# Initialize AWS clients
s3_client = boto3.client('s3')


def find_latest_margin_data():
    """
    Find the latest margin data JSON file.
    
    Returns:
        dict: Loaded margin data
    """
    latest_file_path = os.path.join(INPUT_PATH, LATEST_FILE)
    
    if os.path.exists(latest_file_path):
        logger.info(f"Found latest margin data file: {latest_file_path}")
        with open(latest_file_path, 'r') as f:
            return json.load(f)
    
    # If latest file not found, look for most recent timestamped file
    json_files = glob.glob(os.path.join(INPUT_PATH, 'margin-data-*.json'))
    if not json_files:
        logger.warning("No margin data files found, using defaults")
        return generate_default_data()
        
    # Find most recent file by sorting filenames (which include timestamps)
    latest_file = sorted(json_files)[-1]
    logger.info(f"Using most recent margin data file: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)


def generate_default_data():
    """
    Generate default margin data if no files are found.
    
    Returns:
        dict: Default margin data
    """
    logger.warning("Generating default margin data")
    margin_data = {}
    now = datetime.datetime.now()
    
    for symbol in DEFAULT_SYMBOLS:
        if symbol == "MES":
            margin_data[symbol] = {
                "symbol": symbol,
                "initial_margin": 1320.0,      # $1,320 per contract
                "maintenance_margin": 1200.0,   # $1,200 per contract
                "effective_date": now.strftime(DATE_FORMAT),
                "settlement_price": 4500.0,
                "last_updated": now.isoformat(),
                "source": "default"
            }
        else:
            margin_data[symbol] = {
                "symbol": symbol,
                "initial_margin": 2000.0,      # Generic default
                "maintenance_margin": 1800.0,   # Generic default
                "effective_date": now.strftime(DATE_FORMAT),
                "settlement_price": 1000.0,
                "last_updated": now.isoformat(),
                "source": "default"
            }
    
    return margin_data


def create_margin_dataframe(margin_data):
    """
    Convert margin data dictionary to a DataFrame suitable for Feature Store.
    
    Args:
        margin_data: Dictionary of margin data by symbol
        
    Returns:
        DataFrame: Margin data in DataFrame format
    """
    rows = []
    now = datetime.datetime.now()
    
    # Add an entry for each date over the past and future 30 days
    # (This ensures we have a complete time-series for offline feature lookups)
    date_range = [(now + datetime.timedelta(days=i-30)).date() for i in range(61)]
    
    for date in date_range:
        timestamp = datetime.datetime.combine(date, datetime.time.min)
        
        for symbol, data in margin_data.items():
            rows.append({
                "event_timestamp": timestamp,
                "created_timestamp": now,
                "symbol": symbol,
                "initial_margin": data["initial_margin"],
                "maintenance_margin": data["maintenance_margin"],
                "effective_date": data["effective_date"],
                "settlement_price": data["settlement_price"],
                "source": data["source"]
            })
    
    df = pd.DataFrame(rows)
    
    # Add derived features
    df["margin_ratio"] = df["maintenance_margin"] / df["initial_margin"]
    df["margin_to_price_ratio"] = df["initial_margin"] / df["settlement_price"] * 100
    
    # Sort by timestamp and symbol
    df.sort_values(["event_timestamp", "symbol"], inplace=True)
    
    return df


def write_parquet_files(df):
    """
    Write DataFrame to parquet files for Feature Store offline storage.
    
    Args:
        df: DataFrame with margin data
        
    Returns:
        dict: Summary of files written
    """
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Group by date (year/month/day)
    df["year"] = df["event_timestamp"].dt.year
    df["month"] = df["event_timestamp"].dt.month
    df["day"] = df["event_timestamp"].dt.day
    
    # Write a separate parquet file for each date
    file_count = 0
    total_rows = 0
    
    # Feature Store typically organizes files by date
    for (year, month, day), group_df in df.groupby(["year", "month", "day"]):
        # Remove grouping columns before writing
        output_df = group_df.drop(columns=["year", "month", "day"])
        
        # Create the directory structure
        date_path = os.path.join(OUTPUT_PATH, f"year={year}", f"month={month}", f"day={day}")
        os.makedirs(date_path, exist_ok=True)
        
        # Write the parquet file
        file_path = os.path.join(date_path, f"margin_data_{year}{month:02d}{day:02d}.parquet")
        output_df.to_parquet(file_path, index=False)
        
        file_count += 1
        total_rows += len(output_df)
        logger.info(f"Wrote {len(output_df)} rows to {file_path}")
    
    # Also write a single consolidated file for easy testing
    consolidated_path = os.path.join(OUTPUT_PATH, "margin_data_all.parquet")
    df.drop(columns=["year", "month", "day"]).to_parquet(consolidated_path, index=False)
    
    return {
        "files_written": file_count + 1,  # +1 for the consolidated file
        "total_rows": total_rows,
        "symbols": df["symbol"].unique().tolist()
    }


def main():
    """Main processing function."""
    try:
        logger.info("Starting margin data processing")
        
        # Load margin data
        margin_data = find_latest_margin_data()
        logger.info(f"Loaded margin data for {len(margin_data)} symbols: {list(margin_data.keys())}")
        
        # Create DataFrame for Feature Store
        df = create_margin_dataframe(margin_data)
        logger.info(f"Created DataFrame with {len(df)} rows")
        
        # Write parquet files
        result = write_parquet_files(df)
        logger.info(f"Processing complete: {result}")
        
        return 0
    except Exception as e:
        logger.error(f"Error processing margin data: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main()) 