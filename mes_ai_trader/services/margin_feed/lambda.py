"""
Lambda function to retrieve margin requirements from CME API and save to S3 and Feature Store.
Triggered on a schedule via EventBridge.
"""
import json
import os
import time
import datetime
import logging
import requests
import boto3
import pandas as pd
import numpy as np
from feast import FeatureStore
from botocore.exceptions import ClientError

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Constants
PRICING_API_URL = os.environ.get("PRICING_API_URL", "https://api.cmegroup.com/pricinginformation/v1/margins")
PRICING_API_KEY = os.environ.get("PRICING_API_KEY", "")
S3_BUCKET = os.environ.get("S3_BUCKET", "mes-ai-trader-feature-store")
S3_PREFIX = os.environ.get("S3_PREFIX", "margin-data")
FEATURE_STORE_REPO_PATH = os.environ.get("FEATURE_STORE_REPO_PATH", "/opt/ml/processing/feature_repo")
SYMBOLS = os.environ.get("SYMBOLS", "MES").split(",")  # Comma-separated list of symbols to fetch

# Initialize AWS clients
s3_client = boto3.client('s3')


def lambda_handler(event, context):
    """
    Lambda function handler that retrieves margin data from CME API 
    and saves it to S3 and Feature Store.
    
    Args:
        event: Lambda event object
        context: Lambda context object
        
    Returns:
        dict: Response containing execution status and details
    """
    logger.info(f"Starting margin data retrieval for symbols: {SYMBOLS}")
    
    try:
        # Get margin data for each symbol
        margin_data = {}
        for symbol in SYMBOLS:
            margin_data[symbol] = get_margin_data(symbol)
        
        # Save to S3
        s3_path = save_to_s3(margin_data)
        
        # Save to Feature Store
        feature_store_path = save_to_feature_store(margin_data)
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "status": "success",
                "message": "Successfully retrieved and stored margin data",
                "s3_path": s3_path,
                "feature_store_path": feature_store_path,
                "timestamp": datetime.datetime.now().isoformat(),
                "symbols_processed": SYMBOLS
            })
        }
        
    except Exception as e:
        logger.error(f"Error in margin data retrieval: {e}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({
                "status": "error",
                "message": f"Error retrieving margin data: {str(e)}",
                "timestamp": datetime.datetime.now().isoformat()
            })
        }


def get_margin_data(symbol):
    """
    Retrieve margin data for a specific symbol from the CME API.
    
    Args:
        symbol: Trading symbol (e.g., "MES" for Micro E-mini S&P)
        
    Returns:
        dict: Margin data for the symbol
    """
    try:
        headers = {
            "Content-Type": "application/json"
        }
        
        # Add API key if provided
        if PRICING_API_KEY:
            headers["Authorization"] = f"Bearer {PRICING_API_KEY}"
        
        # Make API request
        response = requests.get(
            f"{PRICING_API_URL}?productCodes={symbol}",
            headers=headers
        )
        
        # Check for successful response
        if response.status_code == 200:
            data = response.json()
            return process_margin_data(data, symbol)
        else:
            logger.error(f"API request failed: {response.status_code} - {response.text}")
            return get_fallback_margin_data(symbol)
            
    except Exception as e:
        logger.error(f"Error retrieving margin data from API: {e}")
        return get_fallback_margin_data(symbol)


def process_margin_data(data, symbol):
    """
    Process API response data into a usable format.
    
    Args:
        data: Raw API response data
        symbol: Symbol being processed
        
    Returns:
        dict: Processed margin data
    """
    try:
        symbol_data = None
        
        # Find the data for this specific symbol
        for product in data.get("products", []):
            if product.get("productCode") == symbol:
                symbol_data = product
                break
                
        if symbol_data:
            # Extract and format the relevant fields
            return {
                "symbol": symbol,
                "initial_margin": float(symbol_data.get("initialMargin", 0)),
                "maintenance_margin": float(symbol_data.get("maintenanceMargin", 0)),
                "effective_date": symbol_data.get("effectiveDate", ""),
                "settlement_price": float(symbol_data.get("settlementPrice", 0)),
                "last_updated": datetime.datetime.now().isoformat(),
                "source": "cme_api"
            }
        else:
            logger.warning(f"No data found for symbol {symbol}")
            return get_fallback_margin_data(symbol)
            
    except Exception as e:
        logger.error(f"Error processing margin data: {e}")
        return get_fallback_margin_data(symbol)


def get_fallback_margin_data(symbol):
    """
    Provide fallback margin data if API retrieval fails.
    
    Args:
        symbol: Trading symbol
        
    Returns:
        dict: Fallback margin data
    """
    # Default values for MES
    if symbol == "MES":
        return {
            "symbol": symbol,
            "initial_margin": 1320.0,      # $1,320 per contract
            "maintenance_margin": 1200.0,   # $1,200 per contract
            "effective_date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "settlement_price": 4500.0,
            "last_updated": datetime.datetime.now().isoformat(),
            "source": "fallback"
        }
    # Default for other symbols (can be expanded)
    else:
        return {
            "symbol": symbol,
            "initial_margin": 2000.0,      # Generic default
            "maintenance_margin": 1800.0,   # Generic default
            "effective_date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "settlement_price": 1000.0,
            "last_updated": datetime.datetime.now().isoformat(),
            "source": "fallback"
        }


def save_to_s3(margin_data):
    """
    Save margin data to S3 for offline storage.
    
    Args:
        margin_data: Dictionary of margin data by symbol
        
    Returns:
        str: S3 path where data was saved
    """
    now = datetime.datetime.now()
    # Create both a timestamped file and a latest file (for easier access)
    timestamped_key = f"{S3_PREFIX}/margin-data-{now.strftime('%Y-%m-%d-%H-%M-%S')}.json"
    latest_key = f"{S3_PREFIX}/margin-data-latest.json"
    
    try:
        # Save JSON data
        json_data = json.dumps(margin_data)
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=timestamped_key,
            Body=json_data,
            ContentType="application/json"
        )
        
        # Also save as latest
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=latest_key,
            Body=json_data,
            ContentType="application/json"
        )
        
        logger.info(f"Saved margin data to S3: s3://{S3_BUCKET}/{timestamped_key}")
        
        # Create a Parquet file for offline Feature Store
        df = create_margin_dataframe(margin_data)
        parquet_key = f"{S3_PREFIX}/parquet/margin-data-{now.strftime('%Y-%m-%d')}.parquet"
        
        # Convert DataFrame to parquet and upload
        parquet_buffer = df.to_parquet()
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=parquet_key,
            Body=parquet_buffer
        )
        
        logger.info(f"Saved margin data as Parquet to S3: s3://{S3_BUCKET}/{parquet_key}")
        
        return f"s3://{S3_BUCKET}/{timestamped_key}"
        
    except Exception as e:
        logger.error(f"Error saving to S3: {e}")
        raise e


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
    
    for symbol, data in margin_data.items():
        rows.append({
            "event_timestamp": now,
            "created_timestamp": now,
            "symbol": symbol,
            "initial_margin": data["initial_margin"],
            "maintenance_margin": data["maintenance_margin"],
            "effective_date": data["effective_date"],
            "settlement_price": data["settlement_price"],
            "source": data["source"]
        })
    
    return pd.DataFrame(rows)


def save_to_feature_store(margin_data):
    """
    Save margin data to Feature Store for online access.
    
    Args:
        margin_data: Dictionary of margin data by symbol
        
    Returns:
        str: Feature Store entity path
    """
    try:
        # Check if Lambda is running in an environment with Feature Store
        if not os.path.exists(FEATURE_STORE_REPO_PATH):
            logger.warning(f"Feature Store repo path not found: {FEATURE_STORE_REPO_PATH}")
            return "Feature Store update skipped - repo not available"
        
        # Initialize Feature Store
        fs = FeatureStore(repo_path=FEATURE_STORE_REPO_PATH)
        
        # Convert to DataFrame format expected by Feature Store
        df = create_margin_dataframe(margin_data)
        
        # Push to online store
        fs.write_to_online_store("margin_features", df)
        
        logger.info(f"Successfully wrote {len(df)} rows to Feature Store")
        return "margin_features"
        
    except Exception as e:
        logger.error(f"Error saving to Feature Store: {e}", exc_info=True)
        # Don't raise exception here - if S3 worked but Feature Store failed, 
        # we still want to consider the Lambda mostly successful
        return f"Feature Store update failed: {str(e)}" 