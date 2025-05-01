#!/usr/bin/env python
"""
Tradovate Micro ES Tick Data Ingestion Pipeline

This script connects to Tradovate's Micro ES feed via websockets,
collects 500 ticks at a time, and sends them to a Kinesis stream.
"""
import asyncio
import json
import logging
import os
import signal
import sys
import time
from typing import Dict, List, Optional

import boto3
import websockets
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration from environment variables
TRADOVATE_TOKEN = os.environ.get("TRADOVATE_TOKEN")
TRADOVATE_WS_ENDPOINT = os.environ.get(
    "TRADOVATE_WS_ENDPOINT", "wss://tradovate-endpoint-placeholder.com/v1/ws"
)
KINESIS_STREAM_NAME = os.environ.get("KINESIS_STREAM_NAME", "mes-raw-ticks")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "500"))
MAX_RECONNECT_ATTEMPTS = int(os.environ.get("MAX_RECONNECT_ATTEMPTS", "5"))
RECONNECT_DELAY_SECONDS = int(os.environ.get("RECONNECT_DELAY_SECONDS", "5"))

# Initialize AWS Kinesis client
kinesis_client = boto3.client("kinesis", region_name=AWS_REGION)


async def send_to_kinesis(records: List[Dict]) -> bool:
    """
    Send a batch of records to Kinesis stream.
    
    Args:
        records: List of tick data records
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not records:
        return True
    
    try:
        # Format records for Kinesis PutRecords
        kinesis_records = [
            {
                "Data": json.dumps(record).encode("utf-8"),
                "PartitionKey": str(record.get("timestamp", time.time())),
            }
            for record in records
        ]
        
        # Send records to Kinesis
        response = kinesis_client.put_records(
            Records=kinesis_records,
            StreamName=KINESIS_STREAM_NAME,
        )
        
        # Check for failed records
        failed_count = response.get("FailedRecordCount", 0)
        if failed_count > 0:
            logger.warning(f"Failed to put {failed_count} records to Kinesis")
            return False
        
        logger.info(f"Successfully sent batch of {len(records)} records to Kinesis")
        return True
        
    except ClientError as e:
        logger.error(f"Error sending records to Kinesis: {e}")
        return False


async def process_message(message: str) -> Optional[Dict]:
    """
    Process a message from the websocket.
    
    Args:
        message: Raw message string from websocket
        
    Returns:
        Dict: Processed tick data or None if not relevant
    """
    try:
        data = json.loads(message)
        
        # Filter for tick data (adjust based on actual Tradovate message format)
        if "e" in data and data["e"] == "tick":
            # Transform the data as needed
            tick_data = {
                "symbol": data.get("s", "MES"),  # Symbol
                "price": data.get("p", 0.0),     # Price
                "size": data.get("q", 0.0),      # Quantity/Size
                "timestamp": data.get("t", int(time.time() * 1000)),  # Timestamp
                "type": data.get("x", ""),       # Exchange or trade type
                "raw": data,                     # Original data
            }
            return tick_data
        return None
    except json.JSONDecodeError:
        logger.error(f"Failed to parse message: {message}")
        return None
    except Exception as e:
        logger.error(f"Error processing message: {e}, message: {message}")
        return None


async def subscribe_to_feed(websocket):
    """Send subscription message to Tradovate."""
    # Adjust this based on Tradovate's actual API
    subscription_msg = {
        "op": "subscribe",
        "args": ["MES"],  # Micro E-mini S&P 500 symbol
        "token": TRADOVATE_TOKEN,
    }
    await websocket.send(json.dumps(subscription_msg))
    logger.info("Subscription request sent")


async def websocket_client():
    """
    Main websocket client function that connects to Tradovate,
    processes tick data, and sends batches to Kinesis.
    """
    if not TRADOVATE_TOKEN:
        logger.error("TRADOVATE_TOKEN environment variable is required")
        sys.exit(1)
        
    reconnect_attempts = 0
    batch = []
    
    while reconnect_attempts < MAX_RECONNECT_ATTEMPTS:
        try:
            logger.info(f"Connecting to Tradovate websocket: {TRADOVATE_WS_ENDPOINT}")
            
            async with websockets.connect(
                TRADOVATE_WS_ENDPOINT,
                ping_interval=30,
                ping_timeout=10,
                close_timeout=10,
            ) as websocket:
                # Reset reconnect attempts on successful connection
                reconnect_attempts = 0
                
                # Subscribe to the feed
                await subscribe_to_feed(websocket)
                
                # Process incoming messages
                while True:
                    try:
                        message = await websocket.recv()
                        tick_data = await process_message(message)
                        
                        if tick_data:
                            batch.append(tick_data)
                            
                            # When batch size is reached, send to Kinesis
                            if len(batch) >= BATCH_SIZE:
                                await send_to_kinesis(batch)
                                batch = []  # Clear the batch after sending
                                
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("Websocket connection closed")
                        # Send any remaining ticks in the batch before breaking
                        if batch:
                            await send_to_kinesis(batch)
                            batch = []
                        break
                        
        except (websockets.exceptions.WebSocketException, OSError) as e:
            reconnect_attempts += 1
            logger.error(
                f"Websocket error (attempt {reconnect_attempts}/{MAX_RECONNECT_ATTEMPTS}): {e}"
            )
            
            # Send any remaining ticks in the batch before reconnecting
            if batch:
                await send_to_kinesis(batch)
                batch = []
                
            # Wait before reconnecting
            await asyncio.sleep(RECONNECT_DELAY_SECONDS)
            
    logger.critical(f"Failed to connect after {MAX_RECONNECT_ATTEMPTS} attempts")
    # Send any final ticks in the batch before exiting
    if batch:
        await send_to_kinesis(batch)


async def shutdown(signal, loop):
    """Handle graceful shutdown."""
    logger.info(f"Received exit signal {signal.name}...")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    
    [task.cancel() for task in tasks]
    logger.info(f"Cancelling {len(tasks)} outstanding tasks")
    
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()


def main():
    """Initialize and run the application."""
    try:
        loop = asyncio.get_event_loop()
        
        # Register signal handlers for graceful shutdown
        signals = (signal.SIGHUP, signal.SIGTERM, signal.SIGINT)
        for s in signals:
            loop.add_signal_handler(
                s, lambda s=s: asyncio.create_task(shutdown(s, loop))
            )
            
        # Start the websocket client
        loop.create_task(websocket_client())
        loop.run_forever()
        
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        sys.exit(1)
    finally:
        loop.close()
        logger.info("Successfully shutdown")


if __name__ == "__main__":
    import signal
    main() 