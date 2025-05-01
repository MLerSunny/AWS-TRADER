# Tradovate Micro ES Tick Ingestion Pipeline

This module connects to Tradovate's Micro ES futures feed via WebSockets, collects market ticks, and streams them to AWS Kinesis for further processing.

## Features

* Real-time connection to Tradovate market data
* Batches 500 ticks at a time for efficient Kinesis ingestion
* Graceful reconnection with configurable retry logic
* Proper error handling and logging

## Configuration

Set the following environment variables:

```
TRADOVATE_TOKEN=your-tradovate-api-token
TRADOVATE_WS_ENDPOINT=wss://tradovate-endpoint.com/v1/ws
KINESIS_STREAM_NAME=mes-raw-ticks
AWS_REGION=us-east-1
BATCH_SIZE=500
MAX_RECONNECT_ATTEMPTS=5
RECONNECT_DELAY_SECONDS=5
```

## Dependencies

* boto3
* websockets
* AWS credentials configured for Kinesis access

## Usage

```bash
# Set required environment variables
export TRADOVATE_TOKEN="your-token-here"

# Run the script
python -m pipelines.tick_ingest.main
```

## Development

Adjust the `process_message` function to match Tradovate's actual message format.
