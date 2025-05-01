"""
Tests for the Tradovate tick ingest pipeline.
"""
import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import websockets
from botocore.exceptions import ClientError

# Import the module under test
from pipelines.tick_ingest.main import (
    BATCH_SIZE,
    process_message,
    send_to_kinesis,
    websocket_client,
    subscribe_to_feed,
)


@pytest.fixture
def mock_env():
    """Mock environment variables needed for the tests."""
    os.environ["TRADOVATE_TOKEN"] = "mock-token"
    os.environ["TRADOVATE_WS_ENDPOINT"] = "wss://mock-endpoint.com/ws"
    os.environ["KINESIS_STREAM_NAME"] = "test-stream"
    os.environ["BATCH_SIZE"] = "500"
    os.environ["MAX_RECONNECT_ATTEMPTS"] = "2"
    os.environ["RECONNECT_DELAY_SECONDS"] = "0"  # No delay for tests
    yield
    # Clean up
    for key in [
        "TRADOVATE_TOKEN",
        "TRADOVATE_WS_ENDPOINT",
        "KINESIS_STREAM_NAME",
        "BATCH_SIZE",
        "MAX_RECONNECT_ATTEMPTS",
        "RECONNECT_DELAY_SECONDS",
    ]:
        if key in os.environ:
            del os.environ[key]


@pytest.fixture
def mock_kinesis():
    """Mock AWS Kinesis client."""
    with patch("pipelines.tick_ingest.main.kinesis_client") as mock_client:
        # Configure the mock to return a successful response
        mock_client.put_records.return_value = {"FailedRecordCount": 0}
        yield mock_client


@pytest.fixture
def mock_websocket():
    """Mock websocket connection."""
    mock_ws = AsyncMock()
    mock_ws.recv.side_effect = generate_mock_tick_messages(600)  # More than 1 batch
    return mock_ws


def generate_mock_tick_messages(count):
    """Generate a series of mock tick messages for testing."""
    messages = []
    for i in range(count):
        tick = {
            "e": "tick",
            "s": "MES",
            "p": 4500.50 + (i * 0.25),  # Simulate price changes
            "q": 1.0,
            "t": 1625097600000 + (i * 1000),  # Timestamp increment
            "x": "CME",
        }
        messages.append(json.dumps(tick))
    
    # Add ConnectionClosed exception at the end to terminate the loop
    messages.append(websockets.exceptions.ConnectionClosed(None, None))
    return messages


class MockConnectionClosed(Exception):
    """Mock websocket connection closed exception."""
    pass


@pytest.mark.asyncio
async def test_process_message_valid_tick():
    """Test that process_message correctly processes valid tick data."""
    # Mock tick message
    tick_msg = json.dumps({
        "e": "tick",
        "s": "MES",
        "p": 4500.75,
        "q": 1.0,
        "t": 1625097600000,
        "x": "CME"
    })
    
    result = await process_message(tick_msg)
    
    assert result is not None
    assert result["symbol"] == "MES"
    assert result["price"] == 4500.75
    assert result["size"] == 1.0
    assert result["timestamp"] == 1625097600000
    assert result["type"] == "CME"


@pytest.mark.asyncio
async def test_process_message_invalid_json():
    """Test that process_message handles invalid JSON input."""
    invalid_msg = "not a valid JSON"
    
    result = await process_message(invalid_msg)
    
    assert result is None


@pytest.mark.asyncio
async def test_process_message_non_tick_message():
    """Test that process_message ignores non-tick messages."""
    non_tick_msg = json.dumps({"e": "other", "data": "some data"})
    
    result = await process_message(non_tick_msg)
    
    assert result is None


@pytest.mark.asyncio
async def test_send_to_kinesis_success(mock_kinesis):
    """Test successful sending of records to Kinesis."""
    records = [{"symbol": "MES", "price": 4500.75, "timestamp": 1625097600000} for _ in range(10)]
    
    result = await send_to_kinesis(records)
    
    assert result is True
    mock_kinesis.put_records.assert_called_once()
    # Check that the correct number of records was sent
    args, kwargs = mock_kinesis.put_records.call_args
    assert len(kwargs["Records"]) == 10


@pytest.mark.asyncio
async def test_send_to_kinesis_failure(mock_kinesis):
    """Test handling of Kinesis failures."""
    # Configure mock to simulate failure
    mock_kinesis.put_records.return_value = {"FailedRecordCount": 5}
    
    records = [{"symbol": "MES", "price": 4500.75, "timestamp": 1625097600000} for _ in range(10)]
    
    result = await send_to_kinesis(records)
    
    assert result is False


@pytest.mark.asyncio
async def test_send_to_kinesis_client_error(mock_kinesis):
    """Test handling of Kinesis client errors."""
    # Configure mock to raise an exception
    mock_kinesis.put_records.side_effect = ClientError(
        {"Error": {"Code": "ResourceNotFoundException", "Message": "Stream not found"}},
        "PutRecords"
    )
    
    records = [{"symbol": "MES", "price": 4500.75, "timestamp": 1625097600000} for _ in range(10)]
    
    result = await send_to_kinesis(records)
    
    assert result is False


@pytest.mark.asyncio
async def test_websocket_client_batch_processing(mock_env, mock_kinesis):
    """Test that websocket_client processes and sends batches of 500 ticks."""
    with patch("websockets.connect", new_callable=AsyncMock) as mock_connect, \
         patch("pipelines.tick_ingest.main.subscribe_to_feed", new_callable=AsyncMock) as mock_subscribe:
        
        # Setup the mock websocket connection
        mock_ws = AsyncMock()
        mock_connect.return_value.__aenter__.return_value = mock_ws
        
        # Generate 600 tick messages (enough for 1 full batch + partial)
        tick_messages = []
        for i in range(600):
            tick = {
                "e": "tick",
                "s": "MES",
                "p": 4500.50 + (i * 0.25),
                "q": 1.0,
                "t": 1625097600000 + (i * 1000),
                "x": "CME",
            }
            tick_messages.append(json.dumps(tick))
        
        # Make the 550th message raise ConnectionClosed to terminate the loop
        class ConnectionClosed(Exception):
            pass
        
        # Set up recv to return tick messages and then raise ConnectionClosed
        mock_ws.recv.side_effect = tick_messages[:550] + [websockets.exceptions.ConnectionClosed(None, None)]
        
        # Start the websocket client in a task and let it run for a bit
        task = asyncio.create_task(websocket_client())
        await asyncio.sleep(0.5)  # Give it time to process
        
        # Cancel the task
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        # Verify that batch was sent
        assert mock_kinesis.put_records.called
        
        # Get all calls to put_records
        calls = mock_kinesis.put_records.call_args_list
        
        # Check at least one call had 500 records (a full batch)
        full_batch_sent = False
        for call in calls:
            args, kwargs = call
            if len(kwargs["Records"]) == BATCH_SIZE:
                full_batch_sent = True
                break
        
        assert full_batch_sent, "No full batch of 500 records was sent to Kinesis"


@pytest.mark.asyncio
async def test_websocket_client_reconnection(mock_env, mock_kinesis):
    """Test that websocket_client attempts to reconnect on connection failure."""
    with patch("websockets.connect", new_callable=AsyncMock) as mock_connect, \
         patch("pipelines.tick_ingest.main.subscribe_to_feed", new_callable=AsyncMock) as mock_subscribe:
        
        # First connection attempt fails, second succeeds
        mock_connect.side_effect = [
            websockets.exceptions.WebSocketException("Connection failed"),
            AsyncMock().__aenter__.return_value,  # Second attempt succeeds
        ]
        
        # Start the task
        task = asyncio.create_task(websocket_client())
        await asyncio.sleep(0.5)  # Give it time to process
        
        # Cancel the task
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        
        # Verify reconnection was attempted
        assert mock_connect.call_count >= 2, "Reconnection was not attempted"


@pytest.mark.asyncio
async def test_max_reconnect_attempts(mock_env, mock_kinesis):
    """Test that websocket_client stops after max reconnection attempts."""
    with patch("websockets.connect", new_callable=AsyncMock) as mock_connect:
        # All connection attempts fail
        mock_connect.side_effect = websockets.exceptions.WebSocketException("Connection failed")
        
        # Run the client
        await websocket_client()
        
        # Check that it attempted to connect MAX_RECONNECT_ATTEMPTS times
        assert mock_connect.call_count == int(os.environ["MAX_RECONNECT_ATTEMPTS"]), \
            f"Expected {os.environ['MAX_RECONNECT_ATTEMPTS']} connection attempts" 