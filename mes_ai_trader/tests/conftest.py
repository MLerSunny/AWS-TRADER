"""
Pytest configuration file with fixtures for testing the MES AI Trader.
"""
import os
import sys
import asyncio
import pytest
from pathlib import Path
import tempfile
import boto3
import json
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Ensure dynaconf loads test settings
os.environ["MES_ENV"] = "testing"

# Import after environment setup
from config import config
from services.order_router.router import app as order_router_app
from services.inference.serve import app as inference_app
from fastapi.testclient import TestClient

# Create test clients
@pytest.fixture
def order_router_client():
    """Create a test client for the order router service."""
    with TestClient(order_router_app) as client:
        yield client

@pytest.fixture
def inference_client():
    """Create a test client for the inference service."""
    with TestClient(inference_app) as client:
        yield client
        
# Database fixtures
@pytest.fixture
async def test_db_pool():
    """Create a test database connection pool."""
    import aiopg
    
    # Create temp database for testing
    dsn = f"host={config.DB_HOST} port={config.DB_PORT} dbname={config.DB_NAME} user={config.DB_USER} password={config.DB_PASSWORD}"
    pool = await aiopg.create_pool(dsn)
    
    # Create test tables
    async with pool.acquire() as conn:
        async with conn.cursor() as cursor:
            await cursor.execute("""
                CREATE TABLE IF NOT EXISTS orders (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP WITH TIME ZONE,
                    symbol VARCHAR(10),
                    action VARCHAR(10),
                    quantity INTEGER,
                    order_type VARCHAR(20),
                    price FLOAT,
                    trading_mode VARCHAR(10),
                    order_id VARCHAR(50),
                    stop_id VARCHAR(50),
                    atr_value FLOAT,
                    order_details JSONB
                )
            """)
    
    yield pool
    
    # Cleanup
    pool.close()
    await pool.wait_closed()

# AWS mocks
@pytest.fixture
def mock_aws_secrets():
    """Mock AWS Secrets Manager."""
    with patch('boto3.client') as mock_client:
        secrets_instance = MagicMock()
        secrets_instance.get_secret_value.return_value = {
            "SecretString": json.dumps({
                "username": "test_user",
                "password": "test_password",
                "app_id": "test_app_id",
                "app_secret": "test_app_secret"
            })
        }
        
        # Configure the mock to return our mock instance
        def get_client(service_name, *args, **kwargs):
            if service_name == 'secretsmanager':
                return secrets_instance
            # Return a different mock for other services
            return MagicMock()
            
        mock_client.side_effect = get_client
        yield secrets_instance

# Model fixtures
@pytest.fixture
def mock_lightgbm_model():
    """Mock LightGBM model."""
    import numpy as np
    
    model = MagicMock()
    model.predict.return_value = np.array([[0.7, 0.3]])
    
    with patch('lightgbm.Booster', return_value=model):
        yield model
        
@pytest.fixture
def mock_tft_model():
    """Mock TFT model."""
    import torch
    
    model = MagicMock()
    # Mock tensor output with shape [batch_size, horizon, quantiles]
    model.return_value = torch.tensor([[[0.1, 0.5, 0.9]]])
    
    with patch('torch.load', return_value=model):
        yield model

@pytest.fixture
def mock_ppo_model():
    """Mock PPO model."""
    model = MagicMock()
    model.predict.return_value = (1, None)  # Action 1 (BUY)
    
    with patch('stable_baselines3.PPO.load', return_value=model):
        yield model

# Feature store mock
@pytest.fixture
def mock_feature_store():
    """Mock Feast feature store."""
    import pandas as pd
    
    store = MagicMock()
    # Mock get_online_features to return a DataFrame with testing features
    store.get_online_features.return_value.to_dict.return_value = {
        "rsi_features:rsi_14": [50.0],
        "rsi_features:rsi_14_trend": [0.1],
        "atr_features:atr_14": [2.5],
        "atr_features:atr_14_normalized": [0.5],
        "vwap_features:vwap": [4500.0],
        "vwap_features:price_to_vwap": [1.001],
        "vwap_features:vwap_trend": [0.2],
        "order_book_features:imbalance": [0.1],
        "order_book_features:bid_volume": [100],
        "order_book_features:ask_volume": [90],
        "order_book_features:imbalance_ma": [0.05],
        "margin_features:initial_margin": [5000],
        "margin_features:margin_to_price_ratio": [1.1]
    }
    
    with patch('feast.FeatureStore', return_value=store):
        yield store

# Tradovate API mock
@pytest.fixture
def mock_tradovate_api():
    """Mock Tradovate API responses."""
    with patch('requests.Session') as mock_session:
        session_instance = MagicMock()
        
        # Mock auth response
        auth_response = MagicMock()
        auth_response.json.return_value = {
            "accessToken": "mock_token",
            "userId": 12345,
            "expirationTime": (
                int((
                    pytest.import_time + 
                    pytest.timezone.timedelta(hours=4)
                ).timestamp() * 1000)
            )
        }
        auth_response.status_code = 200
        
        # Mock account response
        account_response = MagicMock()
        account_response.json.return_value = [
            {"id": 67890, "name": "Test Account", "active": True}
        ]
        account_response.status_code = 200
        
        # Mock contract response
        contract_response = MagicMock()
        contract_response.json.return_value = {"id": 12345, "name": "MES"}
        contract_response.status_code = 200
        
        # Mock order response
        order_response = MagicMock()
        order_response.json.return_value = {"orderId": "mock_order_123"}
        order_response.status_code = 200
        
        # Configure session mock responses
        def mock_request(*args, **kwargs):
            url = args[1] if len(args) > 1 else kwargs.get('url', '')
            
            if 'auth/accessTokenRequest' in url:
                return auth_response
            elif 'account/list' in url:
                return account_response
            elif 'contract/find' in url:
                return contract_response
            elif 'order/place' in url:
                return order_response
            
            # Default response
            default_response = MagicMock()
            default_response.status_code = 200
            default_response.json.return_value = {}
            return default_response
            
        # Set up the mock session
        session_instance.post.side_effect = mock_request
        session_instance.get.side_effect = mock_request
        mock_session.return_value = session_instance
        
        yield session_instance

# Set pytest event loop
@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close() 