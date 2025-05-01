"""
Unit tests for the order router service.
"""
import pytest
import json
from unittest.mock import patch, MagicMock

# Tests for health endpoint
def test_health_check(order_router_client):
    """Test the health check endpoint."""
    response = order_router_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "dependencies" in data
    assert "timestamp" in data

# Tests for API key authentication
def test_api_key_auth_disabled(order_router_client):
    """Test that API requests work when auth is disabled."""
    # Mock config to disable auth
    with patch('config.config.API_AUTH_ENABLED', False):
        response = order_router_client.post(
            "/execute",
            json={
                "symbol": "MES",
                "action": "Buy",
                "quantity": 1,
                "order_type": "Market"
            }
        )
        # The request should pass authentication
        # It might still fail for other reasons, but not return 403
        assert response.status_code != 403

def test_api_key_auth_required(order_router_client):
    """Test that API requests require auth when enabled."""
    # Mock config to enable auth
    with patch('config.config.API_AUTH_ENABLED', True):
        response = order_router_client.post(
            "/execute",
            json={
                "symbol": "MES",
                "action": "Buy",
                "quantity": 1,
                "order_type": "Market"
            }
        )
        # Should return unauthorized
        assert response.status_code == 403

def test_api_key_auth_valid(order_router_client):
    """Test that valid API key works."""
    # Mock config to enable auth
    with patch('config.config.API_AUTH_ENABLED', True):
        response = order_router_client.post(
            "/execute",
            headers={"X-API-Key": "test-key"},
            json={
                "symbol": "MES",
                "action": "Buy",
                "quantity": 1,
                "order_type": "Market"
            }
        )
        # Should not return unauthorized (might still fail for other reasons)
        assert response.status_code != 403

# Tests for order execution
@pytest.mark.asyncio
async def test_execute_order(order_router_client, mock_tradovate_api, mock_aws_secrets, test_db_pool):
    """Test order execution endpoint."""
    # Mock TradovateClient and dependencies
    with patch('services.order_router.router.TradovateClient.get_client') as mock_get_client:
        # Create a mock client
        mock_client = MagicMock()
        mock_client.authenticate.return_value = True
        mock_client.get_account_id.return_value = 67890
        mock_client.get_contract_id.return_value = 12345
        
        # Mock successful order placement
        mock_client.place_order.return_value = {
            "success": True,
            "orderId": "mock_order_123",
            "message": "Order placed successfully"
        }
        
        # Configure get_client to return our mock
        mock_get_client.return_value = mock_client
        
        # Disable auth for test
        with patch('config.config.API_AUTH_ENABLED', False):
            # Test market order
            response = order_router_client.post(
                "/execute",
                json={
                    "symbol": "MES",
                    "action": "Buy",
                    "quantity": 1,
                    "order_type": "Market"
                }
            )
            
            # Verify response
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["order_id"] == "mock_order_123"
            
            # Verify client method calls
            mock_client.get_account_id.assert_called_once()
            mock_client.get_contract_id.assert_called_once()
            mock_client.place_order.assert_called_once()

# Test error handling
@pytest.mark.asyncio
async def test_execute_order_tradovate_error(order_router_client):
    """Test handling of Tradovate API errors."""
    # Mock TradovateClient to raise an exception
    with patch('services.order_router.router.TradovateClient.get_client') as mock_get_client:
        mock_get_client.side_effect = Exception("Tradovate API unavailable")
        
        # Disable auth for test
        with patch('config.config.API_AUTH_ENABLED', False):
            # Test market order
            response = order_router_client.post(
                "/execute",
                json={
                    "symbol": "MES",
                    "action": "Buy",
                    "quantity": 1,
                    "order_type": "Market"
                }
            )
            
            # Verify response indicates error
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data

# Test validation
def test_order_validation(order_router_client):
    """Test order request validation."""
    # Disable auth for test
    with patch('config.config.API_AUTH_ENABLED', False):
        # Test invalid action
        response = order_router_client.post(
            "/execute",
            json={
                "symbol": "MES",
                "action": "InvalidAction",
                "quantity": 1,
                "order_type": "Market"
            }
        )
        
        # Verify validation error
        assert response.status_code == 422
        
        # Test invalid quantity
        response = order_router_client.post(
            "/execute",
            json={
                "symbol": "MES",
                "action": "Buy",
                "quantity": 0,  # Should be > 0
                "order_type": "Market"
            }
        )
        
        # Verify validation error
        assert response.status_code == 422
        
        # Test invalid order type
        response = order_router_client.post(
            "/execute",
            json={
                "symbol": "MES",
                "action": "Buy",
                "quantity": 1,
                "order_type": "InvalidType"
            }
        )
        
        # Verify validation error
        assert response.status_code == 422
        
        # Test missing required fields
        response = order_router_client.post(
            "/execute",
            json={
                "symbol": "MES",
                "action": "Buy",
                # Missing quantity
                "order_type": "Market"
            }
        )
        
        # Verify validation error
        assert response.status_code == 422 