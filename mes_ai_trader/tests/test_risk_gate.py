"""
Tests for risk_gate Lambda function with moto to stub SSM Parameter Store.
"""
import pytest
import json
import os
import boto3
from unittest.mock import patch
from moto import mock_ssm

# Import the Lambda handler
from mes_ai_trader.services.risk_gate.lambda import lambda_handler, calculate_var


@pytest.fixture
def aws_credentials():
    """Mocked AWS Credentials for boto3"""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"


@pytest.fixture
def ssm_client(aws_credentials):
    """Create mocked SSM client."""
    with mock_ssm():
        client = boto3.client("ssm", region_name="us-east-1")
        # Setup the equity parameter
        client.put_parameter(
            Name="/mes-ai-trader/equity",
            Value="50000.0",
            Type="String"
        )
        yield client


def test_calculate_var():
    """Test the VaR calculation function."""
    # Testing with simple values
    sigma = 10.0
    quantity = 5.0
    var = calculate_var(sigma, quantity)
    
    # VAR_CONFIDENCE_FACTOR = 1.65, LEVERAGE_FACTOR = 5
    expected_var = 1.65 * sigma * quantity * 5
    assert var == expected_var


def test_lambda_handler_direct_invocation_approved(ssm_client):
    """Test Lambda handler with direct invocation for an approved trade."""
    # Set up the event for an approved trade (low VaR)
    event = {
        "prob": 0.75,
        "qty": 1.0,
        "sigma": 5.0  # This should result in VaR = 1.65 * 5.0 * 1.0 * 5 = 41.25
    }
    
    with patch.dict(os.environ, {"EQUITY_PARAM_PATH": "/mes-ai-trader/equity"}):
        response = lambda_handler(event, None)
    
    # Equity is 50000, threshold is 50000 * 0.01 = 500, VaR is 41.25, so should be approved
    assert response["status"] == "approved"
    assert "details" in response
    assert response["details"]["var"] == 41.25
    assert response["details"]["equity"] == 50000.0
    assert response["details"]["threshold"] == 500.0  # 1% of 50000


def test_lambda_handler_direct_invocation_rejected(ssm_client):
    """Test Lambda handler with direct invocation for a rejected trade."""
    # Set up the event for a rejected trade (high VaR)
    event = {
        "prob": 0.75,
        "qty": 10.0,
        "sigma": 8.0  # This should result in VaR = 1.65 * 8.0 * 10.0 * 5 = 660
    }
    
    with patch.dict(os.environ, {"EQUITY_PARAM_PATH": "/mes-ai-trader/equity"}):
        response = lambda_handler(event, None)
    
    # Equity is 50000, threshold is 50000 * 0.01 = 500, VaR is 660, so should be rejected
    assert response["status"] == "rejected"
    assert "details" in response
    assert response["details"]["var"] == 660.0
    assert response["details"]["equity"] == 50000.0
    assert response["details"]["threshold"] == 500.0  # 1% of 50000


def test_lambda_handler_api_gateway_format(ssm_client):
    """Test Lambda handler with API Gateway event format."""
    # Set up the API Gateway event format
    event = {
        "body": json.dumps({
            "prob": 0.75,
            "qty": 1.0,
            "sigma": 5.0  # This should result in VaR = 1.65 * 5.0 * 1.0 * 5 = 41.25
        })
    }
    
    with patch.dict(os.environ, {"EQUITY_PARAM_PATH": "/mes-ai-trader/equity"}):
        response = lambda_handler(event, None)
    
    # Check response format for API Gateway
    assert response["statusCode"] == 200
    body = json.loads(response["body"])
    assert body["status"] == "approved"
    assert "details" in body
    assert body["details"]["var"] == 41.25


def test_lambda_handler_invalid_input():
    """Test Lambda handler with invalid input."""
    # Set up an event with missing parameters
    event = {
        "prob": 0.75,
        # Missing qty and sigma
    }
    
    response = lambda_handler(event, None)
    
    # For direct invocation, should still return a dict with error details
    assert "status" in response
    assert response["status"] == "error"
    
    # Set up API Gateway event with invalid JSON
    event = {
        "body": "not valid json"
    }
    
    response = lambda_handler(event, None)
    
    # For API Gateway, should return 400 status code
    assert response["statusCode"] == 400
    assert "body" in response 