"""
Risk Gate Lambda Function

This Lambda function acts as a risk gate that:
1. Calculates Value at Risk (VaR) based on input parameters
2. Compares VaR against a percentage of account equity
3. Returns approval/rejection decision
"""
import json
import os
import boto3
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
ssm_client = boto3.client('ssm')

# Constants
VAR_CONFIDENCE_FACTOR = 1.65  # 95% confidence interval
LEVERAGE_FACTOR = 5          # 5x leverage
EQUITY_THRESHOLD_PCT = 0.01  # 1% of equity

# SSM Parameter paths
EQUITY_PARAM_PATH = os.environ.get('EQUITY_PARAM_PATH', '/mes-ai-trader/equity')


def get_equity():
    """
    Retrieve the current equity value from SSM Parameter Store.
    
    Returns:
        float: Current equity value
    """
    try:
        response = ssm_client.get_parameter(Name=EQUITY_PARAM_PATH)
        equity = float(response['Parameter']['Value'])
        logger.info(f"Retrieved equity: ${equity:.2f}")
        return equity
    except Exception as e:
        logger.error(f"Error retrieving equity from SSM: {e}")
        # Default to a conservative value if parameter can't be retrieved
        return 10000.0  # Default to $10,000 USD


def calculate_var(sigma, quantity):
    """
    Calculate Value at Risk (VaR) using the formula: 1.65 × σ × qty × 5
    
    Args:
        sigma (float): Standard deviation of returns
        quantity (float): Position size/quantity
        
    Returns:
        float: Calculated VaR
    """
    var = VAR_CONFIDENCE_FACTOR * sigma * quantity * LEVERAGE_FACTOR
    return var


def lambda_handler(event, context):
    """
    Lambda handler function for risk gate validation.
    
    Args:
        event (dict): Input event containing prob, qty, sigma
        context: Lambda context
        
    Returns:
        dict: Response with approval status and details
    """
    logger.info(f"Received event: {json.dumps(event)}")
    
    # Extract parameters from event
    try:
        # Handle both direct invocation and API Gateway formats
        if 'body' in event:
            # API Gateway format (body is a JSON string)
            body = json.loads(event['body'])
            prob = float(body.get('prob', 0.0))
            quantity = float(body.get('qty', 0.0))
            sigma = float(body.get('sigma', 0.0))
        else:
            # Direct invocation format
            prob = float(event.get('prob', 0.0))
            quantity = float(event.get('qty', 0.0))
            sigma = float(event.get('sigma', 0.0))
            
    except (ValueError, TypeError) as e:
        error_msg = f"Error parsing input parameters: {e}"
        logger.error(error_msg)
        return {
            'statusCode': 400,
            'body': json.dumps({
                'status': 'error',
                'message': error_msg
            })
        }
    
    # Get current equity
    equity = get_equity()
    
    # Calculate VaR
    var = calculate_var(sigma, quantity)
    
    # Calculate equity threshold
    equity_threshold = equity * EQUITY_THRESHOLD_PCT
    
    # Make decision
    is_approved = var <= equity_threshold
    decision = "approved" if is_approved else "rejected"
    
    logger.info(f"VaR: ${var:.2f}, Threshold: ${equity_threshold:.2f}, Decision: {decision}")
    
    # Prepare response
    response = {
        'status': decision,
        'details': {
            'var': var,
            'equity': equity,
            'threshold': equity_threshold,
            'var_pct_of_equity': (var / equity) * 100 if equity > 0 else float('inf'),
            'prob': prob,
            'qty': quantity,
            'sigma': sigma
        }
    }
    
    # Format response based on invocation type
    if 'body' in event:
        # API Gateway response format
        return {
            'statusCode': 200,
            'body': json.dumps(response)
        }
    else:
        # Direct invocation response
        return response 