"""
Lambda function for rotating Tradovate API token.
Integrates with AWS Secrets Manager rotation.
"""
import boto3
import json
import logging
import os
import requests
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Constants
TRADOVATE_AUTH_URL = os.environ.get('TRADOVATE_AUTH_URL', 'https://demo.tradovateapi.com/v1/auth/accessTokenRequest')
SECRET_ARN = os.environ.get('SECRET_ARN')


def lambda_handler(event, context):
    """
    Lambda handler for Secrets Manager rotation of Tradovate API token.
    
    Args:
        event: Lambda event containing Secrets Manager rotation event
        context: Lambda context
        
    Returns:
        dict: Response including success status
    """
    logger.info(f"Received event: {json.dumps(event)}")
    
    # Parse event and extract parameters
    try:
        arn = event['SecretId']
        token = event['ClientRequestToken']
        step = event['Step']
    except KeyError:
        logger.error("Missing required parameters in event")
        raise Exception("Missing required parameters in event")
    
    # Initialize the Secrets Manager client
    service_client = boto3.client('secretsmanager')
    
    # Validate the request
    metadata = service_client.describe_secret(SecretId=arn)
    if 'RotationEnabled' not in metadata or not metadata['RotationEnabled']:
        logger.error(f"Secret {arn} is not enabled for rotation")
        raise Exception(f"Secret {arn} is not enabled for rotation")
    
    # Execute appropriate step
    if step == "createSecret":
        create_secret(service_client, arn, token)
    elif step == "setSecret":
        set_secret(service_client, arn, token)
    elif step == "testSecret":
        test_secret(service_client, arn, token)
    elif step == "finishSecret":
        finish_secret(service_client, arn, token)
    else:
        logger.error(f"Invalid step parameter: {step}")
        raise Exception(f"Invalid step parameter: {step}")
    
    return {"statusCode": 200, "body": f"Successfully completed step: {step}"}


def create_secret(service_client, arn, token):
    """
    Create a new secret with a newly generated Tradovate token.
    
    Args:
        service_client: Secrets Manager client
        arn: ARN of the secret
        token: Rotation token
    """
    # Get the current secret
    current_secret = get_secret_dict(service_client, arn, None)
    
    # Get credentials for Tradovate from the current secret
    username = current_secret.get('username')
    password = current_secret.get('password')
    
    if not username or not password:
        logger.error("Current secret is missing username or password")
        raise Exception("Current secret is missing username or password")
    
    # Generate new token from Tradovate API
    new_token = get_tradovate_token(username, password)
    
    # Create new secret version
    new_secret = {
        'username': username,
        'password': password,
        'token': new_token,
        'created_at': current_secret.get('created_at')  # Preserve original created timestamp
    }
    
    # Put the new secret version
    try:
        service_client.put_secret_value(
            SecretId=arn,
            ClientRequestToken=token,
            SecretString=json.dumps(new_secret),
            VersionStages=['AWSPENDING']
        )
        logger.info(f"Successfully created new secret version for {arn}")
    except ClientError as e:
        logger.error(f"Error creating new secret version: {e}")
        raise e


def set_secret(service_client, arn, token):
    """
    This step is not necessary for this rotation scheme.
    The token has already been generated in the createSecret step.
    
    Args:
        service_client: Secrets Manager client
        arn: ARN of the secret
        token: Rotation token
    """
    logger.info("setSecret step - token already generated in createSecret step")
    pass  # Nothing to do


def test_secret(service_client, arn, token):
    """
    Test the new secret by making a test API call to Tradovate.
    
    Args:
        service_client: Secrets Manager client
        arn: ARN of the secret
        token: Rotation token
    """
    # Get the pending secret
    pending_secret = get_secret_dict(service_client, arn, 'AWSPENDING')
    
    # Test the token with a simple API call
    test_token = pending_secret.get('token')
    if not test_token:
        logger.error("Pending secret is missing token")
        raise Exception("Pending secret is missing token")
    
    # Make a test API call to validate token
    try:
        response = requests.get(
            "https://demo.tradovateapi.com/v1/account/list",
            headers={
                "Authorization": f"Bearer {test_token}",
                "Content-Type": "application/json"
            }
        )
        
        if response.status_code != 200:
            logger.error(f"Test API call failed with status {response.status_code}: {response.text}")
            raise Exception(f"Test API call failed with status {response.status_code}")
        
        logger.info("Successfully tested new Tradovate token")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error testing Tradovate token: {e}")
        raise Exception(f"Error testing Tradovate token: {e}")


def finish_secret(service_client, arn, token):
    """
    Finalize the rotation by marking the new secret as AWSCURRENT.
    
    Args:
        service_client: Secrets Manager client
        arn: ARN of the secret
        token: Rotation token
    """
    # Get the pending secret version
    metadata = service_client.describe_secret(SecretId=arn)
    
    # Verify that the pending secret exists
    if 'VersionIdsToStages' not in metadata:
        logger.error(f"Secret {arn} has no version stages")
        raise Exception(f"Secret {arn} has no version stages")
    
    # Find the pending version
    pending_version = None
    for version_id, stages in metadata['VersionIdsToStages'].items():
        if 'AWSPENDING' in stages:
            pending_version = version_id
            break
    
    if not pending_version:
        logger.error(f"Secret {arn} has no pending version")
        raise Exception(f"Secret {arn} has no pending version")
    
    # Promote the pending secret to current
    try:
        service_client.update_secret_version_stage(
            SecretId=arn,
            VersionStage='AWSCURRENT',
            MoveToVersionId=pending_version,
            RemoveFromVersionId=get_current_version_id(metadata)
        )
        logger.info(f"Successfully completed rotation for {arn}")
    except ClientError as e:
        logger.error(f"Error promoting secret version: {e}")
        raise e


def get_secret_dict(service_client, arn, stage):
    """
    Get a secret dictionary from Secrets Manager.
    
    Args:
        service_client: Secrets Manager client
        arn: ARN of the secret
        stage: Stage to get (None for AWSCURRENT)
        
    Returns:
        dict: Secret contents as a dictionary
    """
    try:
        if stage:
            secret = service_client.get_secret_value(
                SecretId=arn,
                VersionStage=stage
            )
        else:
            secret = service_client.get_secret_value(
                SecretId=arn
            )
        
        return json.loads(secret['SecretString'])
    except ClientError as e:
        logger.error(f"Error retrieving secret: {e}")
        raise e


def get_current_version_id(metadata):
    """
    Get the current version ID from secret metadata.
    
    Args:
        metadata: Secret metadata from describe_secret
        
    Returns:
        str: Current version ID
    """
    for version_id, stages in metadata['VersionIdsToStages'].items():
        if 'AWSCURRENT' in stages:
            return version_id
    
    logger.error("No current version found for secret")
    raise Exception("No current version found for secret")


def get_tradovate_token(username, password):
    """
    Get a new token from the Tradovate API.
    
    Args:
        username: Tradovate username
        password: Tradovate password
        
    Returns:
        str: New Tradovate access token
    """
    try:
        response = requests.post(
            TRADOVATE_AUTH_URL,
            json={
                "name": username,
                "password": password,
                "appId": "Sample App",
                "appVersion": "1.0"
            },
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code != 200:
            logger.error(f"Tradovate API returned {response.status_code}: {response.text}")
            raise Exception(f"Tradovate API returned {response.status_code}")
        
        data = response.json()
        if 'accessToken' not in data:
            logger.error(f"Tradovate API response missing accessToken: {data}")
            raise Exception("Tradovate API response missing accessToken")
        
        return data['accessToken']
    except requests.exceptions.RequestException as e:
        logger.error(f"Error calling Tradovate API: {e}")
        raise Exception(f"Error calling Tradovate API: {e}") 