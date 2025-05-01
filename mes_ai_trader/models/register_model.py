"""
Model registration utilities for MES AI Trader

This module provides functions to register models in the SageMaker Model Registry
and update their approval status.
"""

import os
import json
import boto3
import logging
from typing import Dict, Any, Optional, List

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize SageMaker client
sagemaker_client = boto3.client('sagemaker')

def register_model(
    model_data_url: str,
    model_type: str,
    model_package_group_name: str,
    inference_image: str,
    metrics_data: Dict[str, Any],
    description: str = None
) -> str:
    """
    Register a model in the SageMaker Model Registry.
    
    Args:
        model_data_url: S3 URL of the model artifacts
        model_type: Type of model (LGBM, TFT, PPO)
        model_package_group_name: Name of the model package group
        inference_image: URI of the inference image
        metrics_data: Dictionary of model metrics
        description: Optional description for the model
        
    Returns:
        str: The ARN of the registered model package
    """
    description = description or f"{model_type} model for MES AI Trader"
    
    # Create model package
    response = sagemaker_client.create_model_package(
        ModelPackageGroupName=model_package_group_name,
        ModelPackageDescription=description,
        ModelApprovalStatus="PendingManualApproval",
        InferenceSpecification={
            "Containers": [
                {
                    "Image": inference_image,
                    "ModelDataUrl": model_data_url
                }
            ],
            "SupportedContentTypes": ["application/json"],
            "SupportedResponseMIMETypes": ["application/json"]
        },
        ModelMetrics={
            "ModelQuality": {
                "Statistics": {
                    "Content": json.dumps(metrics_data),
                    "ContentType": "application/json"
                }
            }
        },
        Tags=[
            {
                "Key": "ModelType",
                "Value": model_type
            }
        ]
    )
    
    logger.info(f"Registered model {model_type} with ARN: {response['ModelPackageArn']}")
    return response['ModelPackageArn']

def update_approval_status(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda function to update the approval status of model packages.
    
    Args:
        event: Lambda event data
        context: Lambda context
        
    Returns:
        Dict: Response data with updated model packages
    """
    logger.info(f"Received event: {json.dumps(event)}")
    
    # Get model package group name from event or environment
    model_package_group_name = event.get('ModelPackageGroupName', 
                                        os.environ.get('MODEL_PACKAGE_GROUP_NAME'))
    
    if not model_package_group_name:
        error_msg = "ModelPackageGroupName not provided in event or environment"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    # Get desired status from event
    status = event.get('Status', 'Approved')
    
    # List all model packages in the group
    model_packages = []
    next_token = None
    
    while True:
        if next_token:
            response = sagemaker_client.list_model_packages(
                ModelPackageGroupName=model_package_group_name,
                ModelApprovalStatus='PendingManualApproval',
                NextToken=next_token
            )
        else:
            response = sagemaker_client.list_model_packages(
                ModelPackageGroupName=model_package_group_name,
                ModelApprovalStatus='PendingManualApproval'
            )
        
        model_packages.extend(response['ModelPackageSummaryList'])
        
        if 'NextToken' in response:
            next_token = response['NextToken']
        else:
            break
    
    logger.info(f"Found {len(model_packages)} model packages with PendingManualApproval status")
    
    # Update the approval status for each model package
    updated_packages = []
    for package in model_packages:
        model_package_arn = package['ModelPackageArn']
        
        try:
            sagemaker_client.update_model_package(
                ModelPackageArn=model_package_arn,
                ModelApprovalStatus=status
            )
            updated_packages.append(model_package_arn)
            logger.info(f"Updated model package {model_package_arn} to {status}")
        except Exception as e:
            logger.error(f"Error updating model package {model_package_arn}: {str(e)}")
    
    return {
        'StatusCode': 200,
        'ExecutionId': event.get('ExecutionId', ''),
        'ModelPackageGroupName': model_package_group_name,
        'UpdatedModelPackages': updated_packages,
        'Status': status
    }

if __name__ == "__main__":
    # This allows testing the script locally
    import argparse
    
    parser = argparse.ArgumentParser(description='Register or update models in SageMaker Model Registry')
    parser.add_argument('--model-package-group-name', required=True, help='Model package group name')
    parser.add_argument('--update-status', action='store_true', help='Update pending models to Approved')
    
    args = parser.parse_args()
    
    if args.update_status:
        event = {
            'ModelPackageGroupName': args.model_package_group_name,
            'Status': 'Approved'
        }
        response = update_approval_status(event, None)
        print(json.dumps(response, indent=2)) 