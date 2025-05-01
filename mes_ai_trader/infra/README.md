# MES AI Trader Infrastructure

This directory contains the AWS CDK (Cloud Development Kit) code for deploying the MES AI Trader services to AWS.

## Overview

The infrastructure consists of multiple stacks:

### InferenceStack
Creates the ML model inference service:

1. ECR (Elastic Container Registry) repository for the inference service Docker image
2. SageMaker model, endpoint configuration, and endpoint using ml.m5.large instance type
3. VPC endpoint for secure SageMaker access
4. IAM roles with appropriate permissions for Feast to read from DynamoDB

### DashboardStack
Creates the Streamlit dashboard and monitoring:

1. App Runner service for hosting the Streamlit dashboard
2. CloudWatch dashboard with performance metrics:
   - p50/p95 inference latency metrics
   - Risk-gate rejection rates
3. CloudWatch alarms for critical metrics
4. IAM roles with appropriate permissions

## Prerequisites

- AWS CLI configured with appropriate credentials
- CDK installed (`npm install -g aws-cdk`)
- Python 3.9+
- Docker installed and running (for building the images)

## Getting Started

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Bootstrap the CDK environment (if not already done):

```bash
cdk bootstrap
```

3. Deploy the stacks:

```bash
# Deploy all stacks
cdk deploy --all

# Or deploy specific stacks
cdk deploy MESTraderInferenceStack
cdk deploy MESTraderDashboardStack
```

4. To destroy the stacks when no longer needed:

```bash
# Destroy all stacks
cdk destroy --all

# Or destroy specific stacks
cdk destroy MESTraderInferenceStack
cdk destroy MESTraderDashboardStack
```

## Environment Variables

The following environment variables can be set to configure the deployment:

- `CDK_DEFAULT_ACCOUNT`: AWS account ID (defaults to the account in your AWS CLI profile)
- `CDK_DEFAULT_REGION`: AWS region for deployment (defaults to "us-east-1")

## Required SSM Parameters

Before deploying the dashboard stack, ensure you have the following parameters in AWS Systems Manager Parameter Store:

- `/mes-ai-trader/db-host`: PostgreSQL database host
- `/mes-ai-trader/db-name`: PostgreSQL database name
- `/mes-ai-trader/db-user`: PostgreSQL database user

And in AWS Secrets Manager:
- `/mes-ai-trader/db-password`: PostgreSQL database password

## Testing the Inference Endpoint

Once deployed, you can test the inference endpoint using the AWS SDK:

```python
import boto3
import json

runtime = boto3.client('sagemaker-runtime')

response = runtime.invoke_endpoint(
    EndpointName='mes-trader-inference',
    ContentType='application/json',
    Body=json.dumps({"model": "lgbm"})  # or 'tft' or 'ppo'
)

result = json.loads(response['Body'].read().decode())
print(result)
```

## Monitoring

The deployment includes a comprehensive CloudWatch dashboard that shows:

- Inference latency metrics (p50/p95)
- Risk gate rejection metrics
- Database connection metrics
- Error logs from the dashboard application

You can access the dashboard from the CloudWatch console or via the URL output when you deploy the dashboard stack. 