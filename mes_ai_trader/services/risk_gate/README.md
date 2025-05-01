# MES AI Trader Risk Gate Service

This service provides a risk gate to validate trades before execution. It calculates Value at Risk (VaR) and ensures it doesn't exceed a configurable percentage of the account equity.

## Architecture

The Risk Gate is implemented as an AWS Lambda function that:

1. Receives trade parameters (probability, quantity, and sigma/standard deviation)
2. Retrieves current account equity from AWS SSM Parameter Store
3. Calculates Value at Risk (VaR) using the formula: `1.65 × σ × qty × 5`
4. Compares VaR to 1% of account equity
5. Returns approval or rejection

## Deployment

The service can be deployed using the AWS Serverless Application Model (SAM):

```bash
# Install SAM CLI if you haven't already
pip install aws-sam-cli

# Build the Lambda package
sam build

# Deploy to AWS
sam deploy --guided
```

During the guided deployment, you'll be asked for:

- Stack name (e.g., "mes-trader-risk-gate")
- AWS Region
- Parameter values (or accept defaults)
- Confirmation for deployment

## Usage

The risk gate function can be invoked in two ways:

### 1. Direct Lambda Invocation

Using AWS CLI:

```bash
aws lambda invoke --function-name MESRiskGateFunction \
  --payload '{"prob": 0.85, "qty": 10, "sigma": 0.02}' \
  response.json
```

Using Python:

```python
import boto3
import json

lambda_client = boto3.client('lambda')

response = lambda_client.invoke(
    FunctionName='MESRiskGateFunction',
    InvocationType='RequestResponse',
    Payload=json.dumps({
        'prob': 0.85,
        'qty': 10, 
        'sigma': 0.02
    })
)

result = json.loads(response['Payload'].read().decode())
print(result)
```

### 2. API Gateway Endpoint

```python
import requests
import json

url = "https://your-api-id.execute-api.region.amazonaws.com/Prod/risk-gate/"
payload = {
    'prob': 0.85,
    'qty': 10,
    'sigma': 0.02
}

response = requests.post(url, json=payload)
result = response.json()
print(result)
```

## Response Format

The function returns a response in this format:

```json
{
  "status": "approved", // or "rejected"
  "details": {
    "var": 0.825, // 1.65 * sigma * qty * 5
    "equity": 10000.0,
    "threshold": 100.0, // 1% of equity
    "var_pct_of_equity": 0.825, // VaR as percentage of equity
    "prob": 0.85,
    "qty": 10,
    "sigma": 0.02
  }
}
```

## Updating Equity

You can update the account equity in the SSM Parameter Store:

```bash
aws ssm put-parameter --name "/mes-ai-trader/equity" --value "15000.0" --type String --overwrite
```

Or using Python:

```python
import boto3

ssm_client = boto3.client('ssm')
ssm_client.put_parameter(
    Name='/mes-ai-trader/equity',
    Value='15000.0',
    Type='String',
    Overwrite=True
)
```
