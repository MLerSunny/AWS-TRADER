# MES AI Trader CI/CD Pipeline

This document describes the CI/CD pipeline for the MES AI Trader project. The pipeline automates the entire workflow from source code to production deployment.

## Pipeline Architecture

The pipeline consists of the following stages:

1. **Source**: Pulls code from GitHub repository
2. **Build**: Runs tests and builds artifacts
   - Python unit tests
   - Scala/sbt compilation and tests
   - Creates deployment packages
3. **Train**: Runs SageMaker Processing jobs for ML model training
   - P-11: Data preprocessing
   - P-12: Feature engineering
   - P-13: Model training
   - Registers trained models in the SageMaker Model Registry
4. **Approval**: Manual approval step to review training results
5. **Deploy**: Blue/green deployment of the new model to production
   - Creates a "blue" deployment with the new model
   - Validates the new deployment
   - Shifts traffic to the new deployment
   - Decommissions the old "green" deployment

## Setup Requirements

Before deploying the pipeline, you need to:

1. Store your GitHub personal access token in AWS Secrets Manager:
   ```bash
   aws secretsmanager create-secret \
     --name mes-ai-trader/github-token \
     --secret-string "your-github-token"
   ```

2. Create processing scripts in the repository:
   - `mes_ai_trader/models/processing/P-11-data-preprocessing.py`
   - `mes_ai_trader/models/processing/P-12-feature-engineering.py`
   - `mes_ai_trader/models/processing/P-13-model-training.py`
   - `mes_ai_trader/models/register_model.py`

3. Create deployment scripts:
   - `mes_ai_trader/scripts/validate_deployment.py`
   - `mes_ai_trader/scripts/shift_traffic.py`

## Deployment

Deploy the pipeline using CDK:

```bash
cd mes_ai_trader/infra
pip install -r requirements.txt
cdk deploy MESTraderPipelineStack
```

You can customize the deployment using context variables:

```bash
cdk deploy MESTraderPipelineStack \
  --context github_owner=your-github-username \
  --context github_repo=your-repo-name \
  --context github_branch=main
```

Or by setting environment variables:

```bash
export GITHUB_OWNER=your-github-username
export GITHUB_REPO=your-repo-name
export GITHUB_BRANCH=main
cdk deploy MESTraderPipelineStack
```

## Customization

### GitHub Configuration

The pipeline sources code from GitHub. You can configure:

- GitHub owner/organization (`github_owner`)
- Repository name (`github_repo`)
- Branch name (`github_branch`)
- Secret containing GitHub token (`github_token_secret_name`)

### Training Configuration

The pipeline runs SageMaker Processing jobs. You can modify:

- Processing script configurations
- Instance types used for training
- The Model Registry setup

### Deployment Configuration

The blue/green deployment strategy can be customized by modifying:

- Validation criteria in `validate_deployment.py`
- Traffic shifting parameters in `shift_traffic.py`
- Deployment parameters in the CDK stack

## Monitoring

You can monitor the pipeline in the AWS CodePipeline console. The URL is provided in the outputs after deployment.

## Troubleshooting

If a pipeline stage fails:

1. Check the CodeBuild logs for the failed build
2. Verify GitHub webhook integration is properly configured
3. Ensure SageMaker has the correct permissions
4. Check that all required scripts are present in the repository 