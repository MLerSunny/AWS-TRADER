#!/usr/bin/env python3
"""
CDK App for MES AI Trader

This file defines the CDK app and instantiates all the required stacks.
"""

import os
import aws_cdk as cdk
from inference_stack import InferenceStack
from dashboard_stack import DashboardStack
from pipeline import PipelineStack
from nightly_retrain_stack import NightlyRetrainStack

app = cdk.App()

# Define the AWS environment
aws_env = cdk.Environment(
    account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
    region=os.environ.get("CDK_DEFAULT_REGION", "us-east-1")
)

# Create the inference stack
inference_stack = InferenceStack(
    app,
    "MESTraderInferenceStack",
    env=aws_env,
    description="SageMaker inference service for MES AI Trader"
)

# Create the dashboard stack
dashboard_stack = DashboardStack(
    app,
    "MESTraderDashboardStack",
    env=aws_env,
    description="Streamlit dashboard and CloudWatch monitoring for MES AI Trader"
)

# Create the CI/CD pipeline stack
pipeline_stack = PipelineStack(
    app,
    "MESTraderPipelineStack",
    env=aws_env,
    description="CI/CD pipeline for MES AI Trader"
)

# Create the nightly retraining stack
nightly_retrain_stack = NightlyRetrainStack(
    app,
    "MESTraderNightlyRetrainStack",
    env=aws_env,
    description="Nightly model retraining for MES AI Trader at 01:00 UTC"
)

app.synth() 