"""
CDK Stack for MES Trader SageMaker Inference Service

This stack creates:
1. ECR repository for the inference service
2. SageMaker model and endpoint configuration
3. VPC endpoint for SageMaker
4. IAM roles with appropriate permissions
"""

import os
from aws_cdk import (
    Stack,
    aws_ec2 as ec2,
    aws_ecr_assets as ecr_assets,
    aws_iam as iam,
    aws_sagemaker as sagemaker,
    aws_ecr as ecr,
    CfnOutput,
    RemovalPolicy,
    Duration,
    aws_ssm as ssm,
)
from constructs import Construct


class InferenceStack(Stack):
    """CDK Stack for deploying the MES trader inference service to SageMaker."""

    def __init__(self, scope: Construct, construct_id: str, vpc=None, **kwargs) -> None:
        """Initialize the inference stack.
        
        Args:
            scope: CDK app scope
            construct_id: Stack identifier
            vpc: Optional VPC to use (will create one if not provided)
            **kwargs: Additional arguments to pass to Stack
        """
        super().__init__(scope, construct_id, **kwargs)

        # Create VPC if one isn't provided
        if not vpc:
            vpc = ec2.Vpc(
                self, "MESTraderVPC",
                max_azs=2,
                nat_gateways=1,
                subnet_configuration=[
                    ec2.SubnetConfiguration(
                        name="Public",
                        subnet_type=ec2.SubnetType.PUBLIC,
                        cidr_mask=24
                    ),
                    ec2.SubnetConfiguration(
                        name="Private",
                        subnet_type=ec2.SubnetType.PRIVATE_WITH_NAT,
                        cidr_mask=24
                    ),
                ],
            )

        # Create an ECR repository for the inference service
        ecr_repository = ecr.Repository(
            self, "MESTraderInferenceRepo",
            repository_name="mes-trader-inference",
            removal_policy=RemovalPolicy.RETAIN,
            image_scan_on_push=True,
        )

        # Build the Docker image and push to ECR
        docker_image_asset = ecr_assets.DockerImageAsset(
            self, "MESTraderInferenceImage",
            directory=os.path.join(os.path.dirname(__file__), '..', '..', 'mes_ai_trader'),
            file="services/inference/Dockerfile",
        )

        # Create IAM role for SageMaker with required permissions
        sagemaker_role = iam.Role(
            self, "SageMakerExecutionRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"),
            ],
        )

        # Add DynamoDB read permissions for Feast
        sagemaker_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "dynamodb:BatchGetItem",
                    "dynamodb:DescribeTable",
                    "dynamodb:GetItem",
                    "dynamodb:Query",
                    "dynamodb:Scan",
                ],
                resources=["arn:aws:dynamodb:*:*:table/feast-*"],
            )
        )

        # Add S3 read permissions for Feast offline store
        sagemaker_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "s3:GetObject",
                    "s3:ListBucket",
                ],
                resources=[
                    "arn:aws:s3:::mes-ai-trader-feast-*",
                    "arn:aws:s3:::mes-ai-trader-feast-*/*",
                ],
            )
        )

        # Create VPC endpoint for SageMaker runtime
        sagemaker_endpoint = ec2.InterfaceVpcEndpoint(
            self, "SageMakerEndpoint",
            vpc=vpc,
            service=ec2.InterfaceVpcEndpointAwsService.SAGEMAKER_RUNTIME,
            private_dns_enabled=True,
            subnets=ec2.SubnetSelection(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_NAT
            ),
        )

        # Create SageMaker model
        model = sagemaker.CfnModel(
            self, "MESTraderModel",
            execution_role_arn=sagemaker_role.role_arn,
            primary_container=sagemaker.CfnModel.ContainerDefinitionProperty(
                image=docker_image_asset.image_uri,
                environment={
                    "FEAST_REPO_PATH": "/app/feature_repo",
                    "LGBM_MODEL_PATH": "/app/models/lgbm/model.lgb",
                    "TFT_MODEL_PATH": "/app/models/tft/model.pt",
                    "PPO_MODEL_PATH": "/app/models/ppo/ppo.zip",
                },
            ),
        )

        # Create SageMaker endpoint configuration
        endpoint_config = sagemaker.CfnEndpointConfig(
            self, "MESTraderEndpointConfig",
            production_variants=[
                sagemaker.CfnEndpointConfig.ProductionVariantProperty(
                    initial_instance_count=1,
                    instance_type="ml.m5.large",
                    model_name=model.attr_model_name,
                    variant_name="AllTraffic",
                    initial_variant_weight=1.0,
                )
            ],
        )

        # Create SageMaker endpoint
        endpoint = sagemaker.CfnEndpoint(
            self, "MESTraderEndpoint",
            endpoint_config_name=endpoint_config.attr_endpoint_config_name,
            endpoint_name="mes-trader-inference",
        )

        # Define dependencies
        endpoint.node.add_dependency(endpoint_config)
        endpoint_config.node.add_dependency(model)

        # Output the endpoint URL and other important information
        CfnOutput(
            self, "SageMakerEndpointName",
            value=endpoint.attr_endpoint_name,
            description="SageMaker Endpoint Name",
        )

        CfnOutput(
            self, "SageMakerRole",
            value=sagemaker_role.role_arn,
            description="IAM Role ARN for SageMaker execution",
        )

        CfnOutput(
            self, "ECRRepository",
            value=ecr_repository.repository_uri,
            description="ECR Repository URI",
        )

        # Create Parameter Store entry for trading mode
        trading_mode_param = ssm.StringParameter(
            self, "TradingModeParameter",
            parameter_name="/mes-ai-trader/trading-mode",
            string_value="shadow",
            description="Trading mode for MES AI Trader (live, shadow)",
            tier=ssm.ParameterTier.STANDARD,
        )

        # Output the trading mode parameter name
        CfnOutput(
            self, "TradingModeParameterName",
            value=trading_mode_param.parameter_name,
            description="Parameter Store key for trading mode"
        ) 