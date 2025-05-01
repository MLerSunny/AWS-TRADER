"""
CDK Stack for MES Trader Dashboard Service

This stack creates:
1. App Runner service for hosting the Streamlit dashboard
2. CloudWatch dashboard with inference latency and risk-gate metrics
3. IAM roles with appropriate permissions
"""

import os
from aws_cdk import (
    Stack,
    aws_apprunner as apprunner,
    aws_cloudwatch as cloudwatch,
    aws_ecr_assets as ecr_assets,
    aws_iam as iam,
    aws_logs as logs,
    CfnOutput,
    Duration,
    RemovalPolicy,
    aws_secretsmanager as secretsmanager,
    aws_ssm as ssm,
)
from constructs import Construct


class DashboardStack(Stack):
    """CDK Stack for deploying the MES trader dashboard to App Runner with monitoring."""

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        """Initialize the dashboard stack.
        
        Args:
            scope: CDK app scope
            construct_id: Stack identifier
            **kwargs: Additional arguments to pass to Stack
        """
        super().__init__(scope, construct_id, **kwargs)

        # Build the Docker image for App Runner
        dashboard_image = ecr_assets.DockerImageAsset(
            self, "DashboardImage",
            directory=os.path.join(os.path.dirname(__file__), '..', 'services', 'dashboard'),
            file="Dockerfile",
        )

        # Create IAM role for App Runner service with required permissions
        app_runner_role = iam.Role(
            self, "AppRunnerServiceRole",
            assumed_by=iam.ServicePrincipal("tasks.apprunner.amazonaws.com"),
        )

        # Add permissions for App Runner to access dashboard assets
        app_runner_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "bedrock:InvokeModel",
                ],
                resources=["*"],  # Ideally restrict to specific model ARNs
            )
        )

        # Add permissions to access RDS PostgreSQL database
        app_runner_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "rds-db:connect",
                ],
                resources=["*"],  # Ideally restrict to specific RDS ARN
            )
        )

        # Add SSM permissions for accessing parameters
        app_runner_role.add_to_policy(
            iam.PolicyStatement(
                actions=[
                    "ssm:GetParameter",
                    "ssm:GetParameters",
                ],
                resources=["arn:aws:ssm:*:*:parameter/mes-ai-trader/*"],
            )
        )

        # Create App Runner service for Streamlit dashboard
        dashboard_service = apprunner.CfnService(
            self, "DashboardService",
            service_name="mes-trader-dashboard",
            source_configuration=apprunner.CfnService.SourceConfigurationProperty(
                authentication_configuration=apprunner.CfnService.AuthenticationConfigurationProperty(
                    access_role_arn=app_runner_role.role_arn,
                ),
                image_repository=apprunner.CfnService.ImageRepositoryProperty(
                    image_identifier=dashboard_image.image_uri,
                    image_repository_type="ECR",
                    image_configuration=apprunner.CfnService.ImageConfigurationProperty(
                        port="8501",
                        runtime_environment_variables={
                            "DB_HOST": ssm.StringParameter.from_string_parameter_name(
                                self, "DbHostParam", "/mes-ai-trader/db-host"
                            ).string_value,
                            "DB_NAME": ssm.StringParameter.from_string_parameter_name(
                                self, "DbNameParam", "/mes-ai-trader/db-name"
                            ).string_value,
                            "DB_USER": ssm.StringParameter.from_string_parameter_name(
                                self, "DbUserParam", "/mes-ai-trader/db-user"
                            ).string_value,
                            "DB_PASSWORD": secretsmanager.Secret.from_secret_name_v2(
                                self, "DbPasswordSecret", "/mes-ai-trader/db-password"
                            ).secret_value.to_string(),
                            "AWS_REGION": self.region,
                            "CLAUDE_MODEL_ID": "anthropic.claude-3-sonnet-20240229-v1:0",
                        },
                    ),
                ),
            ),
            health_check_configuration=apprunner.CfnService.HealthCheckConfigurationProperty(
                protocol="TCP",
                path="/",
                interval=10,
                timeout=5,
                healthy_threshold=1,
                unhealthy_threshold=5,
            ),
            instance_configuration=apprunner.CfnService.InstanceConfigurationProperty(
                cpu="1 vCPU",
                memory="2 GB",
                instance_role_arn=app_runner_role.role_arn,
            ),
            auto_scaling_configuration_arn=apprunner.CfnAutoScalingConfiguration(
                self, "DashboardAutoScaling",
                auto_scaling_configuration_name="mes-dashboard-autoscaling",
                max_concurrency=10,
                max_size=2,
                min_size=1,
            ).attr_auto_scaling_configuration_arn,
        )

        # Create CloudWatch Dashboard for monitoring
        dashboard = cloudwatch.Dashboard(
            self, "MESTraderDashboard",
            dashboard_name="MES-Trader-Metrics",
        )

        # Add inference latency metrics
        inference_latency_widget = cloudwatch.GraphWidget(
            title="Inference Latency",
            left=[
                cloudwatch.Metric(
                    namespace="MESTrader/Inference",
                    metric_name="Latency",
                    statistic="p50",
                    period=Duration.minutes(1),
                    label="p50 Latency",
                ),
                cloudwatch.Metric(
                    namespace="MESTrader/Inference",
                    metric_name="Latency",
                    statistic="p95",
                    period=Duration.minutes(1),
                    label="p95 Latency",
                ),
            ],
            width=12,
            height=6,
        )

        # Add risk gate rejection metrics
        risk_gate_widget = cloudwatch.GraphWidget(
            title="Risk Gate Rejections",
            left=[
                cloudwatch.Metric(
                    namespace="MESTrader/RiskGate",
                    metric_name="Rejections",
                    statistic="sum",
                    period=Duration.minutes(5),
                    label="Rejection Count",
                ),
                cloudwatch.Metric(
                    namespace="MESTrader/RiskGate",
                    metric_name="RejectionRate",
                    statistic="avg",
                    period=Duration.minutes(5),
                    label="Rejection Rate (%)",
                ),
            ],
            width=12,
            height=6,
        )

        # Add widgets to dashboard
        dashboard.add_widgets(
            cloudwatch.TextWidget(
                markdown="# MES AI Trader Monitoring\nInference performance and risk metrics",
                width=24,
                height=2,
            ),
            inference_latency_widget,
            risk_gate_widget,
            cloudwatch.GraphWidget(
                title="Database Connections",
                left=[
                    cloudwatch.Metric(
                        namespace="AWS/RDS",
                        metric_name="DatabaseConnections",
                        statistic="sum",
                        period=Duration.minutes(1),
                        label="DB Connections",
                    ),
                ],
                width=12,
                height=6,
            ),
            cloudwatch.LogQueryWidget(
                title="Recent Dashboard Errors",
                log_group_names=[f"/aws/apprunner/{dashboard_service.attr_service_name}/application"],
                query_lines=[
                    "fields @timestamp, @message",
                    "filter level = 'ERROR'",
                    "sort @timestamp desc",
                    "limit 20",
                ],
                width=24,
                height=8,
            ),
        )

        # Create CloudWatch alarms for critical metrics
        cloudwatch.Alarm(
            self, "HighLatencyAlarm",
            metric=cloudwatch.Metric(
                namespace="MESTrader/Inference",
                metric_name="Latency",
                statistic="p95",
                period=Duration.minutes(5),
            ),
            threshold=1000,  # 1000ms = 1s
            evaluation_periods=3,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            alarm_description="Alert when inference latency is too high",
            alarm_name="MES-Trader-HighLatencyAlarm",
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        )

        cloudwatch.Alarm(
            self, "HighRejectionRateAlarm",
            metric=cloudwatch.Metric(
                namespace="MESTrader/RiskGate",
                metric_name="RejectionRate",
                statistic="avg",
                period=Duration.minutes(10),
            ),
            threshold=50,  # 50% rejection rate
            evaluation_periods=2,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
            alarm_description="Alert when risk gate rejection rate is too high",
            alarm_name="MES-Trader-HighRejectionRateAlarm",
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        )

        # Output the Dashboard URL
        CfnOutput(
            self, "DashboardURL",
            value=f"https://{dashboard_service.attr_service_url}",
            description="URL for the Streamlit dashboard",
        )

        # Output CloudWatch Dashboard URL
        CfnOutput(
            self, "CloudWatchDashboardURL",
            value=f"https://{self.region}.console.aws.amazon.com/cloudwatch/home?region={self.region}#dashboards:name={dashboard.dashboard_name}",
            description="URL for the CloudWatch dashboard",
        ) 