"""
CDK Stack for nightly retraining of ML models

This stack creates:
1. Step Functions workflow for model retraining
2. EventBridge rule to trigger retraining at 01:00 UTC
3. EventBridge rule to monitor for model drift
4. SNS topic for model drift notifications
5. IAM roles for execution
"""

import os
import json
from aws_cdk import (
    Stack,
    aws_stepfunctions as sfn,
    aws_events as events,
    aws_events_targets as targets,
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_sagemaker as sagemaker,
    aws_sns as sns,
    aws_sns_subscriptions as sns_subs,
    CfnOutput,
    Duration,
)
from constructs import Construct


class NightlyRetrainStack(Stack):
    """CDK Stack for nightly retraining of MES AI Trader models."""

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        """Initialize the nightly retraining stack.
        
        Args:
            scope: CDK app scope
            construct_id: Stack identifier
            **kwargs: Additional arguments to pass to Stack
        """
        super().__init__(scope, construct_id, **kwargs)

        # Create IAM role for Step Functions
        step_functions_role = iam.Role(
            self, "StepFunctionsRole",
            assumed_by=iam.ServicePrincipal("states.amazonaws.com"),
            description="Role for Step Functions to execute SageMaker jobs and Lambda functions"
        )
        
        # Add required permissions
        step_functions_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess")
        )
        step_functions_role.add_to_policy(
            iam.PolicyStatement(
                actions=["lambda:InvokeFunction"],
                resources=["*"]
            )
        )
        
        # Create SNS topic for model drift alerts
        model_drift_topic = sns.Topic(
            self, "ModelDriftTopic",
            topic_name="mes-ai-trader-model-drift-alerts",
            display_name="MES AI Trader Model Drift Alerts"
        )
        
        # Create Lambda for submitting model for review upon drift detection
        submit_model_for_review_lambda = lambda_.Function(
            self, "SubmitModelForReviewLambda",
            function_name="SubmitModelForReview",
            runtime=lambda_.Runtime.PYTHON_3_9,
            code=lambda_.Code.from_asset("mes_ai_trader/models"),
            handler="register_model.submit_for_review",
            timeout=Duration.minutes(5),
            environment={
                "MODEL_PACKAGE_GROUP_NAME": "MESTraderModelPackageGroup",
                "NOTIFICATION_TOPIC_ARN": model_drift_topic.topic_arn
            }
        )
        
        # Grant permissions to the Lambda
        submit_model_for_review_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "sagemaker:UpdateModelPackage",
                    "sagemaker:DescribeModelPackage",
                    "s3:GetObject",
                    "sns:Publish"
                ],
                resources=["*"]
            )
        )
        
        # Create Lambda for updating model approval status
        update_model_approval_lambda = lambda_.Function(
            self, "UpdateModelApprovalStatus",
            function_name="UpdateModelApprovalStatus",
            runtime=lambda_.Runtime.PYTHON_3_9,
            code=lambda_.Code.from_asset("mes_ai_trader/models"),
            handler="register_model.update_approval_status",
            timeout=Duration.minutes(5),
            environment={
                "MODEL_PACKAGE_GROUP_NAME": "MESTraderModelPackageGroup"
            }
        )
        
        # Grant permissions to the Lambda
        update_model_approval_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=["sagemaker:UpdateModelPackage"],
                resources=["*"]
            )
        )
        
        # Load the Step Functions definition
        with open("mes_ai_trader/infra/nightly_retrain_sf.json", "r") as f:
            definition_template = json.load(f)
            
        # Replace placeholders in the definition
        definition_str = json.dumps(definition_template)
        definition_str = definition_str.replace("{{account}}", self.account)
        definition_str = definition_str.replace("{{region}}", self.region)
        definition_str = definition_str.replace("{{SageMakerRoleArn}}", step_functions_role.role_arn)
        definition_str = definition_str.replace("{{ModelDriftSNSTopicArn}}", model_drift_topic.topic_arn)
        definition_str = definition_str.replace(
            "{{SubmitModelForReviewFunctionArn}}", 
            submit_model_for_review_lambda.function_arn
        )
        
        # Create the Step Functions state machine
        state_machine = sfn.CfnStateMachine(
            self, "NightlyRetrainStateMachine",
            role_arn=step_functions_role.role_arn,
            definition_string=definition_str,
            state_machine_name="MESTraderNightlyRetrain"
        )
        
        # Create EventBridge rule to trigger the workflow at 01:00 UTC
        nightly_rule = events.Rule(
            self, "NightlyRetrainRule",
            rule_name="MESTraderNightlyRetrainTrigger",
            schedule=events.Schedule.cron(
                minute="0",
                hour="1",
                month="*",
                week_day="*",
                year="*"
            ),
            description="Triggers the MES Trader nightly retraining workflow at 01:00 UTC"
        )
        
        # Add Step Functions state machine as target
        nightly_rule.add_target(
            targets.SfnStateMachine(
                sfn.StateMachine.from_state_machine_arn(
                    self, "ImportedStateMachine",
                    state_machine_arn=state_machine.attr_arn
                )
            )
        )
        
        # Create EventBridge rule to monitor model performance metrics every 4 hours
        drift_monitor_rule = events.Rule(
            self, "DriftMonitorRule",
            rule_name="MESTraderDriftMonitorTrigger",
            schedule=events.Schedule.cron(
                minute="0",
                hour="*/4",  # Every 4 hours
                month="*",
                week_day="*",
                year="*"
            ),
            description="Monitors model metrics for drift every 4 hours"
        )
        
        # Create Lambda function for drift monitoring
        drift_monitor_lambda = lambda_.Function(
            self, "DriftMonitorLambda",
            function_name="ModelDriftMonitor",
            runtime=lambda_.Runtime.PYTHON_3_9,
            code=lambda_.Code.from_asset("mes_ai_trader/utils"),
            handler="model_monitor.check_model_drift",
            timeout=Duration.minutes(5),
            environment={
                "MODEL_PACKAGE_GROUP_NAME": "MESTraderModelPackageGroup",
                "NOTIFICATION_TOPIC_ARN": model_drift_topic.topic_arn,
                "DRIFT_THRESHOLD": "0.1"  # 10% threshold for drift
            }
        )
        
        # Add drift monitor Lambda as target
        drift_monitor_rule.add_target(
            targets.LambdaFunction(drift_monitor_lambda)
        )
        
        # Grant permissions to the drift monitor Lambda
        drift_monitor_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=[
                    "sagemaker:DescribeModelPackage",
                    "cloudwatch:GetMetricData",
                    "s3:GetObject",
                    "sns:Publish"
                ],
                resources=["*"]
            )
        )
        
        # Create SageMaker Model Package Group if it doesn't exist
        model_package_group = sagemaker.CfnModelPackageGroup(
            self, "ModelPackageGroup",
            model_package_group_name="MESTraderModelPackageGroup",
            model_package_group_description="MES Trader ML Models",
            tags=[{"key": "Project", "value": "MESTrader"}],
        )
        
        # Output the State Machine ARN
        CfnOutput(
            self, "StateMachineArn",
            value=state_machine.attr_arn,
            description="ARN of the nightly retraining Step Functions workflow"
        )
        
        # Output the EventBridge rule ARNs
        CfnOutput(
            self, "NightlyRetrainRuleArn",
            value=nightly_rule.rule_arn,
            description="ARN of the EventBridge rule that triggers the nightly retraining"
        )
        
        CfnOutput(
            self, "DriftMonitorRuleArn",
            value=drift_monitor_rule.rule_arn,
            description="ARN of the EventBridge rule that monitors for model drift"
        )
        
        CfnOutput(
            self, "ModelDriftTopicArn",
            value=model_drift_topic.topic_arn,
            description="ARN of the SNS topic for model drift alerts"
        ) 