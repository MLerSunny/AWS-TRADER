"""
CI/CD Pipeline for MES AI Trader

This stack creates a complete CI/CD pipeline that automates:
1. Source code retrieval from GitHub
2. Build and test
3. Model training
4. Model registration
5. Blue/green deployment to production
"""

import os
from aws_cdk import (
    Stack,
    aws_codebuild as codebuild,
    aws_codecommit as codecommit,
    aws_codepipeline as codepipeline,
    aws_codepipeline_actions as codepipeline_actions,
    aws_iam as iam,
    aws_s3 as s3,
    aws_sagemaker as sagemaker,
    aws_ecr as ecr,
    SecretValue,
    CfnOutput,
    Duration,
    RemovalPolicy,
)
from constructs import Construct


class PipelineStack(Stack):
    """CDK Stack for CI/CD pipeline for MES AI Trader."""

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        """Initialize the pipeline stack.
        
        Args:
            scope: CDK app scope
            construct_id: Stack identifier
            **kwargs: Additional arguments to pass to Stack
        """
        super().__init__(scope, construct_id, **kwargs)

        # GitHub configuration - read from environment or context
        github_owner = self.node.try_get_context("github_owner") or os.environ.get("GITHUB_OWNER", "default-owner")
        github_repo = self.node.try_get_context("github_repo") or os.environ.get("GITHUB_REPO", "AWS-Trader")
        github_branch = self.node.try_get_context("github_branch") or os.environ.get("GITHUB_BRANCH", "main")
        github_token_secret_name = self.node.try_get_context("github_token_secret_name") or os.environ.get("GITHUB_TOKEN_SECRET_NAME", "mes-ai-trader/github-token")

        # Pipeline artifacts
        source_output = codepipeline.Artifact("SourceCode")
        build_output = codepipeline.Artifact("BuildOutput")
        train_output = codepipeline.Artifact("TrainOutput")
        model_package_group_name = "MESTraderModelPackageGroup"

        # Create artifact bucket for pipeline
        artifact_bucket = s3.Bucket(
            self, "ArtifactBucket",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
        )

        # Create ECR repository for build images
        build_repository = ecr.Repository(
            self, "BuildRepository",
            repository_name="mes-trader-build",
            removal_policy=RemovalPolicy.RETAIN,
        )

        # Create Model Package Group for SageMaker Model Registry
        model_package_group = sagemaker.CfnModelPackageGroup(
            self, "ModelPackageGroup",
            model_package_group_name=model_package_group_name,
            model_package_group_description="MES Trader ML Models",
            tags=[{"key": "Project", "value": "MESTrader"}],
        )

        # Create CodeBuild project for building the ML pipeline
        build_project = codebuild.PipelineProject(
            self, "BuildProject",
            project_name="MESTraderBuild",
            build_spec=codebuild.BuildSpec.from_object({
                "version": "0.2",
                "phases": {
                    "install": {
                        "runtime-versions": {
                            "java": "corretto11",
                            "python": "3.9"
                        },
                        "commands": [
                            "pip install pytest pytest-cov",
                            "curl -s https://get.sdkman.io | bash",
                            "source $HOME/.sdkman/bin/sdkman-init.sh",
                            "sdk install sbt"
                        ]
                    },
                    "pre_build": {
                        "commands": [
                            "echo Running tests...",
                            "cd mes_ai_trader && python -m pytest tests/",
                            "echo Running Scala tests...",
                            "cd mes_ai_trader/pipelines/flink && sbt test"
                        ]
                    },
                    "build": {
                        "commands": [
                            "echo Building Scala flink job...",
                            "cd mes_ai_trader/pipelines/flink && sbt assembly",
                            "echo Packaging ML training code...",
                            "cd mes_ai_trader/models",
                            "zip -r ../../model-training.zip ."
                        ]
                    },
                    "post_build": {
                        "commands": [
                            "echo Build completed on `date`"
                        ]
                    }
                },
                "artifacts": {
                    "files": [
                        "mes_ai_trader/pipelines/flink/target/scala-2.12/*.jar",
                        "model-training.zip"
                    ]
                }
            }),
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxBuildImage.AMAZON_LINUX_2_3,
                privileged=True,
            ),
            timeout=Duration.minutes(30),
            description="Build and test the MES Trader codebase"
        )

        # Create CodeBuild project for SageMaker training
        training_project = codebuild.PipelineProject(
            self, "TrainingProject",
            project_name="MESTraderTraining",
            build_spec=codebuild.BuildSpec.from_object({
                "version": "0.2",
                "phases": {
                    "install": {
                        "runtime-versions": {
                            "python": "3.9"
                        },
                        "commands": [
                            "pip install sagemaker boto3 pandas numpy scikit-learn"
                        ]
                    },
                    "build": {
                        "commands": [
                            "echo Starting SageMaker Processing Jobs...",
                            "python mes_ai_trader/models/processing/P-11-data-preprocessing.py",
                            "python mes_ai_trader/models/processing/P-12-feature-engineering.py",
                            "python mes_ai_trader/models/processing/P-13-model-training.py",
                            "echo Registering models in Model Registry...",
                            "python mes_ai_trader/models/register_model.py --model-package-group-name " + model_package_group_name
                        ]
                    }
                },
                "artifacts": {
                    "files": [
                        "mes_ai_trader/models/output/model-info.json"
                    ]
                }
            }),
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxBuildImage.AMAZON_LINUX_2_3,
                privileged=True,
            ),
            timeout=Duration.hours(2),
            description="Run SageMaker training jobs for the MES Trader models"
        )

        # Create CodeBuild project for deployment
        deploy_project = codebuild.PipelineProject(
            self, "DeployProject",
            project_name="MESTraderDeploy",
            build_spec=codebuild.BuildSpec.from_object({
                "version": "0.2",
                "phases": {
                    "install": {
                        "runtime-versions": {
                            "python": "3.9",
                            "nodejs": "16"
                        },
                        "commands": [
                            "pip install awscli boto3",
                            "npm install -g aws-cdk"
                        ]
                    },
                    "pre_build": {
                        "commands": [
                            "cd mes_ai_trader/infra",
                            "pip install -r requirements.txt"
                        ]
                    },
                    "build": {
                        "commands": [
                            "echo Deploying with blue/green strategy...",
                            "export DEPLOYMENT_TYPE=blue",
                            "export MODEL_PACKAGE_ARN=$(cat ../models/output/model-info.json | jq -r '.ModelPackageArn')",
                            "cdk deploy MESTraderInferenceStack-Blue --require-approval never --parameters modelPackageArn=$MODEL_PACKAGE_ARN",
                            "echo Validating blue deployment...",
                            "python ../scripts/validate_deployment.py --deployment-type blue",
                            "echo Shifting traffic to blue deployment...",
                            "python ../scripts/shift_traffic.py --target blue --percentage 100",
                            "echo Retiring green deployment...",
                            "cdk destroy MESTraderInferenceStack-Green --force"
                        ]
                    }
                }
            }),
            environment=codebuild.BuildEnvironment(
                build_image=codebuild.LinuxBuildImage.AMAZON_LINUX_2_3,
                privileged=True,
            ),
            timeout=Duration.minutes(30),
            description="Deploy the MES Trader models to production"
        )

        # Grant SageMaker permissions to the training project
        training_project.role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess")
        )

        # Grant permissions to the deploy project
        deploy_project.role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess")
        )
        deploy_project.role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AWSCloudFormationFullAccess")
        )

        # Create the pipeline
        pipeline = codepipeline.Pipeline(
            self, "MESTraderPipeline",
            pipeline_name="MESTraderPipeline",
            artifact_bucket=artifact_bucket,
            restart_execution_on_update=True,
        )

        # Add Source Stage
        source_stage = pipeline.add_stage(
            stage_name="Source",
            actions=[
                codepipeline_actions.GitHubSourceAction(
                    action_name="GitHub_Source",
                    owner=github_owner,
                    repo=github_repo,
                    branch=github_branch,
                    oauth_token=SecretValue.secrets_manager(github_token_secret_name),
                    output=source_output,
                    trigger=codepipeline_actions.GitHubTrigger.WEBHOOK,
                )
            ]
        )

        # Add Build Stage
        build_stage = pipeline.add_stage(
            stage_name="Build",
            actions=[
                codepipeline_actions.CodeBuildAction(
                    action_name="BuildAndTest",
                    project=build_project,
                    input=source_output,
                    outputs=[build_output],
                )
            ]
        )

        # Add Training Stage
        train_stage = pipeline.add_stage(
            stage_name="Train",
            actions=[
                codepipeline_actions.CodeBuildAction(
                    action_name="TrainModels",
                    project=training_project,
                    input=source_output,
                    outputs=[train_output],
                )
            ]
        )

        # Add Approval Stage
        approval_stage = pipeline.add_stage(
            stage_name="Approval",
            actions=[
                codepipeline_actions.ManualApprovalAction(
                    action_name="ApproveDeployment",
                    notification_topic=None,  # Optionally add an SNS topic here
                    additional_information="Please review the training results and approve for deployment to production.",
                )
            ]
        )

        # Add Deploy Stage
        deploy_stage = pipeline.add_stage(
            stage_name="Deploy",
            actions=[
                codepipeline_actions.CodeBuildAction(
                    action_name="DeployToProduction",
                    project=deploy_project,
                    input=train_output,
                )
            ]
        )

        # Output the pipeline URL
        CfnOutput(
            self, "PipelineConsoleUrl",
            value=f"https://{self.region}.console.aws.amazon.com/codesuite/codepipeline/pipelines/{pipeline.pipeline_name}/view?region={self.region}",
            description="URL to the CodePipeline console"
        ) 