"""
SageMaker Pipeline for Skin Lesion Classification

This module defines the end-to-end ML pipeline including:
- Data preprocessing
- Model training with hyperparameter tuning
- Model evaluation
- Model registration
- Conditional deployment
"""

import json
import boto3
from typing import Dict, Any

import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
    ParameterFloat,
    ParameterBoolean,
)
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CreateModelStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.properties import PropertyFile
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics

from .steps import (
    create_preprocessing_step,
    create_training_step,
    create_evaluation_step,
    create_register_model_step,
)


class SkinLesionPipeline:
    """
    SageMaker pipeline for skin lesion classification.
    """

    def __init__(
        self,
        pipeline_name: str,
        role: str,
        bucket: str,
        region: str = "us-east-1",
        config: Dict[str, Any] = None,
    ):
        """
        Initialize the pipeline.

        Args:
            pipeline_name: Name of the pipeline
            role: SageMaker execution role ARN
            bucket: S3 bucket for artifacts
            region: AWS region
            config: Additional configuration
        """
        self.pipeline_name = pipeline_name
        self.role = role
        self.bucket = bucket
        self.region = region
        self.config = config or {}

        self.sagemaker_session = sagemaker.Session()

        # Define pipeline parameters
        self._define_parameters()

    def _define_parameters(self):
        """Define pipeline parameters."""
        # Data parameters
        self.input_data = ParameterString(
            name="InputData",
            default_value=f"s3://{self.bucket}/data/raw",
        )

        self.train_split = ParameterFloat(
            name="TrainSplit",
            default_value=0.8,
        )

        self.val_split = ParameterFloat(
            name="ValSplit",
            default_value=0.1,
        )

        # Processing parameters
        self.processing_instance_type = ParameterString(
            name="ProcessingInstanceType",
            default_value="ml.m5.xlarge",
        )

        self.processing_instance_count = ParameterInteger(
            name="ProcessingInstanceCount",
            default_value=1,
        )

        # Training parameters
        self.training_instance_type = ParameterString(
            name="TrainingInstanceType",
            default_value="ml.p3.2xlarge",
        )

        self.training_instance_count = ParameterInteger(
            name="TrainingInstanceCount",
            default_value=1,
        )

        self.epochs = ParameterInteger(
            name="Epochs",
            default_value=50,
        )

        self.batch_size = ParameterInteger(
            name="BatchSize",
            default_value=32,
        )

        self.learning_rate = ParameterFloat(
            name="LearningRate",
            default_value=0.001,
        )

        self.model_architecture = ParameterString(
            name="ModelArchitecture",
            default_value="resnet50",
        )

        # Model registry parameters
        self.model_package_group_name = ParameterString(
            name="ModelPackageGroupName",
        )

        self.model_approval_status = ParameterString(
            name="ModelApprovalStatus",
            default_value="PendingManualApproval",
        )

        # Evaluation threshold
        self.accuracy_threshold = ParameterFloat(
            name="AccuracyThreshold",
            default_value=0.85,
        )

        # Enable spot instances
        self.use_spot_instances = ParameterBoolean(
            name="UseSpotInstances",
            default_value=True,
        )

    def create_pipeline(self) -> Pipeline:
        """
        Create the SageMaker pipeline.

        Returns:
            SageMaker Pipeline object
        """
        # Step 1: Data preprocessing
        preprocessing_step = create_preprocessing_step(
            role=self.role,
            bucket=self.bucket,
            input_data=self.input_data,
            train_split=self.train_split,
            val_split=self.val_split,
            processing_instance_type=self.processing_instance_type,
            processing_instance_count=self.processing_instance_count,
        )

        # Step 2: Model training
        training_step = create_training_step(
            role=self.role,
            bucket=self.bucket,
            training_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs[
                "train"
            ].S3Output.S3Uri,
            validation_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs[
                "validation"
            ].S3Output.S3Uri,
            training_instance_type=self.training_instance_type,
            training_instance_count=self.training_instance_count,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            model_architecture=self.model_architecture,
            use_spot_instances=self.use_spot_instances,
        )

        # Step 3: Model evaluation
        evaluation_step = create_evaluation_step(
            role=self.role,
            bucket=self.bucket,
            model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            test_data=preprocessing_step.properties.ProcessingOutputConfig.Outputs[
                "test"
            ].S3Output.S3Uri,
            processing_instance_type=self.processing_instance_type,
        )

        # Create evaluation report property file
        evaluation_report = PropertyFile(
            name="EvaluationReport",
            output_name="evaluation",
            path="evaluation.json",
        )
        evaluation_step.add_property_file(evaluation_report)

        # Step 4: Model registration (conditional)
        register_step = create_register_model_step(
            model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            model_package_group_name=self.model_package_group_name,
            model_approval_status=self.model_approval_status,
            evaluation_report=evaluation_report,
        )

        # Condition: Register model only if accuracy meets threshold
        accuracy_condition = ConditionGreaterThanOrEqualTo(
            left=JsonGet(
                step_name=evaluation_step.name,
                property_file=evaluation_report,
                json_path="metrics.accuracy",
            ),
            right=self.accuracy_threshold,
        )

        condition_step = ConditionStep(
            name="CheckAccuracyThreshold",
            conditions=[accuracy_condition],
            if_steps=[register_step],
            else_steps=[],
        )

        # Create pipeline
        pipeline = Pipeline(
            name=self.pipeline_name,
            parameters=[
                self.input_data,
                self.train_split,
                self.val_split,
                self.processing_instance_type,
                self.processing_instance_count,
                self.training_instance_type,
                self.training_instance_count,
                self.epochs,
                self.batch_size,
                self.learning_rate,
                self.model_architecture,
                self.model_package_group_name,
                self.model_approval_status,
                self.accuracy_threshold,
                self.use_spot_instances,
            ],
            steps=[preprocessing_step, training_step, evaluation_step, condition_step],
            sagemaker_session=self.sagemaker_session,
        )

        return pipeline

    def deploy_pipeline(self, role_arn: str = None) -> Dict[str, Any]:
        """
        Deploy the pipeline to SageMaker.

        Args:
            role_arn: Optional role ARN for pipeline execution

        Returns:
            Pipeline deployment response
        """
        pipeline = self.create_pipeline()

        # Upsert pipeline (create or update)
        response = pipeline.upsert(role_arn=role_arn or self.role)

        print(f"Pipeline {self.pipeline_name} deployed successfully")
        print(f"Pipeline ARN: {response['PipelineArn']}")

        return response

    def start_execution(self, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Start a pipeline execution.

        Args:
            parameters: Optional parameters to override defaults

        Returns:
            Execution response
        """
        pipeline = self.create_pipeline()

        execution = pipeline.start(parameters=parameters)

        print(f"Pipeline execution started: {execution.arn}")

        return {
            "execution_arn": execution.arn,
            "execution_id": execution.arn.split("/")[-1],
        }

    def get_execution_status(self, execution_arn: str) -> str:
        """
        Get the status of a pipeline execution.

        Args:
            execution_arn: ARN of the pipeline execution

        Returns:
            Execution status
        """
        client = boto3.client("sagemaker", region_name=self.region)

        response = client.describe_pipeline_execution(
            PipelineExecutionArn=execution_arn
        )

        return response["PipelineExecutionStatus"]


def main():
    """
    Main function to deploy the pipeline.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-name", type=str, required=True)
    parser.add_argument("--role", type=str, required=True)
    parser.add_argument("--bucket", type=str, required=True)
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--model-package-group", type=str, required=True)
    parser.add_argument("--deploy", action="store_true", help="Deploy the pipeline")
    parser.add_argument("--start", action="store_true", help="Start pipeline execution")

    args = parser.parse_args()

    # Create pipeline
    pipeline_builder = SkinLesionPipeline(
        pipeline_name=args.pipeline_name,
        role=args.role,
        bucket=args.bucket,
        region=args.region,
    )

    if args.deploy:
        # Deploy pipeline
        response = pipeline_builder.deploy_pipeline()
        print(json.dumps(response, indent=2))

    if args.start:
        # Start execution
        execution = pipeline_builder.start_execution(
            parameters={
                "ModelPackageGroupName": args.model_package_group,
            }
        )
        print(json.dumps(execution, indent=2))


if __name__ == "__main__":
    main()
