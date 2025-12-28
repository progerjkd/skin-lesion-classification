"""
SageMaker Pipeline Step Definitions

This module contains functions to create individual pipeline steps.
"""

from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.pytorch.estimator import PyTorch
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.properties import PropertyFile


def create_preprocessing_step(
    role: str,
    bucket: str,
    input_data,
    train_split,
    val_split,
    processing_instance_type,
    processing_instance_count,
) -> ProcessingStep:
    """
    Create preprocessing step for data preparation.

    Args:
        role: SageMaker execution role
        bucket: S3 bucket for outputs
        input_data: Input data S3 path parameter
        train_split: Train split ratio parameter
        val_split: Validation split ratio parameter
        processing_instance_type: Instance type parameter
        processing_instance_count: Instance count parameter

    Returns:
        ProcessingStep for preprocessing
    """
    sklearn_processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name="skin-lesion-preprocessing",
        role=role,
    )

    preprocessing_step = ProcessingStep(
        name="PreprocessData",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(
                source=input_data,
                destination="/opt/ml/processing/input",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/train",
                destination=f"s3://{bucket}/data/processed/train",
            ),
            ProcessingOutput(
                output_name="validation",
                source="/opt/ml/processing/validation",
                destination=f"s3://{bucket}/data/processed/validation",
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/test",
                destination=f"s3://{bucket}/data/processed/test",
            ),
        ],
        code="preprocessing/preprocess.py",
        job_arguments=[
            "--train-split",
            str(train_split),
            "--val-split",
            str(val_split),
        ],
    )

    return preprocessing_step


def create_training_step(
    role: str,
    bucket: str,
    training_data,
    validation_data,
    training_instance_type,
    training_instance_count,
    epochs,
    batch_size,
    learning_rate,
    model_architecture,
    use_spot_instances,
) -> TrainingStep:
    """
    Create training step for model training.

    Args:
        role: SageMaker execution role
        bucket: S3 bucket for outputs
        training_data: Training data S3 path
        validation_data: Validation data S3 path
        training_instance_type: Instance type parameter
        training_instance_count: Instance count parameter
        epochs: Number of epochs parameter
        batch_size: Batch size parameter
        learning_rate: Learning rate parameter
        model_architecture: Model architecture parameter
        use_spot_instances: Use spot instances parameter

    Returns:
        TrainingStep for model training
    """
    # PyTorch estimator for training
    pytorch_estimator = PyTorch(
        entry_point="train.py",
        source_dir="training",
        role=role,
        instance_type=training_instance_type,
        instance_count=training_instance_count,
        framework_version="2.1.0",
        py_version="py310",
        base_job_name="skin-lesion-training",
        hyperparameters={
            "epochs": epochs,
            "batch-size": batch_size,
            "learning-rate": learning_rate,
            "model-architecture": model_architecture,
        },
        use_spot_instances=use_spot_instances,
        max_run=86400,  # 24 hours
        max_wait=90000 if use_spot_instances else None,  # 25 hours for spot
        checkpoint_s3_uri=f"s3://{bucket}/checkpoints",
        output_path=f"s3://{bucket}/models",
        metric_definitions=[
            {"Name": "train:loss", "Regex": "Train Loss: ([0-9\\.]+)"},
            {"Name": "train:accuracy", "Regex": "Train Accuracy: ([0-9\\.]+)"},
            {"Name": "validation:loss", "Regex": "Validation Loss: ([0-9\\.]+)"},
            {"Name": "validation:accuracy", "Regex": "Validation Accuracy: ([0-9\\.]+)"},
            {"Name": "validation:auc", "Regex": "Validation AUC: ([0-9\\.]+)"},
        ],
    )

    training_step = TrainingStep(
        name="TrainModel",
        estimator=pytorch_estimator,
        inputs={
            "train": TrainingInput(
                s3_data=training_data,
                content_type="application/x-image",
            ),
            "validation": TrainingInput(
                s3_data=validation_data,
                content_type="application/x-image",
            ),
        },
    )

    return training_step


def create_evaluation_step(
    role: str,
    bucket: str,
    model_data,
    test_data,
    processing_instance_type,
) -> ProcessingStep:
    """
    Create evaluation step for model evaluation.

    Args:
        role: SageMaker execution role
        bucket: S3 bucket for outputs
        model_data: Model artifacts S3 path
        test_data: Test data S3 path
        processing_instance_type: Instance type parameter

    Returns:
        ProcessingStep for evaluation
    """
    sklearn_processor = SKLearnProcessor(
        framework_version="1.2-1",
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name="skin-lesion-evaluation",
        role=role,
    )

    evaluation_step = ProcessingStep(
        name="EvaluateModel",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(
                source=model_data,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=test_data,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation",
                destination=f"s3://{bucket}/evaluation",
            ),
        ],
        code="evaluation/evaluate.py",
    )

    return evaluation_step


def create_register_model_step(
    model_data,
    model_package_group_name,
    model_approval_status,
    evaluation_report: PropertyFile,
) -> RegisterModel:
    """
    Create model registration step.

    Args:
        model_data: Model artifacts S3 path
        model_package_group_name: Model package group name parameter
        model_approval_status: Model approval status parameter
        evaluation_report: Evaluation report property file

    Returns:
        RegisterModel step
    """
    # Define model metrics
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=evaluation_report,
            content_type="application/json",
        )
    )

    register_step = RegisterModel(
        name="RegisterModel",
        estimator=None,  # Will use model_data directly
        model_data=model_data,
        content_types=["application/x-image"],
        response_types=["application/json"],
        inference_instances=["ml.m5.xlarge", "ml.m5.2xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )

    return register_step
