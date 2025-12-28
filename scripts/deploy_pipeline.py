"""
Script to deploy SageMaker Pipeline

This script:
1. Reads configuration
2. Creates/updates SageMaker pipeline
3. Optionally starts a pipeline execution
"""

import argparse
import json
import logging
from pathlib import Path
import yaml
import boto3
from typing import Dict, Any

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline.pipeline import SkinLesionPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_terraform_outputs(outputs_path: str) -> Dict[str, Any]:
    """Load Terraform outputs."""
    if not outputs_path or not Path(outputs_path).exists():
        logger.warning("Terraform outputs not found, using config values")
        return {}

    with open(outputs_path, 'r') as f:
        return json.load(f)


def merge_config_with_terraform(
    config: Dict[str, Any],
    terraform_outputs: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge configuration with Terraform outputs."""
    if not terraform_outputs:
        return config

    # Update AWS configuration with Terraform outputs
    if "data_bucket_name" in terraform_outputs:
        config["aws"]["buckets"]["data"] = terraform_outputs["data_bucket_name"]["value"]

    if "models_bucket_name" in terraform_outputs:
        config["aws"]["buckets"]["models"] = terraform_outputs["models_bucket_name"]["value"]

    if "pipeline_bucket_name" in terraform_outputs:
        config["aws"]["buckets"]["pipeline"] = terraform_outputs["pipeline_bucket_name"]["value"]

    if "sagemaker_execution_role_arn" in terraform_outputs:
        config["aws"]["roles"]["sagemaker_execution"] = terraform_outputs["sagemaker_execution_role_arn"]["value"]

    if "training_ecr_repository_url" in terraform_outputs:
        config["aws"]["ecr"]["training_repo"] = terraform_outputs["training_ecr_repository_url"]["value"]

    if "inference_ecr_repository_url" in terraform_outputs:
        config["aws"]["ecr"]["inference_repo"] = terraform_outputs["inference_ecr_repository_url"]["value"]

    return config


def deploy_pipeline(config: Dict[str, Any], start_execution: bool = False):
    """
    Deploy SageMaker pipeline.

    Args:
        config: Configuration dictionary
        start_execution: Whether to start a pipeline execution
    """
    # Extract configuration
    pipeline_name = config["aws"]["sagemaker"]["pipeline_name"]
    role = config["aws"]["roles"]["sagemaker_execution"]
    bucket = config["aws"]["buckets"]["pipeline"]
    region = config["aws"]["region"]
    model_package_group = config["aws"]["sagemaker"]["model_package_group"]

    logger.info(f"Deploying pipeline: {pipeline_name}")
    logger.info(f"Region: {region}")
    logger.info(f"Bucket: {bucket}")
    logger.info(f"Role: {role}")

    # Create pipeline
    pipeline_builder = SkinLesionPipeline(
        pipeline_name=pipeline_name,
        role=role,
        bucket=bucket,
        region=region,
        config=config,
    )

    # Deploy pipeline
    response = pipeline_builder.deploy_pipeline()

    logger.info("Pipeline deployed successfully!")
    logger.info(f"Pipeline ARN: {response['PipelineArn']}")

    # Start execution if requested
    if start_execution:
        logger.info("Starting pipeline execution...")

        execution = pipeline_builder.start_execution(
            parameters={
                "ModelPackageGroupName": model_package_group,
            }
        )

        logger.info(f"Execution started: {execution['execution_arn']}")
        logger.info(f"Execution ID: {execution['execution_id']}")

    return response


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Deploy SageMaker Pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--terraform-outputs",
        type=str,
        default="",
        help="Path to Terraform outputs JSON"
    )
    parser.add_argument(
        "--start",
        action="store_true",
        help="Start pipeline execution after deployment"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deployed without actually deploying"
    )

    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Load Terraform outputs if available
    if args.terraform_outputs:
        logger.info(f"Loading Terraform outputs from {args.terraform_outputs}")
        terraform_outputs = load_terraform_outputs(args.terraform_outputs)
        config = merge_config_with_terraform(config, terraform_outputs)

    # Validate required configuration
    required_fields = [
        ("aws", "roles", "sagemaker_execution"),
        ("aws", "buckets", "pipeline"),
    ]

    for fields in required_fields:
        value = config
        for field in fields:
            value = value.get(field, {})
        if not value:
            logger.error(f"Missing required configuration: {'.'.join(fields)}")
            return 1

    if args.dry_run:
        logger.info("DRY RUN - Would deploy with configuration:")
        logger.info(json.dumps(config, indent=2))
        return 0

    # Deploy pipeline
    try:
        deploy_pipeline(config, start_execution=args.start)
        logger.info("Deployment completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
